# /Users/allenchou/works/ac/output-audio/output_audio/__init__.py
import queue
import threading
import time
import typing
from enum import Enum

import numpy as np
import openai
import pydantic
import sounddevice as sd
import typing_extensions

# Audio Configuration Constants
SAMPLE_RATE: int = 24_000  # Hz (matches OpenAI PCM output)
CHANNELS: int = 1  # Mono audio
DTYPE: str = "int16"  # 16-bit PCM samples
BLOCK_FRAMES: int = 1024  # PortAudio callback buffer size
CHUNK_BYTES: int = 4096  # TTS HTTP chunk size

# Pre-buffering configuration
PRE_BUFFER_DURATION: float = 0.2  # Seconds of audio to buffer before playback starts

# Audio padding for seamless transitions
ITEM_SILENCE: bytes = b"\x00" * int(SAMPLE_RATE * 0.05) * 2  # 50ms between items
FINAL_SILENCE: bytes = b"\x00" * int(SAMPLE_RATE * 0.2) * 2  # 200ms at end


class ItemState(str, Enum):
    """Playback state for monitoring (not used for synchronization)."""

    IDLE = "idle"
    STREAMING = "streaming"
    DONE = "done"


class AudioConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True
    )


class TTSAudioConfig(AudioConfig):
    # model: str = "tts-1"
    model: str = "gpt-4o-mini-tts"
    voice: str = "nova"
    speed: float = 1.0
    instructions: str = (
        "Voice: Warm, upbeat, and reassuring, with a steady and confident cadence that keeps the conversation calm and productive. "  # noqa: E501
        "Speak at a *very fast* pace while maintaining clarity and emotional warmth. "
        "Tone: Positive and solution-oriented, always focusing on the next steps rather than dwelling on the problem. "  # noqa: E501
        "Dialect: Neutral and professional, avoiding overly casual speech but maintaining a friendly and approachable style."  # noqa: E501
    ).strip()
    openai_client: typing.Union["openai.OpenAI", "openai.AzureOpenAI"] = pydantic.Field(
        default_factory=lambda: openai.OpenAI()
    )


class AudioItem(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(validate_assignment=True)

    audio_config: AudioConfig | None = None

    def read(
        self,
        chunk_size: int = CHUNK_BYTES,
    ) -> typing.Generator[bytes, None, None]:
        raise NotImplementedError


class OpenAITTSAudioItem(AudioItem):
    model_config = pydantic.ConfigDict(validate_assignment=True)

    audio_config: TTSAudioConfig | None = None

    content: str

    @typing_extensions.override
    def read(
        self,
        chunk_size: int = CHUNK_BYTES,
    ) -> typing.Generator[bytes, None, None]:
        audio_config = (
            TTSAudioConfig() if self.audio_config is None else self.audio_config
        )
        openai_client = audio_config.openai_client

        with openai_client.audio.speech.with_streaming_response.create(
            input=self.content,
            model=audio_config.model,
            voice=audio_config.voice,
            instructions=audio_config.instructions,
            response_format="pcm",  # Raw PCM for direct playback
            speed=audio_config.speed,
        ) as response:
            # Stream chunks directly to queue as they arrive
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                yield chunk


class PlaylistItem(pydantic.BaseModel):
    """A single text segment to be converted to speech and played."""

    idx: int = pydantic.Field(..., description="Zero-based position in playlist")
    audio_item: AudioItem = pydantic.Field(..., description="Audio item to play")
    audio_queue: "queue.Queue[bytes | None]" = pydantic.Field(
        default_factory=lambda: queue.Queue(maxsize=150)
    )
    state: ItemState = pydantic.Field(default=ItemState.IDLE)

    model_config = pydantic.ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True
    )


class Playlist(pydantic.BaseModel):
    items: typing.List[PlaylistItem] = pydantic.Field(
        default_factory=list, description="List of playlist items"
    )
    audio_producer_threads: typing.List[threading.Thread] = pydantic.Field(
        default_factory=list, description="List of audio producer threads"
    )
    start_event: threading.Event = pydantic.Field(
        default_factory=threading.Event,
        description="Event to signal when playback should start",
    )
    stop_event: threading.Event = pydantic.Field(
        default_factory=threading.Event,
        description="Event to signal when playback should stop",
    )
    current_idx: int = pydantic.Field(default=0, description="Next item to play")

    model_config = pydantic.ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True
    )

    def __iter__(self) -> typing.Generator[PlaylistItem, None, None]:
        for item in self.items:
            yield item

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PlaylistItem:
        return self.items[idx]

    def __setitem__(self, idx: int, item: PlaylistItem) -> None:
        self.items[idx] = item

    def __delitem__(self, idx: int) -> None:
        del self.items[idx]

    def add_item(self, audio_item: AudioItem) -> None:
        """Dynamically add an item to the playlist."""

        item = PlaylistItem(idx=len(self.items), audio_item=audio_item)
        self.items.append(item)

        if self.start_event.is_set():
            thread = threading.Thread(target=self._run_audio_producer, args=(item,))
            self.audio_producer_threads.append(thread)
            thread.start()

    def play(
        self,
        playback_queue: "queue.Queue[bytes | None]",
        *,
        max_playback_time: float = 10.0,
    ) -> None:
        """
        Merge individual item queues into a single playback_queue
        while allowing dynamic insertion of new items.
        """
        self.start_audio_producer()
        start_time = time.time()

        try:
            while True:
                # Wait until at least one un-played item is available
                while self.current_idx >= len(self.items):
                    if self.stop_event.is_set():
                        raise ManualStopException("Stop requested")

                    if time.time() - start_time > max_playback_time:
                        raise ManualStopException("Playback timed out")

                    time.sleep(0.05)  # small poll-sleep; keeps CPU low

                # Consume the next item
                audio_item = self.items[self.current_idx]
                self.current_idx += 1  # advance for the next loop

                while True:
                    if self.stop_event.is_set():
                        raise ManualStopException("Stop requested")

                    chunk = audio_item.audio_queue.get()

                    if chunk is None:  # end-of-item sentinel
                        playback_queue.put(ITEM_SILENCE)
                        break

                    playback_queue.put(chunk)

            # Add final silence once every queued item has finished
            playback_queue.put(FINAL_SILENCE)

        except ManualStopException:
            self.on_stop()

    def start_audio_producer(self):
        if self.start_event.is_set():
            return

        self.start_event.set()
        self.stop_event.clear()

        for item in self.items:
            thread = threading.Thread(target=self._run_audio_producer, args=(item,))
            self.audio_producer_threads.append(thread)
            thread.start()

    def on_stop(self):
        self.stop_event.set()
        for thread in self.audio_producer_threads:
            thread.join(timeout=0.1)
        # Ensure all threads have completed
        for thread in self.audio_producer_threads:
            if thread.is_alive():
                thread.join()

        for item in self.items:
            while not item.audio_queue.empty():
                item.audio_queue.get()
                item.state = ItemState.IDLE

        self.audio_producer_threads.clear()
        self.start_event.clear()
        self.stop_event.clear()
        self.current_idx = 0

    def _run_audio_producer(self, item: PlaylistItem) -> None:
        """
        Streams TTS audio for a single item into its dedicated queue.
        """
        # Update state for monitoring (non-blocking)
        item.state = ItemState.STREAMING

        # Pre-buffering mechanism: accumulate initial data before playback
        initial_buffer = []
        # Calculate buffer size based on audio parameters
        bytes_per_sample = 2  # 16-bit PCM
        initial_buffer_size = int(
            SAMPLE_RATE * CHANNELS * bytes_per_sample * PRE_BUFFER_DURATION
        )
        current_buffer_size = 0

        try:
            for chunk in item.audio_item.read(chunk_size=CHUNK_BYTES):
                if self.stop_event.is_set():
                    break

                # Accumulate initial buffer
                if current_buffer_size < initial_buffer_size:
                    initial_buffer.append(chunk)
                    current_buffer_size += len(chunk)

                    # Release all buffered data once we have enough
                    if current_buffer_size >= initial_buffer_size:
                        for buffered_chunk in initial_buffer:
                            item.audio_queue.put(buffered_chunk)
                        initial_buffer = []  # Clear buffer list
                else:
                    # Initial buffer already released, put directly to queue
                    item.audio_queue.put(chunk)

        except Exception as exc:
            # On error, inject silence to keep playlist flowing
            print(f"[Producer {item.idx}] Error: {exc!r}")
            # Release any remaining buffered data
            for buffered_chunk in initial_buffer:
                item.audio_queue.put(buffered_chunk)
            item.audio_queue.put(ITEM_SILENCE)

        finally:
            # Ensure all buffered data is released
            for buffered_chunk in initial_buffer:
                item.audio_queue.put(buffered_chunk)
            # Signal completion with sentinel value
            item.audio_queue.put(None)  # Merger will recognize this as end-of-item
            item.state = ItemState.DONE


class ManualStopException(Exception):
    pass


# ──────────────────────────────────────────────────────────────
# PortAudio Callback Builder
# ──────────────────────────────────────────────────────────────


def create_audio_callback(
    playback_queue: "queue.Queue[bytes]",
    buffer_remainder: bytearray,
    playback_started: threading.Event,
):
    """
    Creates PortAudio callback function for real-time audio output.

    Args:
        playback_queue: Source of audio data
        buffer_remainder: Carries over partial frames between callbacks
        playback_started: Event signaled when actual audio begins

    Returns:
        Configured callback function for PortAudio
    """

    def audio_callback(outdata, frames, time_info, status):
        bytes_needed = frames * CHANNELS * 2  # 16-bit samples = 2 bytes each

        # Start with any leftover data from previous callback
        audio_buffer = bytearray(buffer_remainder)
        buffer_remainder.clear()

        # Fill buffer from queue until we have enough data
        while len(audio_buffer) < bytes_needed:
            try:
                audio_buffer.extend(playback_queue.get_nowait())
            except queue.Empty:
                break  # No more data available right now

        # Handle buffer size mismatches
        if len(audio_buffer) < bytes_needed:
            # Pad with silence if insufficient data
            audio_buffer.extend(b"\x00" * (bytes_needed - len(audio_buffer)))
        elif len(audio_buffer) > bytes_needed:
            # Save excess for next callback
            buffer_remainder.extend(audio_buffer[bytes_needed:])
            audio_buffer = audio_buffer[:bytes_needed]

        # Convert to numpy array and output
        outdata[:] = np.frombuffer(audio_buffer, np.int16).reshape(-1, CHANNELS)

        # Signal when first real audio (non-silence) starts playing
        if not playback_started.is_set() and any(audio_buffer):
            playback_started.set()

    return audio_callback


def output_audio(audio_items: typing.Sequence[AudioItem]) -> None:
    if not audio_items:
        return

    # Create playlist
    playlist = Playlist()
    for audio_item in audio_items:
        playlist.add_item(audio_item)

    # Start audio producers
    playlist.start_audio_producer()

    # Global queue that feeds the audio output device
    playback_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=300)

    # Start playlist playback
    playlist_playback_thread = threading.Thread(
        target=playlist.play,
        args=(playback_queue,),
        name="Playlist Playback",
        daemon=True,
    )
    playlist_playback_thread.start()

    # Set up audio output
    buffer_remainder = bytearray()
    playback_started = threading.Event()

    # Start PortAudio stream - this begins immediate playback
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_FRAMES,
        latency=0.2,  # 200ms latency works with pre-buffering mechanism
        callback=create_audio_callback(
            playback_queue, buffer_remainder, playback_started
        ),
    ):
        # Wait for first audio to start playing
        playback_started.wait()

        # Wait for playlist playback to finish
        playlist_playback_thread.join()

        # Continue playing until all audio data is consumed
        while not playback_queue.empty() or buffer_remainder:
            time.sleep(0.01)

        # Allow hardware buffer to drain completely
        time.sleep(0.5)

    # Clean up producer threads
    playlist.on_stop()


def output_playlist_audio(
    playlist: Playlist,
    *,
    playback_stop_event: threading.Event | None = None,
) -> None:
    # Start audio producers
    playlist.start_audio_producer()

    # Global queue that feeds the audio output device
    playback_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=300)

    # Start playlist playback
    playlist_playback_thread = threading.Thread(
        target=playlist.play,
        args=(playback_queue,),
        name="Playlist Playback",
        daemon=True,
    )
    playlist_playback_thread.start()

    # Set up audio output
    buffer_remainder = bytearray()
    playback_started = threading.Event()

    # Start PortAudio stream - this begins immediate playback
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_FRAMES,
        latency=0.2,  # 200ms latency works with pre-buffering mechanism
        callback=create_audio_callback(
            playback_queue, buffer_remainder, playback_started
        ),
    ):
        # Wait for first audio to start playing
        playback_started.wait()

        # Wait for playlist playback to finish
        playlist_playback_thread.join()

        # Continue playing until all audio data is consumed
        while not playback_queue.empty() or buffer_remainder:
            time.sleep(0.01)

        if playback_stop_event is not None:
            playback_stop_event.wait()
        else:
            # Allow hardware buffer to drain completely
            time.sleep(0.5)

    # Clean up producer threads
    playlist.on_stop()
