"""
Stream Player - Low-latency sequential TTS audio streaming
=========================================================

A refactored audio streaming module that provides:
* Strict sequential playback order
* Immediate playback start (as soon as first PCM bytes arrive)
* Background pre-streaming of subsequent items for zero-gap playback
* Pydantic v2 data models for configuration
* Producer-Merger-Player architecture for optimal performance

Dependencies:
    pip install pydantic numpy sounddevice openai typing_extensions

Usage:
    from openai import OpenAI
    from output_audio import PlaylistItem, play_playlist

    client = OpenAI(api_key="...")
    items = [
        PlaylistItem(idx=0, content="Hello world"),
        PlaylistItem(idx=1, content="This is a test"),
    ]
    play_playlist(items, client)
"""

from __future__ import annotations

import queue
import threading
import time
from enum import Enum
from typing import List, Sequence, Union

import numpy as np
import openai
import sounddevice as sd
from pydantic import BaseModel, ConfigDict, Field

# ──────────────────────────────────────────────────────────────
# Audio Configuration Constants
# ──────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 24_000  # Hz (matches OpenAI PCM output)
CHANNELS: int = 1  # Mono audio
DTYPE: str = "int16"  # 16-bit PCM samples
BLOCK_FRAMES: int = 1024  # PortAudio callback buffer size
CHUNK_BYTES: int = 4096  # TTS HTTP chunk size

# Audio padding for seamless transitions
ITEM_SILENCE: bytes = b"\x00" * int(SAMPLE_RATE * 0.05) * 2  # 50ms between items
FINAL_SILENCE: bytes = b"\x00" * int(SAMPLE_RATE * 0.5) * 2  # 500ms at end


# ──────────────────────────────────────────────────────────────
# Pydantic v2 Data Models
# ──────────────────────────────────────────────────────────────


class ItemState(str, Enum):
    """Playback state for monitoring (not used for synchronization)."""

    IDLE = "idle"
    STREAMING = "streaming"
    DONE = "done"


class TTSConfig(BaseModel):
    """OpenAI Text-to-Speech configuration parameters."""

    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    speed: float = 1.3
    instructions: str = (
        "Speak in clear Mandarin with a light Taiwanese accent. "
        "Keep a calm, neutral tone, avoid dramatic pitch swings. "
        "Pretend you are a newscaster reading headlines."
    )


class PlaylistItem(BaseModel):
    """A single text segment to be converted to speech and played."""

    idx: int = Field(..., description="Zero-based position in playlist")
    content: str = Field(..., description="Text content to speak")
    tts_config: TTSConfig = Field(default_factory=TTSConfig)
    state: ItemState = Field(default=ItemState.IDLE)

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


# ──────────────────────────────────────────────────────────────
# Producer Thread (One per PlaylistItem)
# ──────────────────────────────────────────────────────────────


def _run_audio_producer(
    item: PlaylistItem,
    item_queue: "queue.Queue[bytes | None]",
    openai_client: Union["openai.OpenAI", "openai.AzureOpenAI"],
) -> None:
    """
    Streams TTS audio for a single item into its dedicated queue.

    Args:
        item: The playlist item to process
        item_queue: Private queue for this item's audio data
        openai_client: Configured OpenAI client
    """
    # Update state for monitoring (non-blocking)
    item.state = ItemState.STREAMING

    try:
        # Stream TTS audio from OpenAI
        with openai_client.audio.speech.with_streaming_response.create(
            input=item.content,
            model=item.tts_config.model,
            voice=item.tts_config.voice,
            instructions=item.tts_config.instructions,
            response_format="pcm",  # Raw PCM for direct playback
            speed=item.tts_config.speed,
        ) as response:
            # Stream chunks directly to queue as they arrive
            for chunk in response.iter_bytes(chunk_size=CHUNK_BYTES):
                item_queue.put(chunk)

    except Exception as exc:
        # On error, inject silence to keep playlist flowing
        print(f"[Producer {item.idx}] Error: {exc!r}")
        item_queue.put(ITEM_SILENCE)

    finally:
        # Signal completion with sentinel value
        item_queue.put(None)  # Merger will recognize this as end-of-item
        item.state = ItemState.DONE


# ──────────────────────────────────────────────────────────────
# Merger Thread (Sequential Order Controller)
# ──────────────────────────────────────────────────────────────


def _run_audio_merger(
    global_queue: "queue.Queue[bytes]",
    item_queues: Sequence["queue.Queue[bytes | None]"],
) -> None:
    """
    Merges individual item queues into global playback queue in order.

    This ensures strict sequential playback while allowing parallel
    background streaming of subsequent items.

    Args:
        global_queue: Main audio queue consumed by PortAudio
        item_queues: Per-item queues in playlist order
    """
    for item_idx, item_queue in enumerate(item_queues):
        while True:
            chunk = item_queue.get()  # Blocks until data available

            if chunk is None:  # End-of-item sentinel
                # Add brief silence between items to prevent audio artifacts
                global_queue.put(ITEM_SILENCE)
                break

            # Forward audio chunk to global playback queue
            global_queue.put(chunk)

    # Add final silence to ensure complete playback
    global_queue.put(FINAL_SILENCE)


# ──────────────────────────────────────────────────────────────
# PortAudio Callback Builder
# ──────────────────────────────────────────────────────────────


def _create_audio_callback(
    global_queue: "queue.Queue[bytes]",
    buffer_remainder: bytearray,
    playback_started: threading.Event,
):
    """
    Creates PortAudio callback function for real-time audio output.

    Args:
        global_queue: Source of audio data
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
                audio_buffer.extend(global_queue.get_nowait())
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


# ──────────────────────────────────────────────────────────────
# Main Playback Function
# ──────────────────────────────────────────────────────────────


def play_playlist(
    items: List[PlaylistItem],
    openai_client: Union["openai.OpenAI", "openai.AzureOpenAI"],
) -> None:
    """
    Plays a list of TTS items with minimal latency and zero gaps.

    This function:
    1. Starts all TTS producers in parallel (background streaming)
    2. Begins playback as soon as first audio bytes arrive
    3. Maintains strict sequential order via the merger
    4. Blocks until all audio has finished playing

    Args:
        items: List of playlist items to play in order
        openai_client: Configured OpenAI client for TTS
    """
    if not items:
        return

    # Create individual queues for each playlist item
    item_queues = [queue.Queue(maxsize=150) for _ in items]

    # Start producer threads - all run in parallel for background streaming
    producer_threads = []
    for item, item_queue in zip(items, item_queues):
        thread = threading.Thread(
            target=_run_audio_producer,
            args=(item, item_queue, openai_client),
            name=f"Producer-{item.idx}",
            daemon=True,
        )
        producer_threads.append(thread)
        thread.start()

    # Global queue that feeds the audio output device
    global_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=300)

    # Start merger thread - maintains sequential order
    merger_thread = threading.Thread(
        target=_run_audio_merger,
        args=(global_queue, item_queues),
        name="Merger",
        daemon=True,
    )
    merger_thread.start()

    # Set up audio output
    buffer_remainder = bytearray()
    playback_started = threading.Event()

    # Start PortAudio stream - this begins immediate playback
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_FRAMES,
        latency="low",  # Minimize audio latency
        callback=_create_audio_callback(
            global_queue, buffer_remainder, playback_started
        ),
    ):
        # Wait for first audio to start playing
        playback_started.wait()

        # Wait for merger to finish processing all items
        merger_thread.join()

        # Continue playing until all audio data is consumed
        while not global_queue.empty() or buffer_remainder:
            time.sleep(0.01)

        # Allow hardware buffer to drain completely
        time.sleep(1.0)

    # Clean up producer threads
    for thread in producer_threads:
        thread.join(timeout=0.1)


# ──────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────


def create_playlist(texts: Sequence[str], **tts_kwargs) -> List[PlaylistItem]:
    """
    Convenience function to create a playlist from text strings.

    Args:
        texts: Sequence of text strings to convert to speech
        **tts_kwargs: Optional TTS configuration overrides

    Returns:
        List of configured PlaylistItem objects
    """
    tts_config = TTSConfig(**tts_kwargs) if tts_kwargs else TTSConfig()

    return [
        PlaylistItem(idx=i, content=text, tts_config=tts_config)
        for i, text in enumerate(texts)
    ]


# ──────────────────────────────────────────────────────────────
# Demo / Testing
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🎵 Demo Mode: Playing sine wave instead of TTS")
    print("   (Replace with real OpenAI client for actual TTS)\n")

    # Demo with synthetic audio instead of TTS
    import math

    def _demo_producer(item: PlaylistItem, q: "queue.Queue[bytes | None]", _client):
        """Generate a 440Hz sine wave for 2 seconds instead of TTS."""
        item.state = ItemState.STREAMING

        frequency = 440 + (item.idx * 110)  # Different pitch per item
        duration_seconds = 2
        total_samples = int(SAMPLE_RATE * duration_seconds)

        try:
            for start_sample in range(0, total_samples, CHUNK_BYTES // 2):
                chunk_samples = min(CHUNK_BYTES // 2, total_samples - start_sample)

                # Generate sine wave samples
                samples = []
                for i in range(chunk_samples):
                    t = (start_sample + i) / SAMPLE_RATE
                    amplitude = int(0.3 * 32767 * math.sin(2 * math.pi * frequency * t))
                    samples.append(amplitude)

                # Convert to bytes and queue
                audio_data = np.array(samples, dtype=np.int16).tobytes()
                q.put(audio_data)

        finally:
            q.put(None)  # End sentinel
            item.state = ItemState.DONE

    # Replace the real producer for demo
    globals()["_run_audio_producer"] = _demo_producer

    # Create demo playlist
    demo_items = create_playlist(
        [
            "First item (440 Hz)",
            "Second item (550 Hz)",
            "Third item (660 Hz)",
        ]
    )

    # Play demo
    play_playlist(demo_items, openai_client=None)  # type: ignore[arg-type]
    print("✅ Demo completed!")
