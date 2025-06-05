import queue
import threading
import time
import typing
from abc import ABC, abstractmethod

import numpy as np
import openai
import sounddevice as sd
import typing_extensions

SAMPLE_RATE = 24_000
CHANNELS = 1
BLOCK_FRAMES = 1024
DTYPE = "int16"


class PlaylistItem(ABC):
    def __init__(
        self,
        *args,
        playlist: typing.List["PlaylistItem"],
        has_played: bool = False,
        is_loaded: bool = False,
        **kwargs,
    ):
        self.playlist = playlist

        self.__has_played = has_played
        self.__is_loaded = is_loaded

    def set_played(self, has_played: bool) -> bool:
        self.__has_played = has_played
        return self.__has_played

    def set_loaded(self, is_loaded: bool) -> bool:
        self.__is_loaded = is_loaded
        return self.__is_loaded

    def all_previous_loaded(self) -> bool:
        for item in self.playlist:
            if item is not self:
                if item.__is_loaded is True:
                    pass
                else:
                    return False
            elif item is self:
                return True

        else:
            # The playlist item is not in the playlist
            return False

    @abstractmethod
    def read_to_queue(self, audio_queue: "queue.Queue[bytes]") -> None:
        raise NotImplementedError


class OpenAITextToSpeechProducer(PlaylistItem):
    def __init__(
        self,
        *args,
        playlist: typing.List["PlaylistItem"],
        content: str,
        openai_client: typing.Union["openai.OpenAI", "openai.AzureOpenAI"],
        openai_model: str = "gpt-4o-mini-tts",
        openai_voice: str = "alloy",
        **kwargs,
    ):
        super().__init__(*args, playlist=playlist, **kwargs)

        self.content = content
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.openai_voice = openai_voice

    @typing_extensions.override
    def read_to_queue(self, audio_queue: "queue.Queue[bytes]") -> None:
        with self.openai_client.audio.speech.with_streaming_response.create(
            input=self.content,
            model=self.openai_model,
            voice=self.openai_voice,
            instructions="Speak in clear Mandarin with a light Taiwanese accent. Keep a calm, neutral tone, avoid dramatic pitch swings. Pretend you are a newscaster reading headlines.",  # noqa: E501
            response_format="pcm",
            speed=1.5,
        ) as resp:
            for chunk in resp.iter_bytes(chunk_size=4096):
                while not self.all_previous_loaded():
                    time.sleep(0.1)
                audio_queue.put(chunk)
        audio_queue.put(b"\x00" * int(SAMPLE_RATE * 0.05) * 2)

        self.set_loaded(True)


def pa_callback_builder(
    audio_q: "queue.Queue[bytes]",
    leftover: "bytearray",
    played_event: threading.Event,
):

    def pa_callback(outdata, frames, time, status):
        need = frames * CHANNELS * 2  # bytes

        chunk = bytearray(leftover)
        leftover.clear()

        while len(chunk) < need:
            try:
                chunk.extend(audio_q.get_nowait())
            except queue.Empty:
                break

        if len(chunk) < need:
            chunk.extend(b"\x00" * (need - len(chunk)))
        elif len(chunk) > need:
            leftover.extend(chunk[need:])  # for next round
            chunk = chunk[:need]

        outdata[:] = np.frombuffer(chunk, np.int16).reshape(-1, CHANNELS)

        # mark the moment real audio hits the device
        if not played_event.is_set() and any(chunk):  # any() skips all-silence
            played_event.set()

    return pa_callback


def output_audio(
    playlist: typing.List[PlaylistItem],
    *,
    audio_queue: typing.Optional["queue.Queue[bytes]"] = None,
    leftover: typing.Optional["bytearray"] = None,
):
    audio_queue = queue.Queue(maxsize=200) if audio_queue is None else audio_queue
    leftover = bytearray() if leftover is None else leftover

    first_bytes_played = threading.Event()
    all_producers_done = threading.Event()
    active_producers = []

    def producer_wrapper(item):
        try:
            item.read_to_queue(audio_queue)
        finally:
            active_producers.remove(threading.current_thread())
            if not active_producers:
                all_producers_done.set()
                # Add silence at the end to ensure all audio plays
                audio_queue.put(b"\x00" * int(SAMPLE_RATE * 0.5) * 2)

    producers: list[threading.Thread] = []
    for item in playlist:
        t = threading.Thread(
            target=producer_wrapper,
            args=(item,),
            daemon=True,
        )
        active_producers.append(t)
        producers.append(t)
        t.start()

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_FRAMES,
        latency="low",
        callback=pa_callback_builder(
            audio_queue, leftover, played_event=first_bytes_played
        ),
    ):
        first_bytes_played.wait()

        # Keep playing until all producers are done AND queue is empty
        while not all_producers_done.is_set() or not audio_queue.empty() or leftover:
            time.sleep(0.01)

        # Extra time to ensure all audio plays through the buffer
        time.sleep(1.0)

    for t in producers:
        t.join(timeout=0.1)
