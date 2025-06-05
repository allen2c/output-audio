import queue
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
        audio_queue: "queue.Queue[bytes]",
        *args,
        playlist: typing.List["PlaylistItem"],
        has_played: bool = False,
        is_loaded: bool = False,
        **kwargs,
    ):
        self.audio_queue = audio_queue
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
            print(f"The playlist item {self} is not in the playlist!")
            return False

    @abstractmethod
    def read_to_queue(self) -> None:
        raise NotImplementedError


class OpenAITextToSpeechProducer(PlaylistItem):
    def __init__(
        self,
        audio_queue: "queue.Queue[bytes]",
        *args,
        playlist: typing.List["PlaylistItem"],
        content: str,
        openai_client: typing.Union["openai.OpenAI", "openai.AzureOpenAI"],
        openai_model: str = "gpt-4o-mini-tts",
        openai_voice: str = "alloy",
        **kwargs,
    ):
        super().__init__(audio_queue, *args, playlist=playlist, **kwargs)

        self.content = content
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.openai_voice = openai_voice

    @typing_extensions.override
    def read_to_queue(self) -> None:
        with self.openai_client.audio.speech.with_streaming_response.create(
            input=self.content,
            model=self.openai_model,
            voice=self.openai_voice,
            response_format="pcm",
        ) as resp:
            for chunk in resp.iter_bytes(chunk_size=4096):
                while not self.all_previous_loaded():
                    time.sleep(0.1)
                self.audio_queue.put(chunk)
        self.audio_queue.put(b"\x00" * int(SAMPLE_RATE * 0.05) * 2)

        self.set_loaded(True)


def pa_callback_builder(audio_q: "queue.Queue[bytes]", leftover: "bytearray"):

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

    return pa_callback


def output_audio(
    playlist: typing.List[PlaylistItem],
    *,
    audio_queue: typing.Optional["queue.Queue[bytes]"] = None,
    leftover: typing.Optional["bytearray"] = None,
):
    audio_queue = queue.Queue(maxsize=200) if audio_queue is None else audio_queue
    leftover = bytearray() if leftover is None else leftover

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_FRAMES,
        latency="low",
        callback=pa_callback_builder(audio_queue, leftover),
    ):
        # TODO: add a thread to read from the audio queue
        while leftover or not audio_queue.empty():
            time.sleep(0.01)
