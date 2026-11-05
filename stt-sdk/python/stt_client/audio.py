"""Источники аудио: WAV-файл, микрофон."""
import asyncio
import logging
import wave
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)

CHUNK_FRAMES = 1600  # 100 мс @ 16 кГц
TARGET_SAMPLE_RATE = 16000


class AudioSource:
    """Абстрактный источник PCM16 @ 16 кГц mono."""

    async def stream(self) -> AsyncIterator[bytes]:
        """Асинхронный генератор PCM16 чанков."""
        raise NotImplementedError

    @classmethod
    def from_wav(cls, path: str | Path) -> "WavAudioSource":
        """Создать источник из WAV-файла."""
        return WavAudioSource(path)

    @classmethod
    def from_mic(cls) -> "MicAudioSource":
        """Создать источник из микрофона (требует stt-client[mic])."""
        return MicAudioSource()


class WavAudioSource(AudioSource):
    """WAV-файл: проверяет 16 кГц mono, отдаёт чанки по 100 мс."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def stream(self) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()

        # Открыть файл в executor (не блокировать event loop)
        wav = await loop.run_in_executor(
            None, lambda: wave.open(str(self.path), "rb")
        )
        try:
            if wav.getframerate() != TARGET_SAMPLE_RATE:
                raise ValueError(
                    f"Expected {TARGET_SAMPLE_RATE} Hz, got {wav.getframerate()}. "
                    f"Resample before sending (docs/api/websocket.md)."
                )
            if wav.getnchannels() != 1:
                raise ValueError(
                    f"Expected mono, got {wav.getnchannels()} channels."
                )
            if wav.getsampwidth() != 2:
                raise ValueError(
                    f"Expected 16-bit PCM, got {wav.getsampwidth()*8}-bit."
                )
            while True:
                chunk = await loop.run_in_executor(
                    None, wav.readframes, CHUNK_FRAMES
                )
                if not chunk:
                    break
                yield chunk
        finally:
            await loop.run_in_executor(None, wav.close)


class MicAudioSource(AudioSource):
    """Микрофон через sounddevice (опционально, установить stt-client[mic])."""

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate

    async def stream(self) -> AsyncIterator[bytes]:
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError(
                "MicAudioSource requires sounddevice and numpy. "
                "Install: pip install 'stt-client[mic]'"
            )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"sounddevice status: {status}")
            # indata: float32, конвертируем в int16
            pcm16 = (indata * 32767).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(queue.put_nowait, pcm16)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_FRAMES,
            callback=callback,
        ):
            while True:
                yield await queue.get()
