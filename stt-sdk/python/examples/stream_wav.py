#!/usr/bin/env python3
"""Пример: транскрипция WAV-файла через STTClient."""
import asyncio
import logging
import os
import sys
from pathlib import Path

from stt_client import STTClient, WavAudioSource, PartialResult, STTError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python stream_wav.py <path/to/audio.wav>")
        sys.exit(1)

    wav_path = Path(sys.argv[1]).resolve()
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        sys.exit(1)

    client = STTClient(
        base_url="ws://localhost:8100/ws/stt",
        token=os.environ.get("STT_API_KEY", "testtoken"),
    )

    def on_partial(partial: PartialResult):
        print(f"partial: {partial.text}")

    try:
        audio = WavAudioSource(str(wav_path))
        result = await client.stream(audio, on_partial=on_partial)
        print(f"final: {result.text}")
        if result.timings:
            print(f"timings: {result.timings}")
    except STTError as e:
        logger.error(f"STT error: {e.code} — {e.message}")
        raise
    except Exception as e:
        logger.error(f"error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
