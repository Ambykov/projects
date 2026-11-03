"""Integration tests for STTClient against live gateway.

Requires gateway at ws://127.0.0.1:8100/ws/stt.
Token: env STT_API_KEY (fallback: "testtoken" for local dev).
Run with: pytest sdk/python -v --run-integration
"""
import os
from pathlib import Path

import pytest

from stt_client import STTClient, WavAudioSource


# sdk/python/tests/test_integration.py -> parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
WAV_PATH = _REPO_ROOT / "audio-samples" / "commands" / "msc" / "stop" / "2023-06-26-21-14-35.wav"
WS_URL = "ws://127.0.0.1:8100/ws/stt"
TOKEN = os.environ.get("STT_API_KEY", "testtoken")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_real_gateway():
    """Full ready -> partial/final -> stop against live gateway."""
    assert WAV_PATH.exists(), f"WAV file not found: {WAV_PATH}"

    client = STTClient(
        base_url=WS_URL,
        token=TOKEN,
        timeout_sec=30.0,
    )
    audio = WavAudioSource(str(WAV_PATH))
    partials = []

    def on_partial(p):
        partials.append(p)

    result = await client.stream(audio, on_partial=on_partial)

    assert result.text, f"empty final text, partials={partials}"
    assert result.timings is not None
    assert "upstream_final_ms" in result.timings
    assert "стой" in result.text.lower(), f"unexpected final: {result.text!r}"
