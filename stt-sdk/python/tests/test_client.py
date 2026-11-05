"""Tests for STTClient."""
import asyncio
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from stt_client import STTClient, WavAudioSource, PartialResult, FinalResult
from stt_client.errors import STTError, RetryPolicy


@pytest_asyncio.fixture
async def client():
    return STTClient(
        base_url="ws://localhost:8100/ws/stt",
        token="test-token",
        retry_policy=RetryPolicy(max_attempts=3, base_delay_sec=0.1, max_delay_sec=0.5),
    )


@pytest.mark.asyncio
async def test_wav_source_stream(client, tmp_path):
    """WavAudioSource читает валидный WAV файл."""
    # Создаём минимальный WAV файл (16kHz, mono, 16-bit PCM)
    import wave
    import struct

    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        # 0.1 секунды silence (1600 samples)
        samples = [0] * 1600
        wf.writeframes(struct.pack(f"{len(samples)}h", *samples))

    audio = WavAudioSource(str(wav_path))
    chunks = [chunk async for chunk in audio.stream()]

    assert len(chunks) > 0
    assert all(isinstance(chunk, bytes) for chunk in chunks)


@pytest.mark.asyncio
async def test_wav_source_format_validation(client, tmp_path):
    """WavAudioSource бросает ValueError при неверном формате."""
    import wave
    import struct

    # Создаём WAV с неправильным sample rate (8kHz вместо 16kHz)
    wav_path = tmp_path / "wrong_rate.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)  # Wrong!
        samples = [0] * 800
        wf.writeframes(struct.pack(f"{len(samples)}h", *samples))

    audio = WavAudioSource(str(wav_path))

    with pytest.raises(ValueError, match="16000 Hz"):
        async for _ in audio.stream():
            pass


@pytest.mark.asyncio
async def test_retryable_error(client, tmp_path):
    """STTError с retryable кодом вызывает retry."""
    import wave
    import struct

    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = [0] * 1600
        wf.writeframes(struct.pack(f"{len(samples)}h", *samples))

    # Mock WebSocket, который возвращает retryable error
    call_count = 0

    async def mock_recv():
        nonlocal call_count
        call_count += 1
        # Каждый новый WebSocket: ready, затем error
        if call_count % 2 == 1:
            return json.dumps({"type": "ready", "model": "giga", "sample_rate": 16000})
        return json.dumps({
            "type": "error",
            "code": "pool_unavailable",
            "message": "no free sessions"
        })

    mock_ws = AsyncMock()
    mock_ws.recv = mock_recv
    mock_ws.send = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    with patch("stt_client.client.websockets.connect", return_value=mock_ws):
        audio = WavAudioSource(str(wav_path))

        # retry_policy.max_attempts=3, но error всегда retryable
        # Ожидаем STTError после исчерпания попыток
        with pytest.raises(STTError) as exc_info:
            await client.stream(audio)

        assert exc_info.value.code == "pool_unavailable"
        assert exc_info.value.is_retryable is True


@pytest.mark.asyncio
async def test_close_4401_translates_to_unauthorized(client, tmp_path):
    """Close code 4401 -> STTError code='unauthorized', is_retryable=False."""
    import wave
    import struct
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("1600h", *([0] * 1600)))

    exc = ConnectionClosed(Close(4401, "unauthorized"), None)

    async def mock_recv():
        raise exc

    mock_ws = AsyncMock()
    mock_ws.recv = mock_recv
    mock_ws.send = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    with patch("stt_client.client.websockets.connect", return_value=mock_ws):
        audio = WavAudioSource(str(wav_path))
        with pytest.raises(STTError) as exc_info:
            await client.stream(audio)

        assert exc_info.value.code == "unauthorized"
        assert exc_info.value.is_retryable is False


# ===== v0.1.1 tests =====


def test_timeout_sec_validation():
    """STTClient rejects timeout_sec <= 0 (P1)."""
    with pytest.raises(ValueError, match="timeout_sec must be > 0"):
        STTClient(base_url="ws://x", token="t", timeout_sec=0)
    with pytest.raises(ValueError, match="timeout_sec must be > 0"):
        STTClient(base_url="ws://x", token="t", timeout_sec=-5.0)


def test_timings_typed_dict():
    """FinalResult.timings accepts Timings TypedDict (P2)."""
    r = FinalResult(
        text="x",
        timings={"upstream_final_ms": 100, "audio_bytes_received": 3200},
    )
    assert r.timings is not None
    assert r.timings["upstream_final_ms"] == 100
    assert r.timings["audio_bytes_received"] == 3200


def test_timings_accepts_none():
    """FinalResult.timings is optional (P2)."""
    r = FinalResult(text="x")
    assert r.timings is None


@pytest.mark.asyncio
async def test_on_partial_callback_error_does_not_crash(client, tmp_path):
    """Callback raising an exception does not break stream (P3)."""
    import wave
    import struct

    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("1600h", *([0] * 1600)))

    call_count = 0

    async def mock_recv():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({"type": "ready", "model": "g", "sample_rate": 16000})
        if call_count == 2:
            return json.dumps({"type": "partial", "text": "hello"})
        return json.dumps({"type": "final", "text": "hello world"})

    mock_ws = AsyncMock()
    mock_ws.recv = mock_recv
    mock_ws.send = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    called = []

    def on_partial(partial):
        called.append(partial)
        raise RuntimeError("callback boom")

    with patch("stt_client.client.websockets.connect", return_value=mock_ws):
        final = await client.stream(
            WavAudioSource(str(wav_path)), on_partial=on_partial
        )

    assert final.text == "hello world"
    assert len(called) == 1
