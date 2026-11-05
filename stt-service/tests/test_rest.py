"""Tests for REST API endpoint."""

import os
import io
import pytest
from unittest.mock import patch, MagicMock

# IMPORTANT: AUTH_MODE must be "none" for these tests
# The lru_cache in config.py prevents changing AUTH_MODE at runtime
os.environ["AUTH_MODE"] = "none"
os.environ["API_KEY"] = ""


import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.main import create_app


def create_test_wav(duration_ms=100):
    """Create a minimal valid WAV file in memory."""
    sample_rate = 16000
    num_samples = duration_ms * sample_rate // 1000
    num_bytes = num_samples * 2  # 16-bit mono

    # WAV header
    header = b'RIFF'
    header += (36 + num_bytes).to_bytes(4, 'little')
    header += b'WAVE'
    header += b'fmt '
    header += (16).to_bytes(4, 'little')  # PCM format size
    header += (1).to_bytes(2, 'little')  # PCM format
    header += (1).to_bytes(2, 'little')  # Mono
    header += (sample_rate).to_bytes(4, 'little')
    header += (sample_rate * 2).to_bytes(4, 'little')  # Byte rate
    header += (2).to_bytes(2, 'little')  # Block align
    header += (16).to_bytes(2, 'little')  # Bits per sample
    header += b'data'
    header += (num_bytes).to_bytes(4, 'little')

    # Silence data
    data = b'\x00' * num_bytes

    return header + data


class MockResponse:
    """Mock httpx.Response with headers support."""
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._json_data


class TestRestEndpoint:
    """Tests for /v1/transcribe endpoint."""

    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        with TestClient(app) as client:
            yield client

    def test_transcribe_invalid_wav(self, client):
        """Should return 400 for invalid WAV files."""
        files = {"file": ("test.txt", io.BytesIO(b"not a wav"), "text/plain")}
        response = client.post("/v1/transcribe", files=files)
        # With AUTH_MODE=none, auth passes, but file validation should fail
        assert response.status_code == 400
        assert "Invalid audio" in response.json()["detail"]

    def test_transcribe_file_too_large(self, client):
        """Should return 413 for files exceeding MAX_UPLOAD_MB."""
        large_data = b"X" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large.wav", io.BytesIO(large_data), "audio/wav")}
        response = client.post("/v1/transcribe", files=files)
        # With AUTH_MODE=none, auth passes, but size validation should fail
        assert response.status_code in (413, 400)

    def test_transcribe_success(self, client):
        """Should return transcription result for valid WAV."""
        mock_response = MockResponse(
            status_code=200,
            json_data={
                "text": "тестовая расшифровка",
                "alternatives": [{"text": "тестовая расшифровка", "confidence": 0.95}]
            }
        )

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            wav_data = create_test_wav()
            files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
            response = client.post("/v1/transcribe", files=files)
            assert response.status_code == 200
            data = response.json()
            assert "text" in data

    def test_transcribe_upstream_error(self, client):
        """Should return 502 when upstream is unavailable."""
        with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Upstream unavailable")):
            wav_data = create_test_wav()
            files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
            response = client.post("/v1/transcribe", files=files)
            assert response.status_code in (500, 502)
