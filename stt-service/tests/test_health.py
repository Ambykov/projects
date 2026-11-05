"""Tests for health check endpoints."""

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ["AUTH_MODE"] = "none"
os.environ["API_KEY"] = ""


import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.main import create_app


class TestHealthEndpoints:
    """Tests for /health, /ready, /version endpoints."""

    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        with TestClient(app) as client:
            yield client

    def test_health_ok(self, client):
        """/health should return ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_ok(self, client):
        """/ready should return ready when upstream is available."""
        # Mock httpx.AsyncClient.get at the module level
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "model": "test"}

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["upstream"] == "gigastt"

    def test_ready_upstream_unavailable(self, client):
        """/ready should return not_ready when upstream is down."""
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"
            assert data["upstream"] == "unavailable"

    def test_version(self, client):
        """/version should return version info."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "2.22.0"}

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/version")
            assert response.status_code == 200
            data = response.json()
            assert "gateway" in data
            assert data["public_hostname"] is not None
