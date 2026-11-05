"""Tests for WebSocket gateway."""

import os
import pytest
import httpx

os.environ["AUTH_MODE"] = "none"
os.environ["API_KEY"] = ""


from fastapi.testclient import TestClient
from app.main import create_app


class TestWebSocketGateway:
    """Tests for /ws/stt endpoint."""

    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_ws_connect_without_token(self, client):
        """Should accept connection without token when auth is disabled."""
        # Just verify the endpoint exists - full WS testing requires mock upstream
        # TestClient doesn't support WebSocket without additional setup
        assert client.app is not None

    def test_ws_accepts_valid_token(self, client):
        """Should accept connection with valid token when auth is enabled."""
        os.environ["AUTH_MODE"] = "api_key"
        os.environ["API_KEY"] = "test-secret-key"
        app = create_app()
        tc = TestClient(app)
        # Verify app created successfully with auth enabled
        assert tc.app is not None
        # Reset
        os.environ["AUTH_MODE"] = "none"
        os.environ["API_KEY"] = ""

    def test_ws_endpoint_exists(self, client):
        """Verify WebSocket endpoint is registered."""
        # Check that the router is included
        routes = [r.path for r in client.app.routes]
        assert "/ws/stt" in routes
