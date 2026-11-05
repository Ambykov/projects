"""Tests for authentication module."""

import os
import pytest
from unittest.mock import patch, MagicMock

# Default env - will be overridden in tests via mocks
os.environ["AUTH_MODE"] = "none"
os.environ["API_KEY"] = ""


from fastapi import HTTPException


class TestApiKeyAuth:
    """Tests for API key authentication."""

    def test_valid_api_key(self):
        """Should accept valid API key."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.AUTH_MODE = "api_key"
            mock_settings.API_KEY = "test-api-key-12345"

            from app.auth import verify_token
            result = verify_token("test-api-key-12345")
            assert result is None  # api_key mode returns None on success

    def test_invalid_api_key(self):
        """Should reject invalid API key."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.AUTH_MODE = "api_key"
            mock_settings.API_KEY = "test-api-key-12345"

            from app.auth import verify_token
            with pytest.raises(HTTPException) as exc_info:
                verify_token("wrong-key")
            assert exc_info.value.status_code == 401

    def test_missing_token(self):
        """Should reject missing token."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.AUTH_MODE = "api_key"
            mock_settings.API_KEY = "test-api-key-12345"

            from app.auth import verify_token
            with pytest.raises(HTTPException) as exc_info:
                verify_token(None)
            assert exc_info.value.status_code == 401


class TestTokenExtraction:
    """Tests for token extraction helpers."""

    def test_extract_from_ws_query_param(self):
        """Should extract token from WebSocket query param."""
        from app.auth import extract_token_from_ws
        mock_conn = MagicMock()
        mock_conn.query_params.get.return_value = "test-token"
        mock_conn.headers.get.return_value = None

        result = extract_token_from_ws(mock_conn)
        assert result == "test-token"

    def test_extract_from_ws_authorization_header(self):
        """Should extract token from Authorization header."""
        from app.auth import extract_token_from_ws
        mock_conn = MagicMock()
        mock_conn.query_params.get.return_value = None
        mock_conn.headers.get.return_value = "Bearer header-token"

        result = extract_token_from_ws(mock_conn)
        assert result == "header-token"

    def test_extract_from_http_authorization(self):
        """Should extract token from HTTP Authorization header."""
        from app.auth import extract_token_from_http
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer http-token"

        result = extract_token_from_http(mock_request)
        assert result == "http-token"

    def test_extract_from_http_no_auth(self):
        """Should return None when no Authorization header."""
        from app.auth import extract_token_from_http
        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""

        result = extract_token_from_http(mock_request)
        assert result is None
