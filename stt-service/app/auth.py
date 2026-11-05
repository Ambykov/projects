"""Authentication module for STT Gateway."""

import secrets
from typing import Any

import jwt
from fastapi import HTTPException, WebSocket
from fastapi.requests import HTTPConnection

from .config import get_settings

settings = get_settings()


def extract_token_from_ws(connection: HTTPConnection) -> str | None:
    """Extract token from WebSocket connection (query param or authorization header)."""
    # Try query parameter first
    query_params = connection.query_params
    token = query_params.get("token")
    if token:
        return str(token)

    # Try Authorization header
    headers = connection.headers
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]

    return None


def extract_token_from_http(request: Any) -> str | None:
    """Extract token from HTTP request (Authorization header)."""
    auth = request.headers.get("authorization") or ""
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def verify_token(token: str | None) -> dict[str, Any] | None:
    """
    Verify token based on AUTH_MODE.

    Returns payload dict for JWT mode, or None for api_key/none modes.
    Raises HTTPException(401) if authentication fails.
    """
    if settings.AUTH_MODE == "none":
        return None

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    if settings.AUTH_MODE == "api_key":
        # Constant-time comparison
        if not secrets.compare_digest(str(token), settings.API_KEY):
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return None

    if settings.AUTH_MODE == "jwt":
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
                audience=settings.JWT_AUDIENCE,
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidAudienceError:
            raise HTTPException(status_code=401, detail="Invalid token audience")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # Fallback
    raise HTTPException(status_code=401, detail="Authentication not configured")
