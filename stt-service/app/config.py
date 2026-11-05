"""Configuration for STT Gateway using Pydantic Settings."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Gateway settings loaded from environment variables."""

    # GigaSTT upstream URLs
    GIGASTT_WS_URL: str = Field(
        default="ws://127.0.0.1:9876/v1/ws",
        description="WebSocket URL for GigaSTT upstream"
    )
    GIGASTT_REST_URL: str = Field(
        default="http://127.0.0.1:9876/v1/transcribe",
        description="REST API URL for GigaSTT upstream"
    )
    GIGASTT_HEALTH_URL: str = Field(
        default="http://127.0.0.1:9876/health",
        description="Health check URL for GigaSTT upstream"
    )
    GIGASTT_READY_URL: str = Field(
        default="http://127.0.0.1:9876/ready",
        description="Readiness URL for GigaSTT pool check"
    )

    # Gateway binding
    GATEWAY_HOST: str = Field(default="0.0.0.0", description="Host to bind")
    GATEWAY_PORT: int = Field(default=8000, ge=1, le=65535, description="Port to bind")

    # Authentication
    AUTH_MODE: Literal["api_key", "jwt", "none"] = Field(
        default="api_key",
        description="Authentication mode"
    )
    API_KEY: str | None = Field(default=None, description="API key for api_key mode")
    JWT_SECRET: str | None = Field(default=None, description="JWT secret for jwt mode")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_AUDIENCE: str | None = Field(default=None, description="JWT audience")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Log level")

    # Limits
    MAX_CONCURRENT_SESSIONS: int = Field(
        default=20, ge=1, description="Maximum concurrent WebSocket sessions"
    )
    MAX_UPLOAD_MB: int = Field(default=10, ge=1, description="Max REST upload size in MB")
    SESSION_IDLE_TIMEOUT_SEC: int = Field(
        default=30, ge=5, description="Idle timeout for WebSocket sessions"
    )

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated CORS origins"
    )

    # Public hostname for docs/version
    PUBLIC_HOSTNAME: str = Field(default="localhost", description="Public hostname")

    @field_validator("CORS_ORIGINS")
    @classmethod
    def parse_origins(cls, v: str) -> str:
        """Validate CORS origins format."""
        origins = [o.strip() for o in v.split(",")]
        if not origins or any(not o for o in origins):
            raise ValueError("CORS_ORIGINS must contain at least one non-empty origin")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return v.upper()

    def validate_auth(self) -> None:
        """Validate authentication configuration. Raises ValueError if invalid."""
        if self.AUTH_MODE == "api_key" and not self.API_KEY:
            raise ValueError("API_KEY is required when AUTH_MODE=api_key")
        if self.AUTH_MODE == "jwt" and not self.JWT_SECRET:
            raise ValueError("JWT_SECRET is required when AUTH_MODE=jwt")
        if self.AUTH_MODE == "jwt" and not self.JWT_AUDIENCE:
            raise ValueError("JWT_AUDIENCE is required when AUTH_MODE=jwt")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
