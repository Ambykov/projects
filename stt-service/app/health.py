"""Health check endpoints for STT Gateway."""

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    status: str
    upstream: str | None = None


class VersionResponse(BaseModel):
    """Version information response."""
    gateway: str
    gigastt: str | None = None
    public_hostname: str


@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe - returns ok if gateway is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready():
    """Readiness probe - checks if upstream GigaSTT is available."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(settings.GIGASTT_HEALTH_URL)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Upstream healthy: {data.get('model', 'unknown')}")
                return ReadyResponse(status="ready", upstream="gigastt")
    except httpx.TimeoutException:
        logger.warning("Upstream health check timed out")
    except httpx.ConnectError:
        logger.warning("Cannot connect to upstream")
    except Exception as e:
        logger.warning(f"Upstream health check failed: {e}")

    return ReadyResponse(status="not_ready", upstream="unavailable")


@router.get("/version", response_model=VersionResponse)
async def version():
    """Get version information."""
    gigastt_version = None

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(settings.GIGASTT_HEALTH_URL)
            if response.status_code == 200:
                data = response.json()
                gigastt_version = data.get("version")
    except Exception:
        pass

    from . import __version__
    return VersionResponse(
        gateway=__version__,
        gigastt=gigastt_version,
        public_hostname=settings.PUBLIC_HOSTNAME,
    )
