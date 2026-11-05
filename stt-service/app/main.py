"""STT Gateway - FastAPI application factory and entry point."""

import asyncio
import os
import signal
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from . import __version__
from .config import get_settings
from .logging import setup_logging, get_logger
from .health import router as health_router
from .gateway import router as gateway_router, session_lifespan
from .rest import router as rest_router
from .metrics import get_metrics

settings = get_settings()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan events."""
        # Startup
        setup_logging(settings.LOG_LEVEL)
        logger.info(f"Starting STT Gateway v{__version__}")
        logger.info(f"Auth mode: {settings.AUTH_MODE}")
        logger.info(f"Max concurrent sessions: {settings.MAX_CONCURRENT_SESSIONS}")

        # Validate auth configuration
        try:
            settings.validate_auth()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise

        # Check upstream health on startup
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(settings.GIGASTT_HEALTH_URL)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        f"Upstream GigaSTT ready: model={data.get('model', 'unknown')}, version={data.get('version', 'unknown')}"
                    )
                else:
                    logger.warning(f"Upstream returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to upstream: {e}")

        yield

        # Shutdown
        logger.info("Shutting down STT Gateway...")

    app = FastAPI(
        title="STT Gateway",
        description="WebSocket and REST gateway for GigaSTT speech recognition",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(gateway_router)
    app.include_router(rest_router)

    # Mount static files for HTML test page
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Test page
    @app.get("/test")
    async def test_page():
        """HTML test page with microphone."""
        return FileResponse(os.path.join(static_dir, "test.html"))

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return get_metrics()

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "status": "ok",
            "service": "STT Gateway",
            "version": __version__,
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=False,
    )
