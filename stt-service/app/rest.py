"""REST API endpoint for STT transcription."""

import io
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse

from .config import get_settings
from .logging import get_logger
from .auth import extract_token_from_http, verify_token
from . import metrics

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["rest"])


def validate_wav_magic(file_bytes: bytes) -> bool:
    """Validate WAV file magic bytes (RIFF header)."""
    return file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WAVE"


@router.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(None),
    token: str | None = None,  # Query parameter for token
) -> dict[str, Any]:
    """
    Transcribe an audio file via GigaSTT REST API.

    Requires Bearer token in Authorization header or query parameter.
    Supports WAV files up to MAX_UPLOAD_MB.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        # Extract and verify token (query param or Authorization header)
        if token:
            ws_token = token
        elif authorization and authorization.startswith("Bearer "):
            ws_token = authorization[7:]
        else:
            ws_token = None

        try:
            verify_token(ws_token)
        except HTTPException as e:
            metrics.inc_rest_request("error")
            raise e

        # Check file size
        # Read file content
        content = await file.read()
        file_size = len(content)
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

        if file_size > max_bytes:
            logger.warning(
                f"[{request_id}] File too large: {file_size} > {max_bytes}",
                extra={"request_id": request_id, "file_size": file_size},
            )
            metrics.inc_rest_request("error")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_MB}MB",
            )

        # Validate WAV format
        if not validate_wav_magic(content):
            logger.warning(
                f"[{request_id}] Invalid WAV format: {file.filename}",
                extra={"request_id": request_id},
            )
            metrics.inc_rest_request("error")
            raise HTTPException(
                status_code=400,
                detail="Invalid audio format. Expected WAV file.",
            )

        # Forward to upstream
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Re-wrap as bytes for upstream
                upstream_files = {"file": (file.filename, io.BytesIO(content), "audio/wav")}
                response = await client.post(
                    settings.GIGASTT_REST_URL,
                    files=upstream_files,
                )

                if response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Upstream error: {response.status_code}",
                        extra={"request_id": request_id, "status": response.status_code},
                    )
                    metrics.inc_rest_request("error")
                    metrics.inc_upstream_error("rest")
                    raise HTTPException(
                        status_code=502,
                        detail=f"Upstream error: {response.status_code}",
                    )

                result = response.json()
                hypothesis = result.get("text", "")

        except httpx.TimeoutException:
            logger.error(f"[{request_id}] Upstream timeout", extra={"request_id": request_id})
            metrics.inc_rest_request("error")
            metrics.inc_upstream_error("rest")
            raise HTTPException(status_code=504, detail="Upstream timeout")

        duration = time.perf_counter() - start_time
        metrics.observe_rest_duration(duration)
        metrics.inc_rest_request("ok")

        logger.info(
            f"[{request_id}] Transcription complete: {file.filename}, {file_size} bytes, {duration:.2f}s",
            extra={
                "request_id": request_id,
                "file_size": file_size,
                "duration_sec": round(duration, 2),
            },
        )

        return {
            "text": hypothesis,
            "duration_sec": round(duration, 2),
            "latency_ms": round(duration * 1000, 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        metrics.inc_rest_request("error")
        logger.error(
            f"[{request_id}] Transcription error: {e}",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))
