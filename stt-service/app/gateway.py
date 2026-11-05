"""WebSocket gateway module - proxies connections to GigaSTT upstream.

Architecture: "one utterance = one upstream connection"

- Client connection to gateway: long-lived for the entire dialogue session
- Upstream connection to GigaSTT: opened per utterance, closed after final

Protocol:
  Client -> Gateway: binary frames (PCM16 audio)
  Client -> Gateway: {"type": "end_of_utterance"} - end of current utterance
  Client -> Gateway: {"type": "stop"} - end of dialogue session
  Gateway -> Client: {"type": "ready", ...} - connection accepted
  Gateway -> Client: {"type": "partial", "text": "..."} - intermediate result
  Gateway -> Client: {"type": "final", "text": "..."} - final result (connection stays open)

Refactored architecture (fixes race condition):
  - Single WebSocket reader: main loop in websocket_stt is the ONLY reader
  - Audio messages are queued in asyncio.Queue
  - client_to_upstream reads from queue (not from WebSocket), can be cancelled cleanly
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi import FastAPI

from .config import get_settings
from .logging import get_logger
from .auth import extract_token_from_ws, verify_token
from . import metrics

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["websocket"])

# Active sessions tracking
active_sessions: set[str] = set()
shutdown_event = asyncio.Event()


@asynccontextmanager
async def session_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage gateway lifespan - graceful shutdown."""
    logger.info(f"Starting STT Gateway on {settings.GATEWAY_HOST}:{settings.GATEWAY_PORT}")
    logger.info(f"Auth mode: {settings.AUTH_MODE}")
    logger.info(f"Max concurrent sessions: {settings.MAX_CONCURRENT_SESSIONS}")

    yield

    # Graceful shutdown
    logger.info("Shutting down gateway...")
    shutdown_event.set()

    # Close all active sessions
    for session_id in list(active_sessions):
        logger.info(f"Closing session {session_id}")

    logger.info(f"Gateway shutdown complete. {len(active_sessions)} sessions remaining.")


async def process_utterance(
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    session_id: str,
    request_id: str,
    rep_id: str,
    rep_start_time: float,
    first_audio_chunk: bytes | None = None,
) -> tuple[str | None, float | None, float | None, int, dict | None]:
    """
    Process a single utterance: open upstream, proxy audio concurrently, close upstream after final.

    Architecture:
    - client_to_upstream: reads audio from queue, forwards to upstream (can be cancelled cleanly)
    - upstream_to_client: receives partial/final from upstream, forwards to client

    The WebSocket is read ONLY by the main loop in websocket_stt, which puts audio into the queue.
    This ensures that cancelling client_to_upstream doesn't leave WebSocket in a bad state.

    Returns: (error_reason, ttfp_ms, ttft_ms, partial_count, next_message, final_text)
    """
    upstream = None
    ttfp_ms = None
    ttft_ms = None
    partial_count = 0
    error_reason = None
    upstream_open_time = time.perf_counter()
    next_message = None
    final_text = ""
    stop_sent = False

    # Server timings
    timings = {
        "upstream_connect_ms": None,
        "upstream_first_byte_ms": None,
        "upstream_final_ms": None,
        "audio_bytes_received": 0,
    }
    t_connect_start = time.perf_counter()

    try:
        # Check pool availability before connecting
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(settings.GIGASTT_READY_URL, timeout=2.0)
                ready_data = r.json()
                if ready_data.get("pool_available", 1) <= 0:
                    logger.warning(
                        f"[{request_id}] [{rep_id}] pool_unavailable: {ready_data}",
                        extra={"request_id": request_id, "rep_id": rep_id, "event": "pool_unavailable"},
                    )
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "code": "pool_unavailable",
                            "message": "GigaSTT pool is exhausted, retry in a few seconds",
                        }))
                    except Exception as send_err:
                        logger.warning(f"[{request_id}] [{rep_id}] Failed to send error to client: {send_err}")
                    return ("pool_unavailable", None, None, 0, None, "")
        except Exception as check_err:
            logger.error(
                f"[{request_id}] [{rep_id}] Pool check failed: {check_err}",
                extra={"request_id": request_id, "rep_id": rep_id, "event": "pool_check_failed"},
            )
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "code": "upstream_connect_failed",
                    "message": f"GigaSTT unreachable: {str(check_err) or 'connection refused'}",
                }))
            except Exception as send_err:
                logger.warning(f"[{request_id}] [{rep_id}] Failed to send error: {send_err}")
            return ("upstream_connect_failed", None, None, 0, None, "")

        # Open upstream connection for this utterance
        upstream_open_time = time.perf_counter()
        try:
            upstream = await websockets.connect(
                settings.GIGASTT_WS_URL,
                ping_interval=30,
                ping_timeout=10,
            )
        except Exception as connect_err:
            error_reason = f"upstream_connect_failed: {connect_err}"
            logger.error(
                f"[{request_id}] [{rep_id}] Upstream connect failed: {connect_err}",
                extra={"request_id": request_id, "rep_id": rep_id, "event": "upstream_connect_failed", "error": str(connect_err)},
            )
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "code": "upstream_connect_failed",
                    "message": f"Failed to connect to GigaSTT: {connect_err}",
                }))
            except Exception as send_err:
                logger.warning(f"[{request_id}] [{rep_id}] Failed to send error to client: {send_err}")
            return (error_reason, None, None, 0, None, "")
        timings["upstream_connect_ms"] = round((time.perf_counter() - t_connect_start) * 1000, 1)
        logger.info(
            f"[{request_id}] [{rep_id}] Upstream opened",
            extra={"request_id": request_id, "rep_id": rep_id, "event": "upstream_opened"},
        )

        # Send configure to upstream
        configure_msg = json.dumps({"type": "configure", "sample_rate": 16000})
        await upstream.send(configure_msg)
        logger.debug(f"[{request_id}] [{rep_id}] Sent configure to upstream")

        # Wait for ready from upstream (retry once on timeout)
        ready = None
        for attempt in range(2):
            try:
                ready = await asyncio.wait_for(upstream.recv(), timeout=10.0)
                logger.debug(f"[{request_id}] [{rep_id}] Upstream ready (attempt {attempt + 1})")
                break
            except asyncio.TimeoutError:
                if attempt == 0:
                    logger.warning(f"[{request_id}] [{rep_id}] Upstream ready timeout, retrying...")
                    continue
                error_reason = "upstream_ready_timeout"
                logger.error(f"[{request_id}] [{rep_id}] Upstream ready timeout after 2 attempts")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "code": "upstream_ready_timeout",
                        "message": "GigaSTT did not respond to ready in 10s",
                    }))
                except Exception as send_err:
                    logger.warning(f"[{request_id}] [{rep_id}] Failed to send error to client: {send_err}")
                if upstream:
                    await upstream.close()
                return error_reason, ttfp_ms, ttft_ms, partial_count, next_message, final_text

        async def client_to_upstream():
            """Forward audio from queue to upstream until sentinel (None) received."""
            nonlocal error_reason, next_message, stop_sent
            audio_bytes_sent = 0

            # Send first chunk if provided
            if first_audio_chunk:
                audio_bytes_sent += len(first_audio_chunk)
                logger.debug(f"[{request_id}] [{rep_id}] First audio chunk: {len(first_audio_chunk)} bytes")
                await upstream.send(first_audio_chunk)

            while True:
                try:
                    item = await audio_queue.get()
                except asyncio.CancelledError:
                    raise

                if item is None:  # Sentinel = end_of_utterance
                    logger.debug(f"[{request_id}] [{rep_id}] End of utterance (from queue)")
                    if not stop_sent:
                        await upstream.send(json.dumps({"type": "stop"}))
                        stop_sent = True
                    break

                if isinstance(item, dict) and item.get("type") == "stop":
                    logger.debug(f"[{request_id}] [{rep_id}] Stop message (from queue)")
                    next_message = {"type": "stop"}
                    if not stop_sent:
                        await upstream.send(json.dumps({"type": "stop"}))
                        stop_sent = True
                    break

                # Audio bytes
                audio_bytes_sent += len(item)
                timings["audio_bytes_received"] += len(item)
                logger.debug(f"[{request_id}] [{rep_id}] Audio chunk from queue: {len(item)} bytes (total: {audio_bytes_sent})")
                await upstream.send(item)

        async def upstream_to_client():
            """Forward upstream responses to client until final."""
            nonlocal error_reason, ttfp_ms, ttft_ms, partial_count, final_text
            try:
                while True:
                    try:
                        try:
                            data = await asyncio.wait_for(upstream.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            logger.error(f"[{request_id}] [{rep_id}] Upstream recv timeout (30s)")
                            error_reason = "upstream_recv_timeout"
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "code": "upstream_recv_timeout",
                                    "message": "GigaSTT did not send partial/final in 30s",
                                }))
                            except Exception as send_err:
                                logger.warning(f"[{request_id}] [{rep_id}] Failed to send error to client: {send_err}")
                            break
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.info(f"[{request_id}] [{rep_id}] Upstream connection closed: {e}")
                        break

                    if isinstance(data, str):
                        logger.debug(f"[{request_id}] [{rep_id}] Upstream response: {data[:100]}...")

                        try:
                            msg = json.loads(data)
                            msg_type = msg.get("type")

                            if msg_type == "partial":
                                if ttfp_ms is None:
                                    ttfp_ms = (time.perf_counter() - rep_start_time) * 1000
                                    timings["upstream_first_byte_ms"] = round((time.perf_counter() - t_connect_start) * 1000, 1)
                                    metrics.inc_partial()
                                    metrics.observe_ttfp(ttfp_ms / 1000)
                                    logger.info(
                                        f"[{request_id}] [{rep_id}] Partial: {msg.get('text', '')} (TTFP={ttfp_ms:.0f}ms)",
                                        extra={"request_id": request_id, "rep_id": rep_id, "event": "partial", "text": msg.get("text", ""), "ttfp_ms": round(ttfp_ms, 1)},
                                    )
                                else:
                                    metrics.inc_partial()
                                    logger.debug(f"[{request_id}] [{rep_id}] Partial: {msg.get('text', '')}")
                                partial_count += 1
                                await websocket.send_text(data)

                            elif msg_type == "final":
                                ttft_ms = (time.perf_counter() - rep_start_time) * 1000
                                timings["upstream_final_ms"] = round((time.perf_counter() - t_connect_start) * 1000, 1)
                                metrics.observe_ttft(ttft_ms / 1000)
                                final_text = msg.get("text", "")
                                ttfp_str = f"{ttfp_ms:.0f}ms" if ttfp_ms else "N/A"
                                logger.info(
                                    f"[{request_id}] [{rep_id}] Final: {final_text} (TTFP={ttfp_str}, TTFT={ttft_ms:.0f}ms)",
                                    extra={"request_id": request_id, "rep_id": rep_id, "event": "final", "text": final_text, "ttfp_ms": round(ttfp_ms, 1) if ttfp_ms else None, "ttft_ms": round(ttft_ms, 1)},
                                )
                                # Send final with timings to client
                                final_msg = {
                                    "type": "final",
                                    "text": final_text,
                                    "timings": timings,
                                }
                                await websocket.send_text(json.dumps(final_msg))
                                return

                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                error_reason = str(e)
                logger.error(f"[{request_id}] [{rep_id}] Upstream read error: {e}")

        # Run both tasks concurrently
        task_c2u = asyncio.create_task(client_to_upstream())
        task_u2c = asyncio.create_task(upstream_to_client())

        # Wait for upstream_to_client to complete (gets final)
        try:
            await asyncio.wait_for(task_u2c, timeout=15.0)
        except asyncio.TimeoutError:
            logger.error(f"[{request_id}] [{rep_id}] Upstream final timeout (15s)")
            error_reason = "upstream_final_timeout"
            if not task_u2c.done():
                task_u2c.cancel()
                try:
                    await task_u2c
                except asyncio.CancelledError:
                    pass

        # Cancel client_to_upstream if still running
        # This is safe because it reads from queue, not from WebSocket
        if not task_c2u.done():
            task_c2u.cancel()
            try:
                await task_c2u
            except asyncio.CancelledError:
                pass

    except websockets.exceptions.ConnectionClosed as e:
        error_reason = str(e)
        logger.error(
            f"[{request_id}] [{rep_id}] Upstream connection closed: {e}",
            extra={"request_id": request_id, "rep_id": rep_id, "event": "upstream_closed", "reason": "connection_closed", "error": error_reason},
        )
        if upstream:
            await upstream.close()

    except asyncio.TimeoutError as e:
        error_reason = str(e)
        logger.error(
            f"[{request_id}] [{rep_id}] Upstream timeout: {e}",
            extra={"request_id": request_id, "rep_id": rep_id, "event": "upstream_closed", "reason": "timeout", "error": error_reason},
        )
        if upstream:
            await upstream.close()

    except Exception as e:
        error_reason = str(e)
        logger.error(
            f"[{request_id}] [{rep_id}] Upstream processing error: {e}",
            extra={"request_id": request_id, "rep_id": rep_id, "event": "upstream_closed", "reason": "error", "error": error_reason},
        )
        if upstream:
            await upstream.close()

    finally:
        # Always close upstream connection and log
        if upstream is not None:
            upstream_close_reason = "final_received" if final_text else ("error" if error_reason else "unknown")
            upstream_duration = time.perf_counter() - upstream_open_time

            # Log utterance completion
            logger.info(
                f"[{request_id}] [{rep_id}] Utterance completed: TTFP={ttfp_ms if ttfp_ms else 'N/A'}ms, TTFT={ttft_ms if ttft_ms else 'N/A'}ms, partials={partial_count}",
                extra={
                    "request_id": request_id,
                    "rep_id": rep_id,
                    "event": "rep_completed",
                    "final_text": final_text,
                    "ttfp_ms": round(ttfp_ms, 1) if ttfp_ms else None,
                    "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                    "partial_count": partial_count,
                },
            )

            try:
                await asyncio.wait_for(upstream.close(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{request_id}] [{rep_id}] upstream.close() timeout (5s), forcing")
                if hasattr(upstream, 'transport'):
                    upstream.transport.close()
            except Exception as e:
                logger.warning(f"[{request_id}] [{rep_id}] upstream.close() error: {e}")

            logger.info(
                f"[{request_id}] [{rep_id}] Upstream closed: reason={upstream_close_reason}, duration={upstream_duration:.3f}s",
                extra={
                    "request_id": request_id,
                    "rep_id": rep_id,
                    "event": "upstream_closed",
                    "reason": upstream_close_reason,
                    "duration_sec": round(upstream_duration, 3),
                },
            )

    return error_reason, ttfp_ms, ttft_ms, partial_count, next_message, final_text


@router.websocket("/ws/stt")
async def websocket_stt(
    websocket: WebSocket,
    token: str | None = Query(None),
):
    """
    WebSocket endpoint for STT streaming.

    Long-lived client connection with per-utterance upstream reconnection.

    Protocol:
      - Client -> Gateway: binary frames (PCM16 audio)
      - Client -> Gateway: {"type": "end_of_utterance"} - end of current utterance
      - Client -> Gateway: {"type": "stop"} - end of dialogue session
      - Gateway -> Client: {"type": "ready", ...} - connection accepted
      - Gateway -> Client: {"type": "partial", "text": "..."} - intermediate result
      - Gateway -> Client: {"type": "final", "text": "..."} - final result

    Refactored: Single WebSocket reader. All audio is queued in asyncio.Queue,
    and process_utterance reads from the queue (not from WebSocket directly).
    """
    request_id = str(uuid.uuid4())
    client_ip = websocket.client.host if websocket.client else "unknown"
    session_id = str(uuid.uuid4())

    connect_time = time.perf_counter()
    rep_counter = 0
    session_close_reason = "unknown"
    last_activity_time = connect_time
    session_start_time = connect_time

    try:
        # Extract and verify token
        ws_token = token or extract_token_from_ws(websocket)

        try:
            verify_token(ws_token)
        except HTTPException as e:
            await websocket.accept()
            await websocket.close(code=4401, reason="Unauthorized")
            metrics.inc_ws_session_rejected("auth")
            metrics.inc_ws_session("rejected")
            session_close_reason = "auth_error"
            logger.warning(
                f"[{request_id}] Authentication failed from {client_ip}",
                extra={"request_id": request_id, "client_ip": client_ip},
            )
            return

        # Check session limit
        if len(active_sessions) >= settings.MAX_CONCURRENT_SESSIONS:
            await websocket.accept()
            await websocket.close(code=4429, reason="Too many concurrent sessions")
            metrics.inc_ws_session_rejected("limit")
            metrics.inc_ws_session("rejected")
            session_close_reason = "limit_reached"
            logger.warning(
                f"[{request_id}] Session limit reached from {client_ip}",
                extra={"request_id": request_id, "client_ip": client_ip},
            )
            return

        # Accept client connection (long-lived)
        await websocket.accept()
        active_sessions.add(session_id)
        metrics.ws_active_sessions.set(len(active_sessions))
        logger.info(
            f"[{request_id}] WebSocket session started from {client_ip}",
            extra={"request_id": request_id, "client_ip": client_ip, "session_id": session_id},
        )

        # Send ready to client
        ready_msg = json.dumps({
            "type": "ready",
            "model": "gigaam-v3-rnnt",
            "sample_rate": 16000,
            "version": "2.22.0",
        })
        await websocket.send_text(ready_msg)
        logger.debug(f"[{request_id}] Sent ready to client")

        # Audio queue for current utterance - ONLY the main loop reads from WebSocket
        audio_queue: asyncio.Queue | None = None
        current_rep_task: asyncio.Task | None = None  # Track current utterance task

        # Main session loop - process utterances until stop or disconnect
        while True:
            # Check idle timeout
            idle_duration = time.perf_counter() - last_activity_time
            if idle_duration > settings.SESSION_IDLE_TIMEOUT_SEC:
                session_close_reason = "idle_timeout"
                logger.info(
                    f"[{request_id}] Session idle timeout: {idle_duration:.1f}s",
                    extra={"request_id": request_id, "session_id": session_id, "event": "session_closed", "reason": "idle_timeout"},
                )
                await websocket.close(code=4408, reason="Idle timeout")
                break

            try:
                # Wait for next message from client - THIS IS THE ONLY PLACE WE READ FROM WEBSOCKET
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.SESSION_IDLE_TIMEOUT_SEC,
                )
                last_activity_time = time.perf_counter()

                if message.get("type") != "websocket.receive":
                    continue

                # Check for stop message
                if "text" in message:
                    text = message["text"]
                    logger.debug(f"[{request_id}] Client message: {text[:50]}...")

                    try:
                        data = json.loads(text)
                        if data.get("type") == "stop":
                            # Wait for current rep_task if any
                            if audio_queue is not None and current_rep_task and not current_rep_task.done():
                                try:
                                    await asyncio.wait_for(current_rep_task, timeout=5.0)
                                except asyncio.TimeoutError:
                                    current_rep_task.cancel()
                            # Reset queue and task
                            audio_queue = None
                            current_rep_task = None
                            session_close_reason = "client_stop"
                            logger.info(
                                f"[{request_id}] Session stopped by client",
                                extra={"request_id": request_id, "session_id": session_id, "event": "session_closed", "reason": "client_stop"},
                            )
                            await websocket.close()
                            break
                        elif data.get("type") == "end_of_utterance":
                            # Put sentinel in queue to signal end_of_utterance
                            if audio_queue is not None:
                                await audio_queue.put(None)
                            # Wait for rep_task to complete and get results
                            if current_rep_task and not current_rep_task.done():
                                try:
                                    error_reason, ttfp_ms, ttft_ms, partial_count, next_message, final_text = await asyncio.wait_for(
                                        current_rep_task, timeout=20.0
                                    )
                                    logger.info(
                                        f"[{request_id}] Utterance completed: TTFP={ttfp_ms:.0f}ms" if ttfp_ms else f"[{request_id}] Utterance completed",
                                        extra={"request_id": request_id, "event": "rep_completed"}
                                    )
                                    if error_reason and next_message is None:
                                        logger.error(
                                            f"[{request_id}] Utterance failed: {error_reason}",
                                            extra={"request_id": request_id, "event": "utterance_error", "error": error_reason},
                                        )
                                except asyncio.TimeoutError:
                                    logger.error(f"[{request_id}] rep_task timeout after end_of_utterance")
                                    current_rep_task.cancel()
                            # Reset queue and task — otherwise next utterance won't create new queue
                            audio_queue = None
                            current_rep_task = None
                            continue
                    except json.JSONDecodeError:
                        pass

                elif "bytes" in message:
                    if audio_queue is None:
                        # First chunk of new utterance - create queue and task
                        rep_counter += 1
                        rep_id = f"rep-{rep_counter:03d}"
                        rep_start_time = time.perf_counter()
                        audio_queue = asyncio.Queue()

                        logger.info(
                            f"[{request_id}] [{rep_id}] Starting utterance processing",
                            extra={"request_id": request_id, "rep_id": rep_id, "event": "utterance_started"},
                        )

                        # Process utterance as background task - allows main loop to handle end_of_utterance
                        rep_task = asyncio.create_task(
                            process_utterance(
                                websocket, audio_queue, session_id, request_id, rep_id, rep_start_time, first_audio_chunk=None
                            )
                        )
                        current_rep_task = rep_task

                    # All chunks (including first) go to the same queue
                    await audio_queue.put(message["bytes"])

            except asyncio.TimeoutError:
                # Idle timeout
                session_close_reason = "idle_timeout"
                logger.info(
                    f"[{request_id}] Session idle timeout",
                    extra={"request_id": request_id, "session_id": session_id, "event": "session_closed", "reason": "idle_timeout"},
                )
                await websocket.close(code=4408, reason="Idle timeout")
                break

    except WebSocketDisconnect:
        session_close_reason = "client_disconnect"
        duration = time.perf_counter() - connect_time
        logger.info(
            f"[{request_id}] Client disconnected after {duration:.1f}s",
            extra={"request_id": request_id, "session_id": session_id, "event": "session_closed", "reason": "client_disconnect", "duration_sec": round(duration, 2)},
        )

    except Exception as e:
        session_close_reason = "error"
        metrics.inc_ws_session("error")
        logger.error(
            f"[{request_id}] Session error: {e}",
            extra={"request_id": request_id, "session_id": session_id, "event": "session_closed", "reason": "error", "error": str(e)},
        )

    finally:
        # Session closed - log metrics
        if session_id in active_sessions:
            active_sessions.discard(session_id)
            metrics.ws_active_sessions.set(len(active_sessions))

        duration = time.perf_counter() - connect_time
        metrics.inc_ws_session_duration(duration)
        metrics.inc_ws_session("ok")

        logger.info(
            f"[{request_id}] Session closed: reason={session_close_reason}, duration={duration:.1f}s, utterances={rep_counter}",
            extra={
                "request_id": request_id,
                "session_id": session_id,
                "event": "session_closed",
                "reason": session_close_reason,
                "duration_sec": round(duration, 2),
                "utterance_count": rep_counter,
            },
        )
