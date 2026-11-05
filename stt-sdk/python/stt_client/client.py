"""STTClient: WebSocket-клиент с partial/final и retry на error."""
import asyncio
import json
import logging
from typing import AsyncIterator, Callable, Optional
from urllib.parse import urlparse, urlunparse, urlencode, parse_qs

import websockets
from websockets.exceptions import ConnectionClosed

from .audio import AudioSource
from .errors import STTError, RetryPolicy
from .models import FinalResult, PartialResult, ErrorMessage

logger = logging.getLogger(__name__)


def _translate_connection_closed(e: ConnectionClosed) -> STTError:
    """Преобразовать ConnectionClosed (close codes 4401/4408/4429) в типизированный STTError."""
    code = e.rcvd.code if e.rcvd else None
    if code == 4401:
        return STTError("unauthorized", "Invalid API key", retryable=False)
    if code == 4429:
        return STTError("session_limit", "Too many concurrent sessions", retryable=True)
    if code == 4408:
        return STTError("idle_timeout", "Session idle timeout", retryable=True)
    return STTError("connection_closed", f"WebSocket closed: code={code}", retryable=False)


class STTClient:
    """WebSocket-клиент к STT Gateway."""

    def __init__(
        self,
        base_url: str,
        token: str,
        retry_policy: Optional[RetryPolicy] = None,
        timeout_sec: float = 30.0,
    ):
        """
        Args:
            base_url: 'ws://localhost:8100/ws/stt' или 'wss://stt.company.local/ws/stt'
            token: API key
            retry_policy: default RetryPolicy(max_attempts=3)
            timeout_sec: общий таймаут на реплику (final)
        """
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {timeout_sec}")
        self.base_url = base_url
        self.token = token
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_sec = timeout_sec

    def _ws_url(self) -> str:
        """Собрать WebSocket URL с token query параметром."""
        parsed = urlparse(self.base_url)
        query = parse_qs(parsed.query)
        query["token"] = [self.token]
        new_query = urlencode(query, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    async def stream(
        self,
        audio: AudioSource,
        on_partial: Optional[Callable[[PartialResult], None]] = None,
    ) -> FinalResult:
        """
        Отправить аудио, вернуть FinalResult.

        Args:
            audio: AudioSource (WavAudioSource или MicAudioSource)
            on_partial: callback на каждый partial (sync)

        Returns:
            FinalResult с текстом и таймингами

        Raises:
            STTError: если ошибка не retryable или после исчерпания retry
        """

        async def _attempt() -> FinalResult:
            async with websockets.connect(
                self._ws_url(), ping_interval=30, ping_timeout=10
            ) as ws:
                # Ожидать ready от сервера
                try:
                    ready_raw = await ws.recv()
                except ConnectionClosed as e:
                    raise _translate_connection_closed(e)
                ready_data = json.loads(ready_raw)
                if ready_data.get("type") != "ready":
                    raise STTError(
                        "protocol_error", f"Expected ready, got {ready_data}"
                    )
                logger.debug(f"ready: {ready_data}")

                # Отправить аудио чанками
                async def send_audio():
                    async for chunk in audio.stream():
                        await ws.send(chunk)
                    await ws.send(json.dumps({"type": "end_of_utterance"}))

                send_task = asyncio.create_task(send_audio())

                try:
                    # Читать partial/final/error до final или timeout
                    async with asyncio.timeout(self.timeout_sec):
                        while True:
                            try:
                                raw = await ws.recv()
                            except ConnectionClosed as e:
                                raise _translate_connection_closed(e)
                            msg = json.loads(raw)
                            msg_type = msg.get("type")

                            if msg_type == "partial":
                                partial = PartialResult(text=msg.get("text", ""))
                                if on_partial:
                                    try:
                                        on_partial(partial)
                                    except Exception as cb_err:
                                        logger.warning(
                                            f"on_partial callback raised: "
                                            f"{type(cb_err).__name__}: {cb_err}",
                                            exc_info=True,
                                        )
                            elif msg_type == "final":
                                return FinalResult(
                                    text=msg.get("text", ""),
                                    timings=msg.get("timings"),
                                )
                            elif msg_type == "error":
                                err = ErrorMessage(
                                    code=msg.get("code", "unknown"),
                                    message=msg.get("message", ""),
                                )
                                logger.warning(
                                    f"server error: {err.code} — {err.message}"
                                )
                                raise STTError(err.code, err.message)
                            elif msg_type == "ready":
                                continue
                            else:
                                logger.warning(f"unknown message type: {msg_type}")

                except asyncio.TimeoutError:
                    raise STTError(
                        "client_timeout", f"No final in {self.timeout_sec}s"
                    )
                finally:
                    if not send_task.done():
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                    # Закрыть сессию
                    try:
                        await ws.send(json.dumps({"type": "stop"}))
                    except Exception:
                        pass

        try:
            return await self.retry_policy.run(_attempt)
        except STTError:
            raise
        except Exception as e:
            raise STTError("unexpected", str(e))

    async def close(self):
        """Закрыть клиент (no-op, соединения закрываются в stream)."""
        pass
