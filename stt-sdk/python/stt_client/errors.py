"""Ошибки SDK и retry policy."""
import asyncio
import logging
from typing import Callable, Awaitable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class STTError(Exception):
    """Базовая ошибка SDK."""

    def __init__(self, code: str, message: str, retryable: bool | None = None):
        self.code = code
        self.message = message
        self._retryable_override = retryable
        super().__init__(f"{code}: {message}")

    @property
    def is_retryable(self) -> bool:
        if self._retryable_override is not None:
            return self._retryable_override
        return self.code in {
            "pool_unavailable",
            "upstream_ready_timeout",
            "upstream_recv_timeout",
            "upstream_connect_failed",
            "session_limit",
            "idle_timeout",
        }


class RetryPolicy:
    """Retry policy для retryable errors."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 10.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec

    def delay(self, attempt: int) -> float:
        """Экспоненциальный backoff: 1s, 2s, 4s, ... max 10s."""
        return min(self.base_delay_sec * (2**attempt), self.max_delay_sec)

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Выполнить fn с retry на STTError с is_retryable=True."""
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                return await fn()
            except STTError as e:
                if not e.is_retryable:
                    raise
                last_error = e
                if attempt < self.max_attempts - 1:
                    delay = self.delay(attempt)
                    logger.warning(
                        f"Retryable error {e.code}, retry {attempt+1}/{self.max_attempts} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected: no error raised")
