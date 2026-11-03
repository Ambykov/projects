"""Модели сообщений и результатов."""
from dataclasses import dataclass
from typing import Optional, TypedDict


class Timings(TypedDict, total=False):
    """Server-side timings (docs/api/websocket.md §Тайминги).

    Все поля опциональны — включаются в final начиная с Stage 4.B.
    """
    upstream_connect_ms: float
    upstream_first_byte_ms: float
    upstream_final_ms: float
    audio_bytes_received: int


@dataclass
class ReadyMessage:
    """Server -> Client: connection accepted."""
    model: str
    sample_rate: int
    version: str


@dataclass
class PartialResult:
    """Server -> Client: intermediate transcription result."""
    text: str


@dataclass
class FinalResult:
    """Server -> Client: final transcription result."""
    text: str
    timings: Optional[Timings] = None


@dataclass
class ErrorMessage:
    """Server -> Client: structured error (Gap 1)."""
    code: str
    message: str

    @property
    def is_retryable(self) -> bool:
        """Все 4 кода из Gap 1 — retryable (см. docs/api/websocket.md)."""
        return self.code in {
            "pool_unavailable",
            "upstream_ready_timeout",
            "upstream_recv_timeout",
            "upstream_connect_failed",
        }
