"""Prometheus metrics for STT Gateway."""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)
from fastapi.responses import Response

# Registry for our metrics
registry = CollectorRegistry()

# WebSocket session metrics
ws_active_sessions = Gauge(
    "stt_gateway_ws_active_sessions",
    "Number of currently active WebSocket sessions",
    registry=registry,
)

ws_sessions_total = Counter(
    "stt_gateway_ws_sessions_total",
    "Total WebSocket sessions",
    ["status"],  # ok, error, rejected
    registry=registry,
)

ws_sessions_rejected_total = Counter(
    "stt_gateway_ws_sessions_rejected_total",
    "Rejected WebSocket sessions",
    ["reason"],  # auth, limit, upstream
    registry=registry,
)

ws_session_duration_seconds = Histogram(
    "stt_gateway_ws_session_duration_seconds",
    "WebSocket session duration in seconds",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    registry=registry,
)

# Note: partials may not be emitted for very short utterances (GigaSTT limitation)
# See ADR-0005 for details
partials_received_total = Counter(
    "stt_gateway_partials_received_total",
    "Total partial transcripts received from upstream (may be 0 for short utterances)",
    registry=registry,
)

# Time to first partial (only recorded when partials are actually emitted)
ws_time_to_first_partial = Histogram(
    "stt_gateway_ws_time_to_first_partial_seconds",
    "Time to first partial transcript (TTFP)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
    registry=registry,
)

ws_time_to_final = Histogram(
    "stt_gateway_ws_time_to_final_seconds",
    "Time to final transcript",
    buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0),
    registry=registry,
)

# REST API metrics
rest_requests_total = Counter(
    "stt_gateway_rest_requests_total",
    "Total REST API requests",
    ["status"],  # ok, error
    registry=registry,
)

rest_request_duration = Histogram(
    "stt_gateway_rest_request_duration_seconds",
    "REST API request duration",
    buckets=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=registry,
)

# Upstream errors
upstream_errors_total = Counter(
    "stt_gateway_upstream_errors_total",
    "Upstream errors",
    ["type"],  # ws, rest
    registry=registry,
)


def get_metrics() -> Response:
    """Get Prometheus metrics response."""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )


# Helper functions for incrementing metrics
def inc_ws_session(status: str) -> None:
    """Increment WebSocket session counter."""
    ws_sessions_total.labels(status=status).inc()


def inc_ws_session_rejected(reason: str) -> None:
    """Increment rejected WebSocket session counter."""
    ws_sessions_rejected_total.labels(reason=reason).inc()


def inc_ws_session_duration(duration: float) -> None:
    """Record WebSocket session duration."""
    ws_session_duration_seconds.observe(duration)


def inc_partial() -> None:
    """Increment partial transcript counter."""
    partials_received_total.inc()


def observe_ttfp(seconds: float) -> None:
    """Observe time to first partial (TTFP)."""
    ws_time_to_first_partial.observe(seconds)


def observe_ttft(seconds: float) -> None:
    """Observe time to final."""
    ws_time_to_final.observe(seconds)


def inc_rest_request(status: str) -> None:
    """Increment REST request counter."""
    rest_requests_total.labels(status=status).inc()


def observe_rest_duration(seconds: float) -> None:
    """Observe REST request duration."""
    rest_request_duration.observe(seconds)


def inc_upstream_error(typ: str) -> None:
    """Increment upstream error counter."""
    upstream_errors_total.labels(type=typ).inc()
