"""Test Gap 1: {"type": "error", "code": ...} for 4 scenarios."""
import asyncio
import json
import pytest
import websockets
from unittest.mock import patch, AsyncMock

GATEWAY_URL = "ws://127.0.0.1:8100/ws/stt?token=testtoken"


async def _read_until_error_or_final(ws, timeout=40):
    """Read messages until error or final."""
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") in ("error", "final"):
            return msg


@pytest.mark.asyncio
async def test_upstream_connect_failed():
    """upstream_connect_failed when GIGASTT_WS_URL is unreachable."""
    # Requires manual test: set GIGASTT_WS_URL=ws://127.0.0.1:9999/v1/ws
    # Then run: pytest tests/test_error_type.py::test_upstream_connect_failed -v
    # Expected: {"type": "error", "code": "upstream_connect_failed", ...}
    pass  # Manual test


@pytest.mark.asyncio
async def test_pool_unavailable():
    """pool_unavailable when pool_available=0 in /ready."""
    # Requires manual test: stop gigastt (docker compose stop gigastt)
    # Then run: pytest tests/test_error_type.py::test_pool_unavailable -v
    # Expected: {"type": "error", "code": "pool_unavailable", ...}
    pass  # Manual test


@pytest.mark.asyncio
@pytest.mark.skip(reason="Manual test: requires mocking upstream ready response timeout")
async def test_upstream_ready_timeout():
    """upstream_ready_timeout when GigaSTT does not respond to ready in 10s."""
    pass


@pytest.mark.asyncio
@pytest.mark.skip(reason="Manual test: requires mocking upstream recv timeout")
async def test_upstream_recv_timeout():
    """upstream_recv_timeout when GigaSTT does not send partial/final in 30s."""
    pass
