"""Structured JSON logging for STT Gateway (Loki-compatible)."""

import logging
import json
import re
from datetime import datetime, timezone
from typing import Any


# Pattern to detect tokens in log messages
TOKEN_PATTERNS = [
    r"token=[\w-]+",
    r"token=['\"]?[\w-]+['\"]?",
    r"Authorization:\s*Bearer\s+[\w-]+",
    r"\?token=[\w-]+",
]


def mask_token(text: str, max_visible: int = 8) -> str:
    """Mask sensitive token values in log text."""
    for pattern in TOKEN_PATTERNS:
        def replacer(m: re.Match) -> str:
            matched = m.group(0)
            # Extract the token value
            token_match = re.search(r"[\w-]{8,}", matched)
            if token_match:
                token = token_match.group(0)
                if len(token) > max_visible:
                    return matched.replace(token, token[:max_visible] + "...")
            return matched
        text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
    return text


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Mask tokens in message
        message = mask_token(record.getMessage())

        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # Add request_id if present
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        # Add extra fields
        for key in ("client_ip", "session_id", "event", "duration_sec", "error"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        return json.dumps(log_obj)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with JSON formatting."""
    logger = logging.getLogger(name)
    # Don't propagate to avoid duplicate logs
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger
