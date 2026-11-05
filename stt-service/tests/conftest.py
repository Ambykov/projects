"""Pytest configuration and fixtures."""

import os
import pytest

# Set default environment for all tests
os.environ["AUTH_MODE"] = "none"
os.environ["API_KEY"] = ""


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset the lru_cache for get_settings before each test."""
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
