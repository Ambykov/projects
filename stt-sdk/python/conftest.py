"""Pytest configuration for SDK tests.

Integration tests require a live gateway at localhost:8100.
Run with: pytest sdk/python -v --run-integration
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (require live gateway at localhost:8100)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires live gateway at localhost:8100 (run with --run-integration)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="need --run-integration to run (requires live gateway)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
