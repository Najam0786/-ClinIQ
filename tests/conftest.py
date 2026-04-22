"""
tests/conftest.py
-----------------
Pytest configuration for ClinIQ test suite.

Integration tests (those that hit a live API) are automatically skipped
in CI unless a CLINIQ_API_URL env var is set AND the server is reachable.
"""

import os
import socket
import urllib.parse
import pytest


def _api_reachable() -> bool:
    url = os.environ.get("CLINIQ_API_URL", "")
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


_INTEGRATION = pytest.mark.skipif(
    not _api_reachable(),
    reason="Integration tests require CLINIQ_API_URL to point to a running server",
)


def pytest_collection_modifyitems(items):
    """Auto-mark any test file that imports urllib.request as an integration test."""
    for item in items:
        try:
            with open(str(item.fspath), encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
        except OSError:
            continue
        if "urllib.request" in src or "CLINIQ_API_URL" in src:
            item.add_marker(_INTEGRATION)
