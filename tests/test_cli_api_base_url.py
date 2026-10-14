"""Which server the CLI talks to.

The hooks and plugins read ``PALINODE_API_URL``; the CLI historically read only
``PALINODE_API``. A client configured for the hooks therefore had a CLI that
silently fell through to ``config.services.api`` (the server's bind address,
normally loopback) and failed with "connection refused". The CLI now accepts
both names, with its own ``PALINODE_API`` taking precedence.
"""
from __future__ import annotations

import pytest

from palinode.cli._api import PalinodeAPI
from palinode.core.config import config


@pytest.fixture(autouse=True)
def _clear_api_env(monkeypatch):
    monkeypatch.delenv("PALINODE_API", raising=False)
    monkeypatch.delenv("PALINODE_API_URL", raising=False)


def _base_url() -> str:
    api = PalinodeAPI()
    try:
        return api.base_url
    finally:
        api.client.close()


def test_defaults_to_the_configured_api_address():
    expected = f"http://{config.services.api.host}:{config.services.api.port}"
    assert _base_url() == expected


def test_accepts_the_hooks_variable(monkeypatch):
    monkeypatch.setenv("PALINODE_API_URL", "http://memory.example:6340")
    assert _base_url() == "http://memory.example:6340"


def test_cli_variable_wins_over_the_hooks_variable(monkeypatch):
    monkeypatch.setenv("PALINODE_API_URL", "http://hooks.example:6340")
    monkeypatch.setenv("PALINODE_API", "http://cli.example:6340")
    assert _base_url() == "http://cli.example:6340"


def test_empty_values_fall_through(monkeypatch):
    monkeypatch.setenv("PALINODE_API", "")
    monkeypatch.setenv("PALINODE_API_URL", "http://memory.example:6340")
    assert _base_url() == "http://memory.example:6340"
