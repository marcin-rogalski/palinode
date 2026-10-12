"""Hardening for the deferred save embed: the behaviours the first tests left open.

``test_save_defer_embed.py`` pins the flag, the keyword-only argument and the MCP
timeout. Each test here guards one line that a mutation run showed was unguarded;
the docstring names the line.

Real SQLite + tmp_path; the embedder is a recording stub, nothing reaches the
network. Import-time constants are read in a fresh interpreter.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from palinode import mcp
from palinode.core import store
from palinode.core.config import config, load_config
from palinode.core.save import save_memory
from palinode.indexer import reconcile as reconcile_mod
from palinode.indexer.index_file import index_file

REPO_ROOT = Path(__file__).parent.parent

_FAKE_VECTOR = [0.01] * 1024
_FLAG_ENV = "PALINODE_DEFER_EMBED_ON_SAVE"
_CONTENT = "Quokka marmalade ledger for the release checklist."


# ── fixtures and helpers ────────────────────────────────────────────────────

@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


@pytest.fixture()
def embed_stub():
    stub = MagicMock(return_value=_FAKE_VECTOR)
    with patch("palinode.core.embedder.embed", stub):
        yield stub


def _defer_on_save(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(config.embeddings, "defer_embed_on_save", value)


def _save(**kwargs) -> dict:
    kwargs.setdefault("content", _CONTENT)
    kwargs.setdefault("type", "Insight")
    kwargs.setdefault("slug", "release-checklist")
    with patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")):
        return save_memory(**kwargs)


def _vectors_present(file_path: str) -> list[bool]:
    db = store.get_db()
    ids = [r["id"] for r in db.execute(
        "SELECT id FROM chunks WHERE file_path = ?", (file_path,)).fetchall()]
    present = [
        bool(db.execute("SELECT 1 FROM chunks_vec WHERE id = ?", (cid,)).fetchone())
        for cid in ids
    ]
    db.close()
    assert present, "the save wrote no chunks"
    return present


# ── healing while the flag is still on ──────────────────────────────────────

def test_flag_on_deferral_heals_through_a_watcher_that_shares_the_config(
    memory_dir, embed_stub, monkeypatch
):
    """The watcher calls ``index_file(path)`` in a process whose config has the
    flag on. Kills ``index_file``/``apply`` reading ``config.embeddings.
    defer_embed_on_save`` themselves: the heal would defer again, forever."""
    _defer_on_save(monkeypatch, True)
    result = _save()
    assert not any(_vectors_present(result["file_path"]))

    outcome = index_file(result["file_path"])

    assert embed_stub.call_count >= 1
    assert outcome["embedded"] is True
    assert all(_vectors_present(result["file_path"]))


# ── a metadata-only re-save with the flag on ────────────────────────────────

def test_flag_on_resave_with_an_unchanged_body_is_embedded_and_keeps_its_vectors(
    memory_dir, embed_stub, monkeypatch
):
    """Nothing is queued for embedding, so nothing is deferred. Kills
    ``diff.deferred = deferred`` without ``and bool(p.to_index)`` in
    ``reconcile.apply``: the result would say ``embedded`` false and carry an
    ``embed deferred`` error for a chunk that already has its vector."""
    first = _save()
    assert all(_vectors_present(first["file_path"]))
    embed_stub.reset_mock()
    _defer_on_save(monkeypatch, True)

    result = _save(title="Release checklist, renamed")

    embed_stub.assert_not_called()
    assert result["embedded"] is True
    assert "deferred" not in (result.get("index_error") or "")
    assert all(_vectors_present(result["file_path"]))


# ── the reason in the deferral message ──────────────────────────────────────

def test_flag_on_save_says_the_caller_asked_for_the_deferral(
    memory_dir, embed_stub, monkeypatch
):
    """Kills the reason expression in ``apply`` always choosing the probe text
    (``"requested by the caller" if defer_embed else ...`` inverted or dropped)."""
    _defer_on_save(monkeypatch, True)

    result = _save()

    assert "requested by the caller" in result["index_error"]
    assert "probe failed" not in result["index_error"]


def test_cold_embedder_save_says_the_probe_failed(memory_dir, embed_stub, monkeypatch):
    """Kills the same expression always choosing the caller text."""
    monkeypatch.setattr(reconcile_mod, "_embeds_deferred", lambda client: True)

    result = _save()

    assert "probe failed" in result["index_error"]
    assert "requested by the caller" not in result["index_error"]


# ── the environment variable ────────────────────────────────────────────────

def _load_with_env(tmp_path: Path, monkeypatch, value: str, yaml_flag: bool):
    (tmp_path / "palinode.config.yaml").write_text(
        f"embeddings:\n  defer_embed_on_save: {str(yaml_flag).lower()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PALINODE_DIR", str(tmp_path))
    monkeypatch.setenv(_FLAG_ENV, value)
    return load_config().embeddings.defer_embed_on_save


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF ", "False"])
def test_falsy_env_values_turn_a_configured_true_off(tmp_path, monkeypatch, value):
    """Kills the falsy branch being removed or narrowed to a subset of the words."""
    assert _load_with_env(tmp_path, monkeypatch, value, yaml_flag=True) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " ON ", "True"])
def test_truthy_env_values_turn_the_flag_on(tmp_path, monkeypatch, value):
    """Kills the truthy set being narrowed (e.g. to ``"true"`` alone)."""
    assert _load_with_env(tmp_path, monkeypatch, value, yaml_flag=False) is True


@pytest.mark.parametrize("yaml_flag", [True, False])
@pytest.mark.parametrize("value", ["maybe", "2", "", "tru"])
def test_a_malformed_env_value_keeps_the_configured_value(
    tmp_path, monkeypatch, value, yaml_flag
):
    """Kills the ``else`` branch assigning a value (e.g. ``bool(flag)``)."""
    assert _load_with_env(tmp_path, monkeypatch, value, yaml_flag) is yaml_flag


# ── the CLI save timeout ────────────────────────────────────────────────────

_CLI_PROBE = textwrap.dedent(
    """
    import json
    import httpx
    from palinode.cli._api import api_client

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["read_timeout"] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json={"file_path": "/m/insights/x.md"})

    api_client.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://testserver"
    )
    api_client.save("Release checklist.", "Insight")
    print(json.dumps(seen))
    """
)


def _cli_save_read_timeout(env_value: str | None) -> float:
    env = {k: v for k, v in os.environ.items() if k != "PALINODE_SAVE_TIMEOUT"}
    if env_value is not None:
        env["PALINODE_SAVE_TIMEOUT"] = env_value
    env.setdefault("PALINODE_LOG_LEVEL", "ERROR")
    proc = subprocess.run(
        [sys.executable, "-c", _CLI_PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen["path"] == "/save"
    return seen["read_timeout"]


def test_cli_save_timeout_defaults_to_30_seconds():
    """Kills the CLI call dropping its ``timeout`` (the client default is 5 s or
    whatever the transport sets) or hard-coding another default."""
    assert _cli_save_read_timeout(None) == 30.0


def test_cli_save_timeout_follows_the_env_var():
    """Kills the CLI passing a literal instead of ``SAVE_TIMEOUT_SECONDS``."""
    assert _cli_save_read_timeout("90") == 90.0


# ── the MCP receipt ─────────────────────────────────────────────────────────

def _mcp_save_reply(monkeypatch, body: dict) -> str:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    monkeypatch.setattr(mcp, "_http_transport", httpx.MockTransport(handler))
    monkeypatch.setattr(mcp, "_http_client", None)
    monkeypatch.setattr(mcp, "_http_client_loop", None)
    reply = asyncio.run(
        mcp._dispatch_tool("palinode_save", {"content": _CONTENT, "type": "Insight"})
    )
    return reply[0].text


def test_mcp_receipt_names_the_deferral_when_the_flag_asked_for_it(
    memory_dir, embed_stub, monkeypatch
):
    """The body is the server's real answer to a flag-on save. Kills the receipt
    branch on ``index_error.startswith("embed deferred")`` being removed."""
    _defer_on_save(monkeypatch, True)
    served = _save()

    text = _mcp_save_reply(monkeypatch, served)

    assert "embed deferred" in text
    assert "vec index write failed" not in text


def test_mcp_receipt_names_the_deferral_on_a_cold_embedder(
    memory_dir, embed_stub, monkeypatch
):
    """Same receipt, reached by the probe path instead of the flag."""
    monkeypatch.setattr(reconcile_mod, "_embeds_deferred", lambda client: True)
    served = _save()

    text = _mcp_save_reply(monkeypatch, served)

    assert "embed deferred" in text
    assert "vec index write failed" not in text


def test_mcp_receipt_still_warns_of_a_vector_write_failure(monkeypatch):
    """A missing vector with any other error is still a failure. Kills the
    deferral wording being shown whenever ``indexed_vec`` is false."""
    text = _mcp_save_reply(monkeypatch, {
        "file_path": "/m/insights/x.md", "rel_path": "insights/x.md",
        "id": "insights-x", "save_outcome": "created", "indexed": False,
        "embedded": False, "indexed_vec": False, "indexed_fts": True,
        "git_committed": True, "index_error": "vec insert failed: disk full",
    })

    assert "vec index write failed" in text
    assert "embed deferred" not in text
