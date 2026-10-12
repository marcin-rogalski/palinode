"""A save can return before the embed, and the MCP save timeout is configurable.

``save_memory`` indexes inline, so ``POST /save`` returns only after vectors and
FTS rows exist. On a slow CPU embedder that outlasts the MCP client's fixed 30 s
request timeout: the write has landed, the client reports a timeout, a retry
creates a duplicate.

The deferred path already exists for a cold embedder: ``reconcile.apply`` writes
FTS-only rows and a later pass embeds them. This change lets a save opt into it
with ``config.embeddings.defer_embed_on_save`` (default off, env
``PALINODE_DEFER_EMBED_ON_SAVE``), threaded as a keyword-only ``defer_embed``
through ``index_file`` → ``reconcile`` → ``apply``, and makes the MCP save
request timeout configurable with ``PALINODE_SAVE_TIMEOUT`` (default 30).

Where the flag lives and what the env var is called are pinned once each, in
``_FLAG_SECTION``/``_FLAG`` and ``_FLAG_ENV`` below. Tests marked *red* fail on
v0.22.0 for the reason in their docstring; the rest pass on v0.22.0 and must
keep passing.

Real SQLite + tmp_path; the embedder is a recording stub, nothing reaches the
network. The MCP timeout is read in a fresh interpreter through
``palinode.mcp._http_transport`` so no module of this session is reloaded.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from palinode.core import store
from palinode.core.config import config, load_config
from palinode.core.save import save_memory
from palinode.indexer import reconcile as reconcile_mod
from palinode.indexer.index_file import index_file

REPO_ROOT = Path(__file__).parent.parent

_FAKE_VECTOR = [0.01] * 1024

_FLAG_SECTION = "embeddings"
_FLAG = "defer_embed_on_save"
_FLAG_ENV = "PALINODE_DEFER_EMBED_ON_SAVE"

_CONTENT = "Quokka marmalade ledger for the release checklist."
_KEYWORD = "marmalade"


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
    """A recording embedder; ``conftest`` routes ``embed_many`` through it."""
    stub = MagicMock(return_value=_FAKE_VECTOR)
    with patch("palinode.core.embedder.embed", stub):
        yield stub


def _set_defer_on_save(monkeypatch, value: bool = True) -> None:
    monkeypatch.setattr(getattr(config, _FLAG_SECTION), _FLAG, value, raising=False)


def _save(**kwargs) -> dict:
    kwargs.setdefault("content", _CONTENT)
    kwargs.setdefault("type", "Insight")
    kwargs.setdefault("slug", "release-checklist")
    with patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")):
        return save_memory(**kwargs)


def _chunk_ids(file_path: str) -> list[str]:
    db = store.get_db()
    rows = db.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,)).fetchall()
    db.close()
    return [r["id"] for r in rows]


def _vec_present(chunk_id: str) -> bool:
    db = store.get_db()
    row = db.execute("SELECT 1 FROM chunks_vec WHERE id = ?", (chunk_id,)).fetchone()
    db.close()
    return bool(row)


def _fts_matches(term: str) -> int:
    db = store.get_db()
    rows = db.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (term,)
    ).fetchall()
    db.close()
    return len(rows)


# ── characterisation: the inline embed and the cold path today ──────────────

def test_flag_off_save_embeds_inline(memory_dir, embed_stub):
    result = _save()

    assert embed_stub.call_count >= 1
    assert result["embedded"] is True
    assert result["indexed_vec"] is True
    ids = _chunk_ids(result["file_path"])
    assert ids
    assert all(_vec_present(cid) for cid in ids)


def test_cold_embedder_save_reports_the_deferral(memory_dir, embed_stub, monkeypatch):
    """The result shape the flag must reproduce: not embedded, FTS written, and
    an ``index_error`` that names the deferral."""
    monkeypatch.setattr(reconcile_mod, "_embeds_deferred", lambda client: True)

    result = _save()

    embed_stub.assert_not_called()
    assert result["embedded"] is False
    assert result["indexed_fts"] is True
    assert "deferred" in result["index_error"]
    assert _fts_matches(_KEYWORD) >= 1


def test_plain_index_file_embeds_even_when_the_flag_is_on(memory_dir, embed_stub, monkeypatch):
    """The watcher calls ``index_file`` with no argument in a process that reads
    the same config. The flag is the save's, so it must not defer the watcher."""
    result = _save()
    embed_stub.reset_mock()
    _set_defer_on_save(monkeypatch, True)
    path = result["file_path"]
    Path(path).write_text(
        Path(path).read_text(encoding="utf-8") + "\nA second paragraph.\n",
        encoding="utf-8",
    )

    outcome = index_file(path)

    assert embed_stub.call_count >= 1
    assert outcome["embedded"] is True


# ── red: the flag ───────────────────────────────────────────────────────────

def test_flag_defaults_to_off():
    """Red: ``config.embeddings`` has no ``defer_embed_on_save`` (AttributeError)."""
    assert getattr(getattr(config, _FLAG_SECTION), _FLAG) is False


def test_env_var_turns_the_flag_on(monkeypatch, tmp_path):
    """Red: ``load_config()`` ignores the variable (AttributeError on the flag)."""
    monkeypatch.setenv("PALINODE_DIR", str(tmp_path))
    monkeypatch.setenv(_FLAG_ENV, "true")

    loaded = load_config()

    assert getattr(getattr(loaded, _FLAG_SECTION), _FLAG) is True


def test_flag_on_save_returns_without_embedding(memory_dir, embed_stub, monkeypatch):
    """Red: the save still calls the embedder (call_count is 1, not 0)."""
    _set_defer_on_save(monkeypatch, True)

    result = _save()

    embed_stub.assert_not_called()
    assert result["file_path"].endswith("release-checklist.md")
    assert result["embedded"] is False
    assert result["indexed_vec"] is False
    assert result["indexed_fts"] is True
    assert "deferred" in result["index_error"]


def test_flag_on_save_is_keyword_searchable_at_once_and_has_no_vector(
    memory_dir, embed_stub, monkeypatch
):
    """Red: the chunk has a vector row (the inline embed ran)."""
    _set_defer_on_save(monkeypatch, True)

    result = _save()

    assert _fts_matches(_KEYWORD) >= 1
    ids = _chunk_ids(result["file_path"])
    assert ids
    assert not any(_vec_present(cid) for cid in ids)


def test_flag_on_deferral_heals_on_the_next_reconcile_pass(
    memory_dir, embed_stub, monkeypatch
):
    """Red: the first assertion, a save with the flag on leaves no vector. The
    heal after it is the existing missing-vec branch, and passes on v0.22.0."""
    _set_defer_on_save(monkeypatch, True)
    result = _save()
    ids = _chunk_ids(result["file_path"])
    assert not any(_vec_present(cid) for cid in ids)

    _set_defer_on_save(monkeypatch, False)
    outcome = index_file(result["file_path"])

    assert embed_stub.call_count >= 1
    assert outcome["embedded"] is True
    assert outcome["chunks_reembedded"] >= 1
    assert all(_vec_present(cid) for cid in _chunk_ids(result["file_path"]))


def test_flag_on_second_save_of_the_same_slug_stays_deferred(
    memory_dir, embed_stub, monkeypatch
):
    """Red: an edited body is re-chunked; the rewrite is deferred too."""
    _set_defer_on_save(monkeypatch, True)
    _save()

    result = _save(content=_CONTENT + "\n\nA changed paragraph about quinces.")

    embed_stub.assert_not_called()
    assert result["embedded"] is False
    assert _fts_matches("quinces") >= 1


# ── red: the keyword-only argument down the stack ───────────────────────────

@pytest.mark.parametrize(
    "func",
    [index_file, reconcile_mod.reconcile, reconcile_mod.apply],
    ids=["index_file", "reconcile", "apply"],
)
def test_defer_embed_is_a_keyword_only_argument_defaulting_to_false(func):
    """Red: none of the three has a ``defer_embed`` parameter (KeyError)."""
    param = inspect.signature(func).parameters["defer_embed"]

    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False


def test_index_file_with_defer_embed_writes_fts_only_rows(memory_dir, embed_stub):
    """Red: ``index_file`` takes no ``defer_embed`` (TypeError)."""
    path = memory_dir / "insights" / "release-checklist.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nid: insights-release-checklist\ncategory: insights\n---\n\n"
        f"{_CONTENT}\n",
        encoding="utf-8",
    )

    outcome = index_file(str(path), defer_embed=True)

    embed_stub.assert_not_called()
    assert outcome["embedded"] is False
    assert outcome["indexed_vec"] is False
    assert outcome["indexed_fts"] is True
    assert "deferred" in outcome["error"]
    assert _fts_matches(_KEYWORD) >= 1


def test_index_file_with_defer_embed_false_embeds(memory_dir, embed_stub):
    """Red: ``index_file`` takes no ``defer_embed`` (TypeError)."""
    path = memory_dir / "insights" / "release-checklist.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nid: insights-release-checklist\ncategory: insights\n---\n\n"
        f"{_CONTENT}\n",
        encoding="utf-8",
    )

    outcome = index_file(str(path), defer_embed=False)

    assert embed_stub.call_count >= 1
    assert outcome["embedded"] is True


# ── the MCP save timeout ────────────────────────────────────────────────────

_MCP_PROBE = textwrap.dedent(
    """
    import asyncio, json
    import httpx
    import palinode.mcp as mcp

    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["read_timeout"] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json={
            "file_path": "/m/insights/x.md", "rel_path": "insights/x.md",
            "id": "insights-x", "save_outcome": "created", "indexed": True,
            "embedded": True, "indexed_vec": True, "indexed_fts": True,
            "git_committed": True,
        })

    mcp._http_transport = httpx.MockTransport(handler)
    asyncio.run(mcp._tool_save({"content": "Release checklist.", "type": "Insight"}))
    print(json.dumps(seen))
    """
)


def _mcp_save_read_timeout(env_value: str | None) -> float:
    """The read timeout the MCP save puts on its HTTP call, in a fresh
    interpreter, with ``PALINODE_SAVE_TIMEOUT`` unset or set to *env_value*."""
    env = {k: v for k, v in os.environ.items() if k != "PALINODE_SAVE_TIMEOUT"}
    if env_value is not None:
        env["PALINODE_SAVE_TIMEOUT"] = env_value
    env.setdefault("PALINODE_LOG_LEVEL", "ERROR")
    proc = subprocess.run(
        [sys.executable, "-c", _MCP_PROBE],
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


def test_mcp_save_timeout_defaults_to_30_seconds():
    assert _mcp_save_read_timeout(None) == 30.0


def test_mcp_save_timeout_follows_the_env_var():
    """Red: the call still passes the 30 s default (30.0 != 90.0)."""
    assert _mcp_save_read_timeout("90") == 90.0
