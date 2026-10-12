"""An ``update_policy="append"`` save grows the document; it does not rebuild
its front matter from the new save alone.

The reported failure: a record is saved with ``metadata`` (a ``title``, a
``status``), ``core`` and ``entities``, then appended to by a second save that
sends only the new content. The body ends up holding both blocks, but the front
matter holds neither the title, the status, the core flag nor the entity.
Today ``save_memory`` carries forward only ``created_at``, ``update_policy``
and ``epistemic`` from the file it is about to rewrite.

Contract these tests pin:

* when a save resolves to ``appended``, every key of the prior front matter that
  the save does not set carries forward, and a key the save sets wins;
* keys the save derives (``id``, ``category``, ``type``, ``content_hash``,
  ``last_updated``, ``auditable_memory``, ``source``) are never carried;
* ``ttl`` is never carried, ``expires_at`` is, so an append cannot move expiry;
* ``entities``: a non-empty list from the caller wins, otherwise the prior list
  carries;
* ``replace`` and first saves are unchanged.

Tests that carry only the word *red* in their docstring fail on v0.22.0 for the
stated reason; the rest pass on v0.22.0 and must keep passing.

Real SQLite + tmp_path; the embedder is mocked so the test does not need Ollama
(same shape as ``test_append_semantics.py``).
"""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta
from unittest.mock import patch

import frontmatter
import pytest
from fastapi.testclient import TestClient

from palinode.core import store
from palinode.core.config import config
from palinode.core.save import save_memory

_FAKE_VECTOR = [0.01] * 1024

_FIRST = "Item one: confirm the rollout window."
_SECOND = "Item two: the rollout window is confirmed."

_FIRST_METADATA = {"title": "Release checklist", "status": "open"}


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    """A fresh tmp memory_dir with a real SQLite DB; git off."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


@pytest.fixture()
def client(memory_dir, monkeypatch):
    """TestClient over the same tmp memory_dir (see test_append_semantics.py for
    why the server module is reloaded with the bearer token cleared)."""
    for _k in ("PALINODE_API_TOKEN", "PALINODE_API_TOKEN_FILE"):
        monkeypatch.delenv(_k, raising=False)
    import palinode.api.server as srv
    srv = importlib.reload(srv)
    srv._rate_counters.clear()
    with TestClient(srv.app, raise_server_exceptions=True) as c:
        yield c
    srv._rate_counters.clear()


def _save(memory_dir, **kwargs) -> str:
    """Direct ``save_memory`` call; returns the file path."""
    kwargs.setdefault("type", "Insight")
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=_FAKE_VECTOR),
    ):
        return save_memory(**kwargs)["file_path"]


def _post(client, **body):
    body.setdefault("type", "Insight")
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=_FAKE_VECTOR),
    ):
        return client.post("/save", json=body)


def _load(file_path: str):
    return frontmatter.load(file_path)


# ── characterisation: passes on v0.22.0 ─────────────────────────────────────

def test_append_adds_the_new_block_beneath_the_prior_body(memory_dir):
    path = _save(memory_dir, content=_FIRST, slug="rollout", update_policy="append")
    path2 = _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")

    assert path2 == path
    body = _load(path).content
    assert _FIRST in body
    assert _SECOND in body
    assert body.index(_FIRST) < body.index(_SECOND)


def test_append_preserves_created_at(memory_dir):
    path = _save(memory_dir, content=_FIRST, slug="rollout", update_policy="append")
    born = str(_load(path).metadata["created_at"])

    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")

    assert str(_load(path).metadata["created_at"]) == born


def test_append_does_not_carry_the_prior_source(memory_dir):
    """``source`` is derived on every save, so a later append that names none is
    attributed to the default, not to the first writer."""
    from palinode.core.save import default_source

    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 update_policy="append", source="agent-a")
    assert _load(path).metadata["source"] == "agent-a"

    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")

    assert _load(path).metadata["source"] == default_source()


def test_append_with_a_new_source_records_the_new_source(memory_dir):
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 update_policy="append", source="agent-a")
    _save(memory_dir, content=_SECOND, slug="rollout",
          update_policy="append", source="agent-b")

    assert _load(path).metadata["source"] == "agent-b"


def test_append_with_a_non_empty_entity_list_replaces_the_prior_list(memory_dir):
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 entities=["project/my-app"], update_policy="append")
    _save(memory_dir, content=_SECOND, slug="rollout",
          entities=["person/alice"], update_policy="append")

    assert _load(path).metadata["entities"] == ["person/alice"]


def test_explicit_core_false_on_an_append_is_written(memory_dir):
    """A value the save sets is not 'unset', even when it is falsy."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 core=True, update_policy="append")
    _save(memory_dir, content=_SECOND, slug="rollout",
          core=False, update_policy="append")

    assert _load(path).metadata["core"] is False


def test_first_save_with_append_has_only_what_it_set(memory_dir):
    """Nothing to carry on a first save: the front matter is exactly the save's."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, update_policy="append")

    meta = _load(path).metadata
    assert meta["title"] == "Release checklist"
    assert meta["status"] == "open"
    assert "expires_at" not in meta


def test_replace_resave_without_metadata_drops_the_caller_keys(memory_dir):
    """Replace keeps its meaning of an overwrite in place."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, entities=["project/my-app"], core=True,
                 update_policy="replace")
    assert _load(path).metadata["title"] == "Release checklist"

    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="replace")

    meta = _load(path).metadata
    assert "title" not in meta
    assert "status" not in meta
    assert "core" not in meta
    assert meta["entities"] == []
    assert _load(path).content.strip() == _SECOND


def test_flipping_an_append_record_to_replace_drops_the_caller_keys(memory_dir):
    """An explicit ``replace`` over a record that declared ``append`` is still an
    overwrite: nothing carries."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, update_policy="append")
    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="replace")

    meta = _load(path).metadata
    assert "title" not in meta
    assert "status" not in meta


# ── red on v0.22.0: the prior front matter is lost ──────────────────────────

def test_append_without_metadata_keeps_title_status_core_and_entity(memory_dir):
    """Red: title, status, core and the entity are absent after the second save.

    The body assertions pass on v0.22.0; the front matter ones are the failure.
    """
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, core=True,
                 entities=["project/my-app"], update_policy="append")

    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")

    post = _load(path)
    assert _FIRST in post.content
    assert _SECOND in post.content
    assert post.metadata["title"] == "Release checklist"
    assert post.metadata["status"] == "open"
    assert post.metadata["core"] is True
    assert post.metadata["entities"] == ["project/my-app"]


def test_append_carries_a_custom_key_and_a_list_value(memory_dir):
    """Red: any key counts, not only the ones the save path knows by name."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata={"reviewer": "dana", "labels": ["q4", "launch"]},
                 update_policy="append")

    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")

    meta = _load(path).metadata
    assert meta["reviewer"] == "dana"
    assert meta["labels"] == ["q4", "launch"]


def test_append_keeps_the_keys_it_does_not_set_and_the_key_it_does(memory_dir):
    """Red: ``title`` is lost today; ``status`` already follows the second save."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, update_policy="append")

    _save(memory_dir, content=_SECOND, slug="rollout",
          metadata={"status": "resolved"}, update_policy="append")

    meta = _load(path).metadata
    assert meta["status"] == "resolved"
    assert meta["title"] == "Release checklist"


def test_append_carries_the_prior_key_after_a_sticky_append(memory_dir):
    """Red: the second save names no policy; the file's declared ``append``
    (sticky) is what makes it an append, and the carry must follow it."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, update_policy="append")

    _save(memory_dir, content=_SECOND, slug="rollout")

    post = _load(path)
    assert _FIRST in post.content
    assert post.metadata["title"] == "Release checklist"


def test_repeated_appends_keep_the_keys_of_the_first_save(memory_dir):
    """Red: the carry is not one-deep. A third save still holds the first save's keys."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, update_policy="append")
    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")
    _save(memory_dir, content="Item three.", slug="rollout", update_policy="append")

    meta = _load(path).metadata
    assert meta["title"] == "Release checklist"
    assert meta["status"] == "open"


def test_append_does_not_move_expires_at(memory_dir):
    """Red: ``expires_at`` is dropped today. It must be carried unchanged, and
    the ``ttl`` that produced it must not come back."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata={"ttl": "7d"}, update_policy="append")
    expires = _load(path).metadata["expires_at"]
    assert "ttl" not in _load(path).metadata

    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")

    meta = _load(path).metadata
    assert str(meta["expires_at"]) == str(expires)
    assert "ttl" not in meta


def _instant(value) -> datetime:
    return datetime.fromisoformat(str(value))


def test_append_that_sets_a_ttl_resolves_a_new_expires_at(memory_dir):
    """A key the save sets wins, ``ttl`` included: the new ``ttl`` is resolved
    from now and replaces the carried ``expires_at``, and is not written itself.

    Passes on v0.22.0, where nothing is carried. It fails against a carry that
    lets the prior ``expires_at`` shadow the new ``ttl``.
    """
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata={"ttl": "7d"}, update_policy="append")
    first_expiry = _instant(_load(path).metadata["expires_at"])

    _save(memory_dir, content=_SECOND, slug="rollout",
          metadata={"ttl": "30d"}, update_policy="append")

    meta = _load(path).metadata
    new_expiry = _instant(meta["expires_at"])
    assert new_expiry > first_expiry
    assert abs(new_expiry - _instant(meta["last_updated"]) - timedelta(days=30)) < timedelta(hours=1)
    assert "ttl" not in meta


def test_append_that_sets_a_ttl_still_carries_the_other_keys(memory_dir):
    """Red: the same append, with the prior ``title`` and ``status`` checked.
    Both are absent on v0.22.0 (KeyError)."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata={**_FIRST_METADATA, "ttl": "7d"}, update_policy="append")
    first_expiry = _instant(_load(path).metadata["expires_at"])

    _save(memory_dir, content=_SECOND, slug="rollout",
          metadata={"ttl": "30d"}, update_policy="append")

    meta = _load(path).metadata
    assert meta["title"] == "Release checklist"
    assert meta["status"] == "open"
    assert _instant(meta["expires_at"]) > first_expiry
    assert "ttl" not in meta


def test_append_does_not_carry_derived_keys_over_the_new_values(memory_dir):
    """Red only through the carry: a naive carry that copies the whole prior
    front matter over the new one would freeze these at the first save's values.
    On v0.22.0 they are all recomputed, so this half passes; the title assertion
    is what ties it to the fix."""
    path = _save(memory_dir, content=_FIRST, slug="rollout",
                 metadata=_FIRST_METADATA, update_policy="append")
    first_hash = _load(path).metadata["content_hash"]

    _save(memory_dir, content=_SECOND, slug="rollout", update_policy="append")

    meta = _load(path).metadata
    assert meta["title"] == "Release checklist"
    assert meta["content_hash"] != first_hash
    assert meta["id"] == "insights-rollout"
    assert meta["category"] == "insights"
    assert meta["type"] == "Insight"


# ── through POST /save: the path MCP and the CLI call ───────────────────────

def test_rest_append_keeps_the_prior_front_matter(client):
    """Red: the same loss through the REST route."""
    first = _post(client, content=_FIRST, slug="rollout",
                  metadata=_FIRST_METADATA, core=True,
                  entities=["project/my-app"], update_policy="append")
    assert first.status_code == 200, first.text
    path = first.json()["file_path"]

    second = _post(client, content=_SECOND, slug="rollout")
    assert second.status_code == 200, second.text
    assert second.json()["file_path"] == path
    assert second.json()["save_outcome"] == "appended"

    post = _load(path)
    assert _FIRST in post.content
    assert _SECOND in post.content
    assert post.metadata["title"] == "Release checklist"
    assert post.metadata["status"] == "open"
    assert post.metadata["core"] is True
    assert post.metadata["entities"] == ["project/my-app"]


def test_rest_replace_resave_still_drops_the_caller_keys(client):
    first = _post(client, content=_FIRST, slug="rollout",
                  metadata=_FIRST_METADATA, update_policy="replace")
    path = first.json()["file_path"]

    second = _post(client, content=_SECOND, slug="rollout", update_policy="replace")
    assert second.json()["save_outcome"] == "replaced"

    meta = _load(path).metadata
    assert "title" not in meta
    assert "status" not in meta
