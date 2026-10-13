"""An append whose entities hold a non-string item still saves.

``metadata.entities`` is written to the front matter as the caller sent it, on
every save. An append builds its ``## See also`` footer from the entity list its
front matter ends with, and that list can hold an item that is not a string. The
footer code reads each entity as ``category/slug`` text, so the append raises
``AttributeError`` before anything is written, and ``POST /save`` answers 500.
A replace save with the same metadata saves.

Contract these tests pin:

* an append whose entities hold a non-string item saves, and the item stays in
  the front matter as sent;
* the footer lists only the string entities;
* ``POST /save`` answers 200 for the same append;
* a replace save with the same metadata is unchanged.

Tests that carry the word *red* in their docstring fail on the feature head for
the stated reason; the rest pass there and must keep passing.

Real SQLite + tmp_path; the embedder is mocked.
"""
from __future__ import annotations

import importlib
import re
from unittest.mock import patch

import frontmatter
import pytest
from fastapi.testclient import TestClient

from palinode.core import store
from palinode.core.config import config
from palinode.core.save import save_memory

_FAKE_VECTOR = [0.01] * 1024

_FOOTER_MARKER = "<!-- palinode-auto-footer -->"


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


@pytest.fixture()
def client(memory_dir, monkeypatch):
    """A client that answers 500 instead of re-raising, so a crash is a status."""
    for _k in ("PALINODE_API_TOKEN", "PALINODE_API_TOKEN_FILE"):
        monkeypatch.delenv(_k, raising=False)
    import palinode.api.server as srv
    srv = importlib.reload(srv)
    srv._rate_counters.clear()
    with TestClient(srv.app, raise_server_exceptions=False) as c:
        yield c
    srv._rate_counters.clear()


def _save(**kwargs) -> str:
    kwargs.setdefault("type", "Insight")
    kwargs.setdefault("slug", "rollout")
    kwargs.setdefault("update_policy", "append")
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=_FAKE_VECTOR),
    ):
        return save_memory(**kwargs)["file_path"]


def _post(client, **body):
    body.setdefault("type", "Insight")
    body.setdefault("slug", "rollout")
    body.setdefault("update_policy", "append")
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=_FAKE_VECTOR),
    ):
        return client.post("/save", json=body)


def _footer_slugs(path: str) -> list[str]:
    body = frontmatter.load(path).content
    block = body.split(_FOOTER_MARKER)[1] if _FOOTER_MARKER in body else ""
    return re.findall(r"^- \[\[([^\]]+)\]\]$", block, flags=re.MULTILINE)


_NON_STRINGS = [123, None, 1.5, ["x"], {"k": "v"}]
_NON_STRING_IDS = ["int", "null", "float", "list", "mapping"]


# ── direct save ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("item", _NON_STRINGS, ids=_NON_STRING_IDS)
def test_an_append_with_a_non_string_entity_saves_and_keeps_it(memory_dir, item):
    """Red: ``AttributeError: ... has no attribute 'split'``, nothing written."""
    _save(content="First block.", entities=["project/a"])

    path = _save(content="Second block.", metadata={"entities": [item]})

    assert frontmatter.load(path).metadata["entities"] == [item]
    assert "Second block." in frontmatter.load(path).content


def test_an_append_footer_lists_only_the_string_entities(memory_dir):
    """Red: the same ``AttributeError`` before the footer is built."""
    _save(content="First block.", entities=["project/a"])

    path = _save(
        content="Second block.", metadata={"entities": [123, "person/bob", None]}
    )

    assert frontmatter.load(path).metadata["entities"] == [123, "person/bob", None]
    assert _footer_slugs(path) == ["bob"]


def test_an_append_with_only_non_string_entities_has_no_footer(memory_dir):
    """Red: the same ``AttributeError``."""
    _save(content="First block.", entities=["project/a"])

    path = _save(content="Second block.", metadata={"entities": [123]})

    assert _footer_slugs(path) == []
    assert _FOOTER_MARKER not in frontmatter.load(path).content


def test_an_append_carrying_a_prior_non_string_entity_saves(memory_dir):
    """Red: the carried list holds the item, so the footer step raises. The prior
    is written by hand, the only way such a file exists."""
    path = memory_dir / "insights" / "rollout.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nid: insights-rollout\ncategory: insights\ntype: Insight\n"
        "update_policy: append\nentities:\n- 123\n- project/a\n---\n\nFirst block.\n",
        encoding="utf-8",
    )

    saved = _save(content="Second block.")

    assert frontmatter.load(saved).metadata["entities"] == [123, "project/a"]
    assert _footer_slugs(saved) == ["a"]


@pytest.mark.parametrize("item", _NON_STRINGS, ids=_NON_STRING_IDS)
def test_a_replace_save_with_a_non_string_entity_saves(memory_dir, item):
    """Passes on the feature head. The replace path never reads the front matter
    list for its footer, and must keep saving. It also shows that nothing after
    the footer, indexing included, rejects the item."""
    _save(content="First block.", entities=["project/a"], update_policy="replace")

    path = _save(
        content="Second block.", metadata={"entities": [item]}, update_policy="replace"
    )

    assert frontmatter.load(path).metadata["entities"] == [item]


# ── POST /save ──────────────────────────────────────────────────────────────

def test_post_save_append_with_a_non_string_entity_answers_200(client, memory_dir):
    """Red: the route answers 500."""
    first = _post(client, content="First block.", entities=["project/a"])
    assert first.status_code == 200

    response = _post(
        client, content="Second block.", metadata={"entities": [123, "person/bob"]}
    )

    assert response.status_code == 200
    path = response.json()["file_path"]
    assert frontmatter.load(path).metadata["entities"] == [123, "person/bob"]
    assert _footer_slugs(path) == ["bob"]
