"""A same-slug body replacement retains record metadata, not old-body metadata."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import frontmatter
import pytest

from palinode.core import store
from palinode.core.config import config
from palinode.core.save import save_memory


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    monkeypatch.setattr(config.search, "retrieval_mode", "hybrid")
    monkeypatch.setattr(config.auto_summary, "enabled", True)
    monkeypatch.setattr(config.auto_summary, "min_content_chars", 1)
    store.init_db()
    return tmp_path


def _save(**kwargs):
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=[0.01] * 1024),
    ):
        return save_memory(type="Insight", **kwargs)


@pytest.mark.parametrize("policy", ["replace", None])
def test_rewrite_keeps_record_fields_and_drops_body_fields(memory_dir, policy):
    first = _save(
        slug="record", content="Old quoted passage.", update_policy=policy,
        source="agent-a",
        title="Original title", core=True, priority=4, confidence=0.8,
        entities=["person/alice"], external_refs={"ticket": "1713"},
        contradicts=["insights/other"], backed_by=["research/evidence"],
        sources=[{"ref": "research/evidence.md", "quote": "Old quoted passage."}],
        claims=[{"text": "Old claim", "source_id": "research/evidence.md",
                 "span": {"quote": "Old quoted passage."}}],
        metadata={"assignee": "Dana", "status": "open", "expires_at": "2099-01-01T00:00:00Z",
                  "description": "Describes old body", "description_pending": True,
                  "summary": "Summarizes old body", "summary_pending": True},
    )
    first_hash = frontmatter.load(first["file_path"]).metadata["content_hash"]
    second = _save(slug="record", content="A wholly new passage.",
                   update_policy=policy, source="agent-b")

    post = frontmatter.load(first["file_path"])
    meta = post.metadata
    assert second["save_outcome"] == "replaced"
    assert "Old quoted passage" not in post.content
    assert "A wholly new passage" in post.content
    assert meta["content_hash"] != first_hash
    assert meta["source"] == "agent-b"
    assert {key: meta[key] for key in ("title", "core", "priority", "confidence", "assignee", "status", "external_refs", "contradicts", "backed_by")} == {
        "title": "Original title", "core": True, "priority": 4, "confidence": 0.8,
        "assignee": "Dana", "status": "open", "external_refs": {"ticket": "1713"},
        "contradicts": ["insights/other"], "backed_by": ["research/evidence"],
    }
    assert meta["entities"] == ["person/alice"]
    assert str(meta["expires_at"]).startswith("2099-01-01")
    for key in ("description", "description_pending", "summary", "summary_pending", "claims", "sources"):
        assert key not in meta
    assert second["description_pending"] is True
    assert second["summary_pending"] is True
    assert "[[alice]]" in post.content


@pytest.mark.parametrize("policy", ["replace", None])
def test_rewrite_call_values_win_and_project_joins_carried_entities(memory_dir, policy):
    first = _save(slug="record", content="Old body.", update_policy=policy,
                  core=True, entities=["person/alice"],
                  metadata={"title": "Old", "status": "open", "ttl": "7d", "custom": "old"})
    old_expiry = datetime.fromisoformat(str(frontmatter.load(first["file_path"]).metadata["expires_at"]))
    _save(slug="record", content="New body.", update_policy=policy, core=False,
          title="New", project="palinode", metadata={"status": "resolved", "ttl": "30d", "custom": "new"})
    meta = frontmatter.load(first["file_path"]).metadata
    assert meta["core"] is False
    assert meta["title"] == "New"
    assert meta["status"] == "resolved"
    assert meta["custom"] == "new"
    assert meta["entities"] == ["person/alice", "project/palinode"]
    assert datetime.fromisoformat(str(meta["expires_at"])) > old_expiry + timedelta(days=20)
    assert "ttl" not in meta


def test_first_replace_save_has_no_prior_metadata(memory_dir):
    result = _save(slug="new", content="New body.", update_policy="replace")
    meta = frontmatter.load(result["file_path"]).metadata
    assert result["save_outcome"] == "created"
    assert meta["entities"] == []
    assert "title" not in meta
    assert "description" not in meta
    assert result["description_pending"] is True


@pytest.mark.parametrize("policy", ["replace", None])
def test_rewrite_entities_precedence_and_falsy_title(memory_dir, policy):
    first = _save(slug="record", content="Old body.", update_policy=policy,
                  title="Old title", entities=["person/alice"])
    _save(slug="record", content="New body.", update_policy=policy,
          title="", entities=["person/bob"])
    meta = frontmatter.load(first["file_path"]).metadata
    assert meta["title"] == ""
    assert meta["entities"] == ["person/bob"]

    _save(slug="record", content="Another body.", update_policy=policy, entities=[])
    assert frontmatter.load(first["file_path"]).metadata["entities"] == ["person/bob"]


def test_replace_ku_compat_restamps_version_and_rederives_lifecycle(memory_dir, monkeypatch):
    monkeypatch.setattr(config.ku_compat, "enabled", True)
    monkeypatch.setattr(config.ku_compat, "ku_version", "1.0")
    first = _save(slug="record", content="Old body.", update_policy="replace",
                  metadata={"status": "open", "lifecycle": "archived", "ku_version": "0.9"})
    _save(slug="record", content="New body.", update_policy="replace",
          metadata={"status": "active"})
    meta = frontmatter.load(first["file_path"]).metadata
    assert meta["ku_version"] == "1.0"
    assert meta["status"] == "active"
    assert meta["lifecycle"] == "active"


@pytest.mark.parametrize("policy", ["replace", None])
def test_rewrite_explicit_empty_record_links_win(memory_dir, policy):
    first = _save(slug="record", content="Old body.", update_policy=policy,
                  external_refs={"ticket": "1713"},
                  contradicts=["insights/other"], backed_by=["research/evidence"])
    _save(slug="record", content="New body.", update_policy=policy,
          external_refs={}, contradicts=[], backed_by=[])
    meta = frontmatter.load(first["file_path"]).metadata
    assert meta["external_refs"] == {}
    assert meta["contradicts"] == []
    assert meta["backed_by"] == []
