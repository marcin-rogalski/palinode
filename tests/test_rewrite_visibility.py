"""Privacy and status preservation across saves and layer splits."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import frontmatter
import pytest

from palinode.consolidation.layer_split import split_file
from palinode.core import store
from palinode.core.config import config
from palinode.core.save import SaveValidationError, save_memory


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


def _save(**kwargs):
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=[0.01] * 1024),
    ):
        return save_memory(type="Insight", slug="private-note", **kwargs)["file_path"]


def _source(memory_dir, *, body):
    path = memory_dir / "projects" / "private-note.md"
    path.parent.mkdir()
    path.write_text(
        "---\nid: projects-private-note\ncategory: projects\ncore: true\n"
        "visibility: private\nscope: agent/alice\naccess:\n- member/bob\n"
        f"---\n\n{body}",
        encoding="utf-8",
    )
    return path


def _privacy(path):
    meta = frontmatter.load(path).metadata
    return {key: meta[key] for key in ("visibility", "scope", "access")}


PRIVACY = {"visibility": "private", "scope": "agent/alice", "access": ["member/bob"]}


def test_replace_carries_prior_privacy_and_custom_keys(memory_dir):
    path = _save(content="First secret", update_policy="replace", metadata={**PRIVACY, "custom": "old"})

    _save(content="Second secret", update_policy="replace")

    assert _privacy(path) == PRIVACY
    assert frontmatter.load(path).metadata["custom"] == "old"


def test_no_policy_carries_prior_privacy_and_custom_keys(memory_dir):
    path = _save(content="First secret", metadata={**PRIVACY, "custom": "old"})

    _save(content="Second secret")

    assert _privacy(path) == PRIVACY
    assert frontmatter.load(path).metadata["custom"] == "old"

    corrupt = "---\nvisibility: [broken\n---\n\nA private note with damaged metadata.\n"
    Path(path).write_text(corrupt, encoding="utf-8")
    with pytest.raises(SaveValidationError):
        _save(content="Third secret")
    assert Path(path).read_text(encoding="utf-8") == corrupt


def test_split_status_inherits_source_privacy(memory_dir):
    source = _source(memory_dir, body="## Current Status\n\nA private status.\n")

    results = split_file(str(source))

    assert _privacy(results["status"]) == PRIVACY


def test_split_seeded_history_inherits_source_privacy(memory_dir):
    source = _source(memory_dir, body="## Architecture\n\nA private design.\n")

    results = split_file(str(source))

    assert _privacy(results["history"]) == PRIVACY


def test_split_existing_status_preserves_metadata_and_archives_prior_body(memory_dir):
    source = _source(memory_dir, body="## Current Status\n\nNew private status.\n")
    status = source.with_name("private-note-status.md")
    status.write_text(
        "---\nid: projects-private-note-status\ncategory: projects\n"
        "status: active\ncustom: keep-me\nvisibility: private\n"
        "scope: agent/alice\naccess:\n- member/bob\n---\n\n"
        "## Current Status\n\nPrior private status.\n",
        encoding="utf-8",
    )
    history = source.with_name("private-note-history.md")
    history.write_text(
        "---\nid: projects-private-note-history\ncategory: projects\n"
        "status: archived\ncustom: history-key\n---\n\n"
        "## History\n\nOlder archived status.\n",
        encoding="utf-8",
    )

    results = split_file(str(source))

    status_doc = frontmatter.load(status)
    history_doc = frontmatter.load(results["history"])
    assert status_doc.metadata["custom"] == "keep-me"
    assert status_doc.metadata["status"] == "active"
    assert _privacy(status) == PRIVACY
    assert "New private status." in status_doc.content
    assert "Prior private status." not in status_doc.content
    assert "Prior private status." in history_doc.content
    assert "Older archived status." in history_doc.content
    assert history_doc.metadata["status"] == "archived"
    assert history_doc.metadata["custom"] == "history-key"
    assert _privacy(results["history"]) == PRIVACY


def test_rewrite_over_a_file_without_frontmatter_still_saves(memory_dir):
    # No frontmatter means no privacy to lose, so the overwrite proceeds.
    path = memory_dir / "insights" / "private-note.md"
    path.parent.mkdir()
    path.write_text("Hand-written note with no frontmatter.\n", encoding="utf-8")

    _save(content="Saved over it", update_policy="replace")

    assert "Saved over it" in path.read_text(encoding="utf-8")


def test_rewrite_over_unparseable_frontmatter_is_refused(memory_dir):
    # Frontmatter that does not parse might hold `visibility: private`.
    path = memory_dir / "insights" / "private-note.md"
    path.parent.mkdir()
    original = "---\nvisibility: [private\n---\n\nSecret.\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(SaveValidationError):
        _save(content="Overwrite attempt", update_policy="replace")

    assert path.read_text(encoding="utf-8") == original


def test_split_sweep_skips_a_refused_file_and_splits_the_rest(memory_dir):
    from palinode.consolidation.layer_split import split_all_core_files

    projects = memory_dir / "projects"
    projects.mkdir()
    (projects / "clash.md").write_text(
        "---\nid: clash\ncategory: projects\ncore: true\nvisibility: private\nscope: agent/alice\n"
        "---\n\n## Current status\n\nNew status.\n",
        encoding="utf-8",
    )
    # An existing status layer restricted to a different scope: the split must refuse.
    (projects / "clash-status.md").write_text(
        "---\nid: clash-status\nvisibility: private\nscope: agent/bob\n---\n\nOld status.\n",
        encoding="utf-8",
    )
    (projects / "fine.md").write_text(
        "---\nid: fine\ncategory: projects\ncore: true\n---\n\n## Current status\n\nWork ongoing.\n",
        encoding="utf-8",
    )

    with patch("palinode.core.embedder.embed", return_value=[0.01] * 1024):
        stats = split_all_core_files()

    assert stats["files_refused"] == ["projects/clash.md"]
    assert stats["files_split"] == 1
    assert (projects / "fine-status.md").exists()
    assert "Old status." in (projects / "clash-status.md").read_text(encoding="utf-8")
