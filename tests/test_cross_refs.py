"""Tests for the mechanical cross-linking work — mechanical untyped cross-linking (`cross_refs`).

Covers the pure matchers (detect_refs / build_registry), then the file-mutating
orchestrator (update_file_cross_refs) on a real tmp memory_dir with git off and
the embedder untouched (no indexing happens here — we call the hook directly).
"""
from __future__ import annotations

import frontmatter
import pytest

from palinode.core import cross_refs
from palinode.core.config import config


# ── pure matcher: detect_refs ────────────────────────────────────────────────

def _reg(**entries):
    """Build a registry dict {ref: {slug, title}} from kwargs ref->title."""
    out = {}
    for ref, title in entries.items():
        slug = ref.split("/")[-1]
        out[ref] = {"slug": slug, "title": title}
    return out


def test_detect_matches_full_path_ref():
    reg = _reg(**{"decisions/drop-legacy-browser": ""})
    body = "As recorded in decisions/drop-legacy-browser, we moved on."
    assert cross_refs.detect_refs(body, reg) == ["decisions/drop-legacy-browser"]


def test_detect_matches_hyphenated_slug():
    reg = _reg(**{"decisions/drop-legacy-browser": ""})
    body = "We finally did drop-legacy-browser last week."
    assert cross_refs.detect_refs(body, reg) == ["decisions/drop-legacy-browser"]


def test_detect_matches_distinctive_title():
    reg = _reg(**{"decisions/dlb": "Drop Legacy Browser"})
    body = "The Drop Legacy Browser call was unpopular."
    assert cross_refs.detect_refs(body, reg) == ["decisions/dlb"]


def test_detect_skips_short_generic_slug():
    """A bare short slug like `api` must NOT match inside prose."""
    reg = _reg(**{"projects/api": ""})
    body = "The api responded quickly and rapidly."
    assert cross_refs.detect_refs(body, reg) == []


def test_detect_skips_stopword_title():
    reg = _reg(**{"insights/x": "Status"})
    body = "The status of the build is green."
    assert cross_refs.detect_refs(body, reg) == []


def test_detect_whole_token_no_substring_false_positive():
    reg = _reg(**{"projects/log": "Log"})
    body = "We use structured logging throughout the blog."
    # 'log' is short+generic (slug skipped, title is a stopword) → no match,
    # and even a longer token must not match as a substring of 'blog'/'logging'.
    assert cross_refs.detect_refs(body, reg) == []


def test_detect_multiple_and_sorted():
    reg = _reg(**{
        "decisions/drop-legacy-browser": "",
        "insights/embed-host-choice": "",
    })
    body = "embed-host-choice informed decisions/drop-legacy-browser."
    assert cross_refs.detect_refs(body, reg) == [
        "decisions/drop-legacy-browser", "insights/embed-host-choice",
    ]


# ── build_registry ───────────────────────────────────────────────────────────

def _write(path, title, body="body text"):
    path.parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body, title=title, category=path.parent.name)
    path.write_text(frontmatter.dumps(post) + "\n")


def test_build_registry_excludes_self_and_skip_dirs(tmp_path):
    _write(tmp_path / "decisions" / "a.md", "Alpha Decision")
    _write(tmp_path / "insights" / "b.md", "Beta Insight")
    _write(tmp_path / "daily" / "2026-06-28.md", "A Daily Note")  # skip dir

    reg = cross_refs.build_registry(str(tmp_path), exclude_ref="decisions/a")
    assert set(reg.keys()) == {"insights/b"}
    assert reg["insights/b"]["title"] == "Beta Insight"


def test_build_registry_skips_nested_skip_dir(tmp_path):
    """A skip dir at any depth is skipped — store prompts live at
    ``specs/prompts/*.md``, so a ``parts[0]`` test never excluded them."""
    _write(tmp_path / "projects" / "x.md", "Project Ecks")
    _write(tmp_path / "specs" / "prompts" / "compaction.md", "Compaction Prompt")
    _write(tmp_path / "specs" / "amr.md", "Auditable Memory Record")

    reg = cross_refs.build_registry(str(tmp_path))
    assert "specs/prompts/compaction" not in reg
    # Only the `prompts` segment is skipped; `specs/` itself stays linkable.
    assert set(reg.keys()) == {"projects/x", "specs/amr"}


# ── update_file_cross_refs (file mutation) ───────────────────────────────────

@pytest.fixture()
def memdir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config.git, "auto_commit", False)
    monkeypatch.setattr(config.capture.cross_refs, "enabled", True)
    monkeypatch.setattr(config.capture.cross_refs, "min_token_len", 6)
    return tmp_path


def _meta(path):
    return frontmatter.load(str(path)).metadata


def test_update_writes_cross_refs_when_mentioned(memdir):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why We Dropped It",
           body="This follows from decisions/drop-legacy-browser.")

    res = cross_refs.update_file_cross_refs(str(src))
    assert res["changed"] is True
    assert res["refs"] == ["decisions/drop-legacy-browser"]
    assert _meta(src)["cross_refs"] == ["decisions/drop-legacy-browser"]


def test_update_is_idempotent(memdir):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")

    first = cross_refs.update_file_cross_refs(str(src))
    assert first["changed"] is True
    # Second pass over the now-updated file must NOT rewrite (loop terminates).
    second = cross_refs.update_file_cross_refs(str(src))
    assert second["changed"] is False
    assert second["refs"] == ["decisions/drop-legacy-browser"]


def test_update_removes_stale_cross_refs(memdir):
    src = memdir / "insights" / "why.md"
    src.parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post("No mentions here at all.", title="Why",
                            category="insights",
                            cross_refs=["decisions/gone"])
    src.write_text(frontmatter.dumps(post) + "\n")

    res = cross_refs.update_file_cross_refs(str(src))
    assert res["changed"] is True
    assert res["refs"] == []
    assert "cross_refs" not in _meta(src)


def test_update_no_write_when_nothing_found_and_none_existing(memdir):
    src = memdir / "insights" / "lonely.md"
    _write(src, "Lonely", body="A memory that mentions nobody.")
    res = cross_refs.update_file_cross_refs(str(src))
    assert res["changed"] is False
    assert res["refs"] == []
    assert "cross_refs" not in _meta(src)


def test_update_respects_disabled_config(memdir, monkeypatch):
    monkeypatch.setattr(config.capture.cross_refs, "enabled", False)
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    res = cross_refs.update_file_cross_refs(str(src))
    assert res["changed"] is False
    assert "cross_refs" not in _meta(src)


def test_update_skips_store_prompts_and_never_links_to_them(memdir):
    """Store prompts at ``specs/prompts/*.md`` are outside the cross-ref world.

    The old ``parts[0]`` skip test missed them (``parts[0] == "specs"``), so the
    auto-updater rewrote their frontmatter — changing the whole-file sha256 that
    ``prompt sync`` compares against ``shipped-hashes.json``, which made every
    store prompt look operator-edited and unrefreshable.
    """
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    prompt = memdir / "specs" / "prompts" / "compaction.md"
    _write(prompt, "Compaction Prompt",
           body="Apply decisions/drop-legacy-browser when compacting projects/x.")
    before = prompt.read_bytes()

    res = cross_refs.update_file_cross_refs(str(prompt))
    assert res["changed"] is False
    assert res["refs"] == []
    assert prompt.read_bytes() == before  # byte-identical: the shipped hash holds

    # A real memory still cross-links — and is never offered the prompt as a target.
    src = memdir / "projects" / "x.md"
    _write(src, "Project Ecks",
           body="Per specs/prompts/compaction, see decisions/drop-legacy-browser.")
    res = cross_refs.update_file_cross_refs(str(src))
    assert res["changed"] is True
    assert res["refs"] == ["decisions/drop-legacy-browser"]
    assert _meta(src)["cross_refs"] == ["decisions/drop-legacy-browser"]


def test_update_excludes_self_reference(memdir):
    """A memory that mentions its own ref/title doesn't cross-link to itself."""
    src = memdir / "insights" / "self-aware.md"
    _write(src, "Self Aware",
           body="This is insights/self-aware talking about Self Aware.")
    res = cross_refs.update_file_cross_refs(str(src))
    assert res["refs"] == []
    assert res["changed"] is False


# ── update_file_cross_refs racing another writer ─────────────────────────────

def _hook_scan(monkeypatch, on_scan):
    """Run ``on_scan(n)`` at the start of the nth registry scan, the slow step
    between the read and the write."""
    real = cross_refs.build_registry
    scans: list[int] = []

    def scan(*args, **kwargs):
        scans.append(len(scans) + 1)
        on_scan(len(scans))
        return real(*args, **kwargs)

    monkeypatch.setattr(cross_refs, "build_registry", scan)
    return scans


def _write_move(path, body):
    post = frontmatter.Post(body, title="Why", category="insights", state="review")
    path.write_text(frontmatter.dumps(post) + "\n")


def test_update_keeps_a_write_that_lands_during_the_scan(memdir, monkeypatch):
    """Another writer saves the file while the registry is scanned. Writing back
    the copy read before the scan drops that save (a lost hand_task move)."""
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    _write(memdir / "decisions" / "keep-the-old-api.md", "Keep The Old Api")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    moved = "see decisions/drop-legacy-browser and decisions/keep-the-old-api\n\nHanded to the tester."
    _hook_scan(monkeypatch, lambda n: _write_move(src, moved) if n == 1 else None)

    res = cross_refs.update_file_cross_refs(str(src))

    saved = frontmatter.load(str(src))
    assert saved.content == moved
    assert saved.metadata["state"] == "review"
    assert saved.metadata["cross_refs"] == ["decisions/drop-legacy-browser", "decisions/keep-the-old-api"]
    assert res == {
        "changed": True,
        "refs": ["decisions/drop-legacy-browser", "decisions/keep-the-old-api"],
        "error": None,
    }


def test_update_with_content_rescans_a_file_saved_since_that_content(memdir):
    """The watcher reads the file, indexes it, then passes what it read; a save
    in between must survive, and the refs come from the saved file."""
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    _write(memdir / "decisions" / "keep-the-old-api.md", "Keep The Old Api")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    read_by_watcher = src.read_text(encoding="utf-8")
    _write_move(src, "now see decisions/keep-the-old-api")

    res = cross_refs.update_file_cross_refs(str(src), content=read_by_watcher)

    saved = frontmatter.load(str(src))
    assert saved.content == "now see decisions/keep-the-old-api"
    assert saved.metadata["cross_refs"] == ["decisions/keep-the-old-api"]
    assert res["refs"] == ["decisions/keep-the-old-api"]
    assert res["changed"] is True


def test_update_gives_up_on_a_file_saved_during_every_scan(memdir, monkeypatch):
    """A file written without pause must not hold the watcher, and the last
    save must stand untouched."""
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    scans = _hook_scan(monkeypatch, lambda n: _write_move(src, f"see decisions/drop-legacy-browser, save {n}"))

    res = cross_refs.update_file_cross_refs(str(src))

    assert len(scans) == 3
    assert src.read_text(encoding="utf-8") == frontmatter.dumps(
        frontmatter.Post(f"see decisions/drop-legacy-browser, save {len(scans)}", title="Why",
                         category="insights", state="review")
    ) + "\n"
    assert res["changed"] is False
    assert res["error"] == "changed during each of 3 scans"


def test_update_never_recreates_a_file_removed_during_the_scan(memdir, monkeypatch):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    _hook_scan(monkeypatch, lambda n: src.unlink())

    res = cross_refs.update_file_cross_refs(str(src))

    assert not src.exists()
    assert res["changed"] is False
    assert "removed" in res["error"]


def test_update_leaves_alone_a_file_saved_as_bytes_it_cannot_decode(memdir, monkeypatch):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    _hook_scan(monkeypatch, lambda n: src.write_bytes(b"\xff\xfe not utf-8"))

    res = cross_refs.update_file_cross_refs(str(src))

    assert src.read_bytes() == b"\xff\xfe not utf-8"
    assert res["changed"] is False
    assert res["error"].startswith("read failed:")


def test_update_reports_a_read_failure_when_the_file_becomes_a_folder_during_the_scan(memdir, monkeypatch):
    """Any OSError on the re-read but a missing file is a read failure, never an escape."""
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")

    def replace_with_folder(n):
        src.unlink()
        src.mkdir()

    _hook_scan(monkeypatch, replace_with_folder)

    res = cross_refs.update_file_cross_refs(str(src))

    assert src.is_dir()
    assert res["changed"] is False
    assert res["error"].startswith("read failed: ")


def test_update_reports_a_parse_failure_for_a_file_that_will_not_decode(memdir):
    src = memdir / "insights" / "why.md"
    src.parent.mkdir(parents=True, exist_ok=True)
    undecodable = b"---\ntitle: Why\n---\nsee decisions/drop-legacy-browser \xff\xfe\n"
    src.write_bytes(undecodable)

    res = cross_refs.update_file_cross_refs(str(src))

    assert src.read_bytes() == undecodable
    assert res["changed"] is False
    assert res["error"].startswith("parse failed: ")


def test_update_writes_nothing_when_a_save_during_the_scan_already_holds_the_right_refs(memdir, monkeypatch):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    _write(memdir / "decisions" / "keep-the-old-api.md", "Keep The Old Api")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    saved = frontmatter.dumps(
        frontmatter.Post("now see decisions/keep-the-old-api", title="Why", category="insights",
                         cross_refs=["decisions/keep-the-old-api"])
    ) + "\n"
    scans = _hook_scan(monkeypatch, lambda n: src.write_text(saved, encoding="utf-8") if n == 1 else None)

    res = cross_refs.update_file_cross_refs(str(src))

    assert src.read_text(encoding="utf-8") == saved
    assert len(scans) == 2
    assert res == {"changed": False, "refs": ["decisions/keep-the-old-api"], "error": None}


def test_update_leaves_the_bytes_of_a_file_whose_refs_are_unchanged(memdir):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    cross_refs.update_file_cross_refs(str(src))
    written = src.read_bytes()
    mtime_ns = src.stat().st_mtime_ns

    res = cross_refs.update_file_cross_refs(str(src))

    assert res["changed"] is False
    assert src.read_bytes() == written
    assert src.stat().st_mtime_ns == mtime_ns


def test_update_scans_again_once_after_one_save_and_writes_once(memdir, monkeypatch):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    _hook_scan(monkeypatch, lambda n: _write_move(src, "see decisions/drop-legacy-browser, moved") if n == 1 else None)

    scans = []
    real = cross_refs.build_registry
    monkeypatch.setattr(cross_refs, "build_registry", lambda *a, **k: scans.append(1) or real(*a, **k))

    res = cross_refs.update_file_cross_refs(str(src))

    assert len(scans) == 2
    assert frontmatter.load(str(src)).content == "see decisions/drop-legacy-browser, moved"
    assert res["changed"] is True
    assert res["error"] is None


def test_update_still_writes_when_the_third_scan_is_the_first_quiet_one(memdir, monkeypatch):
    _write(memdir / "decisions" / "drop-legacy-browser.md", "Drop Legacy Browser")
    src = memdir / "insights" / "why.md"
    _write(src, "Why", body="see decisions/drop-legacy-browser")
    scans = _hook_scan(
        monkeypatch,
        lambda n: _write_move(src, f"see decisions/drop-legacy-browser, save {n}") if n < 3 else None,
    )

    res = cross_refs.update_file_cross_refs(str(src))

    saved = frontmatter.load(str(src))
    assert len(scans) == 3
    assert saved.content == "see decisions/drop-legacy-browser, save 2"
    assert saved.metadata["cross_refs"] == ["decisions/drop-legacy-browser"]
    assert res["changed"] is True
    assert res["error"] is None
