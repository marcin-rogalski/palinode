"""Tests for palinode.migration.openclaw — OpenClaw MEMORY.md import."""
from __future__ import annotations

import os
import re
import textwrap
from datetime import UTC
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from palinode.cli import main
from palinode.migration.openclaw import (
    _detect_type,
    _slugify,
    parse_memory_md,
    run_migration,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_MEMORY_MD = textwrap.dedent("""\
    # Memory

    Some preamble before any sections.

    ## Alice the Engineer
    Alice is a senior engineer on the platform team. She owns the deploy pipeline.
    She prefers async communication.

    ## Chose PostgreSQL over MySQL
    We decided to use PostgreSQL because it has better JSON support
    and the team already knew it. Rationale: cost + familiarity.

    ## Project Alpha roadmap
    Project Alpha covers the new billing sprint. Current milestone: v1.2.
    Tasks include migrating the payment service.

    ## Useful debugging trick
    When the API hangs, run `strace -p <pid>` to see which syscall is blocking.
""")


@pytest.fixture()
def memory_md_file(tmp_path: Path) -> Path:
    p = tmp_path / "MEMORY.md"
    p.write_text(SAMPLE_MEMORY_MD, encoding="utf-8")
    return p


@pytest.fixture()
def fake_memory_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    mem = tmp_path / "palinode"
    mem.mkdir()
    # Minimal git repo so the git commit call doesn't crash
    os.system(f"git init -q {mem} 2>/dev/null")
    os.system(f"git -C {mem} config user.email 'test@test.com' 2>/dev/null")
    os.system(f"git -C {mem} config user.name 'Test' 2>/dev/null")
    # write_memory_file's traversal guard reads the real global config
    # singleton, not palinode.migration.openclaw's patched `config` name —
    # patching one doesn't touch the other's already-bound reference to the
    # real object, so point the real one at the same directory too.
    from palinode.core.config import config as real_config
    monkeypatch.setattr(real_config, "memory_dir", str(mem))
    return mem


# ── parse_memory_md ───────────────────────────────────────────────────────────

def test_parse_sections(memory_md_file: Path) -> None:
    sections = parse_memory_md(str(memory_md_file))
    assert len(sections) == 4
    headings = [s["heading"] for s in sections]
    assert "Alice the Engineer" in headings
    assert "Chose PostgreSQL over MySQL" in headings
    assert "Project Alpha roadmap" in headings
    assert "Useful debugging trick" in headings


def test_parse_body_content(memory_md_file: Path) -> None:
    sections = parse_memory_md(str(memory_md_file))
    alice = next(s for s in sections if s["heading"] == "Alice the Engineer")
    assert "senior engineer" in alice["body"]
    assert "deploy pipeline" in alice["body"]


def test_parse_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    sections = parse_memory_md(str(p))
    assert sections == []


def test_parse_no_sections(tmp_path: Path) -> None:
    p = tmp_path / "flat.md"
    p.write_text("Just some flat text with no headings.", encoding="utf-8")
    sections = parse_memory_md(str(p))
    assert sections == []


# ── _detect_type ─────────────────────────────────────────────────────────────

def test_type_person_from_heading() -> None:
    assert _detect_type("Who is Bob", "Bob is a developer") == "person"


def test_type_person_from_body() -> None:
    assert _detect_type("Bob", "Bob is a person on the team") == "person"


def test_type_decision_decided() -> None:
    assert _detect_type("Database choice", "We decided to use Postgres") == "decision"


def test_type_decision_because() -> None:
    assert _detect_type("Framework", "Chose FastAPI because it is fast") == "decision"


def test_type_project() -> None:
    assert _detect_type("Alpha Sprint", "Project roadmap for the sprint") == "project"


def test_type_insight_fallback() -> None:
    assert _detect_type("Debugging tip", "Run strace to see syscalls") == "insight"


def test_type_person_wins_over_decision() -> None:
    # Person keyword takes priority even when decision words also present
    result = _detect_type("Who decided", "This person decided to quit")
    assert result == "person"


# ── _slugify ─────────────────────────────────────────────────────────────────

def test_slugify_basic() -> None:
    assert _slugify("Hello World") == "hello-world"


def test_slugify_special_chars() -> None:
    slug = _slugify("Chose PostgreSQL over MySQL!")
    assert slug == "chose-postgresql-over-mysql"


def test_slugify_truncates() -> None:
    long_heading = "a" * 100
    assert len(_slugify(long_heading)) <= 60


# ── run_migration — dry_run ───────────────────────────────────────────────────

def test_dry_run_creates_no_files(
    memory_md_file: Path, fake_memory_dir: Path
) -> None:
    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        result = run_migration(str(memory_md_file), dry_run=True)

    assert result["dry_run"] is True
    assert result["sections_found"] == 4
    assert len(result["files_created"]) == 4
    assert result["log_file"] is None

    # Nothing written to disk
    for subdir in ("people", "decisions", "projects", "insights"):
        assert not (fake_memory_dir / subdir).exists()


def test_dry_run_reports_correct_types(
    memory_md_file: Path, fake_memory_dir: Path
) -> None:
    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        result = run_migration(str(memory_md_file), dry_run=True)

    paths = result["files_created"]
    subdirs = {p.split("/")[0] for p in paths}
    assert "people" in subdirs
    assert "decisions" in subdirs
    assert "projects" in subdirs
    assert "insights" in subdirs


# ── run_migration — real write ────────────────────────────────────────────────

def test_migration_writes_files(
    memory_md_file: Path, fake_memory_dir: Path
) -> None:
    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        result = run_migration(str(memory_md_file), dry_run=False)

    assert result["sections_found"] == 4
    assert len(result["files_created"]) == 4
    assert len(result["files_skipped"]) == 0

    for rel_path in result["files_created"]:
        abs_path = fake_memory_dir / rel_path
        assert abs_path.exists(), f"Expected {abs_path} to exist"
        content = abs_path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "source: openclaw-migration" in content


def test_migration_creates_log_file(
    memory_md_file: Path, fake_memory_dir: Path
) -> None:
    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        result = run_migration(str(memory_md_file), dry_run=False)

    assert result["log_file"] is not None
    log_path = fake_memory_dir / result["log_file"]
    assert log_path.exists()
    log_content = log_path.read_text(encoding="utf-8")
    assert "OpenClaw Migration" in log_content
    assert "Files created: 4" in log_content


def test_migration_frontmatter_fields(
    memory_md_file: Path, fake_memory_dir: Path
) -> None:
    import yaml as _yaml

    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        result = run_migration(str(memory_md_file), dry_run=False)

    for rel_path in result["files_created"]:
        abs_path = fake_memory_dir / rel_path
        content = abs_path.read_text(encoding="utf-8")
        # Strip the --- delimiters and parse
        fm_block = content.split("---\n")[1]
        fm = _yaml.safe_load(fm_block)
        assert "id" in fm
        assert "category" in fm
        assert "name" in fm
        assert "last_updated" in fm
        assert fm["source"] == "openclaw-migration"


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_deduplication_skips_identical_content(
    memory_md_file: Path, fake_memory_dir: Path
) -> None:
    # Freeze datetime.now so both runs produce byte-identical frontmatter
    # (the dedup hash is over the full file content, including the
    # `last_updated` timestamp — a real wall-clock tick between the two
    # calls would defeat dedup and flake the test).
    from datetime import datetime as _real_datetime

    fixed_now = _real_datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    class _FrozenDatetime(_real_datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now if tz is None else fixed_now.astimezone(tz)

    with patch("palinode.migration.openclaw.config") as mock_cfg, \
         patch("palinode.migration.openclaw.datetime", _FrozenDatetime):
        mock_cfg.memory_dir = str(fake_memory_dir)
        _result1 = run_migration(str(memory_md_file), dry_run=False)
        result2 = run_migration(str(memory_md_file), dry_run=False)

    # Second run: all should be skipped, none created
    assert len(result2["files_created"]) == 0
    assert len(result2["files_skipped"]) == 4
    assert result2["files_renamed"] == []


def test_path_collision_preserves_existing_memory(
    tmp_path: Path, fake_memory_dir: Path
) -> None:
    source = tmp_path / "MEMORY.md"
    source.write_text("## Collision\nWe decided on the imported plan.\n", encoding="utf-8")
    original = fake_memory_dir / "decisions" / "collision.md"
    original.parent.mkdir()
    original_bytes = (
        b"---\ntitle: Existing decision\nassignee: Alice\nstatus: active\n"
        b"---\n\n# Collision\n\nKeep this decision.\n"
    )
    original.write_bytes(original_bytes)

    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        result = run_migration(str(source))

    assert original.read_bytes() == original_bytes
    assert result["files_skipped"] == []
    assert len(result["files_renamed"]) == 1
    rename = result["files_renamed"][0]
    assert rename["from"] == "decisions/collision.md"
    assert re.fullmatch(r"decisions/collision-openclaw-[0-9a-f]{8}\.md", rename["to"])
    assert result["files_created"] == [rename["to"]]
    assert "We decided on the imported plan." in (fake_memory_dir / rename["to"]).read_text()
    log = (fake_memory_dir / result["log_file"]).read_text()
    assert "Files renamed: 1" in log
    assert f"`{rename['from']}` → `{rename['to']}`" in log


def test_taken_renamed_path_gets_another_name(
    tmp_path: Path, fake_memory_dir: Path
) -> None:
    from datetime import datetime as _real_datetime

    fixed_now = _real_datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    class _FrozenDatetime(_real_datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now if tz is None else fixed_now.astimezone(tz)

    source = tmp_path / "MEMORY.md"
    source.write_text("## Collision\nWe decided on the imported plan.\n", encoding="utf-8")
    original = fake_memory_dir / "decisions" / "collision.md"
    original.parent.mkdir()
    original.write_bytes(b"Existing decision\n")

    with patch("palinode.migration.openclaw.config") as mock_cfg, \
         patch("palinode.migration.openclaw.datetime", _FrozenDatetime):
        mock_cfg.memory_dir = str(fake_memory_dir)
        preview = run_migration(str(source), dry_run=True)
        first_rename = preview["files_renamed"][0]["to"]
        taken = fake_memory_dir / first_rename
        taken.write_bytes(b"Another existing memory\n")
        result = run_migration(str(source))

    assert original.read_bytes() == b"Existing decision\n"
    assert taken.read_bytes() == b"Another existing memory\n"
    assert result["files_renamed"] == [
        {"from": "decisions/collision.md", "to": first_rename[:-3] + "-2.md"}
    ]
    assert result["files_created"] == [result["files_renamed"][0]["to"]]
    assert (fake_memory_dir / result["files_created"][0]).exists()


def test_cli_reports_path_collision_in_dry_run(
    tmp_path: Path, fake_memory_dir: Path
) -> None:
    source = tmp_path / "MEMORY.md"
    source.write_text("## Collision\nWe decided on the imported plan.\n", encoding="utf-8")
    original = fake_memory_dir / "decisions" / "collision.md"
    original.parent.mkdir()
    original.write_bytes(b"Existing decision\n")

    result = CliRunner().invoke(
        main, ["migrate", "openclaw", str(source), "--dry-run", "--format", "text"]
    )

    assert result.exit_code == 0, result.output
    assert "Files renamed:   1 (path collision)" in result.output
    assert "decisions/collision.md → decisions/collision-openclaw-" in result.output
    assert original.read_bytes() == b"Existing decision\n"
    assert sorted(path.name for path in original.parent.iterdir()) == ["collision.md"]


# ── Path validation ───────────────────────────────────────────────────────────

def test_rejects_null_byte_in_path() -> None:
    from palinode.migration.openclaw import _validate_source_path

    with pytest.raises(ValueError, match="Null bytes"):
        _validate_source_path("/some/path\x00evil")



def test_rerun_skips_sections_already_migrated_without_a_frozen_clock(
    memory_md_file: Path, fake_memory_dir: Path
) -> None:
    """The frontmatter carries a per-run `last_updated`; dedup must compare
    bodies, or a real re-run (clock moving) renames every section into a copy."""
    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        first = run_migration(str(memory_md_file), dry_run=False)
        second = run_migration(str(memory_md_file), dry_run=False)

    assert len(first["files_created"]) == 4
    assert second["files_created"] == []
    assert second["files_renamed"] == []
    assert sorted(second["files_skipped"]) == sorted(first["files_created"])


def test_renamed_file_gets_its_own_id(tmp_path: Path, fake_memory_dir: Path) -> None:
    import frontmatter

    original = fake_memory_dir / "decisions" / "collision.md"
    original.parent.mkdir()
    original.write_text("---\nid: decisions-collision\n---\n\nOriginal fact.\n", encoding="utf-8")
    source = tmp_path / "MEMORY.md"
    # Decision wording, so the section lands in decisions/ and collides.
    source.write_text("## Collision\nWe decided on the imported plan.\n", encoding="utf-8")

    with patch("palinode.migration.openclaw.config") as mock_cfg:
        mock_cfg.memory_dir = str(fake_memory_dir)
        result = run_migration(str(source))

    (rename,) = result["files_renamed"]
    meta = frontmatter.load(fake_memory_dir / rename["to"]).metadata
    assert meta["id"] == rename["to"][:-3].replace("/", "-", 1)
    assert meta["id"] != "decisions-collision"
