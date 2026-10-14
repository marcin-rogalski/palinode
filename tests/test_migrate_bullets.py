"""Flat bullet-list migration parsing and CLI contracts."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from palinode.cli import main
from palinode.core.config import config
from palinode.migration.bullets import _parse_raw, parse_bullet_list, run_migration


FIXTURE = Path(__file__).parent / "fixtures" / "bullet_memories.md"


def test_parser_groups_leading_undated_and_date_boundaries() -> None:
    sections = parse_bullet_list(str(FIXTURE))

    assert [section["heading"] for section in sections] == [
        "Undated memories",
        "2026-03-01",
        "2026-03-14",
    ]
    assert sections[0]["body"].splitlines() == [
        "- No leading date at all.",
        "- Another undated memory about a project task.",
    ]
    assert sections[1]["body"].count("\n") == 1
    assert "Alternative bullet marker" in sections[2]["body"]
    assert "undated continuation" in sections[2]["body"]
    assert sections[2]["type"] == "decision"


def test_empty_file_is_a_clean_noop(tmp_path: Path) -> None:
    source = tmp_path / "empty.md"
    source.write_text("", encoding="utf-8")

    assert parse_bullet_list(str(source)) == []
    result = run_migration(str(source), dry_run=True)
    assert result == {
        "sections_found": 0,
        "files_created": [],
        "files_skipped": [],
        "files_renamed": [],
        "log_file": None,
        "dry_run": True,
    }


def test_repeated_date_is_merged_without_duplicate_output_paths() -> None:
    sections = _parse_raw(
        "- 2026-03-01 First memory.\n"
        "- 2026-03-14 Different date.\n"
        "- 2026-03-01 Later memory for the first date.\n"
    )

    assert [section["heading"] for section in sections] == [
        "2026-03-01",
        "2026-03-14",
    ]
    assert sections[0]["body"].splitlines() == [
        "- First memory.",
        "- Later memory for the first date.",
    ]


def test_cli_dry_run_reports_sections_without_writing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setattr(config, "memory_dir", str(memory_dir))

    result = CliRunner().invoke(
        main,
        ["migrate", "bullets", str(FIXTURE), "--dry-run", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sections_found"] == 3
    assert len(payload["files_created"]) == 3
    assert payload["dry_run"] is True
    assert list(memory_dir.iterdir()) == []


def test_migration_reuses_writer_with_bullet_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setattr(config, "memory_dir", str(memory_dir))
    monkeypatch.setattr(config.git, "auto_commit", False)

    result = run_migration(str(FIXTURE))

    assert result["sections_found"] == 3
    assert len(result["files_created"]) == 3
    assert result["log_file"].startswith("migrations/bullets-")
    for relative_path in result["files_created"]:
        content = (memory_dir / relative_path).read_text(encoding="utf-8")
        assert "source: bullets-migration" in content
    log = (memory_dir / result["log_file"]).read_text(encoding="utf-8")
    assert "# Bullet-list Migration" in log
    assert "Sections found: 3" in log


def test_cli_and_inventory_expose_bullet_migration() -> None:
    from palinode.core.parity import INVENTORY_INFRA

    assert "bullets" in main.commands["migrate"].commands
    assert "migrate bullets" in INVENTORY_INFRA["cli"]
