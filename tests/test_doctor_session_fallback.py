"""Tests for the ``session_floor_fallback`` doctor check.

Real files under ``tmp_path``; the check reads the cwd project's
``.claude/session-floor-fallback.jsonl`` and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from palinode.core.config import Config
from palinode.diagnostics.checks.session_fallback import session_floor_fallback
from palinode.diagnostics.registry import all_checks
from palinode.diagnostics.types import DoctorContext


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


def _ctx(tmp_path: Path) -> DoctorContext:
    mem = tmp_path / "mem"
    return DoctorContext(config=Config(memory_dir=str(mem), db_path=str(mem / ".palinode.db")))


def _fallback(project: Path) -> Path:
    path = project / ".claude" / "session-floor-fallback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_registered_as_fast_check() -> None:
    names = {fn.__name__: tags for fn, tags in all_checks()}
    assert "fast" in names["session_floor_fallback"]


def test_no_file_passes(tmp_path: Path, project: Path) -> None:
    res = session_floor_fallback(_ctx(tmp_path))
    assert res.passed
    assert res.name == "session_floor_fallback"


def test_empty_file_passes(tmp_path: Path, project: Path) -> None:
    _fallback(project).write_text("\n\n")
    assert session_floor_fallback(_ctx(tmp_path)).passed


def test_records_warn_with_count_and_replay_step(tmp_path: Path, project: Path) -> None:
    rec = {"summary": "s", "session_id": "abc", "cwd": str(project),
           "reason": "other", "trigger": "hook"}
    _fallback(project).write_text("\n".join(json.dumps(rec) for _ in range(3)) + "\n")
    res = session_floor_fallback(_ctx(tmp_path))
    assert not res.passed
    assert res.severity == "warn"
    assert "3 session-end capture(s)" in res.message
    assert "session-floor-fallback.jsonl" in res.message
    assert res.remediation and "palinode session-end" in res.remediation
