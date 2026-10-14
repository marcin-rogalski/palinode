"""Tests for `palinode worktree-reconcile` — stale dead-PID worktree cleanup.

Real git + tmp_path (no mocking). A bare repo serves as `origin` so the
upstream check has something real to resolve against.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from click.testing import CliRunner

from palinode.cli import main
from palinode.cli.worktree import (
    reconcile,
    _apply,
    _parse_porcelain,
    _pid_alive,
    _under_claude_worktrees,
)


def _run(args, cwd):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True, check=True
    )


DEAD_PID = 999_999  # not a live process (os.kill → ProcessLookupError)


@pytest.fixture()
def live_pid():
    """PID of a live helper process that the lock-reason regex can parse.

    ``os.getpid()`` is not enough: in a container pytest can run as PID 1–9,
    and ``_PID_RE`` deliberately ignores single-digit numbers, so the lock
    would read as "no PID" instead of "alive". PIDs are allocated upward, so
    spawning until one reaches two digits ends within a few processes.
    """
    procs = []
    try:
        while not procs or procs[-1].pid < 10:
            procs.append(subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            ))
        yield procs[-1].pid
    finally:
        for p in procs:
            p.kill()
            p.wait()


@pytest.fixture()
def repo(tmp_path):
    """A git repo with a bare `origin` and a first commit on `main`."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True,
                   capture_output=True)
    root = tmp_path / "repo"
    _run(["init", "-b", "main", str(root)], cwd=tmp_path)
    (root / "README.md").write_text("hi\n")
    _run(["add", "-A"], cwd=root)
    _run(["commit", "-m", "init"], cwd=root)
    _run(["remote", "add", "origin", str(origin)], cwd=root)
    _run(["push", "-u", "origin", "main"], cwd=root)
    (root / ".claude" / "worktrees").mkdir(parents=True)
    return root


def _add_worktree(root, name, branch, *, lock_pid, push=True, dirty=False):
    path = root / ".claude" / "worktrees" / name
    _run(["worktree", "add", "-b", branch, str(path), "main"], cwd=root)
    if push:
        _run(["push", "-u", "origin", branch], cwd=path)
    if dirty:
        (path / "scratch.txt").write_text("uncommitted\n")
    _run(["worktree", "lock", "--reason", f"claude session pid {lock_pid}", str(path)],
         cwd=root)
    return path.as_posix()


# ---------------------------------------------------------------------------
# Unit: pure helpers
# ---------------------------------------------------------------------------


def test_pid_alive_self_true_and_dead_false():
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(DEAD_PID) is False
    assert _pid_alive(0) is False
    assert _pid_alive(-1) is False


def test_pid_alive_child_process_lifecycle():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    try:
        assert _pid_alive(proc.pid) is True
    finally:
        proc.wait()
    assert _pid_alive(proc.pid) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Win32-specific error code tests")
def test_pid_alive_failsafe_on_access_denied_or_error_win32(monkeypatch):
    import ctypes
    import palinode.cli.worktree as wt

    monkeypatch.setattr(wt._kernel32, "OpenProcess", lambda *_: 0)
    # ERROR_ACCESS_DENIED = 5 -> fail-safe alive
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)
    assert _pid_alive(12345) is True
    # Unexpected error (e.g. 999) -> fail-safe alive
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 999)
    assert _pid_alive(12345) is True
    # ERROR_INVALID_PARAMETER = 87 -> dead (False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 87)
    assert _pid_alive(12345) is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific PermissionError test")
def test_pid_alive_failsafe_on_access_denied_posix(monkeypatch):
    def raise_eperm(pid, sig):
        raise PermissionError("Access denied")
    monkeypatch.setattr(os, "kill", raise_eperm)
    assert _pid_alive(12345) is True


def test_pid_alive_invalid_types_and_bounds():
    assert _pid_alive(True) is False
    assert _pid_alive(False) is False
    assert _pid_alive(None) is False  # type: ignore
    assert _pid_alive("12345") is False  # type: ignore
    assert _pid_alive(3.14) is False  # type: ignore
    assert _pid_alive(0) is False
    assert _pid_alive(-100) is False
    assert _pid_alive(0x1_0000_0000) is False  # > 32-bit DWORD


@pytest.mark.skipif(sys.platform != "win32", reason="Win32-specific process wait code tests")
def test_pid_alive_win32_wait_codes(monkeypatch):
    import palinode.cli.worktree as wt

    closed = []
    monkeypatch.setattr(wt._kernel32, "OpenProcess", lambda *_: 1234)
    monkeypatch.setattr(wt._kernel32, "CloseHandle", lambda h: closed.append(h) or True)

    # WAIT_TIMEOUT (0x102) -> running (True)
    monkeypatch.setattr(wt._kernel32, "WaitForSingleObject", lambda h, ms: 0x00000102)
    assert _pid_alive(100) is True
    assert closed == [1234]

    # WAIT_OBJECT_0 (0x0) -> exited (False)
    closed.clear()
    monkeypatch.setattr(wt._kernel32, "WaitForSingleObject", lambda h, ms: 0x00000000)
    assert _pid_alive(100) is False
    assert closed == [1234]

    # WAIT_FAILED / unexpected -> fail-safe (True)
    closed.clear()
    monkeypatch.setattr(wt._kernel32, "WaitForSingleObject", lambda h, ms: 0xFFFFFFFF)
    assert _pid_alive(100) is True
    assert closed == [1234]


def test_pid_alive_exit_code_259_dead():
    """Processes exiting with code 259 (STILL_ACTIVE on Windows) must be recognized as DEAD."""
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(259)"])
    proc.wait()
    if sys.platform == "win32":
        assert proc.returncode == 259
    assert _pid_alive(proc.pid) is False


def test_under_claude_worktrees_case_and_boundary():
    if sys.platform == "win32":
        # Casing on Windows should match
        assert _under_claude_worktrees("c:/repo", "C:/repo/.claude/worktrees/dead") is True
        assert _under_claude_worktrees("C:/repo", "c:/repo/.claude/worktrees/sub/deep") is True
        assert _under_claude_worktrees("c:/repo", "c:/repo/.claude/worktrees_fake/dead") is False
        assert _under_claude_worktrees("c:/repo", "c:/repo/.claude/other/dead") is False
    else:
        assert _under_claude_worktrees("/repo", "/repo/.claude/worktrees/dead") is True
        assert _under_claude_worktrees("/repo", "/repo/.claude/worktrees/sub/deep") is True
        assert _under_claude_worktrees("/repo", "/repo/.claude/worktrees_fake/dead") is False
        assert _under_claude_worktrees("/repo", "/repo/.claude/other/dead") is False



def test_parse_porcelain_extracts_locked_and_branch():
    text = (
        "worktree /r\n\n"
        "worktree /r/.claude/worktrees/a\nbranch refs/heads/feat/a\nlocked pid 5\n"
    )
    entries = _parse_porcelain(text)
    assert entries[1]["path"].endswith("/a")
    assert entries[1]["branch"] == "refs/heads/feat/a"
    assert entries[1]["locked"] and "pid 5" in entries[1]["lock_reason"]


# ---------------------------------------------------------------------------
# Verdicts across the four cases
# ---------------------------------------------------------------------------


def test_dead_clean_upstream_is_removed(repo):
    dead = _add_worktree(repo, "dead", "feat/dead", lock_pid=DEAD_PID)
    verdicts = {v.path: v for v in reconcile(str(repo))}
    assert verdicts[dead].action == "remove"
    assert verdicts[dead].pid_alive is False and verdicts[dead].clean and verdicts[dead].has_upstream


def test_alive_lock_is_skipped(repo, live_pid):
    alive = _add_worktree(repo, "alive", "feat/alive", lock_pid=live_pid)
    v = {x.path: x for x in reconcile(str(repo))}[alive]
    assert v.action == "skip" and "alive" in v.reason


def test_dirty_tree_is_skipped(repo):
    dirty = _add_worktree(repo, "dirty", "feat/dirty", lock_pid=DEAD_PID, dirty=True)
    v = {x.path: x for x in reconcile(str(repo))}[dirty]
    assert v.action == "skip" and "dirty" in v.reason


def test_no_upstream_is_skipped(repo):
    nou = _add_worktree(repo, "nou", "feat/nou", lock_pid=DEAD_PID, push=False)
    v = {x.path: x for x in reconcile(str(repo))}[nou]
    assert v.action == "skip" and "upstream" in v.reason


# ---------------------------------------------------------------------------
# Apply: only the safe one is removed; branch/commits survive
# ---------------------------------------------------------------------------


def test_apply_removes_only_dead_clean_upstream(repo, live_pid):
    dead = _add_worktree(repo, "dead", "feat/dead", lock_pid=DEAD_PID)
    alive = _add_worktree(repo, "alive", "feat/alive", lock_pid=live_pid)
    verdicts = reconcile(str(repo))
    removed = _apply(str(repo), verdicts)

    assert removed == [dead]
    assert not os.path.exists(dead), "removed worktree dir should be gone"
    assert os.path.exists(alive), "alive worktree must be untouched"
    # The branch of the removed worktree survives (remove drops only the dir).
    branches = _run(["branch", "--list", "feat/dead"], cwd=repo).stdout
    assert "feat/dead" in branches


# ---------------------------------------------------------------------------
# CLI: dry-run is the default and mutates nothing
# ---------------------------------------------------------------------------


def test_cli_dry_run_default_removes_nothing(repo, monkeypatch):
    dead = _add_worktree(repo, "dead", "feat/dead", lock_pid=DEAD_PID)
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(main, ["worktree-reconcile"])
    assert result.exit_code == 0, result.output
    assert "WOULD REMOVE" in result.output
    assert os.path.exists(dead), "dry-run must not remove anything"


def test_cli_execute_removes(repo, monkeypatch):
    dead = _add_worktree(repo, "dead", "feat/dead", lock_pid=DEAD_PID)
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(main, ["worktree-reconcile", "--execute"])
    assert result.exit_code == 0, result.output
    assert not os.path.exists(dead)
