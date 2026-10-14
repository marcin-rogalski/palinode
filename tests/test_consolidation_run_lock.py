"""The store-local consolidation lock is shared by every run surface."""

from __future__ import annotations

import importlib
import json
import logging
import multiprocessing
import os
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import palinode.consolidation.cron as cron
import palinode.consolidation.activity_gate as activity_gate
import palinode.consolidation.run_lock as run_lock
import palinode.consolidation.runner as runner
import palinode.mcp as mcp
from palinode.api import server as srv
from palinode.api.server import app
from palinode.core.config import config
from palinode.mcp import _dispatch_tool


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    monkeypatch.setattr(config.auto_summary, "enabled", False)
    return tmp_path


@pytest.fixture()
def client(memory_dir):
    srv._rate_counters.clear()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    srv._rate_counters.clear()


def _write_lock(path: Path, **overrides) -> dict[str, object]:
    metadata: dict[str, object] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": time.time(),
        "token": "prior-owner",
    }
    metadata.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return metadata


def _hold_process_lock(memory_dir: str, ready, release) -> None:
    with run_lock.consolidation_run_lock(memory_dir):
        ready.set()
        release.wait(timeout=10)


def test_weekly_and_nightly_public_runners_share_one_lock(memory_dir):
    path = run_lock.consolidation_lock_path()

    with run_lock.consolidation_run_lock():
        for run in (runner.run_consolidation, runner.run_nightly):
            with pytest.raises(
                run_lock.ConsolidationAlreadyRunning,
                match="Consolidation is already running",
            ):
                run(dry_run=True)

    assert not path.exists()


def test_second_process_refuses_while_owner_is_alive(memory_dir):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_process_lock,
        args=(str(memory_dir), ready, release),
    )
    process.start()

    try:
        assert ready.wait(timeout=10), "child process did not acquire the lock"
        with pytest.raises(
            run_lock.ConsolidationAlreadyRunning,
            match=f"pid={process.pid}",
        ):
            with run_lock.consolidation_run_lock(memory_dir):
                pass
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert process.exitcode == 0
    assert not run_lock.consolidation_lock_path(memory_dir).exists()


def test_successful_weekly_and_nightly_runs_release_lock(memory_dir):
    path = run_lock.consolidation_lock_path()

    assert runner.run_consolidation(dry_run=True)["status"] == "no notes found"
    assert not path.exists()
    assert runner.run_nightly(dry_run=True)["status"] == "no_new_notes"
    assert not path.exists()


def test_exception_path_releases_lock(memory_dir, monkeypatch):
    path = run_lock.consolidation_lock_path()

    def fail_run(**kwargs):
        raise RuntimeError("injected consolidation failure")

    monkeypatch.setattr(runner, "_run_consolidation_unlocked", fail_run)
    with pytest.raises(RuntimeError, match="injected consolidation failure"):
        runner.run_consolidation()

    assert not path.exists()
    with run_lock.consolidation_run_lock() as reacquired:
        assert reacquired == path


def test_dead_pid_lock_is_reclaimed_with_reason(memory_dir, monkeypatch, caplog):
    path = run_lock.consolidation_lock_path()
    old = _write_lock(path, pid=424242)
    monkeypatch.setattr(run_lock, "_pid_is_alive", lambda pid: False)

    with caplog.at_level(logging.WARNING, logger="palinode.consolidation"):
        with run_lock.consolidation_run_lock():
            current = json.loads(path.read_text(encoding="utf-8"))
            assert current["token"] != old["token"]

    assert not path.exists()
    assert "Reclaimed stale consolidation lock" in caplog.text
    assert "owner pid 424242 is no longer running" in caplog.text


def test_ttl_stale_lock_is_reclaimed_with_reason(memory_dir, caplog):
    path = run_lock.consolidation_lock_path()
    _write_lock(
        path,
        hostname="another-host",
        created_at=time.time() - run_lock.LOCK_TTL_SECONDS - 1,
    )

    with caplog.at_level(logging.WARNING, logger="palinode.consolidation"):
        with run_lock.consolidation_run_lock():
            pass

    assert not path.exists()
    assert "exceeds TTL" in caplog.text


def test_live_local_owner_is_not_reclaimed_only_for_age(memory_dir, monkeypatch):
    path = run_lock.consolidation_lock_path()
    _write_lock(
        path,
        created_at=time.time() - run_lock.LOCK_TTL_SECONDS - 1,
    )
    monkeypatch.setattr(run_lock, "_pid_is_alive", lambda pid: True)

    with pytest.raises(run_lock.ConsolidationAlreadyRunning):
        with run_lock.consolidation_run_lock():
            pass


def test_recent_malformed_lock_fails_closed(memory_dir):
    path = run_lock.consolidation_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(
        run_lock.ConsolidationAlreadyRunning,
        match="owner metadata unavailable",
    ):
        with run_lock.consolidation_run_lock():
            pass


def test_failed_lock_metadata_write_does_not_leave_wedge(memory_dir, monkeypatch):
    path = run_lock.consolidation_lock_path()

    def fail_write(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(run_lock.os, "write", fail_write)
    with pytest.raises(OSError, match="disk full"):
        with run_lock.consolidation_run_lock():
            pass

    assert not path.exists()


def test_release_never_deletes_a_replacement_owner(memory_dir, caplog):
    path = run_lock.consolidation_lock_path()

    with caplog.at_level(logging.WARNING, logger="palinode.consolidation"):
        with run_lock.consolidation_run_lock():
            _write_lock(path, token="replacement-owner")

    assert path.exists()
    assert "owned by another run" in caplog.text
    path.unlink()


@pytest.mark.parametrize("nightly", [False, True])
def test_api_returns_clear_409_while_store_is_locked(client, nightly):
    with run_lock.consolidation_run_lock():
        response = client.post(
            "/consolidate",
            json={"dry_run": True, "nightly": nightly},
        )

    assert response.status_code == 409
    assert "Consolidation is already running" in response.json()["detail"]


def test_cli_relays_busy_detail_and_exits_nonzero():
    consolidate_cli = importlib.import_module("palinode.cli.consolidate")
    detail = "Consolidation is already running (pid=123)."
    request = httpx.Request("POST", "http://palinode.test/consolidate")
    response = httpx.Response(409, json={"detail": detail}, request=request)
    error = httpx.HTTPStatusError(
        "409 Conflict",
        request=request,
        response=response,
    )

    with patch.object(consolidate_cli.api_client, "consolidate", side_effect=error):
        result = CliRunner().invoke(consolidate_cli.consolidate, ["--dry-run"])

    assert result.exit_code == 1
    assert detail in result.output


@pytest.mark.asyncio
async def test_mcp_relays_busy_detail(monkeypatch):
    detail = "Consolidation is already running (pid=123)."

    class BusyResponse:
        status_code = 409
        text = json.dumps({"detail": detail})

    async def busy_post(*args, **kwargs):
        return BusyResponse()

    monkeypatch.setattr(mcp, "_post", busy_post)
    result = await _dispatch_tool("palinode_consolidate", {"dry_run": True})

    assert detail in result[0].text


def test_cron_logs_busy_detail_and_exits_nonzero(memory_dir, monkeypatch, caplog):
    def busy_run(**kwargs):
        raise run_lock.ConsolidationAlreadyRunning("Consolidation is already running")

    monkeypatch.setattr(config.consolidation, "enabled", True)
    monkeypatch.setattr(cron, "run_consolidation", busy_run)
    monkeypatch.setattr(sys, "argv", ["palinode.consolidation.cron"])

    with caplog.at_level(logging.ERROR, logger="palinode.consolidation.cron"):
        with pytest.raises(SystemExit) as raised:
            cron.main()

    assert raised.value.code == 1
    assert "Consolidation is already running" in caplog.text


@pytest.mark.parametrize("mode", ["weekly", "nightly"])
def test_cron_records_lock_refusal(memory_dir, monkeypatch, mode):
    monkeypatch.setattr(config.consolidation, "enabled", True)
    argv = ["palinode.consolidation.cron", "--ignore-gate"]
    if mode == "nightly":
        argv.append("--nightly")
    monkeypatch.setattr(sys, "argv", argv)

    with run_lock.consolidation_run_lock(memory_dir):
        with pytest.raises(SystemExit, match="1"):
            cron.main()

    history = activity_gate.run_history(mode, memory_dir)
    assert len(history) == 1
    assert history[0]["status"] == "lock_refused"
    assert "Consolidation is already running" in history[0]["error"]


def test_lock_uses_gitignored_operational_directory(memory_dir):
    path = run_lock.consolidation_lock_path()
    repository_root = Path(__file__).resolve().parents[1]

    assert path.relative_to(memory_dir).as_posix() == ".palinode/consolidation.lock"
    assert ".palinode/" in (repository_root / ".gitignore").read_text(encoding="utf-8")
