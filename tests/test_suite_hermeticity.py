"""The test suite neither reads nor writes the developer's real palinode install.

Two directions, both fixed at the top of ``tests/conftest.py`` and in the
import-time writers themselves:

* **Reads.** An exported ``PALINODE_DIR`` naming a client-only config (remote
  ``services.api.host``) used to abort every run at collection with the
  bearer-auth bind-gate INTERNALERROR, because importing the API server built
  the config from the developer's shell.
* **Writes.** Importing ``palinode.api.server``, ``palinode.indexer.watcher`` or
  ``palinode.mcp`` created ``logs/`` (with ``operations.jsonl``) and ``.audit/``
  in whatever store the process default resolved to, which outside a sandbox is
  the developer's ``~/palinode``.

Neither failure is visible from inside a normal run, so the guard runs a
throwaway pytest session over a *copy of the real conftest* with a fake ``HOME``
and a hostile environment, then inspects both directories from outside.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import palinode
from palinode.core.log_files import LazyDirFileHandler

REPO_ROOT = Path(palinode.__file__).resolve().parents[1]
CONFTEST = REPO_ROOT / "tests" / "conftest.py"

_HOSTILE_HOST = "memory.example.invalid"

# Imports at module level on purpose: the original failure happened at
# *collection*, before any fixture could redirect config, so that is when the
# import-time writers have to be observed.
_PROBE = '''
import os

import palinode.api.server  # noqa: F401
import palinode.indexer.watcher  # noqa: F401
import palinode.mcp  # noqa: F401
from palinode.core.config import config

_DEFAULT_DIR = config.memory_dir
_API_HOST = config.services.api.host
_CREATED_AT_IMPORT = sorted(
    name for name in (".audit", "logs") if os.path.exists(os.path.join(_DEFAULT_DIR, name))
)


def test_probe():
    hostile = os.environ["PALINODE_TEST_HOSTILE_DIR"]
    home = os.environ["HOME"]
    assert not _DEFAULT_DIR.startswith(hostile), _DEFAULT_DIR
    assert not _DEFAULT_DIR.startswith(home), _DEFAULT_DIR
    assert _API_HOST != "{host}", _API_HOST
    for name in ("PALINODE_API", "PALINODE_API_URL", "PALINODE_PROJECT", "PALINODE_API_HOST"):
        assert name not in os.environ, name
    assert _CREATED_AT_IMPORT == [], (
        f"importing palinode created {{_CREATED_AT_IMPORT}} in {{_DEFAULT_DIR}}"
    )
'''.format(host=_HOSTILE_HOST)


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.fixture(scope="module")
def hostile_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("hermeticity")
    home = root / "home"
    home.mkdir()
    hostile = root / "client-config"
    hostile.mkdir()
    # A CLI-only client config: the API is a remote host this machine talks to.
    (hostile / "palinode.config.yaml").write_text(
        f'services:\n  api:\n    host: "{_HOSTILE_HOST}"\n    port: 6340\n',
        encoding="utf-8",
    )
    suite = root / "suite"
    suite.mkdir()
    shutil.copy(CONFTEST, suite / "conftest.py")
    (suite / "test_probe.py").write_text(_PROBE, encoding="utf-8")

    hostile_before = _tree(hostile)
    env = {
        **os.environ,
        "HOME": str(home),
        "PALINODE_DIR": str(hostile),
        "PALINODE_API": f"http://{_HOSTILE_HOST}:6340",
        "PALINODE_API_URL": f"http://{_HOSTILE_HOST}:6340",
        "PALINODE_API_HOST": _HOSTILE_HOST,
        "PALINODE_PROJECT": "someone-elses-project",
        "PALINODE_TEST_HOSTILE_DIR": str(hostile),
        "PYTHONPATH": str(REPO_ROOT),
    }
    # The session must be configured by its own arguments alone.
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PALINODE_TEST_DIR", None)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(suite)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    return {
        "proc": proc,
        "output": proc.stdout + proc.stderr,
        "home_created": _tree(home),
        "hostile_before": hostile_before,
        "hostile_after": _tree(hostile),
    }


def test_hostile_environment_does_not_abort_the_session(hostile_run: dict) -> None:
    output = hostile_run["output"]
    assert "INTERNALERROR" not in output, output[-4000:]
    assert hostile_run["proc"].returncode == 0, output[-4000:]
    assert "1 passed" in output, output[-4000:]


def test_session_writes_nothing_into_home(hostile_run: dict) -> None:
    # Scoped to palinode-named paths: an interpreter or third-party library
    # caching under ``$HOME`` is not this suite's concern, and asserting an
    # empty HOME would make the guard depend on what else is installed.
    offenders = sorted(p for p in hostile_run["home_created"] if "palinode" in p.lower())
    assert offenders == [], f"test session wrote into $HOME: {offenders}"


def test_session_writes_nothing_into_the_exported_palinode_dir(hostile_run: dict) -> None:
    assert hostile_run["hostile_after"] == hostile_run["hostile_before"], (
        "test session wrote into the exported PALINODE_DIR: "
        f"{sorted(hostile_run['hostile_after'] - hostile_run['hostile_before'])}"
    )


def test_lazy_handler_creates_nothing_until_a_record_is_written(tmp_path: Path) -> None:
    target = tmp_path / "store" / "logs" / "operations.jsonl"
    handler = LazyDirFileHandler(str(target))
    try:
        assert not target.parent.exists()
        handler.emit(logging.LogRecord("palinode", logging.INFO, __file__, 1, "hi", None, None))
        assert target.read_text(encoding="utf-8").strip() == "hi"
    finally:
        handler.close()


def test_git_processes_never_start_auto_maintenance(tmp_path: Path) -> None:
    """Every git the suite starts sees ``gc.auto=0`` and ``maintenance.auto=false``.

    Background maintenance creates and deletes ``objects/maintenance.lock``
    while a test copies or removes the repository, which is how the erasure
    runbook's fixture lost a file mid-``rmtree``. The setting arrives through
    ``GIT_CONFIG_*`` in ``tests/conftest.py``, so it reaches clones and
    palinode's own commits too; this checks a fresh repo that has no local
    config of its own.
    """
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    def _get(key: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "config", "--get", key],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    assert _get("gc.auto") == "0"
    assert _get("maintenance.auto") == "false"
