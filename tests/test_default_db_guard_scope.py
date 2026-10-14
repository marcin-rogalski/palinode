"""Scope of the default-db guard: normal suites guarded, ``tests/live/`` exempt.

The autouse ``_no_writes_to_the_default_db`` fixture in ``tests/conftest.py``
fails any test that mutates the database ``config.db_path`` resolved to before
patching — on a developer machine, their real memory store. ``tests/live/`` is
the one suite where writing that database is the point: the harness exports
``PALINODE_DIR`` at a throwaway memory dir and drives a real server against it,
so the resolved default and the fixture are the same directory.

Both halves are invisible from inside a normal run — the guard only speaks when
it fires, and the exemption only speaks when it doesn't — so neither half has a
natural failure mode that a passing suite would surface. Losing the guard shows
up as a developer's database quietly changing; losing the exemption shows up as
every writing live test passing and then erroring in teardown, which is how the
un-scoped version shipped.

Both are asserted here by running a throwaway pytest session over a *copy of the
real conftest*, in a sandbox memory dir, with two identical offender modules:
one at the tests root, one under ``live/``.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import palinode

REPO_ROOT = Path(palinode.__file__).resolve().parents[1]
CONFTEST = REPO_ROOT / "tests" / "conftest.py"

# A test that writes the process default database — the thing the guard exists
# to catch. It resolves the path through the conftest's own ``_default_db_path``
# fixture, which is how a real offender's store reaches it.
#
# The assertion is a safety interlock, not the subject: if the throwaway session
# ever resolved a default outside its sandbox, this fails loudly instead of
# writing to the developer's real memory database.
_OFFENDER = '''
import os
import sqlite3


def test_writes_the_default_db(_default_db_path):
    sandbox = os.environ["PALINODE_GUARD_SCOPE_SANDBOX"]
    assert str(_default_db_path).startswith(sandbox), (
        f"refusing to write {_default_db_path}: outside sandbox {sandbox}"
    )
    conn = sqlite3.connect(_default_db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS probe (n INTEGER)")
        conn.execute("INSERT INTO probe VALUES (1)")
        conn.commit()
    finally:
        conn.close()
'''

_NORMAL_MODULE = "test_offender_normal.py"
_LIVE_MODULE = "test_offender_live.py"


def _outcomes(output: str, module: str) -> list[str]:
    """Every verbose outcome word reported for *module*, in report order.

    A test that passes and then fails in teardown reports twice — ``PASSED``
    for the call phase, ``ERROR`` for teardown — so the first line alone would
    call the un-scoped guard's behaviour a pass.
    """
    words = ("PASSED", "FAILED", "ERROR", "SKIPPED")
    found = []
    for line in output.splitlines():
        # Progress lines only: ``<nodeid> OUTCOME [ nn%]``. The short summary
        # reports the same failures leading with the outcome word instead, and
        # counting both would double every non-pass.
        fields = line.split()
        if len(fields) < 2 or "::" not in fields[0] or module not in fields[0]:
            continue
        if fields[1] in words:
            found.append(fields[1])
    return found


@pytest.fixture(scope="module")
def guard_scope_run(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Run the two offenders under a copy of the real conftest; return output.

    Module-scoped: the session costs a full palinode import, and both tests read
    the same output.
    """
    root = tmp_path_factory.mktemp("guard-scope")
    memory_dir = root / "memory"
    (memory_dir / ".audit").mkdir(parents=True)
    suite = root / "suite"
    (suite / "live").mkdir(parents=True)

    # Pin the sandbox as the resolved default from both config sources that can
    # set it: the env override, and a config file inside the dir it points at.
    (memory_dir / "palinode.config.yaml").write_text(
        f"memory_dir: {memory_dir}\ndb_path: .palinode.db\n", encoding="utf-8"
    )
    default_db = memory_dir / ".palinode.db"
    # Pre-create it: on a developer machine the default database already exists,
    # so mutation — not creation — is the case the guard has to catch.
    conn = sqlite3.connect(default_db)
    conn.execute("CREATE TABLE IF NOT EXISTS seed (n INTEGER)")
    conn.commit()
    conn.close()

    shutil.copy(CONFTEST, suite / "conftest.py")
    (suite / _NORMAL_MODULE).write_text(_OFFENDER, encoding="utf-8")
    (suite / "live" / _LIVE_MODULE).write_text(_OFFENDER, encoding="utf-8")

    env = {
        **os.environ,
        # The conftest replaces an inherited ``PALINODE_DIR`` with an empty
        # sandbox of its own; ``PALINODE_TEST_DIR`` is how a harness names one.
        "PALINODE_TEST_DIR": str(memory_dir),
        "PALINODE_GUARD_SCOPE_SANDBOX": str(root),
        "PYTHONPATH": str(REPO_ROOT),
    }
    # This session must be configured by nothing but its own arguments — an
    # exported PYTEST_ADDOPTS would otherwise reach in and change what runs.
    env.pop("PYTEST_ADDOPTS", None)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", str(suite)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    output = proc.stdout + proc.stderr
    assert "refusing to write" not in output, (
        f"sandbox interlock tripped — the throwaway session resolved a default "
        f"database outside {root}:\n{output[-4000:]}"
    )
    return output


def test_guard_fires_for_a_normal_test(guard_scope_run: str) -> None:
    """The guarded case: a test outside the exempt dirs writes, and is failed."""
    outcomes = _outcomes(guard_scope_run, _NORMAL_MODULE)
    assert outcomes == ["PASSED", "ERROR"], (
        "a test that wrote the process default database was not failed by the "
        f"guard (outcomes={outcomes}):\n{guard_scope_run[-4000:]}"
    )
    # Flattened: pytest word-wraps a failure message to the terminal width, so
    # the phrase is only contiguous once whitespace is normalised.
    flat = " ".join(guard_scope_run.split())
    assert "the process default database" in flat, (
        "the guard fired without its diagnostic — the message is the only thing "
        f"that tells the next developer what to patch:\n{guard_scope_run[-4000:]}"
    )


def test_guard_is_exempt_under_live(guard_scope_run: str) -> None:
    """The exempt case: the identical test under ``live/`` runs clean."""
    outcomes = _outcomes(guard_scope_run, _LIVE_MODULE)
    assert outcomes == ["PASSED"], (
        "the live suite is not exempt from the default-db guard — writing the "
        "resolved default is what those tests do, so this is a teardown error "
        f"on every passing live save (outcomes={outcomes}):\n"
        f"{guard_scope_run[-4000:]}"
    )
