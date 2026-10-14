---
created: 2026-04-26T00:00:00Z
category: documentation
---

# Palinode — Developer Setup

This document covers the local development workflow, including running the test
suite and the specific setup required when working inside a `git worktree`.

## Basic setup

```bash
git clone https://github.com/phasespace-labs/palinode
cd palinode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the unit tests:

```bash
pytest
```

`pyproject.toml` configures `pythonpath = ["."]` under `[tool.pytest.ini_options]`,
so pytest always prefers the local source tree over whatever is installed in
`site-packages`.  This is the canonical fix for the editable-install gotcha
described below.

## Interpreting CI failures

Documentation-only contributions receive unit, lint, security, and shipping
checks too. The private development gate runs Ubuntu with Python 3.11 and 3.12
and macOS with Python 3.12; its required Ubuntu names remain `unit-tests (3.11)`
and `unit-tests (3.12)`. The macOS lane uses Homebrew Python so SQLite can load
the sqlite-vec extension. The post-merge sweep covers the same three lanes.

The PR lanes are trimmed for latency; nothing is dropped, only moved. They run
in parallel (`pytest -n auto`, from `pytest-xdist` in the `dev` extra), skip
tests marked `slow`, and on macOS run only tests marked `platform_sensitive`
(watcher, hooks, service files, path handling; the module globs are in
`tests/conftest.py`). The post-merge sweep and a nightly run execute the whole
unit suite serially on all three lanes. To reproduce a PR lane locally:

```bash
python -m pytest tests/ -n auto -m "not slow and not serial" --ignore=tests/integration --ignore=tests/live
python -m pytest tests/ -m "not slow and serial" --ignore=tests/integration --ignore=tests/live
```

On macOS, prefix both marker expressions with `platform_sensitive and`. Tests
marked `serial` share a system-wide resource (the real filesystem-event daemon)
and run in their own non-parallel pass.

Open the first failed step and assertion. A Python installation or SQLite
extension-loading failure is a runner/setup problem. A watcher coalescing
assertion tests controlled timer ordering; a real-observer timeout tests actual
filesystem delivery and index convergence. Both deserve investigation. Repeated
stderr messages alone do not prove repeated indexing: compare pytest's captured
log records and the embedding/pass assertions, since duplicate logging handlers
can print a single record many times.

For a focused local check, run:

```bash
python -m pytest tests/test_watcher_debounce_coalesce.py tests/test_watcher_thread_shutdown.py tests/test_watcher_on_moved.py -q
```

If an unrelated check fails on a docs contribution, maintainers should record
the failing lane, assertion, and run link and investigate the test or runtime
defect. Contributors need not change their documentation to satisfy an unrelated
watcher failure. Fork workflow approval may require a maintainer; repeated
reruns, platform skips, or bypassing CI are not the repair.

The watcher defers a zero-byte event read for one additional debounce window,
then reopens the path even if no further event arrives. A persistently empty
file still reconciles, because markdown on disk is authoritative. This bounded
retry reduces transient truncate/write exposure; it cannot guarantee a coherent
snapshot of an arbitrarily slow non-atomic writer. Explicit reindex reads the
current file immediately.

## Working in worktrees

Palinode's agent-based development workflow uses `git worktree` to give each
parallel agent its own branch and checkout:

```bash
git worktree add .claude/worktrees/my-feature my-feature
cd .claude/worktrees/my-feature
```

### The editable-install gotcha

When you (or an agent) run `pytest` inside a worktree **without** additional
setup, Python loads the `palinode` package through the editable-install finder
registered in the **main** repo's `.venv`:

```
.venv/lib/.../site-packages/__editable___palinode_0_7_2_finder.py
  → /Users/you/Code/palinode/palinode   ← main repo, not the worktree!
```

Result: code changes in the worktree are silently invisible to the test suite.
Tests can pass on broken code (and fail on correct code) without any error.
This was independently flagged by two M1-Wave-1 agents working on parallel
issues, and is tracked internally.

The `pythonpath = ["."]` setting in `pyproject.toml` mitigates this for most
pytest invocations — pytest prepends the worktree root to `sys.path`, so the
local source directory shadows the editable finder.  However, the fully
correct fix for an isolated worktree is a per-worktree venv.

### Canonical fix: per-worktree venv via `scripts/setup-worktree.sh`

After creating a new worktree, run:

```bash
cd .claude/worktrees/<branch>
bash scripts/setup-worktree.sh
source .venv-worktree/bin/activate
pytest
```

The script:

1. Detects whether the current directory is a worktree (`.git` is a file) or
   the main working tree (`.git` is a directory) and reports which.
2. Creates a per-worktree venv at `.venv-worktree/` — separate from the main
   repo's `.venv` so the editable install resolves to *this* worktree.
3. Runs `pip install -e .[dev]` from the worktree root.
4. Verifies that `palinode.__file__` resolves under the worktree and warns if
   it doesn't.
5. Prints the exact `source` and `pytest` commands to run next.

The script is idempotent: re-running it on an existing `.venv-worktree/`
upgrades the editable install without recreating the venv.

### Quick one-liner fallback (no venv required)

If you just need a quick sanity check without setting up a full per-worktree
venv, the `PYTHONPATH` override forces the local source tree onto the path:

```bash
PYTHONPATH=. pytest tests/
```

This is equivalent to what `pythonpath = ["."]` does automatically when pytest
reads `pyproject.toml`.  Use the per-worktree venv for sustained development
inside a worktree; the one-liner for quick CI debugging.

### Summary of worktree test commands

| Situation | Command |
|---|---|
| First time in a new worktree | `bash scripts/setup-worktree.sh && source .venv-worktree/bin/activate && pytest` |
| Returning to an existing worktree | `source .venv-worktree/bin/activate && pytest` |
| Quick one-off (no venv) | `PYTHONPATH=. pytest tests/` |
| Main repo (no worktree) | `pytest` (works with the default `.venv`) |

## Running specific test subsets

```bash
# All non-live tests (default)
pytest

# A single file
pytest tests/test_store.py -v

# Tests matching a keyword
pytest -k "session_end" -v

# Integration tests (require a running palinode-api)
pytest tests/integration/ -v
```

## Release Checks

Before pushing a PR, run the focused tests for the code you changed and any
release checklist items that apply to the affected surface.
