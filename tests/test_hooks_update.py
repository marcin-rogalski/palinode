"""Installed hook drift: the `hooks_current` doctor check and `palinode hooks update`.

Real files in tmp_path throughout: a fake HOME with a user-level
``~/.claude/settings.json`` and a project directory with its own
``.claude/settings.json``, so discovery walks the same settings cascade Claude
Code does.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from palinode.cli.hooks import (
    STATE_CURRENT,
    STATE_FOREIGN,
    STATE_MISSING,
    STATE_STALE,
    STATE_SYMLINK,
    hook_statuses,
    hook_version,
    hooks,
    packaged_hooks,
)
from palinode.cli.init import init
from palinode.diagnostics.checks.hooks_current import hooks_current
from palinode.diagnostics.types import DoctorContext

REPO_ROOT = Path(__file__).resolve().parent.parent
END = "palinode-session-end.sh"
PROMPT = "palinode-user-prompt-submit.sh"
START = "palinode-session-start.sh"

#: What the maintainer's machine actually had: a pre-stamp hook from an older
#: release. Any palinode-authored content that differs from the package will do.
STALE_HOOK = "#!/bin/bash\n# palinode-session-end.sh — an older release's copy\nexit 0\n"
FOREIGN_HOOK = "#!/bin/bash\necho 'my own lint hook'\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".claude" / "hooks").mkdir(parents=True)
    (project / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    return home, project


def _register(settings: Path, *commands: str) -> None:
    settings.parent.mkdir(parents=True, exist_ok=True)
    data = {"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": c} for c in commands]}
    ]}}
    settings.write_text(json.dumps(data), encoding="utf-8")


def _drift_fixture(home: Path) -> dict[str, Path]:
    """User-level install: one stale palinode hook, one current, one foreign hook."""
    hooks_dir = home / ".claude" / "hooks"
    stale = hooks_dir / END
    stale.write_text(STALE_HOOK, encoding="utf-8")
    stale.chmod(0o755)
    current = hooks_dir / PROMPT
    current.write_text(packaged_hooks()[PROMPT], encoding="utf-8")
    foreign = hooks_dir / "my-lint.sh"
    foreign.write_text(FOREIGN_HOOK, encoding="utf-8")
    _register(
        home / ".claude" / "settings.json",
        "~/.claude/hooks/palinode-session-end.sh",
        '"$HOME"/.claude/hooks/palinode-user-prompt-submit.sh',
        "bash ~/.claude/hooks/my-lint.sh --strict",
    )
    return {"stale": stale, "current": current, "foreign": foreign}


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _update(*args: str):
    return CliRunner().invoke(hooks, ["update", "--format", "json", *args])


# -- version stamp ---------------------------------------------------------


def test_every_packaged_hook_carries_a_version_stamp_on_line_two():
    for name, content in packaged_hooks().items():
        lines = content.splitlines()
        assert lines[0].startswith("#!"), name
        assert re.fullmatch(r"# palinode-hook-version: \d+\.\d+\.\d+", lines[1]), name
        assert hook_version(content) is not None


def test_packaged_hooks_match_examples_byte_for_byte():
    """The stamp must not break init/examples parity — the refresh source is init's copy."""
    for name, content in packaged_hooks().items():
        assert content == (REPO_ROOT / "examples" / "hooks" / name).read_text(encoding="utf-8"), name


# -- discovery -------------------------------------------------------------


def test_discovery_walks_the_settings_cascade_and_skips_non_palinode(env):
    home, project = env
    _drift_fixture(home)
    _register(
        project / ".claude" / "settings.local.json",
        "${CLAUDE_PROJECT_DIR}/.claude/hooks/palinode-session-start.sh",
    )
    statuses = {Path(s.path).name: s for s in hook_statuses(home, project)}
    assert set(statuses) == {END, PROMPT, START}
    assert statuses[END].state == STATE_STALE
    assert statuses[END].installed_version is None
    assert statuses[PROMPT].state == STATE_CURRENT
    assert statuses[START].state == STATE_MISSING
    assert statuses[START].path == str(project / ".claude" / "hooks" / START)


def test_a_script_registered_twice_is_one_hook(env):
    home, project = env
    _drift_fixture(home)
    _register(project / ".claude" / "settings.json", f"{home}/.claude/hooks/{END}")
    ends = [s for s in hook_statuses(home, project) if s.name == END]
    assert len(ends) == 1
    assert len(ends[0].settings) == 2


# -- doctor ----------------------------------------------------------------


def test_doctor_warns_on_drift_and_names_the_refresh_command(env):
    home, _ = env
    paths = _drift_fixture(home)
    result = hooks_current(DoctorContext(config=None))
    assert not result.passed
    assert result.severity == "warn"
    assert str(paths["stale"]) in result.message
    assert "unstamped" in result.message
    assert "palinode hooks update" in result.remediation
    assert "my-lint.sh" not in result.message


def test_doctor_passes_when_current_and_when_nothing_is_registered(env):
    home, _ = env
    assert hooks_current(DoctorContext(config=None)).passed  # nothing registered
    _drift_fixture(home)
    assert _update().exit_code == 0
    result = hooks_current(DoctorContext(config=None))
    assert result.passed, result.message
    assert result.severity == "info"


def test_doctor_runs_hooks_current_by_name(env):
    from palinode.diagnostics.runner import run_one

    _drift_fixture(env[0])
    assert run_one(DoctorContext(config=None), "hooks_current").severity == "warn"


# -- update ----------------------------------------------------------------


def test_update_makes_hook_byte_identical_and_keeps_a_backup(env):
    home, _ = env
    paths = _drift_fixture(home)
    res = _update()
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    entry = next(h for h in out["hooks"] if h["name"] == END)
    assert entry["action"] == "updated"

    assert paths["stale"].read_text(encoding="utf-8") == packaged_hooks()[END]
    assert os.access(paths["stale"], os.X_OK)
    backup = Path(entry["backup"])
    assert backup.parent == paths["stale"].parent
    assert re.fullmatch(rf"{re.escape(END)}\.bak-\d{{8}}-\d{{6}}", backup.name)
    assert backup.read_text(encoding="utf-8") == STALE_HOOK


def test_second_update_is_a_no_op(env):
    home, _ = env
    _drift_fixture(home)
    assert _update().exit_code == 0
    before = _snapshot(home)
    res = _update()
    assert res.exit_code == 0
    assert {h["action"] for h in json.loads(res.output)["hooks"]} == {"skipped"}
    assert _snapshot(home) == before


def test_update_never_touches_non_palinode_hooks(env):
    home, _ = env
    paths = _drift_fixture(home)
    # A script that borrows a palinode file name but is not palinode's.
    impostor = home / "elsewhere" / START
    impostor.parent.mkdir()
    impostor.write_text(FOREIGN_HOOK, encoding="utf-8")
    settings = home / ".claude" / "settings.json"
    data = json.loads(settings.read_text())
    data["hooks"]["SessionStart"] = [{"hooks": [{"type": "command", "command": str(impostor)}]}]
    settings.write_text(json.dumps(data))
    settings_before = settings.read_bytes()

    res = _update()
    assert res.exit_code == 0, res.output
    assert paths["foreign"].read_text(encoding="utf-8") == FOREIGN_HOOK
    assert impostor.read_text(encoding="utf-8") == FOREIGN_HOOK
    assert settings.read_bytes() == settings_before
    states = {h["name"]: h["state"] for h in json.loads(res.output)["hooks"]}
    assert states[START] == STATE_FOREIGN
    assert "my-lint.sh" not in states
    assert not list((home / ".claude" / "hooks").glob("my-lint.sh.bak-*"))
    assert not list(impostor.parent.glob("*.bak-*"))


def test_dry_run_changes_nothing_and_shows_the_diff(env):
    home, project = env
    _drift_fixture(home)
    _register(project / ".claude" / "settings.json", f"$CLAUDE_PROJECT_DIR/.claude/hooks/{START}")
    before_home, before_project = _snapshot(home), _snapshot(project)

    res = _update("--dry-run")
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["dry_run"] is True
    end = next(h for h in out["hooks"] if h["name"] == END)
    assert end["action"] == "would update"
    assert "+# palinode-hook-version:" in end["diff"]
    assert "-# palinode-session-end.sh — an older release's copy" in end["diff"]
    assert next(h for h in out["hooks"] if h["name"] == START)["action"] == "would install"

    assert _snapshot(home) == before_home
    assert _snapshot(project) == before_project
    assert not (project / ".claude" / "hooks").exists()


def test_dry_run_text_output_prints_a_diff(env):
    _drift_fixture(env[0])
    res = CliRunner().invoke(hooks, ["update", "--dry-run", "--format", "text"])
    assert res.exit_code == 0, res.output
    assert "would update:" in res.output
    assert "+# palinode-hook-version:" in res.output
    assert "nothing written" in res.output


def test_update_installs_a_registered_but_missing_hook(env):
    home, project = env
    _register(project / ".claude" / "settings.json", f"$CLAUDE_PROJECT_DIR/.claude/hooks/{START}")
    res = _update()
    assert res.exit_code == 0, res.output
    installed = project / ".claude" / "hooks" / START
    assert installed.read_text(encoding="utf-8") == packaged_hooks()[START]
    assert os.access(installed, os.X_OK)


def test_update_leaves_symlinked_hooks_alone(env):
    home, _ = env
    target = home / "checkout" / END
    target.parent.mkdir()
    target.write_text(STALE_HOOK, encoding="utf-8")
    link = home / ".claude" / "hooks" / END
    link.symlink_to(target)
    _register(home / ".claude" / "settings.json", str(link))

    res = _update()
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["hooks"][0]["state"] == STATE_SYMLINK
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == STALE_HOOK
    assert not hooks_current(DoctorContext(config=None)).passed


def test_init_installed_hooks_are_current_then_refresh_after_drift(env):
    """End to end: `palinode init` → doctor clean → drift → update → clean."""
    _, project = env
    res = CliRunner().invoke(
        init,
        ["--dir", str(project), "--hook", "--no-mcp", "--no-claudemd", "--no-slash", "--no-prompts"],
    )
    assert res.exit_code == 0, res.output
    assert "sha256" in res.output  # init records what it installed
    assert hooks_current(DoctorContext(config=None)).passed

    hook = project / ".claude" / "hooks" / END
    hook.write_text(STALE_HOOK, encoding="utf-8")
    assert not hooks_current(DoctorContext(config=None)).passed

    assert _update().exit_code == 0
    assert hook.read_text(encoding="utf-8") == packaged_hooks()[END]
    assert hooks_current(DoctorContext(config=None)).passed
