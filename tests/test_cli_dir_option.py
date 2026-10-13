"""``palinode --dir <path>`` names the store, as ``PALINODE_DIR`` does.

The configuration loads when ``palinode.core.config`` is first imported, so a
flag cannot be read by click: the console script is ``<ENTRY_MODULE>:main``,
which reads a leading ``--dir`` from argv, sets ``PALINODE_DIR``, and only then
imports the CLI. These tests run it as a subprocess (``python -m
<ENTRY_MODULE>``), because an in-process call sees the configuration of the
first import. The entry module cannot live inside ``palinode.cli``: importing
any module of a package runs the package's ``__init__`` first, and
``palinode/cli/__init__.py`` loads the configuration.

Every observation that proves the flag works is one the same run gives with
``PALINODE_DIR`` set and no flag. The ``control`` runs below pin that, on
today's CLI, so a red test cannot be red for a reason other than the missing
flag: the module not existing fails the assertion on the observation, not on
a missing file.

No test reaches the network: a request is recorded and answered by a stub
transport installed in the subprocess before the CLI is imported.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from click.testing import CliRunner

REPO = Path(__file__).resolve().parent.parent

ENTRY_MODULE = "palinode.entry"

_BANNER = "Palinode config:"
_NO_CONFIG_WARNING = "no palinode.config.yaml found"

# Installed before palinode is imported: records each request body, answers [].
_RUNNER = '''
import json, os, runpy, sys
import httpx

def handle_request(self, request):
    with open(os.environ["RUN_REQUEST_LOG"], "a") as log:
        log.write(json.dumps(json.loads(request.content or b"null")) + "\\n")
    return httpx.Response(200, json=[], request=request)

httpx.HTTPTransport.handle_request = handle_request
way = sys.argv.pop(1)
if way == "control":
    from palinode.cli import main
    sys.argv[0] = "palinode"
    main()
else:
    runpy.run_module(os.environ["RUN_ENTRY_MODULE"], run_name="__main__", alter_sys=True)
'''


class Ran(NamedTuple):
    returncode: int
    stdout: str
    stderr: str
    requests: list[dict]


@pytest.fixture()
def world(tmp_path: Path) -> SimpleNamespace:
    """Three stores, a home directory, and two working directories."""
    root = tmp_path.resolve()

    def store(name: str, config: str | None) -> Path:
        path = root / "stores" / name
        path.mkdir(parents=True)
        if config is not None:
            (path / "palinode.config.yaml").write_text(config, encoding="utf-8")
        return path

    runner = root / "runner.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    home = root / "home"
    home.mkdir()
    (home / "notes").mkdir()
    (home / "notes" / "palinode.config.yaml").write_text(
        "context:\n  project_map:\n    checklist: my-app\n", encoding="utf-8"
    )
    work = root / "work"
    (work / "checklist").mkdir(parents=True)
    (work / "elsewhere").mkdir()
    return SimpleNamespace(
        root=root,
        runner=runner,
        home=home,
        work=work,
        # maps the directory "checklist" to project "my-app"
        named=store("named", "context:\n  project_map:\n    checklist: my-app\n"),
        # maps the same directory to another project
        other=store("other", "context:\n  project_map:\n    checklist: other-app\n"),
        # turns project resolution off
        quiet=store("quiet", "context:\n  enabled: false\n"),
    )


def _run(
    world: SimpleNamespace,
    args: list[str],
    *,
    way: str = "entry",
    stubbed: bool = False,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> Ran:
    """Run the CLI in a fresh interpreter that inherits nothing from the test session.

    ``way="entry"`` is ``python -m <ENTRY_MODULE>``; ``way="control"`` is
    today's ``palinode.cli.main``, which has no flag. ``stubbed`` installs the
    transport stub, which the control always has.
    """
    log = world.root / "requests.jsonl"
    log.unlink(missing_ok=True)
    full_env = {
        "PATH": os.environ["PATH"],
        "HOME": str(world.home),
        "RUN_REQUEST_LOG": str(log),
        "RUN_ENTRY_MODULE": ENTRY_MODULE,
        "PYTHONPATH": str(REPO),
        **(env or {}),
    }
    if way == "entry" and not stubbed:
        command = [sys.executable, "-m", ENTRY_MODULE, *args]
    else:
        command = [sys.executable, str(world.runner), way, *args]
    done = subprocess.run(  # nosec B603 - argv form, no shell
        command,
        cwd=cwd or world.work / "checklist",
        env=full_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    requests = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    return Ran(done.returncode, done.stdout, done.stderr, requests)


def _search(world, *flag, way="entry", env=None, cwd=None) -> Ran:
    """``palinode [flag] search`` with the transport stubbed."""
    return _run(
        world,
        [*flag, "search", "release checklist"],
        way=way,
        stubbed=True,
        env=env,
        cwd=cwd,
    )


def _context_sent(ran: Ran) -> list[str] | None:
    """The project scope the search request carried; ``None`` when none was sent."""
    return ran.requests[0].get("context") if ran.requests else None


_MCP_CONFIG = ["mcp-config", "--stdio", "--json", "--executable", os.path.realpath(sys.executable)]


def _store_written_by_mcp_config(ran: Ran) -> str | None:
    if ran.returncode != 0:
        return None
    entry = json.loads(ran.stdout)["mcpServers"]["palinode"]
    return entry["env"]["PALINODE_DIR"]


# ── the controls: what the flag must reproduce, on today's CLI ──────────────

def test_control_with_no_flag_and_no_variable_the_cli_warns_and_infers_from_the_directory(world):
    ran = _search(world, way="control")

    assert ran.returncode == 0, ran.stderr
    assert _NO_CONFIG_WARNING in ran.stderr
    assert _context_sent(ran) == ["project/checklist"]


def test_control_the_variable_selects_the_config_in_force(world):
    named = _search(world, way="control", env={"PALINODE_DIR": str(world.named)})
    other = _search(world, way="control", env={"PALINODE_DIR": str(world.other)})
    quiet = _search(world, way="control", env={"PALINODE_DIR": str(world.quiet)})

    assert _NO_CONFIG_WARNING not in named.stderr
    assert _context_sent(named) == ["project/my-app"]
    assert _context_sent(other) == ["project/other-app"]
    assert _context_sent(quiet) == []


def test_control_the_variable_is_what_mcp_config_writes(world):
    ran = _run(world, _MCP_CONFIG, way="control", env={"PALINODE_DIR": str(world.named)})

    assert _store_written_by_mcp_config(ran) == str(world.named)


def test_control_an_option_after_the_subcommand_is_clicks_no_such_option(world):
    ran = _run(world, ["search", "--dir", str(world.named), "q"], way="control")

    assert ran.returncode == 2
    assert "No such option" in ran.stderr
    assert ran.requests == []


# ── 1. one global option, --dir PATH, no short form ─────────────────────────

def test_help_through_the_entry_point_lists_the_dir_option(world):
    ran = _run(world, ["--help"], env={"PALINODE_DIR": str(world.named)})

    assert ran.returncode == 0, ran.stderr
    assert re.search(r"^\s+--dir\b", ran.stdout, re.M), ran.stdout
    assert not re.search(r"-\w, --dir\b", ran.stdout)


def test_the_group_help_lists_the_dir_option(world):
    from palinode.cli import main

    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert re.search(r"^\s+--dir\b", result.output, re.M), result.output
    assert not re.search(r"-\w, --dir\b", result.output)


# ── 2. the option wins over PALINODE_DIR ────────────────────────────────────

def test_the_option_wins_over_the_variable_for_the_store_mcp_config_writes(world):
    ran = _run(world, ["--dir", str(world.named), *_MCP_CONFIG],
               env={"PALINODE_DIR": str(world.other)})

    assert _store_written_by_mcp_config(ran) == str(world.named), ran.stderr


def test_the_option_wins_over_the_variable_for_the_config_in_force(world):
    ran = _search(world, "--dir", str(world.named), env={"PALINODE_DIR": str(world.other)})

    assert _context_sent(ran) == ["project/my-app"], ran.stderr
    assert str(world.named / "palinode.config.yaml") in ran.stderr


# ── 3. one mechanism: the process environment, set before anything imports ──

@pytest.mark.parametrize("form", ["separate", "equals", "relative", "tilde"])
def test_the_flag_alone_gives_the_store_mcp_config_writes(world, form):
    home_notes = world.home / "notes"
    forms = {
        "separate": (["--dir", str(world.named)], world.named, world.work),
        "equals": ([f"--dir={world.named}"], world.named, world.work),
        "relative": (["--dir", "stores/named"], world.named, world.root),
        "tilde": (["--dir=~/notes"], home_notes, world.work),
    }
    flag, expected, cwd = forms[form]

    ran = _run(world, [*flag, *_MCP_CONFIG], cwd=cwd)

    assert _store_written_by_mcp_config(ran) == str(expected), ran.stderr


def test_a_relative_path_reaches_a_reader_that_prints_it_as_an_absolute_one(world):
    """``mcp-config`` makes any path absolute itself, so only a reader that echoes
    the variable shows whether the entry point did."""
    ran = _run(world, ["--dir", "stores/named", "obsidian-sync"], cwd=world.root)

    assert f"No .md files found under {world.named} " in ran.stdout, ran.stderr


def test_the_option_is_removed_before_the_group_parses(world):
    ran = _run(world, ["--dir", str(world.named), "--version"])

    assert ran.returncode == 0, ran.stderr
    assert re.search(r"palinode \d+\.\d+", ran.stdout), ran.stdout


def test_obsidian_sync_reads_the_flag_not_the_variable(world):
    ran = _run(world, ["--dir", str(world.named), "obsidian-sync"],
               env={"PALINODE_DIR": str(world.other)})

    assert f"No .md files found under {world.named} " in ran.stdout, ran.stderr


def test_import_from_vault_targets_the_flag_not_the_variable(world):
    vault = world.root / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Release checklist\n", encoding="utf-8")

    ran = _run(world, ["--dir", str(world.named), "import", "from-vault", "--from-vault", str(vault)],
               env={"PALINODE_DIR": str(world.other)})

    assert f"Target memory_dir: {world.named}\n" in ran.stdout, ran.stderr


def test_mcp_smoke_records_in_the_flag_store_not_the_variable_store(world):
    _run(world, ["--dir", str(world.named), "mcp-smoke", "claude-code", "--record"],
         env={"PALINODE_DIR": str(world.other)})

    log = Path(".palinode") / "harness-smoke-runs.jsonl"
    assert (world.named / log).is_file()
    assert not (world.other / log).exists()


# ── 4. the config file is found in the named directory ──────────────────────

def test_the_config_in_the_named_directory_is_the_one_in_force(world):
    ran = _search(world, "--dir", str(world.quiet))

    assert _NO_CONFIG_WARNING not in ran.stderr
    assert str(world.quiet / "palinode.config.yaml") in ran.stderr
    assert _context_sent(ran) == []


def test_help_with_the_flag_does_not_warn_about_a_missing_config(world):
    ran = _run(world, ["--dir", str(world.named), "--help"])

    assert ran.returncode == 0, ran.stderr
    assert _NO_CONFIG_WARNING not in ran.stderr
    assert str(world.named / "palinode.config.yaml") in ran.stderr


# ── 5. a bad path is a usage error before anything is imported ──────────────

def _bad_paths(world) -> dict[str, tuple[list[str], str]]:
    (world.root / "a-file").write_text("not a directory", encoding="utf-8")
    missing = str(world.root / "missing" / "store")
    a_file = str(world.root / "a-file")
    return {
        "missing": (["--dir", missing, "search", "q"], missing),
        "file": (["--dir", a_file, "search", "q"], a_file),
        "equals-missing": ([f"--dir={missing}", "search", "q"], missing),
        "no-value": (["--dir"], "--dir"),
        "empty-value": (["--dir="], "--dir"),
    }


@pytest.mark.parametrize("case", ["missing", "file", "equals-missing", "no-value", "empty-value"])
def test_a_bad_path_is_a_usage_error_that_names_it_and_loads_nothing(world, case):
    args, named_in_error = _bad_paths(world)[case]

    ran = _run(world, args, stubbed=True, env={"PALINODE_DIR": str(world.named)})

    assert ran.returncode == 2, ran.stderr
    assert named_in_error in ran.stderr
    assert _BANNER not in ran.stderr
    assert _NO_CONFIG_WARNING not in ran.stderr
    assert ran.stdout == ""
    assert ran.requests == []


# ── 6. only before the subcommand ───────────────────────────────────────────

@pytest.mark.parametrize("args", [
    ["search", "--dir", "{named}", "q"],
    ["--dir", "{named}", "search", "--dir", "{other}", "q"],
], ids=["after-the-subcommand", "twice"])
def test_the_option_after_the_subcommand_is_no_such_option(world, args):
    args = [a.format(named=world.named, other=world.other) for a in args]

    ran = _run(world, args, stubbed=True)

    assert ran.returncode == 2, ran.stderr
    assert "No such option" in ran.stderr
    assert "--dir" in ran.stderr
    assert ran.requests == []


# ── 7. the group declares --dir and refuses it when the entry point was bypassed

def test_the_group_refuses_dir_and_says_it_must_come_first(monkeypatch, world):
    from palinode.cli import main

    monkeypatch.delenv("PALINODE_DIR", raising=False)
    result = CliRunner().invoke(main, ["--dir", str(world.named), "banner"])

    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "--dir" in result.output
    assert "first" in result.output.lower()
    assert "palinode" in result.output.lower()
    assert "PALINODE_DIR" not in os.environ


def test_the_group_refuses_dir_even_for_a_path_that_does_not_exist(world):
    from palinode.cli import main

    result = CliRunner().invoke(main, ["--dir", str(world.root / "missing"), "banner"])

    assert result.exit_code == 2
    assert "No such option" not in result.output


# ── 8. the working directory is untouched ───────────────────────────────────

def test_the_project_is_still_read_from_the_working_directory(world):
    ran = _search(world, "--dir", str(world.named), cwd=world.work / "checklist")

    assert _context_sent(ran) == ["project/my-app"], ran.stderr


def test_a_directory_the_named_config_does_not_map_still_resolves_from_its_own_name(world):
    ran = _search(world, "--dir", str(world.named), cwd=world.work / "elsewhere")

    assert _context_sent(ran) == ["project/elsewhere"], ran.stderr


# ── 9. nothing else changes ─────────────────────────────────────────────────

@pytest.mark.parametrize("variable", [None, "named"], ids=["no-variable", "variable"])
def test_without_the_flag_the_entry_point_behaves_like_the_group(world, variable):
    env = {"PALINODE_DIR": str(world.named)} if variable else {}

    entry = _run(world, _MCP_CONFIG, env=env)
    control = _run(world, _MCP_CONFIG, way="control", env=env)

    assert control.returncode == 0
    assert (entry.returncode, entry.stdout, entry.stderr) == (
        control.returncode, control.stdout, control.stderr
    )


def _scripts() -> dict[str, str]:
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def test_the_palinode_command_starts_at_the_entry_point():
    assert _scripts()["palinode"] == f"{ENTRY_MODULE}:main"


def test_importing_the_entry_module_loads_no_configuration(world):
    """The whole mechanism rests on this: nothing may import the config first."""
    probe = (
        "import importlib, sys; "
        f"importlib.import_module({ENTRY_MODULE!r}); "
        "print('palinode.core.config' in sys.modules)"
    )
    done = subprocess.run(  # nosec B603 - argv form, no shell
        [sys.executable, "-c", probe],
        cwd=world.root,
        env={"PATH": os.environ["PATH"], "HOME": str(world.home),
             "PYTHONPATH": str(REPO), "PALINODE_DIR": str(world.named)},
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )

    assert done.stdout.strip() == "False", done.stderr


def test_the_other_console_scripts_are_untouched():
    scripts = _scripts()
    scripts.pop("palinode")

    assert scripts == {
        "palinode-watcher": "palinode.indexer.watcher:main",
        "palinode-api": "palinode.api.server:main",
        "palinode-mcp": "palinode.mcp:main",
        "palinode-mcp-http": "palinode.mcp:main_http",
        "palinode-mcp-sse": "palinode.mcp:main_sse",
    }


# ── edges on the entry point ────────────────────────────────────────────────

def _nothing_was_loaded(ran: Ran) -> None:
    assert _BANNER not in ran.stderr
    assert _NO_CONFIG_WARNING not in ran.stderr
    assert ran.stdout == ""
    assert ran.requests == []


@pytest.mark.parametrize("form", ["separate", "equals", "mixed"])
def test_two_leading_dir_options_are_a_usage_error_and_load_nothing(world, form):
    first, second = str(world.named), str(world.other)
    leading = {
        "separate": ["--dir", first, "--dir", second],
        "equals": [f"--dir={first}", f"--dir={second}"],
        "mixed": ["--dir", first, f"--dir={second}"],
    }[form]

    ran = _run(world, [*leading, "search", "q"], stubbed=True, env={"PALINODE_DIR": first})

    assert ran.returncode == 2, ran.stderr
    assert "given twice" in ran.stderr
    _nothing_was_loaded(ran)


def test_a_value_that_looks_like_an_option_is_taken_as_the_path(world):
    """``--dir --help`` names a directory called ``--help``; it is not a request for help."""
    ran = _run(world, ["--dir", "--help"], stubbed=True, env={"PALINODE_DIR": str(world.named)})

    error = [line for line in ran.stderr.splitlines() if line.startswith("Error")]
    assert ran.returncode == 2, ran.stderr
    assert error and "--help" in error[0], ran.stderr
    _nothing_was_loaded(ran)


def test_a_symlink_to_a_directory_is_used_as_the_path_it_was_given_not_resolved(world):
    link = world.root / "link-to-named"
    link.symlink_to(world.named, target_is_directory=True)

    ran = _run(world, ["--dir", str(link), "obsidian-sync"], env={"PALINODE_DIR": str(world.other)})

    assert f"No .md files found under {link} " in ran.stdout, ran.stderr
    assert str(link / "palinode.config.yaml") in ran.stderr


def test_a_dangling_symlink_is_a_usage_error_that_names_it_and_loads_nothing(world):
    dangling = world.root / "dangling-link"
    dangling.symlink_to(world.root / "nowhere", target_is_directory=True)

    ran = _run(world, ["--dir", str(dangling), "search", "q"], stubbed=True,
               env={"PALINODE_DIR": str(world.named)})

    assert ran.returncode == 2, ran.stderr
    assert str(dangling) in ran.stderr
    _nothing_was_loaded(ran)


@pytest.mark.parametrize("form", ["separate", "equals"])
def test_a_path_with_a_space_in_it_names_the_store(world, form):
    spaced = world.root / "stores" / "release notes"
    spaced.mkdir()
    (spaced / "palinode.config.yaml").write_text("context:\n  enabled: false\n", encoding="utf-8")
    flag = ["--dir", str(spaced)] if form == "separate" else [f"--dir={spaced}"]

    ran = _search(world, *flag, env={"PALINODE_DIR": str(world.other)})
    written = _run(world, [*flag, *_MCP_CONFIG], env={"PALINODE_DIR": str(world.other)})

    assert _NO_CONFIG_WARNING not in ran.stderr
    assert str(spaced / "palinode.config.yaml") in ran.stderr
    assert _context_sent(ran) == []
    assert _store_written_by_mcp_config(written) == str(spaced)


# ── 10. docs ────────────────────────────────────────────────────────────────

def _section(text: str, heading: str) -> str:
    """The body under ``heading`` (any depth), up to the next heading of the same or a higher level."""
    match = re.search(rf"^(#+) {re.escape(heading)}\s*$", text, re.M)
    if not match:
        return ""
    rest = text[match.end():]
    end = re.search(rf"^#{{1,{len(match.group(1))}}} ", rest, re.M)
    return rest[: end.start()] if end else rest


def test_the_cli_reference_documents_the_dir_option_under_global_options():
    text = (REPO / "docs" / "CLI.md").read_text(encoding="utf-8")

    section = _section(text, "Global options")

    assert "--dir" in section
    assert "PALINODE_DIR" in section


def test_the_changelog_lists_the_dir_option_as_added():
    text = (REPO / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")

    added = _section(_section(text, "Unreleased"), "Added")

    assert "--dir" in added
