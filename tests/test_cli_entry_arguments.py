"""The ``palinode`` command hands its arguments to click as click reads them itself.

Click expands wildcards, ``~`` and environment variables in the arguments only
when ``main()`` is called without ``args``, and only on Windows, where the shell
does not. The command reads a leading ``--dir PATH`` before click does, so what
remains of ``sys.argv`` must still reach click's own path; an explicit ``args``
list skips the expansion and ``palinode ingest *.md`` receives the literal ``*.md``.

Windows is simulated in this process: ``click.core`` sees ``os.name == "nt"`` and
nothing else does. ``click.core._expand_args`` is click's real function, wrapped to
record what it received. The CLI is not given a store to read: the subcommand added
here only records its arguments.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import click
import pytest

import click.core


class _WindowsOs:
    """``os`` for ``click.core`` alone, saying ``nt`` and delegating the rest."""

    name = "nt"

    def __getattr__(self, attribute: str):
        return getattr(os, attribute)


@pytest.fixture()
def windows_click(monkeypatch, tmp_path):
    """``run(*argv)`` calls ``palinode.entry.main`` and reports what click and the command saw."""
    from palinode.cli import main as group

    home = tmp_path / "home"
    work = tmp_path / "work"
    store = tmp_path / "store"
    for directory in (home, work, store):
        directory.mkdir()
    for name in ("a.md", "b.md", "c.txt"):
        (work / name).write_text("x", encoding="utf-8")
    (home / "notes.md").write_text("x", encoding="utf-8")
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("PALINODE_DIR", raising=False)

    expanded: list[list[str]] = []
    received: list[tuple[str, ...]] = []
    real_expand_args = click.core._expand_args

    def recording_expand_args(args, **kwargs):
        expanded.append(list(args))
        return real_expand_args(args, **kwargs)

    @click.command("probe-arguments")
    @click.argument("names", nargs=-1)
    def probe(names: tuple[str, ...]) -> None:
        received.append(names)

    monkeypatch.setattr(click.core, "os", _WindowsOs())
    monkeypatch.setattr(click.core, "_expand_args", recording_expand_args)
    group.add_command(probe)

    def run(*argv: str) -> SimpleNamespace:
        from palinode import entry

        monkeypatch.setattr("sys.argv", ["palinode", *argv])
        with pytest.raises(SystemExit) as exit_info:
            entry.main()
        return SimpleNamespace(code=exit_info.value.code, expanded=expanded, received=received)

    try:
        yield SimpleNamespace(run=run, store=store, home=home)
    finally:
        group.commands.pop("probe-arguments", None)


_DIR_FORMS = {
    "no-dir": lambda store: [],
    "dir-separate": lambda store: ["--dir", str(store)],
    "dir-equals": lambda store: [f"--dir={store}"],
}


@pytest.mark.parametrize("form", _DIR_FORMS.values(), ids=_DIR_FORMS.keys())
def test_a_wildcard_after_the_command_is_expanded_as_click_expands_it_on_windows(
    windows_click, form
):
    """With ``args`` passed explicitly click never expands, and the command is
    handed ``*.md`` itself. Kills the remaining arguments being given to click as a list."""
    result = windows_click.run(*form(windows_click.store), "probe-arguments", "*.md")

    assert result.code == 0
    assert sorted(result.received[0]) == ["a.md", "b.md"]


@pytest.mark.parametrize("form", _DIR_FORMS.values(), ids=_DIR_FORMS.keys())
def test_click_is_given_the_arguments_without_the_dir_pair(windows_click, form):
    """The ``--dir PATH`` pair is the command's; the group refuses it, so it must not
    reach click. Kills a fix that leaves the pair in ``sys.argv``."""
    result = windows_click.run(*form(windows_click.store), "probe-arguments", "*.md", "c.txt")

    assert result.code == 0
    assert result.expanded == [["probe-arguments", "*.md", "c.txt"]]


def test_a_tilde_after_the_command_is_expanded_to_the_home_directory(windows_click):
    """Kills the same loss for ``~``, which the shell expands on Unix and click on Windows."""
    result = windows_click.run("--dir", str(windows_click.store), "probe-arguments", "~/notes.md")

    assert result.code == 0
    assert result.received[0] == (str(windows_click.home / "notes.md"),)


def test_an_argument_that_matches_nothing_reaches_the_command_as_typed(windows_click):
    """A guard that must keep passing: click keeps a pattern that matches no file."""
    result = windows_click.run("--dir", str(windows_click.store), "probe-arguments", "*.nothing")

    assert result.code == 0
    assert result.received[0] == ("*.nothing",)


def test_off_windows_a_pattern_reaches_the_command_as_typed(windows_click, monkeypatch):
    """A guard that must keep passing: elsewhere the shell expands, and a quoted
    pattern is the user's. Kills expanding in the command whatever the system."""
    monkeypatch.setattr(click.core, "os", os)

    result = windows_click.run("--dir", str(windows_click.store), "probe-arguments", "*.md")

    assert result.code == 0
    assert result.expanded == []
    assert result.received[0] == ("*.md",)
