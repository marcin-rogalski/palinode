"""The ``palinode`` console script.

The configuration loads when :mod:`palinode.core.config` is first imported, so a leading
``--dir PATH`` is read here and set as ``PALINODE_DIR`` before any palinode import.
"""
from __future__ import annotations

import os
import sys

import click

_OPTION = "--dir"


def _split_store_dir(args: list[str]) -> tuple[str | None, list[str]]:
    if not args:
        return None, args
    first = args[0]
    if first == _OPTION:
        value, rest = (args[1], args[2:]) if len(args) > 1 else ("", [])
    elif first.startswith(f"{_OPTION}="):
        value, rest = first.partition("=")[2], args[1:]
    else:
        return None, args
    if not value:
        raise click.UsageError(f"Option '{_OPTION}' requires an argument.")
    if rest and (rest[0] == _OPTION or rest[0].startswith(f"{_OPTION}=")):
        raise click.UsageError(f"Option '{_OPTION}' was given twice.")
    return value, rest


def _resolve_store_dir(value: str) -> str:
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.exists(path):
        raise click.BadParameter(f"Directory '{value}' does not exist.", param_hint=f"'{_OPTION}'")
    if not os.path.isdir(path):
        raise click.BadParameter(f"'{value}' is not a directory.", param_hint=f"'{_OPTION}'")
    return path


def main() -> None:
    """Run the ``palinode`` CLI, with ``--dir PATH`` before the subcommand naming the palinode directory."""
    try:
        store_dir, args = _split_store_dir(sys.argv[1:])
        if store_dir is not None:
            os.environ["PALINODE_DIR"] = _resolve_store_dir(store_dir)
    except click.UsageError as error:
        error.show()
        sys.exit(error.exit_code)

    from palinode.cli import main as cli_main

    # click expands wildcards on Windows only in the arguments it reads from sys.argv itself.
    sys.argv[1:] = args
    cli_main(prog_name="palinode")


if __name__ == "__main__":
    main()
