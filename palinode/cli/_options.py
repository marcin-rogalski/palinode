"""Click callbacks shared by more than one command."""
from __future__ import annotations

import click

from palinode.core.where import WhereError, parse_where


def validate_where(
    ctx: click.Context, param: click.Parameter, value: tuple[str, ...]
) -> tuple[str, ...]:
    try:
        parse_where(value)
    except WhereError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc
    return value
