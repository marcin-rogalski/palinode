"""Front-matter ``where`` filters for list and its surfaces.

An entry is ``key=value``. Every entry must match: a scalar by its text (booleans
as ``true``/``false``), a list when any element does, and a date or timestamp as
the search date window reads it. A key that is missing, null or a mapping never matches.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from palinode.core.ranker import parse_iso_utc

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
# Only the dashed form: an id such as 20261012 or 2026-W42 must not be read as a date.
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}.*)?")


class WhereError(ValueError):
    """A ``where`` entry that is not ``key=value`` with a well-formed key."""


def parse_where(entries: Iterable[Any] | str | None) -> list[tuple[str, str]]:
    """``(key, value)`` pairs for *entries*; raises :class:`WhereError` on a malformed one.

    A single string is one entry.
    """
    if isinstance(entries, str):
        entries = [entries]
    conditions: list[tuple[str, str]] = []
    for entry in entries or ():
        key, sep, value = str(entry).partition("=")
        if not sep or not value or not _KEY_RE.fullmatch(key):
            raise WhereError(
                f"malformed where entry {str(entry)!r}: expected key=value, "
                f"the key matching {_KEY_RE.pattern}"
            )
        conditions.append((key, value))
    return conditions


def _as_text(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _is_in_date_span(text: str, expected: str) -> bool:
    if not (_DATE_RE.fullmatch(text) and _DATE_RE.fullmatch(expected)):
        return False
    # As in_date_window(text, expected, expected): a bare date spans its whole UTC day.
    stamp = parse_iso_utc(text)
    start = parse_iso_utc(expected)
    end = parse_iso_utc(expected, end_of_day=True)
    if stamp is None or start is None or end is None:
        return False
    return start <= stamp <= end


def _matches_value(candidate: Any, expected: str) -> bool:
    text = _as_text(candidate)
    if text is None:
        return False
    return text == expected or _is_in_date_span(text, expected)


def matches_where(metadata: dict[str, Any], conditions: list[tuple[str, str]]) -> bool:
    """True when *metadata* satisfies every condition."""
    for key, expected in conditions:
        if key not in metadata:
            return False
        value = metadata[key]
        candidates = value if isinstance(value, list) else [value]
        if not any(_matches_value(c, expected) for c in candidates):
            return False
    return True
