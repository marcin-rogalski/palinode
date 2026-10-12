"""Front-matter ``where`` filters shared by search, list and their surfaces.

An entry is ``key=value``. Every entry must match: a scalar by its text (booleans
as ``true``/``false``, timestamps as the same instant), a list when any element
does. A record without the key, or with a null or a mapping there, never matches.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
# A date and a time, in either of the spellings a stored timestamp comes back in.
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


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


def _as_instant(text: str) -> datetime | None:
    if not _TIMESTAMP_RE.match(text):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _matches_value(candidate: Any, expected: str) -> bool:
    text = _as_text(candidate)
    if text is None:
        return False
    if text == expected:
        return True
    # A file read back gives "T", the index's JSON copy a space, and a writer may use "Z".
    instant = _as_instant(text)
    return instant is not None and instant == _as_instant(expected)


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
