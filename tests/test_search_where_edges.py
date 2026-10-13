"""Edges of ``where`` on search: timestamp spellings, MCP argument shapes, non-scalar values.

``test_search_where.py`` and ``test_search_where_surfaces.py`` pin the contract.
This file pins three places where the matching or the argument handling has more
than one way to go wrong, on ``POST /search`` and on the recency listing (an
empty query):

* a timestamp is matched as an instant, so the ``T``, space, ``Z`` and offset
  spellings select the same record, for a value stored as text and for one
  stored as a YAML timestamp, and a value with no offset is UTC. A bare date is a
  day: it matches any time on it, and as a stored value it is midnight. Only
  dates written with dashes are read so: ``20261012`` and ``2026-W42`` are
  compared as text, and so is a value whose UTC reading is outside years 1 to 9999.
  Whitespace around a date or timestamp is ignored, as the window ignores it.
  The index's copy of the front matter is what search matches;
* the ``palinode_search`` ``where`` argument is a list, a bare string or a
  JSON-string array; an object or an item that is not a string is refused naming
  the problem, with no request made;
* a key holding null, a mapping, or a list of mappings never matches and never
  fails the request.

Each docstring names the mutant it kills.

Real SQLite + tmp_path through ``TestClient`` in lexical retrieval mode, so no
embedder is involved; a prior record that only a YAML parser could have written
(an unquoted timestamp, a mapping) is written by hand and indexed.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from palinode import mcp
from palinode.api import server
from palinode.core import embedder, store
from palinode.core.config import config
from palinode.indexer import reconcile
from palinode.indexer.index_file import index_file

_KEYWORD = "harborlight"


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.search, "retrieval_mode", "lexical")
    monkeypatch.setattr(config.git, "auto_commit", False)
    no_model = Mock(side_effect=AssertionError("a model was contacted"))
    monkeypatch.setattr(embedder, "embed", no_model)
    monkeypatch.setattr(embedder, "embed_many", no_model)
    monkeypatch.setattr(reconcile, "get_ollama_client", no_model)
    store.init_db()
    return tmp_path


@pytest.fixture()
def api(memory_dir):
    server._rate_counters.clear()
    with TestClient(server.app, raise_server_exceptions=False) as client:
        yield client
    server._rate_counters.clear()


def _write(memory_dir: Path, slug: str, front_matter: str) -> str:
    """A record written by hand, then indexed; returns its memory-relative path."""
    path = memory_dir / "insights" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: insights-{slug}\ncategory: insights\ntype: Insight\n"
        f"{front_matter}---\n\n{_KEYWORD} note {slug}\n",
        encoding="utf-8",
    )
    index_file(str(path))
    return f"insights/{slug}.md"


def _save(api, slug: str, **kwargs) -> str:
    body = {"content": f"{_KEYWORD} note {slug}", "type": "Insight", "slug": slug, **kwargs}
    response = api.post("/save", json=body)
    assert response.status_code == 200, response.text
    return response.json()["rel_path"]


def _found_all(api, surface: str, *entries: str) -> list[str]:
    """The paths the ``where`` entries select together on *surface*, asserting a 200."""
    query = _KEYWORD if surface == "search" else ""
    response = api.post(
        "/search",
        json={"query": query, "where": list(entries), "limit": 50, "threshold": 0.0},
    )
    assert response.status_code == 200, f"{surface} {entries!r}: {response.text}"
    payload = response.json()
    rows = payload["results"] if isinstance(payload, dict) else payload
    return [row["rel_path"] for row in rows]


def _found(api, surface: str, entry: str) -> list[str]:
    """The paths one ``where`` entry selects on *surface*, asserting a 200."""
    return _found_all(api, surface, entry)


_SURFACES = ["search", "recent"]


# ── timestamps ──────────────────────────────────────────────────────────────

_INSTANT = "2026-03-01T10:00:00Z"
_INSTANT_SPELLINGS = [
    "2026-03-01T10:00:00+00:00",
    "2026-03-01 10:00:00+00:00",
    "2026-03-01T10:00:00Z",
    "2026-03-01T11:00:00+01:00",
    "2026-03-01T10:00:00.000000+00:00",
    "2026-03-01T10:00:00",
    "2026-03-01 10:00:00",
    "2026-03-01T10:00:00.000000",
    "2026-03-01T12:00:00+02:00",
    "2026-03-01 12:00:00+02:00",
]
_INSTANT_IDS = [
    "T-offset", "space-offset", "Z", "other-offset", "microseconds",
    "naive-T", "naive-space", "naive-microseconds", "plus-two", "space-plus-two",
]
# How `_INSTANT` can be stored: YAML timestamps, which a parser turns into datetimes,
# and text, which a save keeps as written.
_STORED_AS_YAML = {
    "yaml-naive": "due: 2026-03-01 10:00:00\n",
    "yaml-utc": "due: 2026-03-01T10:00:00Z\n",
    "yaml-plus-two": "due: 2026-03-01T12:00:00+02:00\n",
}
_STORED_AS_TEXT = {
    "text-naive-T": "2026-03-01T10:00:00",
    "text-naive-space": "2026-03-01 10:00:00",
    "text-offset": "2026-03-01T10:00:00+00:00",
}


@pytest.fixture()
def stamped(api, memory_dir) -> dict[str, str]:
    """``due`` is `_INSTANT` in six stored forms, and one second later in a seventh."""
    records = {
        name: _write(memory_dir, name, front_matter)
        for name, front_matter in _STORED_AS_YAML.items()
    }
    for name, due in _STORED_AS_TEXT.items():
        records[name] = _save(api, name, metadata={"due": due})
    records["other"] = _save(api, "due-other", metadata={"due": "2026-03-01T10:00:01+00:00"})
    return records


def _the_instant(stamped: dict[str, str]) -> list[str]:
    return sorted(path for name, path in stamped.items() if name != "other")


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize("spelling", _INSTANT_SPELLINGS, ids=_INSTANT_IDS)
def test_every_spelling_of_an_instant_selects_every_way_it_is_stored(
    api, surface, stamped, spelling
):
    """A stored value with no offset equals an entry with one, and the other way
    round. Kills any change that compares a timestamp by its text, or by
    one spelling: the instant dropped, the ``[T ]`` pattern narrowed, ``Z``
    unsupported, a missing offset left unread."""
    found = _found(api, surface, f"due={spelling}")

    assert sorted(found) == _the_instant(stamped)


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_different_instant_selects_neither(api, surface, stamped):
    """Kills the instant comparison collapsing to a prefix or to a date match."""
    assert _found(api, surface, "due=2026-03-01T10:00:02Z") == []
    assert _found(api, surface, "due=2026-03-01T10:00:02") == []


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_value_with_no_offset_is_not_read_in_the_offset_of_the_other(api, surface, stamped):
    """Kills a missing offset being compared by its wall clock: ten o'clock at
    ``+02:00`` is eight o'clock UTC, and nothing was stored then."""
    assert _found(api, surface, "due=2026-03-01T10:00:00+02:00") == []
    assert _found(api, surface, "due=2026-03-01T08:00:00") == []


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")
@pytest.mark.parametrize("zone", ["Europe/Warsaw", "America/New_York", "Asia/Kolkata"])
def test_a_value_with_no_offset_is_utc_in_every_local_zone(api, surface, stamped, zone):
    """Kills a missing offset being read in the machine's zone, which a laptop
    set to UTC would never show."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        found = _found(api, surface, f"due={_INSTANT}")
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()

    assert sorted(found) == _the_instant(stamped)


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize("spelling", ["2026-04-01 11:00:00+01:00", "2026-04-01T10:00:00"])
def test_a_list_of_timestamps_matches_by_element_as_an_instant(api, surface, memory_dir, spelling):
    """Kills list elements being compared by text while scalars compare as instants.
    The second element has no offset."""
    path = _write(
        memory_dir, "due-many",
        "due:\n- 2026-03-01T10:00:00Z\n- 2026-04-01 10:00:00\n",
    )
    _write(memory_dir, "due-one", "due: 2026-05-01T10:00:00Z\n")

    assert _found(api, surface, f"due={spelling}") == [path]


@pytest.fixture()
def days(api, memory_dir) -> dict[str, str]:
    """``day`` in the spellings a date takes, around 12 October 2026 (UTC)."""
    return {
        "date": _write(memory_dir, "day-date", "day: 2026-10-12\n"),
        "text-date": _write(memory_dir, "day-text-date", 'day: "2026-10-12"\n'),
        "midnight-utc": _write(memory_dir, "day-midnight-utc", "day: 2026-10-12T00:00:00Z\n"),
        "midnight-naive": _write(memory_dir, "day-midnight-naive", "day: 2026-10-12 00:00:00\n"),
        "ten": _write(memory_dir, "day-ten", "day: 2026-10-12T10:00:00Z\n"),
        "last-second": _write(memory_dir, "day-last-second", "day: 2026-10-12T23:59:59Z\n"),
        "next-day": _write(memory_dir, "day-next-day", "day: 2026-10-13T00:00:00Z\n"),
        "compact": _write(memory_dir, "day-compact", "day: 20261012\n"),
        "week": _write(memory_dir, "day-week", "day: 2026-W42\n"),
        "basic": _write(memory_dir, "day-basic", "day: 20261012T000000Z\n"),
    }


_THE_DAY = ["date", "text-date", "midnight-utc", "midnight-naive", "ten", "last-second"]
_MIDNIGHT = ["date", "text-date", "midnight-utc", "midnight-naive"]


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize(
    "entry, expected",
    [
        ("day=2026-10-12", _THE_DAY),
        ("day=2026-10-13", ["next-day"]),
        ("day=2026-10-12T00:00:00Z", _MIDNIGHT),
        ("day=2026-10-12T00:00:00", _MIDNIGHT),
        ("day=2026-10-12 00:00:00+00:00", _MIDNIGHT),
        ("day=2026-10-12T02:00:00+02:00", _MIDNIGHT),
        ("day=2026-10-12T10:00:00Z", ["ten"]),
        ("day=2026-10-12T23:59:59Z", ["last-second"]),
        ("day=20261012", ["compact"]),
        ("day=2026-W42", ["week"]),
        ("day=20261012T000000Z", ["basic"]),
    ],
)
def test_dates_and_timestamps_read_as_the_date_window_reads_them(
    api, surface, days, entry, expected
):
    """A date or a timestamp V matches as ``in_date_window(stored, V, V)`` keeps it:
    a timestamp V only that instant, a bare date V any time on that UTC day, and a
    stored bare date is midnight UTC. ``20261012``, ``2026-W42`` (Monday 12 October)
    and ``20261012T000000Z`` are text, which the window would read as dates.

    Kills the date gate being dropped on either side, a bare date V read as one
    instant or a timestamp V read as a day, a stored bare date left as text, and
    the end of the day being a day early or late."""
    found = _found(api, surface, entry)

    assert sorted(found) == sorted(days[name] for name in expected)


# ── the edges of the calendar ───────────────────────────────────────────────

# Valid ISO values whose UTC reading falls outside years 1 to 9999. They are not dates
# to compare, so they are text, like any value that is not a date.
_EARLY = "0001-01-01T00:00+01:00"
_LATE = "9999-12-31T23:59-01:00"
_EXTREMES = {"early": _EARLY, "late": _LATE}
_ON_THE_DAY = "2026-10-12T05:00:00Z"


@pytest.fixture()
def extremes(api, memory_dir) -> dict[str, str]:
    """Records with an extreme ``d``, alone and in a list beside a date on 12 October."""
    records = {
        "on-the-day": _write(memory_dir, "x-day", f"status: open\nd: {_ON_THE_DAY}\n"),
        "elsewhere": _write(memory_dir, "x-elsewhere", "status: open\nd: 2026-10-13T05:00:00Z\n"),
    }
    for tag, value in _EXTREMES.items():
        records[tag] = _write(memory_dir, f"x-{tag}", f'status: open\nd: "{value}"\n')
        records[f"{tag}-then-day"] = _write(
            memory_dir, f"x-{tag}-then-day", f'status: open\nd:\n- "{value}"\n- {_ON_THE_DAY}\n'
        )
        records[f"day-then-{tag}"] = _write(
            memory_dir, f"x-day-then-{tag}", f'status: open\nd:\n- {_ON_THE_DAY}\n- "{value}"\n'
        )
    return records


def _the_day_and_its_lists(extremes: dict[str, str]) -> list[str]:
    names = ["on-the-day"]
    for tag in _EXTREMES:
        names += [f"{tag}-then-day", f"day-then-{tag}"]
    return sorted(extremes[name] for name in names)


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_record_with_an_extreme_value_fails_only_a_date_condition(api, surface, extremes):
    """A guard that must keep passing: the extreme record is listed when another
    condition selects it, and is not a date on 12 October."""
    assert extremes["early"] in _found(api, surface, "status=open")
    assert extremes["late"] in _found(api, surface, "status=open")
    assert extremes["early"] not in _found(api, surface, "d=2026-10-12")
    assert extremes["late"] not in _found(api, surface, "d=2026-10-12")


@pytest.mark.parametrize("surface", _SURFACES)
def test_an_extreme_value_in_a_list_does_not_hide_the_element_that_matches(api, surface, extremes):
    """An extreme element must not stop the next one being read: the route
    swallows an error per file, so a raise would drop the record without a word.
    A scalar cannot show it, since a raise reads as a miss; a list can."""
    expected = _the_day_and_its_lists(extremes)

    assert sorted(_found(api, surface, "d=2026-10-12")) == expected
    assert sorted(_found_all(api, surface, "status=open", "d=2026-10-12")) == expected
    assert _found_all(api, surface, "d=2026-10-12", "status=resolved") == []


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize("tag", _EXTREMES)
def test_an_extreme_entry_lists_the_records_whose_text_equals_it(api, surface, extremes, tag):
    """The entry is text, so it selects the records that hold exactly it and no
    date, including a record that holds a date before the equal text: comparing
    the entry with that date must not raise."""
    found = _found(api, surface, f"d={_EXTREMES[tag]}")

    assert sorted(found) == sorted(
        extremes[name] for name in (tag, f"{tag}-then-day", f"day-then-{tag}")
    )


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize("value", ["2026-02-30", "2026-13-45", "2026-00-10"])
def test_an_invalid_date_matches_only_equal_text(api, surface, memory_dir, value):
    """A guard that must keep passing. Kills an invalid date being rolled
    over to a real one (30 February to 2 March), or compared by its first ten
    characters."""
    same = _write(memory_dir, "bad-same", f'd: "{value}"\n')
    timed = _write(memory_dir, "bad-timed", f'd: "{value}T10:00:00Z"\n')
    march = _write(memory_dir, "real-march", 'd: "2026-03-02"\n')
    ok = _write(memory_dir, "real-ok", 'd: "2026-02-28"\n')

    assert _found(api, surface, f"d={value}") == [same]
    assert _found(api, surface, f"d={value}T10:00:00Z") == [timed]
    assert sorted(_found(api, surface, "d=2026-03-02")) == [march]
    assert _found(api, surface, "d=2026-02-28") == [ok]


@pytest.mark.parametrize("surface", _SURFACES)
def test_an_hour_without_minutes_is_text(api, surface, memory_dir):
    """A guard that must keep passing. A time needs hh:mm to be read, so
    ``2026-10-12T10`` is no date: it selects itself, and the day does not select it."""
    hour = _write(memory_dir, "hour-only", 'd: "2026-10-12T10"\n')
    full = _write(memory_dir, "hour-full", "d: 2026-10-12T10:00:00Z\n")

    assert _found(api, surface, "d=2026-10-12T10") == [hour]
    assert _found(api, surface, "d=2026-10-12") == [full]
    assert _found(api, surface, "d=2026-10-12T10:00:00Z") == [full]


@pytest.mark.parametrize("surface", _SURFACES)
def test_the_last_microsecond_of_the_day_is_in_the_day(api, surface, memory_dir):
    """A guard that must keep passing. Kills the end of the day being
    cut at the second, or a microsecond early."""
    last = _write(memory_dir, "last-micro", "d: 2026-10-12T23:59:59.999999Z\n")
    next_day = _write(memory_dir, "next-micro", "d: 2026-10-13T00:00:00Z\n")

    found = _found(api, surface, "d=2026-10-12")

    assert last in found
    assert next_day not in found


@pytest.mark.parametrize("surface", _SURFACES)
def test_fractions_of_a_second_are_compared_exactly_against_a_timestamp(api, surface, memory_dir):
    """A guard that must keep passing. Kills fractions being dropped or
    rounded before the comparison."""
    half = _write(memory_dir, "half", "d: 2026-10-12T10:00:00.500000Z\n")
    half_offset = _write(memory_dir, "half-offset", 'd: "2026-10-12T12:00:00.5+02:00"\n')
    whole = _write(memory_dir, "whole", "d: 2026-10-12T10:00:00Z\n")
    micro = _write(memory_dir, "micro", "d: 2026-10-12T10:00:00.000001Z\n")

    assert sorted(_found(api, surface, "d=2026-10-12T10:00:00.5Z")) == sorted([half, half_offset])
    assert _found(api, surface, "d=2026-10-12T10:00:00Z") == [whole]
    assert _found(api, surface, "d=2026-10-12T10:00:00.000001Z") == [micro]


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")
@pytest.mark.parametrize(
    "zone", ["UTC", "Europe/Warsaw", "America/New_York", "Asia/Kolkata", "Pacific/Auckland"]
)
def test_a_day_is_the_utc_day_and_not_the_local_one(api, surface, memory_dir, zone):
    """A guard that must keep passing. ``00:30+02:00`` on the 12th is the
    11th in UTC; ``23:30-02:00`` on the 11th is the 12th. Kills the day being cut
    in the machine's zone or in the stored offset."""
    early = _write(memory_dir, "tz-early", 'd: "2026-10-12T00:30+02:00"\n')
    late = _write(memory_dir, "tz-late", 'd: "2026-10-11T23:30-02:00"\n')
    before = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        on_the_12th = _found(api, surface, "d=2026-10-12")
        on_the_11th = _found(api, surface, "d=2026-10-11")
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()

    assert on_the_12th == [late]
    assert on_the_11th == [early]


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_day_selects_a_list_when_one_element_is_on_it(api, surface, memory_dir):
    """A guard that must keep passing. Kills a list needing every element
    on the day, or only its first or last."""
    middle = _write(
        memory_dir, "list-middle",
        "d:\n- 2026-10-11T23:59:59Z\n- 2026-10-12T08:00:00Z\n- 2026-10-13T00:00:00Z\n",
    )
    by_offset = _write(
        memory_dir, "list-offset",
        'd:\n- "2026-10-12T00:30+02:00"\n- "2026-10-13T01:00+02:00"\n',
    )
    _write(memory_dir, "list-none", "d:\n- 2026-10-11T23:59:59Z\n- 2026-10-13T00:00:00Z\n")
    _write(memory_dir, "list-offset-none", 'd:\n- "2026-10-12T00:30+02:00"\n- "2026-10-13T02:00+02:00"\n')

    assert sorted(_found(api, surface, "d=2026-10-12")) == sorted([middle, by_offset])


# ── whitespace around a date ────────────────────────────────────────────────

@pytest.fixture()
def padded(api, memory_dir) -> dict[str, str]:
    """``day`` stored with whitespace around a date, a timestamp, and junk."""
    return {
        "trail-date": _write(memory_dir, "pad-trail-date", 'day: "2026-10-12 "\n'),
        "lead-date": _write(memory_dir, "pad-lead-date", 'day: " 2026-10-12"\n'),
        "tab-date": _write(memory_dir, "pad-tab-date", 'day: "2026-10-12\\t"\n'),
        "lead-timed": _write(memory_dir, "pad-lead-timed", 'day: " 2026-10-12T10:00:00Z"\n'),
        "trail-timed": _write(memory_dir, "pad-trail-timed", 'day: "2026-10-12T10:00:00Z "\n'),
        "newline-timed": _write(memory_dir, "pad-newline-timed", 'day: "2026-10-12T10:00\\n"\n'),
        "offset": _write(memory_dir, "pad-offset", 'day: "2026-10-12T10:00:00 +02:00"\n'),
        "junk": _write(memory_dir, "pad-junk", 'day: "2026-10-12T10:00x"\n'),
        "blank": _write(memory_dir, "pad-blank", 'day: "   "\n'),
        "date": _write(memory_dir, "pad-date", "day: 2026-10-12\n"),
        "ten": _write(memory_dir, "pad-ten", "day: 2026-10-12T10:00:00Z\n"),
    }


_PADDED_DATES = ["trail-date", "lead-date", "tab-date", "date"]
_PADDED_TEN = ["lead-timed", "trail-timed", "newline-timed", "ten"]
_PADDED_THE_DAY = _PADDED_DATES + _PADDED_TEN + ["offset"]


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize(
    "entry, expected",
    [
        ("day=2026-10-12", _PADDED_THE_DAY),
        ("day=2026-10-12 ", _PADDED_THE_DAY),
        ("day= 2026-10-12", _PADDED_THE_DAY),
        ("day=2026-10-12\t", _PADDED_THE_DAY),
        ("day=2026-10-12T00:00:00Z", _PADDED_DATES),
        ("day= 2026-10-12T00:00:00Z", _PADDED_DATES),
        ("day=2026-10-12T10:00:00Z", _PADDED_TEN),
        ("day= 2026-10-12T10:00:00Z", _PADDED_TEN),
        ("day=2026-10-12T10:00 ", _PADDED_TEN),
        ("day=2026-10-12T08:00:00Z", ["offset"]),
        ("day=2026-10-12T10:00x", ["junk"]),
        ("day=   ", ["blank"]),
        ("day=2026-10-13", []),
    ],
)
def test_whitespace_around_a_date_does_not_change_what_search_selects(
    api, surface, padded, entry, expected
):
    """Padded dates and timestamps are read as the window reads them, on the stored
    side and the entry side alike; junk and blanks stay text, a space before an
    offset stays a timestamp. Kills the date test reading either side unstripped, and
    a date pattern that refuses an inner space."""
    found = _found(api, surface, entry)

    assert sorted(found) == sorted(padded[name] for name in expected)


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize(
    "stored, entry, expected",
    [
        ("ready ", "ready", False),
        (" ready", "ready", False),
        ("ready", "ready ", False),
        ("ready ", "ready ", True),
    ],
)
def test_text_that_is_not_a_date_is_compared_exactly(api, surface, memory_dir, stored, entry, expected):
    """A guard that must keep passing: only the date test ignores whitespace. Kills
    the strip being applied to the equality comparison."""
    path = _write(memory_dir, "plain-text", f'status: "{stored}"\n')

    assert (path in _found(api, surface, f"status={entry}")) is expected


# ── non-scalar values ───────────────────────────────────────────────────────

@pytest.fixture()
def shaped(api, memory_dir) -> dict[str, str]:
    return {
        "null": _write(memory_dir, "null", "owner:\n"),
        "mapping": _write(memory_dir, "mapping", "meta:\n  name: alice\n"),
        "mappings": _write(
            memory_dir, "mappings",
            "reviewers:\n- name: alice\n- name: bob\n",
        ),
        "mixed": _write(
            memory_dir, "mixed",
            "reviewers:\n- alice\n- name: bob\n- null\n- - carol\n",
        ),
    }


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize(
    "entry",
    ["owner=None", "owner=null", "owner=~", "owner=owner"],
)
def test_a_null_value_never_matches(api, shaped, surface, entry):
    """Kills ``_as_text`` turning ``None`` into the string ``"None"``."""
    assert _found(api, surface, entry) == []


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize(
    "entry",
    ["meta=alice", "meta=name", "meta=name: alice", "meta={'name': 'alice'}"],
)
def test_a_mapping_value_never_matches(api, shaped, surface, entry):
    """Kills a mapping being matched by its keys, values or Python text."""
    assert _found(api, surface, entry) == []


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize(
    "entry",
    ["reviewers=alice", "reviewers=name", "reviewers={'name': 'alice'}"],
)
def test_a_list_of_mappings_never_matches(api, shaped, surface, entry):
    """Kills the mappings in a list being matched through their Python text or
    their keys and values."""
    found = _found(api, surface, entry)

    assert shaped["mappings"] not in found


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_list_that_mixes_strings_and_mappings_matches_its_strings_only(
    api, shaped, surface
):
    """Kills a list with a non-scalar element being skipped as a whole, and a
    mapping element being matched."""
    assert _found(api, surface, "reviewers=alice") == [shaped["mixed"]]
    assert _found(api, surface, "reviewers=bob") == []
    assert _found(api, surface, "reviewers=None") == []
    assert _found(api, surface, "reviewers=['carol']") == []


# ── the MCP argument ────────────────────────────────────────────────────────

class _Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=[])


@pytest.fixture()
def wire(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(mcp, "_http_transport", httpx.MockTransport(recorder.handle))
    monkeypatch.setattr(mcp, "_http_client", None)
    monkeypatch.setattr(mcp, "_http_client_loop", None)
    return recorder


def _call(tool: str, arguments: dict) -> str:
    return asyncio.run(mcp._dispatch_tool(tool, arguments))[0].text


def _sent_where(wire: _Recorder, tool: str) -> list[str]:
    return json.loads(wire.requests[-1].content).get("where", [])


def _arguments(tool: str, where) -> dict:
    return {"query": _KEYWORD, "where": where}


_TOOLS = ["palinode_search"]


@pytest.mark.parametrize("tool", _TOOLS)
def test_a_bare_string_is_one_entry(wire, tool):
    """Kills the bare string being iterated into characters."""
    _call(tool, _arguments(tool, "status=open"))

    assert _sent_where(wire, tool) == ["status=open"]


@pytest.mark.parametrize("tool", _TOOLS)
def test_a_json_string_array_is_decoded_into_its_entries(wire, tool):
    """Kills ``coerce_str_array`` being dropped from the where argument."""
    _call(tool, _arguments(tool, '["status=open", "kind=note"]'))

    assert _sent_where(wire, tool) == ["status=open", "kind=note"]


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize("empty", [None, [], "[]", ""], ids=["none", "list", "json", "text"])
def test_an_empty_where_sends_no_filter(wire, tool, empty):
    """Kills an empty argument being sent, which would change the request key."""
    _call(tool, _arguments(tool, empty))

    assert len(wire.requests) == 1
    assert _sent_where(wire, tool) == []


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize(
    "where",
    [
        {"status": "open"},
        {"status=open": "x"},
        '{"status": "open"}',
        '"status=open"',
        "status",
        '["status"]',
    ],
    ids=["object", "object-keyed-like-an-entry", "json-object", "json-string",
         "bare-key", "json-array-bare-key"],
)
def test_a_where_that_is_not_a_list_of_entries_is_refused_before_any_request(
    wire, tool, where
):
    """Kills the list check being dropped (an object keyed like an entry would
    pass the parser and be sent) or the entry check being dropped."""
    text = _call(tool, _arguments(tool, where))

    assert text.startswith("Error:")
    assert wire.requests == []


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize(
    "item",
    [123, None, True, 1.5, ["status=open"], {"status": "open"}],
    ids=["int", "null", "bool", "float", "list", "mapping"],
)
def test_a_non_string_item_is_refused_before_any_request(wire, tool, item):
    """Kills the items being stringified into an entry that passes the parser."""
    text = _call(tool, _arguments(tool, ["status=open", item]))

    assert text.startswith("Error:")
    assert wire.requests == []
