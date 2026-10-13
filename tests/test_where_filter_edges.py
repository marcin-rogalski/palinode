"""Edges of ``where`` on the listing: timestamps, MCP argument shapes, non-scalar values.

``test_where_filter.py`` and ``test_where_filter_surfaces.py`` pin the contract.
This file pins three places where the matching or the argument handling has more
than one way to go wrong:

* a timestamp is matched as an instant: the ``T``, space, ``Z`` and offset
  spellings select the same record on ``GET /list``, for a value stored as text
  and for one stored as a YAML timestamp, and a value with no offset is UTC. Only
  a date and a time are timestamps: a bare date, ``20261012`` and ``2026-W42``
  are compared as text, so none of them equals ``2026-10-12`` or a midnight;
* the ``palinode_list`` ``where`` argument is a list, a bare string or a
  JSON-string array; an object or an item that is not a string is refused naming
  the problem, with no request made;
* a key holding null, a mapping, or a list of mappings never matches and never
  fails the request.

Each docstring names the mutant it kills. A test that is red on the head it was
written against says why.

Real SQLite + tmp_path through ``TestClient`` in lexical retrieval mode, so no
embedder is involved; a prior record that only a YAML parser could have written
(an unquoted timestamp, a mapping) is written by hand and indexed.
"""
from __future__ import annotations

import asyncio
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


def _found(api, entry: str) -> list[str]:
    """The paths one ``where`` entry selects on ``GET /list``, asserting a 200."""
    response = api.get("/list", params={"where": entry})
    assert response.status_code == 200, f"{entry!r}: {response.text}"
    return [row["file"] for row in response.json()]


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


@pytest.mark.parametrize("spelling", _INSTANT_SPELLINGS, ids=_INSTANT_IDS)
def test_every_spelling_of_an_instant_selects_every_way_it_is_stored(
    api, stamped, spelling
):
    """Red: a stored value with no offset never equals an entry with one, and the
    other way round. Kills any change that compares a timestamp by its text, or by
    one spelling: the instant dropped, the ``[T ]`` pattern narrowed, ``Z``
    unsupported, a missing offset left unread."""
    found = _found(api, f"due={spelling}")

    assert sorted(found) == _the_instant(stamped)


def test_a_different_instant_selects_neither(api, stamped):
    """Kills the instant comparison collapsing to a prefix or to a date match."""
    assert _found(api, "due=2026-03-01T10:00:02Z") == []
    assert _found(api, "due=2026-03-01T10:00:02") == []


def test_a_value_with_no_offset_is_not_read_in_the_offset_of_the_other(api, stamped):
    """Kills a missing offset being compared by its wall clock: ten o'clock at
    ``+02:00`` is eight o'clock UTC, and nothing was stored then."""
    assert _found(api, "due=2026-03-01T10:00:00+02:00") == []
    assert _found(api, "due=2026-03-01T08:00:00") == []


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")
@pytest.mark.parametrize("zone", ["Europe/Warsaw", "America/New_York", "Asia/Kolkata"])
def test_a_value_with_no_offset_is_utc_in_every_local_zone(api, stamped, zone):
    """Kills a missing offset being read in the machine's zone, which a laptop
    set to UTC would never show."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        found = _found(api, f"due={_INSTANT}")
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()

    assert sorted(found) == _the_instant(stamped)


@pytest.mark.parametrize("spelling", ["2026-04-01 11:00:00+01:00", "2026-04-01T10:00:00"])
def test_a_list_of_timestamps_matches_by_element_as_an_instant(api, memory_dir, spelling):
    """Kills list elements being compared by text while scalars compare as instants.
    The second element has no offset."""
    path = _write(
        memory_dir, "due-many",
        "due:\n- 2026-03-01T10:00:00Z\n- 2026-04-01 10:00:00\n",
    )
    _write(memory_dir, "due-one", "due: 2026-05-01T10:00:00Z\n")

    assert _found(api, f"due={spelling}") == [path]


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("day=2026-10-12", ["date"]),
        ("day=20261012", ["compact"]),
        ("day=2026-W42", ["week"]),
        ("day=20261012T000000Z", ["basic"]),
        ("day=2026-10-12T00:00:00Z", ["midnight-utc", "midnight-naive"]),
        ("day=2026-10-12 00:00:00", ["midnight-utc", "midnight-naive"]),
        ("day=2026-10-12T00:00:00", ["midnight-utc", "midnight-naive"]),
    ],
)
def test_only_a_date_and_a_time_are_compared_as_instants(api, memory_dir, entry, expected):
    """Everything else is text. ``parse_iso_utc`` reads a bare date as midnight and
    accepts ``20261012``, ``2026-W42`` (Monday 12 October) and ``20261012T000000Z``,
    so a matcher that hands it every value makes five records one.

    The two midnight rows are red for the missing offset only; the rest pass on
    v0.22.0 and must keep passing. Kills the ``[T ]`` pattern gate being dropped
    on either side."""
    records = {
        "date": _write(memory_dir, "day-date", "day: 2026-10-12\n"),
        "compact": _write(memory_dir, "day-compact", "day: 20261012\n"),
        "week": _write(memory_dir, "day-week", "day: 2026-W42\n"),
        "basic": _write(memory_dir, "day-basic", "day: 20261012T000000Z\n"),
        "midnight-utc": _write(memory_dir, "day-midnight-utc", "day: 2026-10-12T00:00:00Z\n"),
        "midnight-naive": _write(memory_dir, "day-midnight-naive", "day: 2026-10-12 00:00:00\n"),
    }

    found = _found(api, entry)

    assert sorted(found) == sorted(records[name] for name in expected)


def test_a_stored_text_date_is_a_day_and_not_a_midnight(api, memory_dir):
    """The same as the table above for a date stored as text, which is how a save
    keeps it. Kills the gate being applied to the entry only."""
    day = _write(memory_dir, "day-text", 'day: "2026-10-12"\n')
    midnight = _write(memory_dir, "midnight-text", 'day: "2026-10-12T00:00:00Z"\n')

    assert _found(api, "day=2026-10-12") == [day]
    assert _found(api, "day=2026-10-12T00:00:00Z") == [midnight]


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


@pytest.mark.parametrize(
    "entry",
    ["owner=None", "owner=null", "owner=~", "owner=owner"],
)
def test_a_null_value_never_matches(api, shaped, entry):
    """Kills ``_as_text`` turning ``None`` into the string ``"None"``."""
    assert _found(api, entry) == []


@pytest.mark.parametrize(
    "entry",
    ["meta=alice", "meta=name", "meta=name: alice", "meta={'name': 'alice'}"],
)
def test_a_mapping_value_never_matches(api, shaped, entry):
    """Kills a mapping being matched by its keys, values or Python text."""
    assert _found(api, entry) == []


@pytest.mark.parametrize(
    "entry",
    ["reviewers=alice", "reviewers=name", "reviewers={'name': 'alice'}"],
)
def test_a_list_of_mappings_never_matches(api, shaped, entry):
    """Kills the mappings in a list being matched through their Python text or
    their keys and values."""
    found = _found(api, entry)

    assert shaped["mappings"] not in found


def test_a_list_that_mixes_strings_and_mappings_matches_its_strings_only(api, shaped):
    """Kills a list with a non-scalar element being skipped as a whole, and a
    mapping element being matched."""
    assert _found(api, "reviewers=alice") == [shaped["mixed"]]
    assert _found(api, "reviewers=bob") == []
    assert _found(api, "reviewers=None") == []
    assert _found(api, "reviewers=['carol']") == []


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


def _list_with(where) -> str:
    return asyncio.run(mcp._dispatch_tool("palinode_list", {"where": where}))[0].text


def _sent_where(wire: _Recorder) -> list[str]:
    return wire.requests[-1].url.params.get_list("where")


def test_a_bare_string_is_one_entry(wire):
    """Kills the bare string being iterated into characters."""
    _list_with("status=open")

    assert _sent_where(wire) == ["status=open"]


def test_a_json_string_array_is_decoded_into_its_entries(wire):
    """Kills ``coerce_str_array`` being dropped from the where argument."""
    _list_with('["status=open", "kind=note"]')

    assert _sent_where(wire) == ["status=open", "kind=note"]


@pytest.mark.parametrize("empty", [None, [], "[]", ""], ids=["none", "list", "json", "text"])
def test_an_empty_where_sends_no_filter(wire, empty):
    """Kills an empty argument being sent, which would change the request key."""
    _list_with(empty)

    assert len(wire.requests) == 1
    assert _sent_where(wire) == []


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
def test_a_where_that_is_not_a_list_of_entries_is_refused_before_any_request(wire, where):
    """Kills the list check being dropped (an object keyed like an entry would
    pass the parser and be sent) or the entry check being dropped."""
    text = _list_with(where)

    assert text.startswith("Error:")
    assert wire.requests == []


@pytest.mark.parametrize(
    "item",
    [123, None, True, 1.5, ["status=open"], {"status": "open"}],
    ids=["int", "null", "bool", "float", "list", "mapping"],
)
def test_a_non_string_item_is_refused_before_any_request(wire, item):
    """Kills the items being stringified into an entry that passes the parser."""
    text = _list_with(["status=open", item])

    assert text.startswith("Error:")
    assert wire.requests == []
