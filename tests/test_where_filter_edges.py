"""Edges of ``where``: timestamp spellings, MCP argument shapes, non-scalar values.

``test_where_filter.py`` and ``test_where_filter_surfaces.py`` pin the contract.
This file pins three places where the matching or the argument handling has more
than one way to go wrong, each across every surface that must agree:

* a timestamp is matched as an instant, so the ``T``, space, ``Z`` and offset
  spellings select the same record on ``GET /list``, on ``POST /search`` and on
  the recency listing (an empty query), for a value stored as text and for one
  stored as a YAML timestamp;
* the MCP ``where`` argument is a list, a bare string or a JSON-string array;
  an object or an item that is not a string is refused naming the problem, with
  no request made;
* a key holding null, a mapping, or a list of mappings never matches, on any
  surface, and never fails the request.

Every test here passes on the feature head. Each docstring names the mutant it
kills.

Real SQLite + tmp_path through ``TestClient`` in lexical retrieval mode, so no
embedder is involved; a prior record that only a YAML parser could have written
(an unquoted timestamp, a mapping) is written by hand and indexed.
"""
from __future__ import annotations

import asyncio
import json
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


def _found(api, surface: str, entry: str) -> list[str]:
    """The paths one ``where`` entry selects on *surface*, asserting a 200."""
    if surface == "list":
        response = api.get("/list", params={"where": entry})
        rows = response.json() if response.status_code == 200 else None
        paths = [row["file"] for row in rows or []]
    else:
        query = _KEYWORD if surface == "search" else ""
        response = api.post(
            "/search",
            json={"query": query, "where": [entry], "limit": 10, "threshold": 0.0},
        )
        payload = response.json() if response.status_code == 200 else None
        rows = payload["results"] if isinstance(payload, dict) else payload
        paths = [row["rel_path"] for row in rows or []]
    assert response.status_code == 200, f"{surface} {entry!r}: {response.text}"
    return paths


_SURFACES = ["list", "search", "recent"]


# ── timestamps ──────────────────────────────────────────────────────────────

_INSTANT_SPELLINGS = [
    "2026-03-01T10:00:00+00:00",
    "2026-03-01 10:00:00+00:00",
    "2026-03-01T10:00:00Z",
    "2026-03-01T11:00:00+01:00",
    "2026-03-01T10:00:00.000000+00:00",
]
_INSTANT_IDS = ["T-offset", "space-offset", "Z", "other-offset", "microseconds"]


@pytest.fixture()
def stamped(api, memory_dir) -> dict[str, str]:
    """``due`` stored as text (saved through the API) and as a YAML timestamp."""
    return {
        "text": _save(api, "due-text", metadata={"due": "2026-03-01T10:00:00+00:00"}),
        "yaml": _write(memory_dir, "due-yaml", "due: 2026-03-01T10:00:00Z\n"),
        "other": _save(api, "due-other", metadata={"due": "2026-03-01T10:00:01+00:00"}),
    }


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize("spelling", _INSTANT_SPELLINGS, ids=_INSTANT_IDS)
def test_every_spelling_of_an_instant_selects_the_same_records(
    api, stamped, surface, spelling
):
    """Kills any change that compares a timestamp by its text, or by one spelling:
    ``_as_instant`` dropped, the ``[T ]`` pattern narrowed, ``Z`` unsupported."""
    found = _found(api, surface, f"due={spelling}")

    assert sorted(found) == sorted([stamped["text"], stamped["yaml"]])


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_different_instant_selects_neither(api, stamped, surface):
    """Kills the instant comparison collapsing to a prefix or to a date match."""
    found = _found(api, surface, "due=2026-03-01T10:00:02Z")

    assert found == []


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_list_of_timestamps_matches_by_element_as_an_instant(
    api, memory_dir, surface
):
    """Kills list elements being compared by text while scalars compare as instants."""
    path = _write(
        memory_dir, "due-many",
        "due:\n- 2026-03-01T10:00:00Z\n- 2026-04-01T10:00:00Z\n",
    )

    found = _found(api, surface, "due=2026-04-01 11:00:00+01:00")

    assert found == [path]


@pytest.mark.parametrize("surface", _SURFACES)
def test_a_timestamp_with_no_offset_never_fails_the_request(api, memory_dir, surface):
    """A stored value with no offset against an entry that has one cannot be
    compared as instants. Kills the comparison raising on mixed awareness."""
    _write(memory_dir, "due-naive", "due: 2026-03-01 10:00:00\n")

    found = _found(api, surface, "due=2026-03-01T10:00:00Z")

    assert isinstance(found, list)


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
    request = wire.requests[-1]
    if tool == "palinode_list":
        return request.url.params.get_list("where")
    return json.loads(request.content).get("where", [])


def _arguments(tool: str, where) -> dict:
    return {"where": where} if tool == "palinode_list" else {"query": _KEYWORD, "where": where}


_TOOLS = ["palinode_list", "palinode_search"]


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
