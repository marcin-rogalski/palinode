"""``where`` filters search and list by front matter; ``include_inbox`` lists ActionItems.

Two gaps, one change.

*Listing ActionItems.* ``type: ActionItem`` saves to ``inbox/``, and ``GET /list``
always skips ``inbox/`` and ``daily/`` and ``archive/``, so a saved ActionItem
can never be listed. ``include_inbox=true`` removes ``inbox`` (and only
``inbox``) from that skip set.

*Filtering by any front-matter key.* Search filters by ``type`` and by nothing
else, so "the open items" can only be found by reading every hit. ``where`` is a
list of ``key=value`` strings on search and list:

* every entry must match (AND);
* a scalar front-matter value matches when its string form equals the value; a
  list value matches when any element does;
* a key the record does not have never matches;
* a key must match ``^[A-Za-z_][A-Za-z0-9_-]*$``; a malformed entry is a 422.

Tests marked *red* fail on v0.22.0 for the reason in their docstring; the rest
pass on v0.22.0 and must keep passing. The surfaces (CLI, MCP, parity) are in
``test_where_filter_surfaces.py``.

Real SQLite + tmp_path through ``TestClient``, in lexical retrieval mode so no
embedder is involved (the stub raises if anything reaches for one).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from palinode.api import server
from palinode.core import embedder, store
from palinode.core.config import config
from palinode.indexer import reconcile

_KEYWORD = "harborlight"


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.search, "retrieval_mode", "lexical")
    monkeypatch.setattr(config.git, "auto_commit", False)
    no_model = Mock(side_effect=AssertionError("a model was contacted"))
    monkeypatch.setattr(embedder, "embed", no_model)
    monkeypatch.setattr(embedder, "embed_many", no_model)
    monkeypatch.setattr(reconcile, "get_ollama_client", no_model)
    store.init_db()
    server._rate_counters.clear()
    with TestClient(server.app, raise_server_exceptions=False) as client:
        yield client
    server._rate_counters.clear()


def _save(api, slug: str, *, type: str = "Insight", content: str | None = None, **kwargs) -> str:
    """Save through POST /save; returns the memory-relative path."""
    body = {
        "content": content or f"{_KEYWORD} note {slug}",
        "type": type,
        "slug": slug,
        **kwargs,
    }
    response = api.post("/save", json=body)
    assert response.status_code == 200, response.text
    return response.json()["rel_path"]


def _list(api, **params) -> list[str]:
    response = api.get("/list", params=params)
    assert response.status_code == 200, response.text
    return [row["file"] for row in response.json()]


def _search(api, **body) -> list[str]:
    body.setdefault("query", _KEYWORD)
    body.setdefault("limit", 10)
    body.setdefault("threshold", 0.0)
    response = api.post("/search", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    rows = payload["results"] if isinstance(payload, dict) else payload
    return [row["rel_path"] for row in rows]


@pytest.fixture()
def three_statuses(api) -> dict[str, str]:
    """Three Insights: status open, status resolved, no status."""
    return {
        "open": _save(api, "status-open", metadata={"status": "open"}),
        "resolved": _save(api, "status-resolved", metadata={"status": "resolved"}),
        "none": _save(api, "status-none"),
    }


@pytest.fixture()
def tagged(api) -> dict[str, str]:
    """Four open/resolved records with different entity lists."""
    return {
        "open_app": _save(api, "open-app", metadata={"status": "open"},
                          entities=["project/my-app"]),
        "open_person": _save(api, "open-person", metadata={"status": "open"},
                             entities=["person/alice"]),
        "open_both": _save(api, "open-both", metadata={"status": "open"},
                           entities=["project/my-app", "person/alice"]),
        "resolved_app": _save(api, "resolved-app", metadata={"status": "resolved"},
                              entities=["project/my-app"]),
    }


# ── list: ActionItems in inbox/ ─────────────────────────────────────────────

def test_a_saved_actionitem_is_not_listed_by_default(api):
    item = _save(api, "release-checklist", type="ActionItem")
    other = _save(api, "release-notes")

    listed = _list(api)

    assert item.startswith("inbox/")
    assert other in listed
    assert item not in listed


def test_include_inbox_lists_the_actionitem(api):
    """Red: ``include_inbox`` is ignored, so the inbox item is still skipped."""
    item = _save(api, "release-checklist", type="ActionItem")

    assert item in _list(api, include_inbox="true")


def test_include_inbox_false_still_skips_the_inbox(api):
    item = _save(api, "release-checklist", type="ActionItem")

    response = api.get("/list", params={"include_inbox": "false"})

    assert response.status_code == 200, response.text
    assert item not in [row["file"] for row in response.json()]


def test_include_inbox_lifts_the_inbox_skip_and_nothing_else(api):
    """Red: the inbox item is missing. The daily and archive records, written
    straight to disk, must stay skipped either way."""
    item = _save(api, "release-checklist", type="ActionItem")
    root = Path(config.memory_dir)
    for rel in ("daily/2026-10-01.md", "archive/old-note.md"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\nid: {path.stem}\ncategory: {path.parent.name}\n---\n\nA retired note.\n",
            encoding="utf-8",
        )

    listed = _list(api, include_inbox="true")

    assert item in listed
    assert "daily/2026-10-01.md" not in listed
    assert "archive/old-note.md" not in listed


def test_include_inbox_composes_with_category(api):
    """Red: the inbox item is missing from the unscoped listing. A ``category``
    filter still narrows the result, flag or not."""
    item = _save(api, "release-checklist", type="ActionItem")
    insight = _save(api, "release-notes")

    listed = _list(api, include_inbox="true", category="insights")

    assert insight in listed
    assert item not in listed
    assert item in _list(api, include_inbox="true")


# ── list: where ─────────────────────────────────────────────────────────────

def test_unfiltered_list_returns_all_three_statuses(api, three_statuses):
    assert set(three_statuses.values()) <= set(_list(api))


def test_list_where_status_returns_only_the_matching_record(api, three_statuses):
    """Red: ``where`` is ignored, so all three come back."""
    assert _list(api, where=["status=open"]) == [three_statuses["open"]]


def test_list_where_missing_key_matches_nothing(api, three_statuses):
    """Red: ``where`` is ignored, so every record comes back."""
    assert _list(api, where=["reviewer=dana"]) == []


def test_list_where_matches_a_custom_key(api):
    """Red: ``where`` is ignored."""
    reviewed = _save(api, "reviewed", metadata={"reviewer": "dana"})
    _save(api, "unreviewed", metadata={"reviewer": "eli"})

    assert _list(api, where=["reviewer=dana"]) == [reviewed]


def test_list_where_scalar_matches_by_string_form(api):
    """Red: ``where`` is ignored. ``priority`` is stored as the integer 2."""
    urgent = _save(api, "urgent", priority=2)
    _save(api, "routine", priority=4)

    assert _list(api, where=["priority=2"]) == [urgent]


def test_list_where_entries_are_anded(api, tagged):
    """Red: ``where`` is ignored."""
    listed = _list(api, where=["status=open", "entities=project/my-app"])

    assert sorted(listed) == sorted([tagged["open_app"], tagged["open_both"]])


def test_list_where_list_value_matches_by_element(api, tagged):
    """Red: ``where`` is ignored."""
    listed = _list(api, where=["entities=person/alice"])

    assert sorted(listed) == sorted([tagged["open_person"], tagged["open_both"]])


def test_list_where_combines_with_include_inbox(api):
    """Red: both parameters are ignored."""
    open_item = _save(api, "open-item", type="ActionItem", metadata={"status": "open"})
    _save(api, "resolved-item", type="ActionItem", metadata={"status": "resolved"})
    _save(api, "open-insight", metadata={"status": "open"})

    listed = _list(api, include_inbox="true", where=["status=open", "type=ActionItem"])

    assert listed == [open_item]


# ── search: where ───────────────────────────────────────────────────────────

def test_unfiltered_search_returns_all_three_statuses(api, three_statuses):
    assert set(three_statuses.values()) <= set(_search(api))


def test_search_where_status_returns_only_the_matching_record(api, three_statuses):
    """Red: ``where`` is ignored, so all three come back."""
    assert _search(api, where=["status=open"]) == [three_statuses["open"]]


def test_search_where_missing_key_matches_nothing(api, three_statuses):
    """Red: ``where`` is ignored."""
    assert _search(api, where=["reviewer=dana"]) == []


def test_search_where_entries_are_anded(api, tagged):
    """Red: ``where`` is ignored."""
    found = _search(api, where=["status=open", "entities=project/my-app"])

    assert sorted(found) == sorted([tagged["open_app"], tagged["open_both"]])


def test_search_where_list_value_matches_by_element(api, tagged):
    """Red: ``where`` is ignored."""
    found = _search(api, where=["entities=person/alice"])

    assert sorted(found) == sorted([tagged["open_person"], tagged["open_both"]])


def test_search_where_composes_with_types(api):
    """Red: ``where`` is ignored."""
    item = _save(api, "open-item", type="ActionItem", metadata={"status": "open"})
    _save(api, "resolved-item", type="ActionItem", metadata={"status": "resolved"})
    _save(api, "open-insight", metadata={"status": "open"})

    assert _search(api, where=["status=open"], types=["ActionItem"]) == [item]


def test_search_where_still_fills_the_limit_when_the_first_page_does_not_match(api):
    """Red: ``where`` is ignored. Pins the over-fetch: the four strongest hits do
    not match, so a filter applied to a page of exactly ``limit`` would return
    nothing, not the two matching records.

    The first assertion (no ``where``) shows the premise and passes on v0.22.0.
    """
    loud = f"{_KEYWORD} " * 30
    for index in range(4):
        _save(api, f"loud-{index}", content=loud, metadata={"status": "resolved"})
    quiet = [
        _save(api, f"quiet-{index}", content=f"{_KEYWORD} once", metadata={"status": "open"})
        for index in range(2)
    ]

    top_two = _search(api, limit=2)
    assert not set(top_two) & set(quiet), "premise: the loud records outrank the quiet ones"

    found = _search(api, limit=2, where=["status=open"])

    assert sorted(found) == sorted(quiet)


def test_search_where_applies_to_a_recency_listing(api, three_statuses):
    """Red: ``where`` is ignored. An empty query lists recent records instead of
    searching, and takes the same filter."""
    found = _search(api, query="", where=["status=open"])

    assert found == [three_statuses["open"]]


# ── booleans and values that contain "=" ────────────────────────────────────

@pytest.fixture()
def flagged(api) -> dict[str, str]:
    """Two records with a boolean ``reviewed`` key and ``core``: true and false."""
    return {
        "yes": _save(api, "flag-yes", core=True, metadata={"reviewed": True}),
        "no": _save(api, "flag-no", core=False, metadata={"reviewed": False}),
    }


@pytest.mark.parametrize("key", ["core", "reviewed"])
def test_list_where_boolean_matches_its_lower_case_yaml_spelling(api, flagged, key):
    """Red: ``where`` is ignored, so both records come back."""
    assert _list(api, where=[f"{key}=true"]) == [flagged["yes"]]
    assert _list(api, where=[f"{key}=false"]) == [flagged["no"]]


@pytest.mark.parametrize("key", ["core", "reviewed"])
def test_list_where_boolean_does_not_match_python_spelling(api, flagged, key):
    """Red: ``where`` is ignored, so both records come back."""
    assert _list(api, where=[f"{key}=True"]) == []
    assert _list(api, where=[f"{key}=False"]) == []


@pytest.mark.parametrize("key", ["core", "reviewed"])
def test_search_where_boolean_matches_its_lower_case_yaml_spelling(api, flagged, key):
    """Red: ``where`` is ignored, so both records come back."""
    assert _search(api, where=[f"{key}=true"]) == [flagged["yes"]]
    assert _search(api, where=[f"{key}=false"]) == [flagged["no"]]


@pytest.mark.parametrize("key", ["core", "reviewed"])
def test_search_where_boolean_does_not_match_python_spelling(api, flagged, key):
    """Red: ``where`` is ignored, so both records come back."""
    assert _search(api, where=[f"{key}=True"]) == []
    assert _search(api, where=[f"{key}=False"]) == []


@pytest.fixture()
def notes(api) -> dict[str, str]:
    return {
        "equals": _save(api, "note-equals", metadata={"note": "a=b"}),
        "prefix": _save(api, "note-prefix", metadata={"note": "a"}),
        "suffix": _save(api, "note-suffix", metadata={"note": "b"}),
    }


def test_list_where_splits_on_the_first_equals_only(api, notes):
    """Red: ``where`` is ignored, so all three come back."""
    assert _list(api, where=["note=a=b"]) == [notes["equals"]]


def test_search_where_splits_on_the_first_equals_only(api, notes):
    """Red: ``where`` is ignored, so all three come back."""
    assert _search(api, where=["note=a=b"]) == [notes["equals"]]


# ── malformed where is a 422 ────────────────────────────────────────────────

_MALFORMED = [
    "no-equals", "=value", "bad key=x", "1leading-digit=x", "dotted.key=x", "", "status=",
]


@pytest.mark.parametrize("entry", _MALFORMED)
def test_malformed_where_is_rejected_by_list(api, three_statuses, entry):
    """Red: the entry is ignored and the listing answers 200."""
    response = api.get("/list", params={"where": [entry]})

    assert response.status_code == 422, response.text


@pytest.mark.parametrize("entry", _MALFORMED)
def test_malformed_where_is_rejected_by_search(api, three_statuses, entry):
    """Red: the field is ignored and the search answers 200."""
    response = api.post("/search", json={"query": _KEYWORD, "where": [entry]})

    assert response.status_code == 422, response.text


def test_one_malformed_entry_rejects_the_whole_request(api, three_statuses):
    """Red: ``where`` is ignored."""
    response = api.get("/list", params={"where": ["status=open", "no-equals"]})

    assert response.status_code == 422, response.text


@pytest.mark.parametrize("key", ["status", "_private", "my-key", "Key_2"])
def test_well_formed_keys_are_accepted(api, three_statuses, key):
    """Passes on v0.22.0 (unknown parameters are ignored) and must keep passing."""
    assert api.get("/list", params={"where": [f"{key}=x"]}).status_code == 200
    assert api.post(
        "/search", json={"query": _KEYWORD, "where": [f"{key}=x"]}
    ).status_code == 200
