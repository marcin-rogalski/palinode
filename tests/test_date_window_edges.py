"""Edges of the search date window: the calendar's ends and date-only spellings.

* A stamp whose UTC reading falls outside years 1 to 9999 is not a timestamp, so a
  record carrying one is kept by a date window, like any stamp that cannot be read.
  As a bound it is refused with a 422 instead of failing the request.
* A date-only upper bound covers its whole UTC day in every spelling the ISO parser
  accepts: ``2026-10-12``, ``20261012`` and the week dates.

Real SQLite + tmp_path through ``TestClient`` in lexical retrieval mode, so no
embedder is involved; the vector arm and the recency list are driven through the store.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from palinode.api import server
from palinode.core import embedder, ranker, store
from palinode.core.config import config
from palinode.indexer import reconcile
from palinode.indexer.index_file import index_file
from tests._store_helpers import upsert_chunks

_EARLY = "0001-01-01T00:00+01:00"
_LATE = "9999-12-31T23:59-01:00"
_EDGES = [_EARLY, _LATE]
_EDGE_IDS = ["early", "late"]

_STAMP = "2026-10-12T15:00:00Z"
# 12 October 2026 is a Monday, so ISO week 42 day 1 is that day.
_SPELLINGS_OF_THE_DAY = ["2026-10-12", "20261012", "2026-W42-1", "2026W421"]
_KEYWORD = "harborlight"


def _record(path: str, last_updated: str) -> dict:
    return {
        "id": path,
        "file_path": path,
        "section_id": None,
        "category": "insights",
        "content": f"{_KEYWORD} {path}",
        "metadata": {"last_updated": last_updated},
        "created_at": "2026-10-01T00:00:00+00:00",
        "last_updated": last_updated,
        "embedding": [1.0] + [0.0] * 1023,
    }


@pytest.fixture()
def store_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(store, "_db_checked", False)
    store.init_db()
    return tmp_path


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
        yield client, tmp_path
    server._rate_counters.clear()


def _index(memory_dir: Path, slug: str, last_updated: str) -> None:
    path = memory_dir / "insights" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: insights-{slug}\ncategory: insights\ntype: Insight\n"
        f'last_updated: "{last_updated}"\n---\n\n{_KEYWORD} note {slug}\n',
        encoding="utf-8",
    )
    index_file(str(path))


def _search(client, query: str = _KEYWORD, **window):
    return client.post("/search", json={"query": query, "threshold": 0.0, **window})


def _found(client, query: str = _KEYWORD, **window) -> set[str]:
    response = _search(client, query, **window)
    assert response.status_code == 200, response.text
    return {Path(hit["file_path"]).name for hit in response.json()}


# ── a stamp at the calendar's edge ──────────────────────────────────────────


@pytest.mark.parametrize("edge", _EDGES, ids=_EDGE_IDS)
def test_an_edge_value_is_not_a_timestamp(edge):
    """Kills parse_iso_utc catching ValueError only: the conversion to UTC raises
    OverflowError, which an unreadable value must not."""
    assert ranker.parse_iso_utc(edge) is None
    assert ranker.parse_iso_utc(edge, end_of_day=True) is None


@pytest.mark.parametrize("edge", _EDGES, ids=_EDGE_IDS)
def test_a_record_stamped_at_the_edge_is_kept_by_any_window(edge):
    """A stamp that cannot be read is kept, as the window already keeps an empty one."""
    assert ranker.in_date_window(edge, "2026-01-01", None) is True
    assert ranker.in_date_window(edge, None, "2026-01-01") is True
    assert ranker.in_date_window(edge, "2026-01-01", "2026-12-31") is True


def test_the_first_and_last_days_of_the_calendar_are_still_dates():
    """A guard that must keep passing: only values that overflow are unreadable."""
    assert ranker.parse_iso_utc("0001-01-01") == datetime(1, 1, 1, tzinfo=UTC)
    assert ranker.parse_iso_utc("9999-12-31", end_of_day=True) == datetime(
        9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC
    )


@pytest.mark.parametrize("edge", _EDGES, ids=_EDGE_IDS)
def test_vector_search_survives_a_record_stamped_at_the_edge(store_db, edge):
    """Kills the vector arm letting OverflowError out of the window."""
    upsert_chunks([_record("edge.md", edge), _record("day.md", _STAMP)])

    def ids(**window):
        hits = store.search(query_embedding=[1.0] + [0.0] * 1023, threshold=0.0, **window)
        return {h["file_path"] for h in hits}

    assert ids(date_after="2026-10-01") == {"edge.md", "day.md"}
    assert ids(date_before="2026-10-31") == {"edge.md", "day.md"}
    assert ids(date_after="2026-10-13") == {"edge.md"}


@pytest.mark.parametrize("edge", _EDGES, ids=_EDGE_IDS)
def test_the_recency_list_survives_a_record_stamped_at_the_edge(store_db, edge):
    """Kills the empty-query path letting OverflowError out of the window."""
    upsert_chunks([_record("edge.md", edge), _record("day.md", _STAMP)])

    def ids(**window):
        return {r["file_path"] for r in store.list_recent(**window)}

    assert ids(date_after="2026-10-01") == {"edge.md", "day.md"}
    assert ids(date_before="2026-10-13") == {"edge.md", "day.md"}
    assert ids(date_after="2026-10-13") == {"edge.md"}


@pytest.mark.parametrize("edge", _EDGES, ids=_EDGE_IDS)
def test_the_hybrid_ranker_survives_a_record_stamped_at_the_edge(edge):
    """Kills rank_hybrid letting OverflowError out of the window."""
    def result(path: str, last_updated: str) -> dict:
        return {
            "file_path": path, "section_id": "root", "score": 0.5, "raw_score": 0.5,
            "content": path, "metadata": {"last_updated": last_updated},
        }

    out = ranker.rank_hybrid(
        [result("edge.md", edge), result("day.md", _STAMP)], [],
        top_k=10, threshold=0.0, hybrid_weight=0.5, priority_weight=0.025,
        date_after="2026-10-01",
    )
    assert {r["file_path"] for r in out} == {"edge.md", "day.md"}


@pytest.mark.parametrize("edge", _EDGES, ids=_EDGE_IDS)
def test_a_search_with_a_date_bound_survives_a_record_stamped_at_the_edge(api, edge):
    """One such record must not make every date-filtered search fail, with a query
    or without one."""
    client, memory_dir = api
    _index(memory_dir, "edge", edge)
    _index(memory_dir, "day", _STAMP)
    assert _found(client, date_after="2026-10-01") == {"edge.md", "day.md"}
    assert _found(client, date_before="2026-10-31") == {"edge.md", "day.md"}
    assert _found(client, "", date_after="2026-10-01") == {"edge.md", "day.md"}
    assert _found(client, date_after="2026-10-13") == {"edge.md"}


# ── an edge value as a bound ────────────────────────────────────────────────


@pytest.mark.parametrize("field", ["date_after", "date_before"])
@pytest.mark.parametrize("edge", _EDGES, ids=_EDGE_IDS)
def test_an_edge_value_as_a_bound_is_a_422(api, field, edge):
    """Kills the bound check letting OverflowError out of the validator, which
    answers 500 where every other unreadable bound answers 422."""
    client, _ = api
    response = _search(client, **{field: edge})
    assert response.status_code == 422, response.text
    assert field in response.text


# ── date-only upper bounds ──────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", _SPELLINGS_OF_THE_DAY)
def test_a_date_only_upper_bound_covers_its_whole_day(spelling):
    """Kills end-of-day applying to the dashed spelling only."""
    assert ranker.parse_iso_utc(spelling) == datetime(2026, 10, 12, tzinfo=UTC)
    assert ranker.parse_iso_utc(spelling, end_of_day=True) == datetime(
        2026, 10, 12, 23, 59, 59, 999999, tzinfo=UTC
    )
    assert ranker.in_date_window(_STAMP, None, spelling) is True
    assert ranker.in_date_window("2026-10-12T23:59:59.999999Z", None, spelling) is True
    assert ranker.in_date_window("2026-10-13T00:00:00Z", None, spelling) is False


@pytest.mark.parametrize("spelling", ["2026-10-11", "20261011"])
def test_the_day_before_a_compact_upper_bound_is_excluded(spelling):
    assert ranker.in_date_window(_STAMP, None, spelling) is False


@pytest.mark.parametrize("value", ["2026-10-12T00:00:00Z", "2026-10-12 00:00:00", "2026-10-12T00:00"])
def test_a_timestamp_upper_bound_is_not_widened_to_its_day(value):
    """A guard that must keep passing: only a date-only value means a whole day."""
    assert ranker.parse_iso_utc(value, end_of_day=True) == datetime(2026, 10, 12, tzinfo=UTC)
    assert ranker.in_date_window(_STAMP, None, value) is False


@pytest.mark.parametrize("spelling", _SPELLINGS_OF_THE_DAY)
def test_a_date_only_upper_bound_keeps_the_record_saved_that_day(api, spelling):
    client, memory_dir = api
    _index(memory_dir, "day", _STAMP)
    assert _found(client, date_before=spelling) == {"day.md"}
    assert _found(client, date_before="2026-10-11") == set()
