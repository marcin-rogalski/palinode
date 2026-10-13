"""Search applies ``where`` before it ranks, so a matching record is never missed for ranking low.

A record that matches ``where`` must be returned however low it ranks: one
``status: open`` record behind a dozen ``status: resolved`` ones that rank higher,
with ``limit: 2``, is still found, because ranking is not what picks the
candidates.

Contract these tests pin:

1. one matcher: the records ``GET /list`` returns for a ``where`` entry are the
   records search returns, for a timestamp, a boolean, a list element and a number;
2. filter first, rank after: the lone match is returned at ``limit`` 1 and 2 in
   hybrid, vector-only and lexical retrieval, and a handful of matches behind a
   dozen others come back newest-or-best first up to ``limit``;
3. the recency listing (an empty query) filters first too;
4. no cap: a match set larger than SQLite's bound-variable limit is returned;
5. an empty match set answers at once: 200, no results, no query embedded and
   no keyword query run;
6. every other filter still narrows the matching set.

The ranking premise (the resolved records outrank the open one) is asserted in its
own test, so a green result cannot be an accident of ranking.

Real SQLite + tmp_path through ``TestClient``. The embedder is a fake that puts
the ``resolvedmark`` records beside the query and the ``openmark`` record away
from it, so the vector arm ranks like the keyword arm does; nothing reaches the
network.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from palinode.api import server
from palinode.core import embedder, store
from palinode.core.config import config
from palinode.indexer import reconcile
from palinode.indexer.index_file import index_file

_KEYWORD = "harborlight"
_DIMENSIONS = 1024


def _fake_embed(text: str) -> list[float]:
    """The query and ``nearmark`` are one point, ``resolvedmark`` is beside it,
    ``slightmark`` is a little further, and ``openmark`` is away."""
    vector = [0.0] * _DIMENSIONS
    lowered = text.lower()
    if "slightmark" in lowered:
        vector[0], vector[1] = 0.95, 0.31224989991991997
    elif "nearmark" in lowered or "openmark" not in lowered and "resolvedmark" not in lowered:
        vector[0] = 1.0
    elif "resolvedmark" in lowered:
        vector[0], vector[1] = 0.99, 0.14106735979665885
    else:
        vector[0], vector[1] = 0.6, 0.8
    return vector


_MODES = ["hybrid", "vector", "lexical"]


def _started(mode, tmp_path, monkeypatch):
    """A client, and the request fields that select the retrieval mode."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    if mode == "lexical":
        monkeypatch.setattr(config.search, "retrieval_mode", "lexical")
        no_model = Mock(side_effect=AssertionError("a model was contacted"))
        monkeypatch.setattr(embedder, "embed", no_model)
        monkeypatch.setattr(embedder, "embed_many", no_model)
        monkeypatch.setattr(reconcile, "get_ollama_client", no_model)
        fields: dict = {}
    else:
        monkeypatch.setattr(config.search, "retrieval_mode", "hybrid")
        monkeypatch.setattr(embedder, "embed", MagicMock(side_effect=_fake_embed))
        fields = {"hybrid": False} if mode == "vector" else {}
    store.init_db()
    server._rate_counters.clear()
    with TestClient(server.app, raise_server_exceptions=False) as client:
        client.mode = mode
        client.fields = fields
        yield client
    server._rate_counters.clear()


@pytest.fixture(params=_MODES)
def engine(request, tmp_path, monkeypatch):
    yield from _started(request.param, tmp_path, monkeypatch)


@pytest.fixture()
def lexical_engine(tmp_path, monkeypatch):
    yield from _started("lexical", tmp_path, monkeypatch)


@pytest.fixture(params=["hybrid", "vector"])
def semantic_engine(request, tmp_path, monkeypatch):
    yield from _started(request.param, tmp_path, monkeypatch)


def _save(client, slug: str, content: str, **kwargs) -> str:
    body = {"content": content, "type": "Insight", "slug": slug, **kwargs}
    server._rate_counters.clear()
    response = client.post("/save", json=body)
    assert response.status_code == 200, response.text
    return response.json()["rel_path"]


def _resolved_text(n: int) -> str:
    return f"{_KEYWORD} {_KEYWORD} {_KEYWORD} resolvedmark note {n}"


def _open_text(n: int) -> str:
    return f"openmark note {n} mentions {_KEYWORD} once among many other plain words here"


def _slight_text(n: int) -> str:
    """Ranks below ``resolvedmark`` records but within the vector relative floor of
    a ``nearmark`` one."""
    return f"slightmark note {n} mentions {_KEYWORD} once among many other plain words here"


def _flat_text(n: int) -> str:
    """Ranks like an open record: the same vector and the same keyword weight."""
    return _open_text(n)


def _search(client, **body) -> list[str]:
    body.setdefault("query", _KEYWORD)
    body.setdefault("threshold", 0.0)
    body = {**client.fields, **body}
    response = client.post("/search", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    rows = payload["results"] if isinstance(payload, dict) else payload
    return [row["rel_path"] for row in rows]


def _list(client, entry: str) -> list[str]:
    response = client.get("/list", params={"where": entry})
    assert response.status_code == 200, response.text
    return [row["file"] for row in response.json()]


@pytest.fixture()
def one_open_behind_twelve(engine) -> dict:
    """The open record is saved first (so it is the oldest) and ranks last."""
    paths = {"open": _save(engine, "open-1", _open_text(1), metadata={"status": "open"})}
    paths["resolved"] = [
        _save(engine, f"resolved-{n:02d}", _resolved_text(n), metadata={"status": "resolved"})
        for n in range(1, 13)
    ]
    return paths


# ── the ranking premise ─────────────────────────────────────────────────────

@pytest.mark.parametrize("limit", [1, 2])
def test_unfiltered_search_ranks_the_resolved_records_above_the_open_one(
    engine, one_open_behind_twelve, limit
):
    """Without this premise, the tests below could pass by luck of ranking."""
    found = _search(engine, limit=limit)

    assert found
    assert one_open_behind_twelve["open"] not in found
    assert set(found) <= set(one_open_behind_twelve["resolved"])


# ── 2. filter first, rank after ─────────────────────────────────────────────

@pytest.mark.parametrize("limit", [1, 2])
def test_a_lone_match_behind_higher_ranked_records_is_found(
    engine, one_open_behind_twelve, limit
):
    """A search that ranked its best ``5 x limit`` candidates and then dropped the
    ones ``where`` rejects would lose the open record, thirteenth by rank, and
    answer ``[]``."""
    found = _search(engine, limit=limit, where=["status=open"])

    assert found == [one_open_behind_twelve["open"]]


def test_several_matches_behind_higher_ranked_records_fill_the_limit(engine):
    """With ``limit`` 2 a window of ten ranked candidates holds only resolved
    records. The three open records are saved first and rank last."""
    opens = [
        _save(engine, f"open-{n}", _open_text(n), metadata={"status": "open"})
        for n in range(1, 4)
    ]
    for n in range(1, 13):
        _save(engine, f"resolved-{n:02d}", _resolved_text(n), metadata={"status": "resolved"})

    two = _search(engine, limit=2, where=["status=open"])
    all_three = _search(engine, limit=5, where=["status=open"])

    assert len(two) == 2
    assert set(two) <= set(opens)
    assert sorted(all_three) == sorted(opens)


def test_where_still_narrows_when_every_record_matches(engine):
    """A restriction that dropped the ranking would reorder the answer; the record
    nearest the query, with most keyword hits, comes first."""
    for n in range(1, 4):
        _save(engine, f"resolved-{n:02d}", _resolved_text(n), metadata={"status": "resolved"})
    nearest = _save(
        engine, "resolved-best", f"{_KEYWORD} {_KEYWORD} {_KEYWORD} {_KEYWORD} nearmark",
        metadata={"status": "resolved"},
    )

    found = _search(engine, limit=1, where=["status=resolved"])

    assert found == [nearest]


# ── 3. the recency listing ──────────────────────────────────────────────────

@pytest.fixture()
def recent_open_behind_twelve(engine) -> dict:
    opens = [
        _save(engine, f"open-{n}", _open_text(n), metadata={"status": "open"})
        for n in range(1, 4)
    ]
    for n in range(1, 13):
        _save(engine, f"resolved-{n:02d}", _resolved_text(n), metadata={"status": "resolved"})
    return {"opens": opens}


def test_the_recency_premise_the_newest_records_are_resolved(
    engine, recent_open_behind_twelve
):
    """The twelve resolved records are newer than the open ones."""
    found = _search(engine, query="", limit=1)

    assert found
    assert found[0] not in recent_open_behind_twelve["opens"]


def test_an_empty_query_with_where_returns_the_newest_matching_records(
    engine, recent_open_behind_twelve
):
    """Fetching the newest ``limit`` records and filtering them afterwards would
    find no open record, because none of the newest is open."""
    opens = recent_open_behind_twelve["opens"]

    one = _search(engine, query="", limit=1, where=["status=open"])
    two = _search(engine, query="", limit=2, where=["status=open"])

    assert one == [opens[2]]
    assert two == [opens[2], opens[1]]


# ── 4. no cap on the match set ──────────────────────────────────────────────

_VARIABLE_LIMIT = 10


def _lower_the_bound_variable_limit(monkeypatch) -> None:
    """Every connection the store opens from now on allows only a few variables."""
    real_get_db = store.get_db

    def get_db_with_a_low_limit(*args, **kwargs):
        db = real_get_db(*args, **kwargs)
        db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _VARIABLE_LIMIT)
        return db

    monkeypatch.setattr(store, "get_db", get_db_with_a_low_limit)


def test_the_lowered_variable_limit_is_in_force(engine, monkeypatch):
    """Guards the next tests against a limit that silently stopped applying."""
    _lower_the_bound_variable_limit(monkeypatch)
    db = store.get_db()
    try:
        placeholders = ", ".join("?" for _ in range(_VARIABLE_LIMIT + 1))
        with pytest.raises(sqlite3.OperationalError, match="too many SQL variables"):
            db.execute(f"SELECT {placeholders}", [0] * (_VARIABLE_LIMIT + 1))  # nosec B608
    finally:
        db.close()


def test_a_match_set_larger_than_the_variable_limit_is_returned(engine, monkeypatch):
    """Every record ranks alike, so the ranking cannot hide a match. Kills an
    implementation that passes the ids as one placeholder each: SQLite refuses it
    at ``_VARIABLE_LIMIT`` variables."""
    matching = [
        _save(engine, f"open-{n:02d}", _open_text(n), metadata={"status": "open"})
        for n in range(1, 41)
    ]
    for n in range(1, 4):
        _save(engine, f"resolved-{n:02d}", _flat_text(n), metadata={"status": "resolved"})
    _lower_the_bound_variable_limit(monkeypatch)

    found = _search(engine, limit=40, where=["status=open"])

    assert sorted(found) == sorted(matching)


def test_the_recency_listing_survives_a_match_set_larger_than_the_variable_limit(
    engine, monkeypatch
):
    """The same, for the empty query."""
    matching = [
        _save(engine, f"open-{n:02d}", _open_text(n), metadata={"status": "open"})
        for n in range(1, 41)
    ]
    _save(engine, "resolved-01", _resolved_text(1), metadata={"status": "resolved"})
    _lower_the_bound_variable_limit(monkeypatch)

    found = _search(engine, query="", limit=40, where=["status=open"])

    assert sorted(found) == sorted(matching)


# ── 5. an empty match set answers at once ───────────────────────────────────

@pytest.fixture()
def spies(engine, one_open_behind_twelve, monkeypatch):
    """Counters for the query embedding and the keyword query, started after seeding."""
    embed_calls = Mock()
    real_embed = embedder.embed
    keyword_calls = Mock()
    real_search_fts = store.search_fts

    def counting_embed(text):
        embed_calls(text)
        return real_embed(text)

    def counting_search_fts(*args, **kwargs):
        keyword_calls(*args)
        return real_search_fts(*args, **kwargs)

    monkeypatch.setattr(embedder, "embed", counting_embed)
    monkeypatch.setattr(store, "search_fts", counting_search_fts)
    return {"embed": embed_calls, "keyword": keyword_calls}


def test_a_where_that_matches_nothing_answers_without_embedding_or_a_keyword_query(
    engine, spies
):
    """The query is neither embedded nor run as a keyword query once the empty
    result is known, in hybrid, vector and lexical retrieval alike."""
    found = _search(engine, limit=5, where=["status=nonexistent"])

    assert found == []
    spies["embed"].assert_not_called()
    spies["keyword"].assert_not_called()


def test_an_empty_query_where_that_matches_nothing_answers_with_no_results(
    engine, spies
):
    """The recency listing answers an empty match set with no results."""
    assert _search(engine, query="", limit=5, where=["status=nonexistent"]) == []


@pytest.fixture()
def retrievals(engine, one_open_behind_twelve, monkeypatch):
    """Counters for the vector query and the recency read, started after seeding."""
    vector_calls = Mock()
    recency_calls = Mock()
    real_search = store.search
    real_list_recent = store.list_recent

    def counting_search(*args, **kwargs):
        vector_calls(*args)
        return real_search(*args, **kwargs)

    def counting_list_recent(*args, **kwargs):
        recency_calls(*args)
        return real_list_recent(*args, **kwargs)

    monkeypatch.setattr(store, "search", counting_search)
    monkeypatch.setattr(store, "list_recent", counting_list_recent)
    return {"vector": vector_calls, "recency": recency_calls}


def test_a_where_that_matches_nothing_runs_no_vector_query(engine, retrievals):
    """In hybrid and vector retrieval ``store.search`` must not run over the whole
    index, or over an empty id list, before the empty result is known; lexical
    retrieval never runs it. The answer comes at once."""
    found = _search(engine, limit=5, where=["status=nonexistent"])

    assert found == []
    retrievals["vector"].assert_not_called()


def test_an_empty_query_where_that_matches_nothing_reads_no_recency_list(
    engine, retrievals
):
    """``store.list_recent`` must not run and have its rows filtered afterwards."""
    found = _search(engine, query="", limit=5, where=["status=nonexistent"])

    assert found == []
    retrievals["recency"].assert_not_called()
    retrievals["vector"].assert_not_called()


def test_a_where_that_matches_something_still_reads_what_it_needs(engine, retrievals):
    """Guards the two spies above against being tests of a counter that never
    counts. The results are not asserted; the tests above pin them."""
    _search(engine, limit=5, where=["status=open"])
    _search(engine, query="", limit=5, where=["status=open"])

    assert retrievals["recency"].called
    if engine.mode != "lexical":
        assert retrievals["vector"].called


# ── 6. every other filter still applies to the matching set ─────────────────

_OLD = "2025-08-01T00:00:00+00:00"


@pytest.fixture()
def streams(engine) -> dict:
    """Every record matches ``stream=alpha``; each one differs in a single way."""
    def save(slug: str, text: str = None, **kwargs) -> str:
        metadata = {"stream": "alpha", **kwargs.pop("metadata", {})}
        return _save(engine, slug, text or _resolved_text(0), metadata=metadata, **kwargs)

    paths = {
        "insight": save("a-insight", priority=3),
        "decision": save("b-decision", type="Decision", priority=3),
        "old": save("c-old", priority=3, metadata={"last_updated": _OLD}),
        "low": save("d-low", priority=1),
        "archived": save("e-archived", metadata={"status": "archived"}),
        "telemetry": save("f-telemetry", metadata={"kind": "telemetry"}),
        "private": save(
            "g-private", metadata={"visibility": "private", "scope": "agent/owner"}
        ),
        "other_project": save("h-other", entities=["project/beta"]),
        "outside": _save(engine, "i-outside", _resolved_text(0), priority=3),
    }
    return paths


def _alpha(engine, **body) -> set[str]:
    return set(_search(engine, where=["stream=alpha"], limit=20, **body))


def test_the_status_kind_and_visibility_exclusions_still_apply(engine, streams):
    """A restriction to the matching ids must not bypass the
    exclusions the search applies to every candidate."""
    found = _alpha(engine)

    assert streams["archived"] not in found
    assert streams["telemetry"] not in found
    assert streams["private"] not in found
    assert streams["outside"] not in found
    assert {streams["insight"], streams["decision"], streams["old"], streams["low"]} <= found


def test_a_record_of_another_project_is_left_out_of_a_scoped_search(engine, streams):
    scoped = _alpha(engine, context=["project/alpha"])
    unscoped = _alpha(engine)

    assert streams["other_project"] not in scoped
    assert streams["other_project"] in unscoped


def test_types_narrows_the_matching_set(engine, streams):
    found = _alpha(engine, types=["Decision"])

    assert found == {streams["decision"]}


def test_type_deny_narrows_the_matching_set(engine, streams):
    found = _alpha(engine, type_deny=["Decision"])

    assert streams["decision"] not in found
    assert streams["insight"] in found


def test_category_narrows_the_matching_set(engine, streams):
    found = _alpha(engine, category="decisions")

    assert found == {streams["decision"]}


def test_date_after_leaves_out_an_older_matching_record(engine, streams):
    found = _alpha(engine, date_after="2026-01-01T00:00:00+00:00")

    assert streams["old"] not in found
    assert streams["insight"] in found


def test_date_before_leaves_out_a_newer_matching_record(engine, streams):
    found = _alpha(engine, date_before="2025-12-31T00:00:00+00:00")

    assert found == {streams["old"]}


def test_since_days_leaves_out_an_older_matching_record(engine, streams):
    found = _alpha(engine, since_days=30)

    assert streams["old"] not in found
    assert streams["insight"] in found


def test_min_priority_leaves_out_a_lower_priority_matching_record(engine, streams):
    found = _alpha(engine, min_priority=3)

    assert streams["low"] not in found
    assert streams["insight"] in found


# ── 1. one matcher ──────────────────────────────────────────────────────────

def _write(memory_dir: Path, slug: str, front_matter: str) -> str:
    path = memory_dir / "insights" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: insights-{slug}\ncategory: insights\ntype: Insight\n"
        f"{front_matter}---\n\n{_KEYWORD} resolvedmark note {slug}\n",
        encoding="utf-8",
    )
    index_file(str(path))
    return f"insights/{slug}.md"


@pytest.fixture()
def typed_values(engine) -> dict:
    memory_dir = Path(config.memory_dir)
    return {
        "stamped": _write(
            memory_dir, "stamped",
            "due: 2026-03-01T10:00:00Z\nreviewed: true\nrank: 2\ntags:\n- alpha\n- beta\n",
        ),
        "plain": _write(
            memory_dir, "plain",
            "due: 2026-03-02T10:00:00Z\nreviewed: false\nrank: 3\ntags:\n- gamma\n",
        ),
        "bare": _write(memory_dir, "bare", ""),
    }


@pytest.mark.parametrize(
    "entry",
    [
        "due=2026-03-01T10:00:00+00:00",
        "due=2026-03-01 10:00:00+00:00",
        "due=2026-03-01T11:00:00+01:00",
        "reviewed=true",
        "reviewed=false",
        "rank=2",
        "tags=beta",
        "tags=gamma",
        "tags=delta",
    ],
)
def test_search_returns_the_records_list_returns_for_the_same_entry(
    engine, typed_values, entry
):
    """Kills a matcher restated in SQL: ``json_extract`` reads
    a boolean as 1, a timestamp by its text, and cannot see inside a list."""
    listed = _list(engine, entry)

    assert sorted(_search(engine, limit=10, where=[entry])) == sorted(listed)
    assert listed or entry == "tags=delta"


# ── edges on the implementation ─────────────────────────────────────────────

_PRIVATE = {"visibility": "private", "scope": "agent/owner"}


def _fts_step_of_the_keyword_query(monkeypatch, chunk_ids) -> str:
    """The plan step SQLite runs on the full-text table for ``search_fts(chunk_ids=...)``."""
    statements: list[str] = []
    real_get_db = store.get_db

    def tracing_get_db(*args, **kwargs):
        db = real_get_db(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(store, "get_db", tracing_get_db)
    store.search_fts(_KEYWORD, chunk_ids=chunk_ids)
    monkeypatch.setattr(store, "get_db", real_get_db)

    [query] = [s for s in statements if "MATCH" in s and "JOIN chunks" in s]
    db = real_get_db()
    try:
        plan = [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + query)]
    finally:
        db.close()
    [step] = [d for d in plan if re.search(r"\bfts VIRTUAL TABLE INDEX", d)]
    return step


def test_the_keyword_arm_restricts_on_the_chunk_row_not_on_the_fts_rowid(
    lexical_engine, monkeypatch
):
    """A restriction on ``fts.rowid`` is pushed into the virtual table as an equality
    constraint (``=M``), and FTS5 then reruns the MATCH once per id."""
    for n in range(1, 4):
        _save(lexical_engine, f"open-{n}", _open_text(n), metadata={"status": "open"})
    db = store.get_db()
    try:
        ids = [row["id"] for row in db.execute("SELECT id FROM chunks")]
    finally:
        db.close()

    step = _fts_step_of_the_keyword_query(monkeypatch, ids[:2])

    assert "=M" not in step, step


@pytest.fixture()
def private_matches_ahead(engine) -> dict:
    """Two private open records rank first, six resolved ones next, the one public
    open record last. Saved oldest first: the public open record, the resolved
    ones, then the private ones."""
    paths = {"open": _save(engine, "open-public", _slight_text(1), metadata={"status": "open"})}
    paths["resolved"] = [
        _save(engine, f"resolved-{n:02d}", _resolved_text(n), metadata={"status": "resolved"})
        for n in range(1, 7)
    ]
    for n in range(1, 3):
        _save(
            engine, f"open-private-{n}", f"nearmark {_KEYWORD} {_KEYWORD} {_KEYWORD} {n}",
            metadata={"status": "open", **_PRIVATE},
        )
    return paths


def test_the_visibility_premise_resolved_records_would_crowd_a_widened_window(
    engine, private_matches_ahead
):
    """Without this premise the next tests could pass by luck of ranking."""
    for query in (_KEYWORD, ""):
        found = _search(engine, query=query, limit=1)

        assert found
        assert found[0] in private_matches_ahead["resolved"]


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_withheld_private_matches_widen_the_window_within_the_matches(
    engine, private_matches_ahead, query
):
    """The first fetch holds only a private match, so the gate widens it fivefold.
    The wider fetch must still be restricted to the matches: unrestricted, it
    holds only resolved records, and one of them would be the answer."""
    found = _search(engine, query=query, limit=1, where=["status=open"])

    assert found == [private_matches_ahead["open"]]


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_an_empty_match_set_with_the_embedder_down_answers_200(semantic_engine, monkeypatch, query):
    """An empty match set needs no embedding, so
    an unavailable embedder is no failure. The same request with a match is 503."""
    _save(semantic_engine, "open-1", _open_text(1), metadata={"status": "open"})

    def embedder_down(text):
        raise embedder.EmbeddingUnavailable(
            backend="local", model="bge-m3", text_len=len(text), cause="connection refused"
        )

    monkeypatch.setattr(embedder, "embed", embedder_down)
    body = {**semantic_engine.fields, "query": query, "threshold": 0.0, "receipt": True}

    nothing = semantic_engine.post("/search", json={**body, "where": ["status=nonexistent"]})
    something = semantic_engine.post("/search", json={**body, "where": ["status=open"]})

    assert nothing.status_code == 200, nothing.text
    assert nothing.json()["results"] == []
    assert nothing.json()["receipt"]["retrieval"]["outcome"] == "no_match"
    assert something.status_code == (503 if query else 200)


def test_an_empty_match_set_reports_the_mode_the_request_asked_for(semantic_engine):
    """The receipt of an empty answer names the retrieval mode as it does for a
    full one: ``vector`` when the request turned the keyword arm off."""
    _save(semantic_engine, "open-1", _open_text(1), metadata={"status": "open"})
    body = {**semantic_engine.fields, "query": _KEYWORD, "threshold": 0.0, "receipt": True}

    nothing = semantic_engine.post("/search", json={**body, "where": ["status=nonexistent"]})
    something = semantic_engine.post("/search", json={**body, "where": ["status=open"]})

    reported = nothing.json()["receipt"]["retrieval"]
    expected = something.json()["receipt"]["retrieval"]
    assert reported["active_mode"] == expected["active_mode"]
    assert reported["configured_mode"] == expected["configured_mode"]
    assert reported["active_mode"] == ("vector" if semantic_engine.mode == "vector" else "hybrid")


@pytest.mark.parametrize("stored", [None, ""], ids=["null", "empty-string"])
@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_a_chunk_with_no_metadata_never_matches(engine, stored, query):
    """It would rank first, so the only thing keeping it out is
    the restriction; kills a matcher that reads missing metadata as a match."""
    matching = _save(engine, "open-1", _open_text(1), metadata={"status": "open"})
    bare = _save(engine, "bare", f"nearmark {_KEYWORD} {_KEYWORD} {_KEYWORD}")
    db = store.get_db()
    try:
        db.execute("UPDATE chunks SET metadata = ? WHERE file_path LIKE ?", (stored, "%/bare.md"))
        db.commit()
    finally:
        db.close()

    found = _search(engine, query=query, limit=5, where=["status=open"])

    assert found == [matching]
    assert bare not in found
    assert store.list_chunk_ids_matching([("status", "open")]) != []
    assert all("bare" not in i for i in store.list_chunk_ids_matching([("status", "open")]))


@pytest.fixture()
def qualifying_open_behind_low_priority_ones(engine) -> str:
    """The one open record of priority 3 is the oldest and ranks last; three newer,
    better-ranked open records have priority 1."""
    qualifying = _save(
        engine, "open-qualifying", _slight_text(1), priority=3, metadata={"status": "open"}
    )
    for n in range(1, 4):
        _save(
            engine, f"open-low-{n}", f"nearmark {_KEYWORD} {_KEYWORD} {n}",
            priority=1, metadata={"status": "open"},
        )
    return qualifying


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_where_with_min_priority_still_overfetches_so_a_low_priority_match_does_not_crowd(
    engine, qualifying_open_behind_low_priority_ones, query
):
    """``min_priority`` filters after the fetch, so the fetch is fivefold wider. The
    lone ``limit`` records fetched are the low-priority ones and nothing qualifies."""
    found = _search(engine, query=query, limit=1, where=["status=open"], min_priority=3)

    assert found == [qualifying_open_behind_low_priority_ones]


@pytest.mark.parametrize(
    "extra", [{}, {"types": ["Insight"]}, {"min_priority": 1}], ids=["plain", "types", "priority"]
)
def test_a_large_limit_with_where_does_not_raise(engine, extra):
    """``limit`` 1000 reaches the vector query as ``k`` of
    6000 or more; the clamp at ``VEC_KNN_MAX_K`` keeps it legal, with a restriction too."""
    matching = _save(engine, "open-1", _open_text(1), metadata={"status": "open"})
    _save(engine, "resolved-1", _resolved_text(1), metadata={"status": "resolved"})

    found = _search(engine, limit=1000, where=["status=open"], **extra)

    assert found == [matching]


# ── 7. a match with no vector ───────────────────────────────────────────────

_NO_SHARED_WORD = "unrelatedwordsonly"


def _drop_vector(slug: str) -> None:
    """The record keeps its chunk and its keyword row but loses its embedding."""
    db = store.get_db()
    try:
        [chunk_id] = [
            row["id"] for row in db.execute(
                "SELECT id FROM chunks WHERE file_path LIKE ?", (f"%/{slug}.md",)
            )
        ]
        db.execute("DELETE FROM chunks_vec WHERE id = ?", (chunk_id,))
        db.commit()
    finally:
        db.close()


def _has_vector(slug: str) -> bool:
    db = store.get_db()
    try:
        return db.execute(
            "SELECT 1 FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE file_path LIKE ?)",
            (f"%/{slug}.md",),
        ).fetchone() is not None
    finally:
        db.close()


def test_one_match_without_a_vector_does_not_empty_the_vector_arm(semantic_engine):
    """sqlite-vec answers a nearest-neighbour query over a list of ids with no rows
    at all when one id has no vector. The query shares no word with any record, so
    only the vector arm can find the embedded match. Kills a restriction that
    passes every matching id to the vector query."""
    embedded = _save(semantic_engine, "open-1", _open_text(1), metadata={"status": "open"})
    _save(semantic_engine, "open-2", _open_text(2), metadata={"status": "open"})
    _drop_vector("open-2")

    found = _search(semantic_engine, query=_NO_SHARED_WORD, limit=5, where=["status=open"])

    assert found == [embedded]


def test_a_match_saved_with_its_embedding_deferred_does_not_empty_the_vector_arm(
    semantic_engine, monkeypatch
):
    """The state above, reached by a save: ``defer_embed_on_save`` writes the keyword
    row now and the vector later."""
    embedded = _save(semantic_engine, "open-1", _open_text(1), metadata={"status": "open"})
    monkeypatch.setattr(config.embeddings, "defer_embed_on_save", True)
    _save(semantic_engine, "open-2", _open_text(2), metadata={"status": "open"})
    assert _has_vector("open-1")
    assert not _has_vector("open-2")

    found = _search(semantic_engine, query=_NO_SHARED_WORD, limit=5, where=["status=open"])

    assert found == [embedded]


def test_a_match_without_a_vector_is_still_found_by_the_keyword_arm(engine):
    """Only the vector arm lacks the record, so a
    query that shares a word with it finds it wherever a keyword arm runs."""
    embedded = _save(engine, "open-1", _open_text(1), metadata={"status": "open"})
    unembedded = _save(engine, "open-2", _open_text(2), metadata={"status": "open"})
    _drop_vector("open-2")

    found = _search(engine, limit=5, where=["status=open"])

    if engine.mode == "vector":
        assert found == [embedded]
    else:
        assert sorted(found) == sorted([embedded, unembedded])


def test_matches_that_all_lack_a_vector_leave_vector_search_empty_without_failing(
    semantic_engine,
):
    """A guard that must keep passing: nothing to rank is an empty answer, not an error."""
    _save(semantic_engine, "open-1", _open_text(1), metadata={"status": "open"})
    _drop_vector("open-1")

    found = _search(semantic_engine, query=_NO_SHARED_WORD, limit=5, where=["status=open"])

    assert found == []


# ── 8. the keyword retry, the request key, and sections ─────────────────────

@pytest.fixture()
def hybrid_engine(tmp_path, monkeypatch):
    yield from _started("hybrid", tmp_path, monkeypatch)


def test_the_keyword_retry_after_a_corrupt_index_stays_within_the_matches(
    hybrid_engine, monkeypatch
):
    """The first keyword query fails, the index is rebuilt and the query runs again.
    The second run must carry the restriction: without it the keyword arm adds every
    record that shares the word, and the answer holds records ``where`` rejects."""
    opens = [
        _save(hybrid_engine, f"open-{n}", _open_text(n), metadata={"status": "open"})
        for n in range(1, 3)
    ]
    for n in range(1, 6):
        _save(hybrid_engine, f"resolved-{n:02d}", _resolved_text(n), metadata={"status": "resolved"})
    real_search_fts = store.search_fts
    calls: list[dict] = []

    def failing_once(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return real_search_fts(*args, **kwargs)

    monkeypatch.setattr(store, "search_fts", failing_once)

    found = _search(hybrid_engine, limit=20, where=["status=open"])

    assert len(calls) == 2
    assert sorted(found) == sorted(opens)


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_an_empty_where_list_is_no_filter_and_not_part_of_the_request_key(
    engine, monkeypatch, query
):
    """``where: []`` selects what leaving it out selects, so the request the receipt
    is built from must not name it: the receipt's identity is its request and its
    results. Kills the empty list reaching the receipt."""
    from palinode.core import receipt as receipt_module

    _save(engine, "open-1", _open_text(1), metadata={"status": "open"})
    _save(engine, "resolved-1", _resolved_text(1), metadata={"status": "resolved"})
    requests: list[dict] = []
    real_build_receipt = receipt_module.build_receipt

    def recording(results, *, request, **kwargs):
        requests.append(dict(request))
        return real_build_receipt(results, request=request, **kwargs)

    monkeypatch.setattr(receipt_module, "build_receipt", recording)

    without = _search(engine, query=query, limit=5)
    empty = _search(engine, query=query, limit=5, where=[])
    filtered = _search(engine, query=query, limit=5, where=["status=open"])

    assert sorted(empty) == sorted(without)
    assert len(filtered) == 1
    assert "where" not in requests[0]
    assert "where" not in requests[1]
    assert requests[2]["where"] == ["status=open"]


def _write_sections(memory_dir: Path, slug: str, status: str) -> str:
    """A file long enough to be split in three chunks, the last one holding a word
    nothing else holds."""
    filler = " ".join(f"plain{n}" for n in range(260))
    body = (
        f"{_KEYWORD} first section\n\n{filler}\n\n"
        f"## Second\n\n{_KEYWORD} second section\n\n{filler}\n\n"
        f"## Third\n\n{_KEYWORD} zebrafinch third section\n\n{filler}\n"
    )
    path = memory_dir / "insights" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: insights-{slug}\ncategory: insights\ntype: Insight\nstatus: {status}\n---\n\n{body}",
        encoding="utf-8",
    )
    index_file(str(path))
    return f"insights/{slug}.md"


def test_every_section_of_a_matching_file_matches(engine):
    """The premise: a file of three sections is three chunks, each carrying the
    file's front matter. A match on a later section is a match, so a restriction
    that keeps only a file's first chunk finds nothing for a word in its third."""
    path = _write_sections(Path(config.memory_dir), "sectioned", "open")
    _write_sections(Path(config.memory_dir), "sectioned-resolved", "resolved")
    db = store.get_db()
    try:
        chunks = db.execute(
            "SELECT id FROM chunks WHERE file_path LIKE ?", ("%/sectioned.md",)
        ).fetchall()
    finally:
        db.close()
    assert len(chunks) == 3
    matching = store.list_chunk_ids_matching([("status", "open")])
    assert {row["id"] for row in chunks} <= set(matching)

    found = _search(engine, query="zebrafinch", limit=5, where=["status=open"])

    assert found
    assert set(found) == {path}


def test_the_recency_listing_returns_a_file_with_every_section_matching(engine):
    """The file appears for each of its chunks, as it does without ``where``, and
    the file of the rejected record does not."""
    path = _write_sections(Path(config.memory_dir), "sectioned", "open")
    _write_sections(Path(config.memory_dir), "sectioned-resolved", "resolved")

    found = _search(engine, query="", limit=10, where=["status=open"])

    assert found
    assert set(found) == {path}


def test_search_matches_the_front_matter_the_index_holds_not_the_file(engine):
    """A hand edit that has not been indexed yet: search still answers from the
    index, and ``GET /list`` from the file. Kills a search that reads the files."""
    edited = _save(engine, "edited", _open_text(1), metadata={"status": "open"})
    path = Path(config.memory_dir) / edited
    path.write_text(path.read_text(encoding="utf-8").replace("status: open", "status: resolved"), encoding="utf-8")

    assert _list(engine, "status=resolved") == [edited]
    assert _list(engine, "status=open") == []
    assert _search(engine, limit=5, where=["status=open"]) == [edited]
    assert _search(engine, limit=5, where=["status=resolved"]) == []

    index_file(str(path))

    assert _search(engine, limit=5, where=["status=open"]) == []
    assert _search(engine, limit=5, where=["status=resolved"]) == [edited]


# ── 9. entries that combine, and what an empty answer reports ───────────────

@pytest.fixture()
def two_statuses(engine) -> dict:
    return {
        "open": _save(engine, "open-1", _open_text(1), metadata={"status": "open", "stream": "alpha"}),
        "resolved": _save(
            engine, "resolved-2", _flat_text(2), metadata={"status": "resolved", "stream": "alpha"}
        ),
    }


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_conflicting_entries_select_nothing_and_run_no_query(engine, two_statuses, spies, query):
    """Every entry must hold, and no record has two statuses: the answer is empty
    at once, as for an entry that matches nothing."""
    found = _search(engine, query=query, limit=5, where=["status=open", "status=resolved"])

    assert found == []
    spies["embed"].assert_not_called()
    spies["keyword"].assert_not_called()


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_a_repeated_entry_acts_as_one(engine, two_statuses, query):
    """Kills an entry counted twice, or a repeat read as a conflict."""
    once = _search(engine, query=query, limit=5, where=["status=open"])
    twice = _search(engine, query=query, limit=5, where=["status=open", "status=open"])

    assert once == [two_statuses["open"]]
    assert twice == once


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_entries_on_different_keys_must_all_hold(engine, two_statuses, query):
    """Kills the entries being joined by ``or``."""
    both = _search(engine, query=query, limit=5, where=["stream=alpha", "status=open"])
    either = _search(engine, query=query, limit=5, where=["stream=alpha"])

    assert both == [two_statuses["open"]]
    assert sorted(either) == sorted(two_statuses.values())


@pytest.mark.parametrize("query", [_KEYWORD, ""], ids=["ranked", "recency"])
def test_an_empty_answer_in_lexical_retrieval_reports_lexical(lexical_engine, query):
    """The receipt of an empty answer names the mode that ran for a full one, and
    says nothing matched."""
    _save(lexical_engine, "open-1", _open_text(1), metadata={"status": "open"})
    body = {"query": query, "threshold": 0.0, "receipt": True}

    nothing = lexical_engine.post("/search", json={**body, "where": ["status=nonexistent"]})
    something = lexical_engine.post("/search", json={**body, "where": ["status=open"]})

    assert nothing.status_code == 200, nothing.text
    assert nothing.json()["results"] == []
    reported = nothing.json()["receipt"]["retrieval"]
    assert reported["outcome"] == "no_match"
    assert reported["configured_mode"] == "lexical"
    assert reported["active_mode"] == something.json()["receipt"]["retrieval"]["active_mode"]
    assert reported["active_mode"] == ("lexical" if query else "recency")


def test_a_query_the_embedder_rejects_still_falls_back_to_the_keyword_arm_under_where(
    hybrid_engine, monkeypatch
):
    """The embedder rejects this one input: the answer comes from the keyword arm,
    restricted to the matches, and the receipt says ``keyword-fallback``."""
    opens = _save(hybrid_engine, "open-1", _open_text(1), metadata={"status": "open"})
    _save(hybrid_engine, "resolved-1", _resolved_text(1), metadata={"status": "resolved"})

    def rejected(text):
        raise embedder.EmbeddingInputError(model="bge-m3", text_len=len(text), ollama_message="NaN")

    monkeypatch.setattr(embedder, "embed", rejected)
    body = {"query": _KEYWORD, "threshold": 0.0, "receipt": True}

    response = hybrid_engine.post("/search", json={**body, "where": ["status=open"]})
    nothing = hybrid_engine.post("/search", json={**body, "where": ["status=nonexistent"]})

    assert response.status_code == 200, response.text
    assert [row["rel_path"] for row in response.json()["results"]] == [opens]
    assert response.json()["receipt"]["retrieval"]["active_mode"] == "keyword-fallback"
    assert nothing.status_code == 200, nothing.text
    assert nothing.json()["results"] == []


def test_an_empty_answer_is_logged_as_one_call_level_entry(engine, monkeypatch):
    """The retrieval log gets the call's query and no results, once, as for any
    search that finds nothing."""
    from palinode.api import _util

    _save(engine, "open-1", _open_text(1), metadata={"status": "open"})
    logged: list[tuple[list, dict]] = []
    monkeypatch.setattr(
        _util._retrieval_logger, "record_search_results",
        lambda results, **kwargs: logged.append((results, kwargs)),
    )

    found = _search(engine, limit=5, where=["status=nonexistent"])

    assert found == []
    assert len(logged) == 1
    assert logged[0][0] == []
    assert logged[0][1]["query"] == _KEYWORD
    assert logged[0][1]["receipt"] is not None
