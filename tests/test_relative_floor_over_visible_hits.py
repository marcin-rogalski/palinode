"""Each arm's relative floor is measured against what the caller may see.

The vector arm keeps a candidate within ``vector_relative_floor`` of the best
cosine in its candidate set, and the keyword arm does the same with its own
fraction of the best BM25 score. The ADR-009 visibility gate and project
isolation run on what the ranker delivers. A record the caller may not see must
therefore not be the best a floor is measured against, or it pushes out a
visible match and is then withheld itself, leaving nothing.

Real SQLite and real memory files under ``tmp_path``; the query vector and each
record's vector sit at an exact cosine, so every cutoff is arithmetic.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from palinode.api import server
from palinode.api.routers import search as search_router
from palinode.core import embedder, store
from palinode.core.config import config
from palinode.core.retrieval_log import RetrievalLogger
from tests._store_helpers import upsert_chunks
from tests.test_vector_arm_relative_floor import _at_cosine, _query_vec

QUERY = "orionledger retention window"
#: Vocabulary disjoint from the query: in hybrid mode the vector arm alone
#: retrieves these, so the vector floor alone decides.
_UNRELATED = "The atrium lighting refit is scheduled around the quarterly close."


@pytest.fixture()
def memory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    monkeypatch.setattr(config.audit, "log_path", str(tmp_path / ".audit" / "mcp-calls.jsonl"))
    monkeypatch.delenv("PALINODE_PROJECT", raising=False)
    monkeypatch.setattr(embedder, "embed", lambda text, *a, **kw: _query_vec())
    monkeypatch.setattr(search_router.embedder, "embed", lambda text, *a, **kw: _query_vec())
    monkeypatch.setattr(search_router, "_retrieval_logger", RetrievalLogger(str(tmp_path)))
    store.init_db()
    return tmp_path


@pytest.fixture()
def client(memory):
    server._rate_counters.clear()
    with TestClient(server.app, raise_server_exceptions=False) as c:
        yield c


def _record(memory, name: str, cosine: float, axis: int, *,
            content: str = _UNRELATED, **frontmatter) -> None:
    """A memory file and its indexed chunk, at an exact cosine to the query."""
    path = memory / "insights" / f"{name}.md"
    path.parent.mkdir(exist_ok=True)
    lines = ["---", "category: insights"]
    lines += [f"{key}: {value}" for key, value in frontmatter.items()]
    path.write_text("\n".join([*lines, "---", f"# {name}", "", content, ""]), encoding="utf-8")
    upsert_chunks([{
        "id": name,
        "file_path": str(path),
        "section_id": "root",
        "category": "insights",
        "content": content,
        "metadata": {"category": "insights", **frontmatter},
        "created_at": "2026-10-01T00:00:00+00:00",
        "last_updated": "2026-10-01T00:00:00+00:00",
        "embedding": _at_cosine(cosine, axis),
    }], skip_unchanged=False)


def _search(client, **body) -> dict | list:
    resp = client.post("/search", json={"query": QUERY, "threshold": 0.4, "limit": 5, **body})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _ids(results: list[dict]) -> list[str]:
    return [r["id"] for r in results]


def _recall_count(chunk_id: str) -> int:
    db = store.get_db()
    try:
        return db.execute(
            "SELECT recall_count FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()["recall_count"]
    finally:
        db.close()


_WITHHELD = {
    "private": {"visibility": "private"},
    "restricted": {"visibility": "restricted", "access": "[agent/someone-else]"},
}


@pytest.mark.parametrize("hybrid", [True, False], ids=["hybrid", "vector"])
@pytest.mark.parametrize("withheld", sorted(_WITHHELD))
def test_a_withheld_better_match_does_not_hide_a_visible_one(memory, client, withheld, hybrid):
    # Measured against 0.80 the floor is 0.68 and drops the visible 0.60.
    _record(memory, "withheld", 0.80, 1, **_WITHHELD[withheld])
    _record(memory, "visible", 0.60, 2)
    assert _ids(_search(client, context=[], hybrid=hybrid)) == ["visible"]


@pytest.mark.parametrize("hybrid", [True, False], ids=["hybrid", "vector"])
def test_another_projects_better_match_does_not_hide_this_projects(memory, client, hybrid):
    _record(memory, "beta", 0.80, 1, entities="[project/beta]")
    _record(memory, "alpha", 0.60, 2, entities="[project/alpha]")
    payload = _search(client, context=["project/alpha"], hybrid=hybrid, receipt=True)
    assert _ids(payload["results"]) == ["alpha"]
    assert payload["receipt"]["retrieval"]["other_projects_withheld"] == 1


def test_the_widened_window_delivers_the_visible_match_and_counts_it_once(memory, client):
    # limit=1: the withheld record fills the first window, so the gate re-fetches wider.
    _record(memory, "withheld", 0.80, 1, visibility="private")
    _record(memory, "visible", 0.60, 2)
    assert _ids(_search(client, context=[], hybrid=False, limit=1)) == ["visible"]
    assert _recall_count("visible") == 1


def test_the_keyword_arm_floor_ignores_a_withheld_better_match(memory, client, monkeypatch):
    monkeypatch.setattr(config.search, "retrieval_mode", "lexical")
    _record(memory, "withheld", 0.80, 1, visibility="private",
            content="orionledger retention window. " * 6)
    _record(memory, "visible", 0.60, 2,
            content=f"orionledger is mentioned once. {_UNRELATED} " * 4)
    unfloored = store.search_hybrid(QUERY, None, top_k=5, record_access=False,
                                    fts_threshold=0.0)
    scores = {h["id"]: h["keyword_score"] for h in unfloored}
    assert scores["visible"] < config.search.lexical_fts_threshold * scores["withheld"]
    assert _ids(_search(client, context=[])) == ["visible"]


def test_a_visible_better_match_still_sets_the_floor(memory, client):
    _record(memory, "best", 0.80, 1)
    _record(memory, "weak", 0.60, 2)
    assert _ids(_search(client, context=[], hybrid=False)) == ["best"]


def test_an_opted_in_other_project_still_sets_the_floor(memory, client):
    _record(memory, "beta", 0.80, 1, entities="[project/beta]")
    _record(memory, "alpha", 0.60, 2, entities="[project/alpha]")
    payload = _search(client, context=["project/alpha"], hybrid=False,
                      include_other_projects=True)
    assert _ids(payload) == ["beta"]
