"""The keyword arm vouches only for a combination of terms rarer than chance.

The measured defect: short operations requests ("delete the fallback file",
"write the release notes for this") reached ``confident`` on the keyword arm
alone. Normalized BM25 is coverage of the query's IDF, so a chunk repeating one
incidental term, or a chunk holding a few everyday terms that a large store of
work notes uses everywhere, clears the confident mark with nothing about the
query in it. ``store.keyword_chance`` gives each row the number of documents
chance alone would give its combination of query terms, and the keyword arm
may claim ``confident`` only on a row below one.

Real SQLite + FTS5 under ``tmp_path`` throughout; no DB mocking.
"""
from __future__ import annotations

import pytest

from palinode.core import confidence, store
from palinode.core.config import config
from palinode.core.scoring import describe_diagnostics
from tests._store_helpers import upsert_chunks

_DIM = 1024
_TS = "2026-10-06T00:00:00+00:00"


def _chunk(chunk_id: str, content: str) -> dict:
    embedding = [0.0] * _DIM
    embedding[1] = 1.0
    return {
        "id": chunk_id,
        "file_path": f"projects/{chunk_id}.md",
        "section_id": "root",
        "category": "projects",
        "content": content,
        "metadata": {},
        "created_at": _TS,
        "last_updated": _TS,
        "embedding": embedding,
    }


#: Forty work-log chunks. "commit" and "push" are each in twenty of them, as
#: everyday words are in a store of session notes; the watcher record is the
#: only one about the watcher.
_NOTES = [
    _chunk(
        f"log-{i}",
        ("Commit the replay fixtures. " if i % 2 == 0 else "Reviewed the planner. ")
        + ("Push the branch after review. " if i < 20 else "Updated the dashboard. ")
        + f"Session {i} notes on unrelated chores and the weekly sync.",
    )
    for i in range(38)
]
_WATCHER = _chunk(
    "watcher-debounce",
    "The watcher debounce coalesces two filesystem events so a save reindexes once.",
)
#: One chunk that repeats an incidental term; the other two query terms live
#: elsewhere in the store.
_DELETE_LOG = _chunk(
    "delete-run",
    "Delete path one, delete path two, the delete in compaction, the delete in "
    "purge: every delete leaves an index row.",
)


@pytest.fixture()
def notes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    monkeypatch.setattr(config.decay, "enabled", False)
    store.init_db()
    # Copies: the loop below extends them, and the module-level rows are shared.
    rows = [dict(row) for row in (*_NOTES, _WATCHER, _DELETE_LOG)]
    # "fallback" and "file" are everyday words here too, each in half the
    # notes, so a note holding both is what chance hands any query.
    for i, row in enumerate(rows[:38]):
        if i % 2:
            row["content"] += " Fallback stays on."
        if i >= 19:
            row["content"] += " Attached the log file."
    upsert_chunks(rows, skip_unchanged=False)
    return len(rows)


def _by_id(hits: list[dict]) -> dict[str, dict]:
    return {hit["id"]: hit for hit in hits}


class TestTheFigure:
    def test_a_pair_of_everyday_terms_is_what_chance_gives(self, notes):
        hits = _by_id(store.search_fts("commit and push", top_k=50))
        both = [h for h in hits.values() if h["id"] in {"log-0", "log-2"}]
        assert both, "the fixture must hold a chunk with both terms"
        # N x (20/N) x (19 or 20)/N — about ten chunks would hold the pair by chance.
        for hit in both:
            assert hit["keyword_chance"] == pytest.approx(
                notes * (19 / notes) * (20 / notes), rel=0.15
            )
            assert hit["keyword_chance"] >= confidence.KEYWORD_CHANCE_BELOW

    def test_a_specific_combination_is_rarer_than_chance(self, notes):
        hits = _by_id(store.search_fts("watcher debounce reindexes", top_k=10))
        assert hits["watcher-debounce"]["keyword_chance"] < confidence.KEYWORD_CHANCE_BELOW

    def test_one_matched_term_is_never_below_one(self, notes):
        hits = _by_id(store.search_fts("delete the fallback file", top_k=10))
        row = hits["delete-run"]
        # Its own document count: it is the only chunk saying "delete".
        assert row["keyword_chance"] == pytest.approx(1.0)
        # And yet its score clears the confident mark: the repetition the
        # figure exists to discount.
        assert row["score"] >= confidence.KEYWORD_CONFIDENT

    def test_a_one_term_query_has_no_co_occurrence_to_measure(self, notes):
        hits = store.search_fts("debounce", top_k=10)
        assert hits and all(hit["keyword_chance"] is None for hit in hits)

    def test_the_figure_rides_through_the_ranker(self, notes):
        rows = store.search_hybrid("delete the fallback file", None, top_k=5,
                                   record_access=False)
        delete = [r for r in rows if r["id"] == "delete-run"]
        assert delete and delete[0]["keyword_chance"] == pytest.approx(1.0)


class TestTheVerdict:
    def test_a_chance_level_keyword_match_is_weak(self):
        rows = [{"raw_score": None, "keyword_score": 1.3, "keyword_chance": 3.0}]
        for mode in ("lexical", "hybrid"):
            value = confidence.assess(rows, active_mode=mode)
            assert value["confidence"] == "weak"
            assert value["co_occurrence"] == "chance"
            # The arm still reports what it saw.
            assert value["arms"]["keyword"]["verdict"] == "confident"

    def test_one_row_rarer_than_chance_is_enough(self):
        rows = [
            {"raw_score": None, "keyword_score": 1.3, "keyword_chance": 3.0},
            {"raw_score": None, "keyword_score": 0.7, "keyword_chance": 0.02},
        ]
        value = confidence.assess(rows, active_mode="lexical")
        assert value["confidence"] == "confident"
        assert "co_occurrence" not in value

    def test_the_rare_row_must_itself_reach_the_mark(self):
        rows = [
            {"raw_score": None, "keyword_score": 1.3, "keyword_chance": 3.0},
            {"raw_score": None, "keyword_score": 0.5, "keyword_chance": 0.02},
        ]
        assert confidence.assess(rows, active_mode="lexical")["confidence"] == "weak"

    def test_a_row_without_the_figure_is_judged_on_its_score(self):
        rows = [{"raw_score": None, "keyword_score": 0.9, "keyword_chance": None}]
        assert confidence.assess(rows, active_mode="lexical")["confidence"] == "confident"

    def test_a_chance_level_row_still_corroborates_the_vector_arm(self):
        # Corroboration is unchanged: the figure governs the keyword arm's own
        # claim, not its recognition of a query the vector arm vouches for.
        rows = [{"raw_score": 0.7, "keyword_score": 0.4, "keyword_chance": 12.0}]
        value = confidence.assess(rows, active_mode="hybrid")
        assert value["confidence"] == "confident"
        assert "corroboration" not in value and "co_occurrence" not in value

    def test_the_diagnostics_line_says_why(self):
        rows = [{"raw_score": 0.55, "keyword_score": 1.3, "keyword_chance": 3.0}]
        value = confidence.assess(rows, active_mode="hybrid")
        line = describe_diagnostics(
            {"active_mode": "hybrid", "index_state": "ready", "outcome": "matched", **value}
        )
        assert "match confidence: weak" in line
        assert "chance" in line


def test_end_to_end_the_incidental_command_is_not_confident(notes):
    """Through the real store and ranker: the repeated term clears the score
    mark and is still not confidence."""
    rows = store.search_hybrid("delete the fallback file", None, top_k=5,
                               record_access=False)
    value = confidence.assess(rows, active_mode="lexical")
    assert value["arms"]["keyword"]["best"] >= confidence.KEYWORD_CONFIDENT
    assert value["confidence"] == "weak"
    assert value["co_occurrence"] == "chance"


def test_end_to_end_a_specific_request_stays_confident(notes):
    rows = store.search_hybrid("fix the watcher debounce", None, top_k=5,
                               record_access=False)
    value = confidence.assess(rows, active_mode="lexical")
    assert rows[0]["id"] == "watcher-debounce"
    assert value["confidence"] == "confident"
