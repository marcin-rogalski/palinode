"""Selection regressions with synthetic files and real SQLite/FTS/vector rows."""
from __future__ import annotations

import math
import json

import pytest
import yaml
from fastapi.testclient import TestClient
from click.testing import CliRunner

from palinode.api.server import app
from palinode.core import store
from palinode.core.bundle import BundleBudget, BundleRequest, build_bundle
from palinode.core.config import config
from tests._store_helpers import upsert_chunks, upsert_entities

QUERY = "nightly catchup bounded batches watermark"


@pytest.fixture()
def selection_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr("palinode.core.embedder.embed", lambda _: [1.0] + [0.0] * 1023)
    store.init_db()
    return tmp_path


def write_record(root, ref, body=QUERY, *, cosine=1.0, **meta):
    path = root / f"{ref}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "active", "type": "Decision", "date": "2026-09-01", **meta}
    path.write_text(f"---\n{yaml.safe_dump(metadata)}---\n\n{body}\n")
    upsert_chunks([{
        "id": ref, "file_path": str(path), "section_id": "root", "content": body,
        "metadata": metadata,
        "embedding": [cosine, math.sqrt(1 - cosine ** 2)] + [0.0] * 1022,
    }])
    upsert_entities(str(path), metadata)
    return path


def test_retired_seeds_cannot_starve_a_current_answer(selection_store):
    for i in range(3):
        write_record(selection_store, f"archive/2026/old-{i}")
    write_record(selection_store, "decisions/batch-bound", cosine=0.8,
                 body=QUERY + " advances in finite windows.")
    seeds = store.search_hybrid(QUERY, [1.0] + [0.0] * 1023, top_k=3, record_access=False)
    assert len(seeds) == 3 and all(s["currency"] == "retired" for s in seeds)

    bundle = build_bundle(BundleRequest(query=QUERY, prompt=True,
                                       budget=BundleBudget(max_items=3, max_chars=1500)))
    assert bundle.retrieval["confidence"] == "confident"
    assert "decisions/batch-bound" in {s.assertion.ref for s in bundle.selected}
    assert "archive/" not in bundle.to_dict()["text"]


def test_retired_only_selection_explains_why_without_relabelling_confidence(selection_store):
    write_record(selection_store, "archive/2026/old-bound")
    bundle = build_bundle(BundleRequest(query=QUERY, prompt=True))
    assert not bundle.selected
    assert bundle.retrieval["confidence"] == "confident"
    assert bundle.retrieval["selection"]["reasons"] == ["retired_withheld"]
    assert bundle.retrieval == bundle.receipt["retrieval"]
    assert "no current selection" in bundle.to_dict()["text"]


def test_widening_does_not_deliver_orphaned_index_rows_or_revisions(selection_store):
    for i in range(3):
        write_record(selection_store, f"archive/2026/old-{i}")
    orphan = write_record(selection_store, "insights/orphan", cosine=0.9)
    orphan.unlink()
    write_record(selection_store, "decisions/batch-bound", cosine=0.8,
                 body=QUERY + " advances in finite windows.")
    data = build_bundle(BundleRequest(query=QUERY, prompt=True,
                                     budget=BundleBudget(max_items=3, max_chars=2000))).to_dict()
    assert "decisions/batch-bound" in {s["ref"] for s in data["selected"]}
    assert "insights/orphan" not in data["source_revisions"]
    assert "insights/orphan" not in data["text"]
    assert "target_missing" in data["coverage"]["reasons"]


@pytest.mark.parametrize("flags", [{"automatic": True}, {"mode": "passive"}])
def test_automatic_fallback_does_not_deliver_an_orphaned_archive_row(selection_store, flags):
    orphan = write_record(selection_store, "archive/2026/orphan")
    orphan.unlink()
    write_record(selection_store, "decisions/batch-bound", cosine=0.8,
                 body=QUERY + " advances in finite windows.")
    with TestClient(app) as client:
        explicit = client.post("/search", json={"query": QUERY, "receipt": True}).json()
        passive = client.post("/search", json={"query": QUERY, "limit": 1, "receipt": True,
                                                **flags}).json()
    assert any(r["rel_path"] == "archive/2026/orphan.md" for r in explicit["results"])
    assert [r["rel_path"] for r in passive["results"]] == ["decisions/batch-bound.md"]


def test_budget_drop_is_explained_and_preserves_qualifications(selection_store):
    write_record(selection_store, "decisions/batch-bound", body=QUERY + " details " * 90,
                 epistemic="hypothesis")
    with TestClient(app) as client:
        body = client.post("/resolve", json={"query": QUERY, "prompt": True,
                                             "max_items": 3, "max_chars": 600}).json()
    assert not body["selected"]
    assert body["retrieval"]["confidence"] == "confident"
    assert "budget_exhausted:chars" in body["retrieval"]["selection"]["reasons"]
    assert "no current selection" in body["text"]
    assert len(body["text"]) <= 600
    # A record answers this; it did not fit. Saying nothing answers is false.
    assert "Nothing in memory answers this." not in body["text"]
    assert "Matches did not fit the budget." in body["text"]
    roomy = build_bundle(BundleRequest(query=QUERY, budget=BundleBudget(max_chars=3000)))
    assert roomy.selected
    assert "epistemic:hypothesis" in roomy.selected[0].assertion.qualifiers
    assert "epistemic:hypothesis" in roomy.to_dict()["text"]


def test_dense_assertion_is_demoted_without_losing_structured_evidence(selection_store):
    write_record(selection_store, "decisions/batch-bound", body=QUERY + " details " * 90,
                 title="Batch scheduling background " * 12, entities=["project/orbitkit"])
    for i in range(4):
        write_record(selection_store, f"insights/background-{i}",
                     body="An independent scheduling observation with contextual details. " * 8,
                     cosine=0.1, type="Insight", entities=["project/orbitkit"])
    small = build_bundle(BundleRequest(query=QUERY, prompt=True,
                                      budget=BundleBudget(max_items=3, max_chars=1500)))
    assert len(small.selected) == 1
    assert "demoted" in small.coverage["reasons"]
    assert small.selected[0].to_dict()["delivery"]["form"] == "minimal"
    assert len(small.selected[0].to_dict()["discovered"]) == 4
    assert small.to_dict()["text"].count("also found (unlinked)") == 0
    assert "+4 more unlinked" in small.to_dict()["text"]
    assert len(small.to_dict()["text"]) <= 1500
    roomy = build_bundle(BundleRequest(query=QUERY, prompt=True,
                                      budget=BundleBudget(max_items=3, max_chars=8000)))
    assert len(roomy.selected) == 1
    assert len(roomy.selected[0].discovered) == 4
    assert roomy.to_dict()["text"].count("also found (unlinked)") == 3
    assert "+1 more unlinked" in roomy.to_dict()["text"]


def test_hook_budget_delivers_assertion_with_bounded_unlinked_evidence(selection_store):
    write_record(selection_store, "decisions/batch-bound", entities=["project/orbitkit"])
    for i in range(12):
        write_record(selection_store, f"insights/background-{i}", cosine=0.1,
                     body="Independent scheduling observation with contextual details. " * 8,
                     type="Insight", entities=["project/orbitkit"])
    bundle = build_bundle(BundleRequest(query=QUERY, prompt=True,
                                       budget=BundleBudget(max_items=3, max_chars=2700)))
    assert bundle.retrieval["confidence"] == "confident"
    assert len(bundle.selected) == 1
    assert len(bundle.selected[0].discovered) == 12
    text = bundle.to_dict()["text"]
    assert "decisions/batch-bound" in text
    assert text.count("also found (unlinked)") == 3
    assert "+9 more unlinked" in text
    assert len(text) <= 2700
    delivered = {r["ref"] for r in bundle.receipt["supplied"]}
    shown = {d.ref for d in bundle.selected[0].discovered[:3]}
    assert delivered == shown | {"decisions/batch-bound"}


@pytest.mark.parametrize("max_chars,max_tokens", [(900, 0), (8000, 225)])
def test_minimal_form_preserves_qualifications_and_lineage(selection_store, max_chars, max_tokens):
    write_record(selection_store, "decisions/old-bound", superseded_by="decisions/batch-bound")
    write_record(selection_store, "decisions/batch-bound", body=QUERY + " details " * 90,
                 title="Batch scheduling background " * 12, epistemic="hypothesis")
    bundle = build_bundle(BundleRequest(ref="decisions/old-bound", budget=BundleBudget(
        max_items=1, max_chars=max_chars, max_tokens=max_tokens,
    )))
    assert len(bundle.selected) == 1
    assert bundle.selected[0].trimmed_statement is not None
    text = bundle.to_dict()["text"]
    assert "epistemic:hypothesis" in text
    assert "replaces: decisions/old-bound" in text
    assert "[trimmed]" in text and "demoted" in bundle.coverage["reasons"]
    assert len(text) <= max_chars
    if max_tokens:
        assert bundle.budget["tokens"] <= max_tokens


def test_unlinked_rendering_prefers_query_matches_and_keeps_contested_evidence(selection_store):
    write_record(selection_store, "decisions/batch-bound", entities=["project/orbitkit"])
    write_record(selection_store, "insights/relevant", body="watermark bounded batches",
                 cosine=0.1, entities=["project/orbitkit"])
    write_record(selection_store, "insights/contested", body="An independent observation.",
                 cosine=0.1, contradicts=["insights/other"], entities=["project/orbitkit"])
    for i in range(8):
        write_record(selection_store, f"insights/background-{i}", cosine=0.1,
                     body="Unrelated observation.", entities=["project/orbitkit"])
    bundle = build_bundle(BundleRequest(ref="decisions/batch-bound", query="watermark",
                                       budget=BundleBudget(max_items=1, max_chars=2700)))
    text = bundle.to_dict()["text"]
    assert "also found (unlinked): [insights/relevant]" in text
    assert "also found (unlinked): [insights/contested] [contested]" in text
    assert text.count("also found (unlinked)") == 4


@pytest.mark.asyncio
async def test_empty_selection_diagnostics_match_api_mcp_and_cli(selection_store, monkeypatch):
    import palinode.mcp as mcp
    from palinode.cli.resolve import api_client, resolve

    write_record(selection_store, "archive/2026/old-bound")
    monkeypatch.setenv("PALINODE_PROJECT", "orbitkit")
    with TestClient(app) as client:
        expected = client.post("/resolve", json={"query": QUERY, "prompt": True}).json()
        monkeypatch.setattr(api_client, "client", client)
        result = CliRunner().invoke(resolve, [QUERY, "--prompt", "--format", "json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["retrieval"] == expected["retrieval"]

        async def post(path, json=None, timeout=30.0):
            return client.post(path, json=json)

        monkeypatch.setattr(mcp, "_post", post)
        output = await mcp._tool_resolve({"query": QUERY, "prompt": True})
    assert "no current selection (retired records withheld)" in output[0].text
    assert "match confidence: confident" in output[0].text.lower()


@pytest.mark.parametrize("retirement", [
    {}, {"status": "archived"}, {"lifecycle": "superseded"},
    {"status": "retracted"}, {"superseded_by": "decisions/not-indexed"},
    {"expires_at": "2020-01-01T00:00:00Z"},
])
@pytest.mark.parametrize("flags", [{"automatic": True}, {"mode": "passive"}])
@pytest.mark.parametrize("query", [QUERY, ""])
def test_automatic_search_withholds_live_retirement_even_with_stale_index(
    selection_store, retirement, flags, query,
):
    ref = "decisions/old-bound" if retirement else "archive/2026/old-bound"
    path = write_record(selection_store, ref)
    # Change only frontmatter after indexing: cached status remains active.
    if retirement:
        metadata = {"type": "Decision", "status": "active", **retirement}
        path.write_text(f"---\n{yaml.safe_dump(metadata)}---\n\n{QUERY}\n")
    write_record(selection_store, "decisions/current-bound", cosine=0.8,
                 body=QUERY + " advances in finite windows.")
    with TestClient(app) as client:
        explicit = client.post("/search", json={"query": query, "receipt": True}).json()
        automatic = client.post("/search", json={"query": query, "receipt": True,
                                                  "limit": 1, "include_retired": True,
                                                  **flags}).json()
    assert any(r["rel_path"] == f"{ref}.md" for r in explicit["results"])
    assert [r["rel_path"] for r in automatic["results"]] == ["decisions/current-bound.md"]
    assert ref not in str(automatic)


@pytest.mark.parametrize("retrieval_mode,hybrid", [("lexical", True), ("hybrid", False)])
def test_retired_only_fallback_has_no_confident_delivery(selection_store, monkeypatch,
                                                       retrieval_mode, hybrid):
    write_record(selection_store, "archive/2026/old-bound")
    monkeypatch.setattr(config.search, "retrieval_mode", retrieval_mode)
    with TestClient(app) as client:
        body = client.post("/search", json={"query": QUERY, "prompt": True, "receipt": True,
                                             "automatic": True, "hybrid": hybrid}).json()
    assert not body["results"]
    assert body["receipt"]["retrieval"]["confidence"] == "none"


def test_confident_budget_refusal_is_not_delivered_as_isolation_news(selection_store, tmp_path):
    """A scoped record answers but cannot fit, and another project's
    record is withheld. The bundle must not claim nothing answers, and the
    hook must not deliver it as the isolation-only notice; it stays silent.
    A retired-only scoped answer still gets the notice: there nothing does."""
    from test_user_prompt_submit_hook import _context_of, _run_hook

    write_record(selection_store, "decisions/batch-bound", body=QUERY + " details " * 90,
                 entities=["project/alpha"])
    write_record(selection_store, "decisions/beta-bound", entities=["project/beta"])
    with TestClient(app) as client:
        refused = client.post("/resolve", json={
            "query": QUERY, "prompt": True, "project": "alpha",
            "max_items": 3, "max_chars": 700,
        }).json()
    assert refused["retrieval"]["confidence"] == "confident"
    assert not refused["selected"] and refused["other_projects_withheld"] >= 1
    assert "budget_exhausted:chars" in refused["retrieval"]["selection"]["reasons"]
    assert "Nothing in memory answers this." not in refused["text"]

    work = tmp_path / "hook-refused"
    work.mkdir()
    proc, log = _run_hook(work, resolve_response=refused,
                          env={"XDG_CACHE_HOME": str(tmp_path / "cache")})
    assert proc.returncode == 0 and proc.stdout == "", proc.stdout
    assert "/search" not in log.read_text()

    (selection_store / "decisions/batch-bound.md").unlink()
    write_record(selection_store, "archive/2026/batch-bound", entities=["project/alpha"])
    with TestClient(app) as client:
        retired = client.post("/resolve", json={
            "query": QUERY, "prompt": True, "project": "alpha",
            "max_items": 3, "max_chars": 2700,
        }).json()
    assert not retired["selected"] and retired["other_projects_withheld"] >= 1
    assert "Nothing in memory answers this." in retired["text"]
    work = tmp_path / "hook-retired"
    work.mkdir()
    proc, _ = _run_hook(work, resolve_response=retired,
                        env={"XDG_CACHE_HOME": str(tmp_path / "cache")})
    assert "from other projects withheld" in _context_of(proc)
