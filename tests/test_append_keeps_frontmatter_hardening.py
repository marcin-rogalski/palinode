"""Hardening for the append carry: the behaviours the first tests left open.

``test_append_keeps_frontmatter.py`` pins the carry itself. Each test here guards
one rule of ``_carry_prior_frontmatter`` or of its place in ``save_memory``; the
docstring names the mutant it kills. All of them pass on the feature head.

Real SQLite + tmp_path; the embedder is mocked. A prior file the library would
never write (a string ``entities``, a hand-written ``ttl``, an invalid
``epistemic``) is written by hand, because that is the only way such a file
exists.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import frontmatter
import pytest

from palinode.core import store
from palinode.core.config import config
from palinode.core.save import save_memory

_FAKE_VECTOR = [0.01] * 1024

_FIRST = "Item one: confirm the rollout window."
_SECOND = "Item two: the rollout window is confirmed."
_LONG = "Rollout window notes. " * 20


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


def _save(**kwargs) -> dict:
    kwargs.setdefault("type", "Insight")
    kwargs.setdefault("slug", "rollout")
    kwargs.setdefault("update_policy", "append")
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=_FAKE_VECTOR),
    ):
        return save_memory(**kwargs)


def _meta(result: dict) -> dict:
    return dict(frontmatter.load(result["file_path"]).metadata)


def _write_prior(memory_dir: Path, front_matter: str) -> None:
    path = memory_dir / "insights" / "rollout.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nid: insights-rollout\ncategory: insights\ntype: Insight\n"
        f"update_policy: append\n{front_matter}---\n\n{_FIRST}\n",
        encoding="utf-8",
    )


# ── ku_compat ───────────────────────────────────────────────────────────────

@pytest.fixture()
def ku_compat(monkeypatch):
    monkeypatch.setattr(config.ku_compat, "enabled", True)
    monkeypatch.setattr(config.ku_compat, "ku_version", "1.0")


def test_append_restamps_ku_version_instead_of_carrying_it(memory_dir, ku_compat):
    """Kills ``ku_version`` leaving the ku_compat never-carried set: the prior
    ``0.9-old`` would survive the append."""
    _save(content=_FIRST, metadata={"ku_version": "0.9-old"})

    result = _save(content=_SECOND)

    assert _meta(result)["ku_version"] == "1.0"


def test_append_that_sets_a_status_rederives_lifecycle(memory_dir, ku_compat):
    """Kills the ``"status" in frontmatter_dict`` guard being dropped, so that
    the prior ``deprecated`` lifecycle is carried over the new ``active`` status."""
    _save(content=_FIRST, metadata={"status": "deprecated"})

    result = _save(content=_SECOND, metadata={"status": "active"})

    meta = _meta(result)
    assert meta["status"] == "active"
    assert meta["lifecycle"] == "active"


def test_append_carries_a_prior_lifecycle_when_it_sets_no_status(memory_dir, ku_compat):
    """Kills ``lifecycle`` being added to the never-carried set unconditionally:
    the lifecycle would be re-derived from the carried ``open`` status as
    ``active`` and lose the prior ``archived``."""
    _save(content=_FIRST, metadata={"status": "open", "lifecycle": "archived"})

    result = _save(content=_SECOND)

    assert _meta(result)["lifecycle"] == "archived"


def test_lifecycle_follows_the_carried_status_when_the_prior_had_none(
    memory_dir, monkeypatch
):
    """The prior was written with ku_compat off, so it has a status and no
    lifecycle. Kills the carry running after the ku_compat block: the block
    would see no status and derive ``active``."""
    _save(content=_FIRST, metadata={"status": "deprecated"})
    monkeypatch.setattr(config.ku_compat, "enabled", True)
    result = _save(content=_SECOND)

    meta = _meta(result)
    assert meta["status"] == "deprecated"
    assert meta["lifecycle"] == "deprecated"
    assert meta["ku_version"] == "1.0"


# ── the project joins the carried entities ──────────────────────────────────

def test_a_project_joins_the_carried_entities(memory_dir):
    """Kills the join becoming a replace: ``["project/z"]`` alone."""
    _save(content=_FIRST, entities=["project/a"])

    result = _save(content=_SECOND, project="z")

    assert _meta(result)["entities"] == ["project/a", "project/z"]


def test_a_project_already_among_the_carried_entities_is_not_added_twice(memory_dir):
    """Kills the ``if e not in prior_entities`` filter being dropped."""
    _save(content=_FIRST, entities=["project/a", "person/alice"])

    result = _save(content=_SECOND, project="a")

    assert _meta(result)["entities"] == ["project/a", "person/alice"]


def test_explicit_entities_and_a_project_do_not_pick_up_the_prior_list(memory_dir):
    """Kills ``has_entities`` being ignored when the ``entities`` parameter is
    given: the prior ``project/a`` would come back."""
    _save(content=_FIRST, entities=["project/a"])

    result = _save(content=_SECOND, entities=["person/bob"], project="z")

    assert _meta(result)["entities"] == ["person/bob", "project/z"]


def test_metadata_entities_as_a_list_wins_outright(memory_dir):
    """Kills ``has_entities`` ignoring ``metadata.entities``: the prior list
    would be put in front of it."""
    _save(content=_FIRST, entities=["person/alice"])

    result = _save(content=_SECOND, metadata={"entities": ["person/bob"]})

    assert _meta(result)["entities"] == ["person/bob"]


def test_metadata_entities_as_a_string_is_written_as_on_any_save(memory_dir):
    """Kills the carry touching a caller-written ``entities`` value of any shape."""
    _save(content=_FIRST, entities=["person/alice"])

    result = _save(content=_SECOND, metadata={"entities": "person/bob"})

    assert _meta(result)["entities"] == "person/bob"


@pytest.mark.parametrize("prior", ["entities: person/alice\n", "entities:\n"],
                         ids=["string", "null"])
def test_a_prior_entities_that_is_not_a_list_is_not_carried(memory_dir, prior):
    """Kills the ``isinstance(prior_entities, list)`` guard being dropped (a
    string plus a list raises) and the generic loop carrying it over the new list."""
    _write_prior(memory_dir, prior)

    result = _save(content=_SECOND, project="z")

    assert _meta(result)["entities"] == ["project/z"]


# ── epistemic and ttl are never carried by the generic carry ────────────────

def test_an_invalid_prior_epistemic_is_not_carried(memory_dir):
    """The sticky rule drops a value that is not a valid marker. Kills
    ``epistemic`` leaving the never-carried set: the generic loop would copy it."""
    _write_prior(memory_dir, "epistemic: bogus\n")

    result = _save(content=_SECOND)

    assert "epistemic" not in _meta(result)


def test_a_valid_prior_epistemic_still_carries_by_its_own_rule(memory_dir):
    """Guards the same set from the other side: the sticky rule, not the generic
    carry, keeps a valid marker."""
    _save(content=_FIRST, metadata={"epistemic": "inference"})

    result = _save(content=_SECOND)

    assert _meta(result)["epistemic"] == "inference"


def test_a_hand_written_prior_ttl_is_not_carried(memory_dir):
    """Kills ``ttl`` leaving the never-carried set: a carried ``ttl`` would
    be resolved by the next reader into an expiry the append never asked for."""
    _write_prior(memory_dir, "ttl: 7d\nexpires_at: '2099-01-01T00:00:00+00:00'\n")

    result = _save(content=_SECOND)

    meta = _meta(result)
    assert "ttl" not in meta
    assert str(meta["expires_at"]).startswith("2099-01-01")


# ── description and summary ─────────────────────────────────────────────────

def test_a_carried_description_is_not_pending(memory_dir):
    """Kills the carry running after the ``description_pending`` check, or
    excluding ``description``: the append would queue a regeneration."""
    _save(content=_FIRST, metadata={"description": "Rollout notes"})

    result = _save(content=_SECOND)

    assert _meta(result)["description"] == "Rollout notes"
    assert "description_pending" not in result


def test_an_empty_description_asks_for_regeneration(memory_dir):
    """Kills the carry overwriting a key the caller sent, empty or not."""
    _save(content=_FIRST, metadata={"description": "Rollout notes"})

    result = _save(content=_SECOND, metadata={"description": ""})

    assert _meta(result)["description"] == ""
    assert result["description_pending"] is True


def test_a_carried_summary_is_not_pending(memory_dir):
    """Kills the carry running after the ``summary_pending`` check, or
    excluding ``summary``. Core and long enough that it would otherwise queue."""
    _save(content=_LONG, core=True, metadata={"summary": "Rollout window"})

    result = _save(content=_LONG + "More.")

    assert _meta(result)["summary"] == "Rollout window"
    assert "summary_pending" not in result


def test_an_empty_summary_asks_for_regeneration(memory_dir):
    """Kills the carry overwriting a key the caller sent."""
    _save(content=_LONG, core=True, metadata={"summary": "Rollout window"})

    result = _save(content=_LONG + "More.", metadata={"summary": ""})

    assert _meta(result)["summary"] == ""
    assert result["summary_pending"] is True
