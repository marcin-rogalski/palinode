"""An append's ``## See also`` footer lists the entities its front matter ends with.

An append now carries the prior ``entities`` into the front matter, and a
``project`` on the append joins them. The footer is built from the entities of
the save alone, after the prior footer has been stripped from the composed body:
the front matter says ``[project/a, project/z]`` and the footer says ``[[z]]``;
an append that sends no entities has front matter ``[project/a]`` and no footer.

Contract these tests pin:

* after an append, the footer lists the slug of every entity in the front matter
  that the body does not already link inline, in front matter order;
* the footer is one block, at the end of the document, however often it appends;
* an entity linked inline, in the prior body or the new block, gets no entry.

Tests that carry the word *red* in their docstring fail on the feature head for
the stated reason; the rest pass there and must keep passing.

Real SQLite + tmp_path; the embedder is mocked.
"""
from __future__ import annotations

import re
from unittest.mock import patch

import frontmatter
import pytest

from palinode.core import store
from palinode.core.config import config
from palinode.core.save import save_memory

_FAKE_VECTOR = [0.01] * 1024

_FOOTER_MARKER = "<!-- palinode-auto-footer -->"


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


def _save(**kwargs) -> str:
    kwargs.setdefault("type", "Insight")
    kwargs.setdefault("slug", "rollout")
    kwargs.setdefault("update_policy", "append")
    with (
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
        patch("palinode.core.embedder.embed", return_value=_FAKE_VECTOR),
    ):
        return save_memory(**kwargs)["file_path"]


def _footer_slugs(path: str) -> list[str]:
    body = frontmatter.load(path).content
    block = body.split(_FOOTER_MARKER)[1] if _FOOTER_MARKER in body else ""
    return re.findall(r"^- \[\[([^\]]+)\]\]$", block, flags=re.MULTILINE)


def _entity_slugs(path: str) -> list[str]:
    return [e.split("/")[-1] for e in frontmatter.load(path).metadata["entities"]]


# ── the footer follows the front matter ─────────────────────────────────────

def test_an_append_that_sends_no_entities_keeps_the_footer_of_the_carried_ones(
    memory_dir,
):
    """Red: the front matter carries ``project/a`` and the body has no footer."""
    _save(content="First block.", entities=["project/a"])

    path = _save(content="Second block.")

    assert _footer_slugs(path) == ["a"]


def test_a_project_on_an_append_adds_to_the_footer_of_the_carried_entities(memory_dir):
    """Red: the footer lists ``z`` alone while the front matter has ``a`` and ``z``."""
    _save(content="First block.", entities=["project/a"])

    path = _save(content="Second block.", project="z")

    assert _footer_slugs(path) == ["a", "z"]


@pytest.mark.parametrize(
    "extra",
    [{}, {"project": "z"}, {"project": "a"}, {"entities": ["person/b"]}],
    ids=["nothing", "new-project", "carried-project", "explicit-entities"],
)
def test_the_footer_lists_the_entities_the_front_matter_ends_with(memory_dir, extra):
    """Red for ``nothing`` and ``new-project``: the footer is built from the
    save's own entities, so it disagrees with the carried front matter. The other
    two pass, because the save's own entities already equal the front matter's."""
    _save(content="First block.", entities=["project/a"])

    path = _save(content="Second block.", **extra)

    assert _footer_slugs(path) == _entity_slugs(path)


# ── one footer, at the end ──────────────────────────────────────────────────

def test_repeated_appends_leave_one_footer_block_at_the_end(memory_dir):
    """Red: no append after the first writes a footer at all."""
    _save(content="First block.", entities=["project/a"])
    _save(content="Second block.")

    path = _save(content="Third block.")

    body = frontmatter.load(path).content
    assert body.count(_FOOTER_MARKER) == 1
    assert body.count("## See also") == 1
    assert body.index("Third block.") < body.index(_FOOTER_MARKER)
    assert _footer_slugs(path) == ["a"]


# ── an entity linked inline gets no footer entry ────────────────────────────

def test_an_entity_linked_inline_in_the_new_block_gets_no_footer_entry(memory_dir):
    """Passes on the feature head. Guards a fix that lists every front matter
    entity without checking the composed body for inline links."""
    _save(content="First block.", entities=["project/a"])

    path = _save(content="Second block, about [[a]] and more.", project="z")

    assert _footer_slugs(path) == ["z"]


def test_an_entity_linked_inline_in_the_prior_body_gets_no_footer_entry(memory_dir):
    """Passes on the feature head. Guards a fix that scans only the new block
    for inline links."""
    _save(content="First block about [[a]].", entities=["project/a"])

    path = _save(content="Second block.", project="z")

    assert _footer_slugs(path) == ["z"]
