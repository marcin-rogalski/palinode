"""Selective project-memory handoff: import side.

A bundle from :mod:`palinode.core.handoff` arrives from someone else, so every
part of it is untrusted. This module plans what an import would do, and
applies a plan the recipient reviewed. Nothing the bundle says can raise its
records' authority or touch the recipient's existing records:

- **Fields.** Only :data:`handoff.KEPT_FIELDS` are read. Anything else a bundle
  carries (``core``, ``authority``, ``scope``, ``visibility``, …) is stripped and
  named, whatever the sending side claimed to have dropped.
- **Content.** Each record's body passes the same security scan and tool-envelope
  check as a save. Text addressed to AI agents is recomputed here and flagged.
- **Paths.** A record lands at its ``origin_path`` only if that is a plain
  ``category/slug.md`` path inside the store. An occupied path is never
  overwritten: the record is renamed with a ``-handoff-<bundle8>`` suffix and
  links to it from the rest of the bundle are remapped.
- **Provenance.** Imported records carry ``source: handoff-import`` and an
  ``imported_from`` block (bundle id, origin path, content hash, export time).
  The same origin record with the same content is recognised on a repeat
  import, from any bundle, and skipped as a duplicate.
- **Scope.** The sender's project entity is rewritten to the recipient's
  project. Without an explicit destination, the import refuses a project the
  recipient's store doesn't know.

:func:`plan_import` writes nothing. :func:`apply_import` re-plans, refuses if
the plan's digest differs from the one the recipient reviewed, writes each new
file atomically and commits them all in one commit. Existing records are never
edited, so no reciprocal links are added.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import yaml

from palinode.core import git_tools, scope, store
from palinode.core.agent_directed import agent_directed_spans
from palinode.core.claims import ClaimError, normalize_claims
from palinode.core.config import config
from palinode.core.envelope import envelope_complaint
from palinode.core.handoff import (
    FORMAT,
    KEPT_FIELDS,
    LINK_FIELDS,
    PERSONAL_DIRS,
    HandoffError,
    _read,
    _root,
    _walk,
)
from palinode.core.revalidation import normalize_ref
from palinode.core.skip_dirs import is_skipped_path

logger = logging.getLogger(__name__)

#: The ``source`` every imported record carries.
IMPORT_SOURCE = "handoff-import"

#: Bundle limits. The API's 5 MB request cap bounds the whole bundle; these
#: bound what one bundle can ask the store to hold.
MAX_RECORDS = 2000
MAX_BODY_BYTES = 512 * 1024

IMPORT_NOTICE = (
    "Imported records are someone else's notes. They arrive with no authority: "
    "they can't be core, can't change how a record may be updated, and can't "
    "widen who sees anything. Each one is marked source: handoff-import. "
    "Nothing in your store is overwritten or edited."
)

_BUNDLE_ID_RE = re.compile(r"^hb-[0-9a-f]{16}$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


# ── bundle validation ───────────────────────────────────────────────────────
def _bundle_parts(bundle: Any) -> tuple[dict[str, Any], list[Any]]:
    """``(manifest, records)``, or a 400 when the bundle isn't one."""
    if not isinstance(bundle, dict):
        raise HandoffError("a handoff bundle is a JSON object with manifest and records")
    manifest, records = bundle.get("manifest"), bundle.get("records")
    if not isinstance(manifest, dict) or not isinstance(records, list):
        raise HandoffError("a handoff bundle is a JSON object with manifest and records")
    if manifest.get("format") != FORMAT:
        raise HandoffError(
            f"unsupported bundle format {manifest.get('format')!r}; this release reads {FORMAT}"
        )
    if not isinstance(manifest.get("bundle_id"), str) or not _BUNDLE_ID_RE.match(
        manifest["bundle_id"]
    ):
        raise HandoffError("manifest.bundle_id is missing or malformed")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("project"), str) \
            or not selection["project"].strip():
        raise HandoffError("manifest.selection.project is missing")
    if len(records) > MAX_RECORDS:
        raise HandoffError(f"a bundle holds at most {MAX_RECORDS} records; this one has "
                           f"{len(records)}", 413)
    origins = [r.get("origin_path") for r in records if isinstance(r, dict)]
    if len(origins) != len(set(origins)):
        raise HandoffError("the bundle names the same origin_path twice")
    return manifest, records


def _path_problem(rel: str) -> str | None:
    """Why ``rel`` can't be a destination path, or ``None``."""
    if not isinstance(rel, str) or not rel.endswith(".md"):
        return "origin_path is not a .md path"
    segments = rel.split("/")
    if len(segments) < 2:
        return "origin_path has no category directory"
    if any(not _SEGMENT_RE.match(s) or s in {".", ".."} for s in segments):
        return "origin_path is not a plain relative path"
    if is_skipped_path(rel):
        return "origin_path is not a memory directory"
    if rel.endswith("-history.md"):
        return "origin_path is a history sibling"
    return None


def _refs(value: Any) -> list[str] | None:
    """A link field as a list of refs, or ``None`` when it's malformed."""
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list):
        return None
    if not all(isinstance(r, str) and _REF_RE.match(r.strip()) and ".." not in r
               for r in items):
        return None
    return [r.strip() for r in items]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── the recipient's store ───────────────────────────────────────────────────
def _store_view(root: str) -> tuple[dict[tuple[str, str], str], set[str]]:
    """``(earlier imports by (origin_path, content_hash), projects in use)``."""
    imported: dict[tuple[str, str], str] = {}
    projects: set[str] = set()
    index = scope.alias_index()
    for rel in _walk(root):
        loaded = _read(root, rel)
        if loaded is None:
            continue
        meta = loaded[0]
        for entity in scope.project_entities(meta):
            projects.add(scope.canonical_project(entity, index))
        origin = meta.get("imported_from")
        if isinstance(origin, dict):
            key = (str(origin.get("origin_path")), str(origin.get("content_hash")))
            imported.setdefault(key, rel.replace(os.sep, "/"))
    return imported, projects


def _uncommitted(rels: list[str]) -> set[str]:
    """Which of ``rels`` git doesn't have committed yet (empty without a repo)."""
    if not rels or not config.git.auto_commit:
        return set()
    try:
        prefix = git_tools._run_git("rev-parse", "--show-prefix")
        out = git_tools._run_git("status", "--porcelain", "--untracked-files=all", "--", *rels)
    except Exception:  # noqa: BLE001 — not a repo, or git missing: nothing to report
        return set()
    if prefix.returncode != 0 or out.returncode != 0:
        return set()
    # Porcelain paths are relative to the repository root, which may sit above the store.
    lead = prefix.stdout.strip()
    paths = {line[3:].strip().strip('"') for line in out.stdout.splitlines() if len(line) > 3}
    return {p[len(lead):] for p in paths if p.startswith(lead)}


# ── the plan ────────────────────────────────────────────────────────────────
def _remap(ref: str, renamed: dict[str, str]) -> str:
    new = renamed.get(normalize_ref(ref))
    if new is None:
        return ref
    return f"{new}.md" if ref.endswith(".md") else new


def _rename(rel: str, bundle_id: str) -> str:
    return f"{rel[:-3]}-handoff-{bundle_id[3:11]}.md"


def plan_import(
    bundle: Any,
    *,
    project: str | None = None,
) -> dict[str, Any]:
    """What importing ``bundle`` would write, and why each record is or isn't.

    Writes nothing. Each record gets an ``action``: ``new`` (lands at its origin
    path), ``collision`` (that path is taken, so it lands renamed), ``duplicate``
    (this origin record with this content is already here) or ``rejected``
    (with a ``reason``). ``digest`` identifies the plan for :func:`apply_import`.
    """
    manifest, records = _bundle_parts(bundle)
    bundle_id = manifest["bundle_id"]
    root = _root()
    index = scope.alias_index()
    source_project = scope.canonical_project(manifest["selection"]["project"], index)
    imported, known_projects = _store_view(root)

    if project is not None and (not isinstance(project, str) or not project.strip()):
        raise HandoffError("project must be a project slug")
    destination = scope.canonical_project(project, index) if project else source_project
    if project is None and source_project not in known_projects:
        raise HandoffError(
            f"project/{source_project} isn't in this store. Pass a project to file "
            f"the records under one of yours, or pass {source_project} to start it here."
        )

    warnings: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(records):
        row: dict[str, Any] = {"index": i}
        rows.append(row)
        if not isinstance(raw, dict):
            row.update(action="rejected", reason="record is not an object")
            continue
        origin = raw.get("origin_path")
        row["origin_path"] = origin
        row["role"] = raw.get("role") if raw.get("role") in {"selected", "dependency"} \
            else "selected"
        problem = _path_problem(origin)
        if problem:
            row.update(action="rejected", reason=problem)
            continue
        body, fm = raw.get("body"), raw.get("frontmatter")
        if not isinstance(body, str) or not isinstance(fm, dict):
            row.update(action="rejected", reason="record needs a string body and a "
                                                 "frontmatter object")
            continue
        if len(body.encode("utf-8")) > MAX_BODY_BYTES:
            row.update(action="rejected", reason=f"body is larger than {MAX_BODY_BYTES} bytes")
            continue
        content_hash = _sha(body)
        if raw.get("content_hash") != content_hash:
            row.update(action="rejected",
                       reason="content_hash doesn't match the body (altered in transit?)")
            continue
        row["content_hash"] = content_hash
        row["stripped_fields"] = sorted(k for k in fm if k not in KEPT_FIELDS)
        meta = {k: fm[k] for k in sorted(fm) if k in KEPT_FIELDS}

        text_fields = [body] + [str(meta[k]) for k in ("title", "description") if k in meta]
        safe, why = store.scan_memory_content("\n".join(text_fields))
        if not safe:
            row.update(action="rejected", reason=f"security scan: {why}")
            continue
        complaint = envelope_complaint(body, "body")
        if complaint:
            row.update(action="rejected", reason="body carries tool-call markup")
            continue
        bad_link = next((f for f in LINK_FIELDS if f in meta and _refs(meta[f]) is None), None)
        if bad_link:
            row.update(action="rejected", reason=f"{bad_link} is not a list of refs")
            continue
        sources = meta.get("sources")
        if sources is not None and (
            not isinstance(sources, list)
            or not all(isinstance(s, dict) and _refs(s.get("ref")) for s in sources)
        ):
            row.update(action="rejected", reason="sources is not a list of anchors")
            continue

        top = origin.split("/")[0]
        if top in PERSONAL_DIRS:
            warnings.append({"path": origin,
                             "warning": f"lands in {top}/ of your store, beside your own"})
        row["agent_directed"] = [body[s:e] for s, e in agent_directed_spans(body)]
        row["state"] = raw.get("state")
        row["meta"], row["body"] = meta, body

        earlier = imported.get((origin, content_hash))
        dest = origin
        if earlier is None and os.path.lexists(os.path.join(root, origin)):
            existing = _read(root, origin)
            if existing is not None and _sha(existing[1]) == content_hash:
                earlier = origin
            else:
                dest = _rename(origin, bundle_id)
                if os.path.lexists(os.path.join(root, dest)):
                    row.update(action="rejected",
                               reason=f"{origin} and {dest} are both taken")
                    continue
        if earlier is not None:
            row.update(action="duplicate", path=earlier)
            continue
        if os.path.commonpath([root, os.path.realpath(os.path.join(root, dest))]) != root:
            row.update(action="rejected", reason=f"{dest} resolves outside the store")
            continue
        row.update(action="collision" if dest != origin else "new", path=dest)

    renamed = {normalize_ref(r["origin_path"]): normalize_ref(r["path"])
               for r in rows if r.get("action") == "collision"}
    landing = {normalize_ref(r["origin_path"]): normalize_ref(r["path"])
               for r in rows if r.get("action") in {"new", "collision", "duplicate"}}
    unresolved: list[dict[str, str]] = []
    for row in rows:
        if row.get("action") not in {"new", "collision"}:
            continue
        meta = _finish_frontmatter(row, manifest, renamed, source_project, destination)
        if meta is None:
            continue
        for field in LINK_FIELDS:
            for ref in _refs(meta.get(field, [])) or []:
                target = normalize_ref(ref)
                if target not in landing.values() and not os.path.exists(
                        os.path.join(root, f"{target}.md")):
                    unresolved.append({"from": row["path"], "relation": field, "ref": ref})
        row["frontmatter"] = meta

    for row in rows:
        row.pop("meta", None)
        if row.get("action") not in {"new", "collision"}:
            row.pop("body", None)

    if rows:
        stray = _uncommitted([r["path"] for r in rows if r.get("action") == "duplicate"])
        for row in rows:
            if row.get("action") == "duplicate" and row["path"] in stray:
                row["uncommitted"] = True
                warnings.append({"path": row["path"], "warning": (
                    "already on disk from an earlier import but not committed; "
                    "applying this plan commits it")})

    counts = {a: sum(1 for r in rows if r.get("action") == a)
              for a in ("new", "collision", "duplicate", "rejected")}
    content = {
        "format": FORMAT,
        "bundle_id": bundle_id,
        "project": {"source": source_project, "destination": destination},
        "records": rows,
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    return {
        **content,
        "origin": {
            "exported_at": manifest.get("exported_at"),
            "exclusions": manifest.get("exclusions") or {},
            "unresolved": manifest.get("unresolved") or [],
            "sender_warnings": manifest.get("warnings") or [],
            "notice": manifest.get("notice"),
        },
        "counts": counts,
        "unresolved": unresolved,
        "warnings": warnings,
        "digest": digest,
        "notice": IMPORT_NOTICE,
    }


def _finish_frontmatter(
    row: dict[str, Any],
    manifest: dict[str, Any],
    renamed: dict[str, str],
    source_project: str,
    destination: str,
) -> dict[str, Any] | None:
    """The frontmatter ``row`` is written with, or ``None`` after rejecting it."""
    meta = dict(row["meta"])
    dest = row["path"]
    for field in LINK_FIELDS:
        if field in meta:
            refs = [_remap(r, renamed) for r in _refs(meta[field]) or []]
            meta[field] = refs[0] if isinstance(row["meta"][field], str) and refs else refs
    if isinstance(meta.get("sources"), list):
        meta["sources"] = [{**s, "ref": _remap(s["ref"], renamed)} for s in meta["sources"]]
    if "claims" in meta:
        try:
            claims = [{k: v for k, v in c.items() if k != "claim_id"} if isinstance(c, dict)
                      else c for c in (meta["claims"] if isinstance(meta["claims"], list)
                                       else [meta["claims"]])]
            for claim in claims:
                if isinstance(claim, dict) and isinstance(claim.get("source_id"), str):
                    claim["source_id"] = _remap(claim["source_id"], renamed)
            meta["claims"] = normalize_claims(claims, dest)
        except ClaimError as exc:
            row.update(action="rejected", reason=f"claims: {exc}")
            row.pop("path", None)
            return None
    entities = meta.get("entities")
    if isinstance(entities, str):
        entities = [entities]
    if isinstance(entities, list):
        rewritten = []
        for entity in entities:
            if isinstance(entity, str) and entity.strip().lower().startswith("project/") \
                    and scope.canonical_project(entity) == source_project:
                entity = f"project/{destination}"
            if entity not in rewritten:
                rewritten.append(entity)
        meta["entities"] = rewritten
    if row["action"] == "collision":
        meta["id"] = f"{dest.split('/')[0]}-{os.path.basename(dest)[:-3]}"
    meta["source"] = IMPORT_SOURCE
    meta["imported_from"] = {
        "bundle_id": manifest["bundle_id"],
        "origin_path": row["origin_path"],
        "content_hash": row["content_hash"],
        "exported_at": manifest.get("exported_at"),
    }
    return json.loads(json.dumps(meta, default=str))


# ── apply ───────────────────────────────────────────────────────────────────
def apply_import(
    bundle: Any,
    expect_digest: str,
    *,
    project: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write a previewed import plan: new files only, one commit.

    Refused (409) when the plan no longer hashes to ``expect_digest``. Each
    ``new``/``collision`` record is written atomically. The commit takes those
    files plus any duplicates an interrupted earlier apply left uncommitted.
    A failed commit is reported, never raised: the files are on disk and the
    next preview shows them.
    """
    if not isinstance(expect_digest, str) or not expect_digest.strip():
        raise HandoffError("expect_digest is required: preview the import first")
    plan = plan_import(bundle, project=project)
    if plan["digest"] != expect_digest.strip():
        raise HandoffError(
            "the bundle or your store changed since the preview (digest differs); "
            "preview again", 409,
        )
    root = _root()
    imported_at = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    written: list[str] = []
    appeared: list[str] = []
    for row in plan["records"]:
        if row.get("action") not in {"new", "collision"}:
            continue
        meta = dict(row["frontmatter"])
        meta["imported_from"] = {**meta["imported_from"], "imported_at": imported_at}
        dumped = yaml.safe_dump(meta, default_flow_style=False, allow_unicode=True,
                                sort_keys=False)
        target = os.path.join(root, row["path"])
        if os.path.lexists(target):
            appeared.append(row["path"])  # written by someone since the re-plan
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        git_tools.write_memory_file(target, f"---\n{dumped}---\n\n{row['body']}\n")
        written.append(row["path"])

    to_commit = written + [r["path"] for r in plan["records"] if r.get("uncommitted")]
    committed, git_error = False, None
    if to_commit and config.git.auto_commit:
        message = (
            f"{config.git.commit_prefix} handoff import: {plan['bundle_id']} "
            f"({len(written)} records into project/{plan['project']['destination']})\n\n"
            f"Palinode-Handoff-Bundle: {plan['bundle_id']}\n"
            f"Palinode-Handoff-Origin-Project: project/{plan['project']['source']}\n"
        )
        outcome = git_tools.try_commit_memory_files(
            [os.path.join(root, p) for p in to_commit], message,
        )
        committed, git_error = outcome.committed, outcome.error

    index_errors: list[dict[str, str]] = []
    for rel in written:
        try:
            from palinode.indexer.index_file import index_file
            outcome = index_file(os.path.join(root, rel))
            if outcome.get("error"):
                index_errors.append({"path": rel, "error": str(outcome["error"])})
        except Exception as exc:  # noqa: BLE001 — the watcher indexes it later
            logger.warning("handoff import: inline index failed for %s: %s", rel, exc)
            index_errors.append({"path": rel, "error": str(exc)})

    result: dict[str, Any] = {
        "bundle_id": plan["bundle_id"],
        "project": plan["project"],
        "written": written,
        "skipped_duplicates": [r["path"] for r in plan["records"]
                               if r.get("action") == "duplicate"],
        "rejected": [{"origin_path": r.get("origin_path"), "reason": r["reason"]}
                     for r in plan["records"] if r.get("action") == "rejected"],
        "not_written": appeared,
        "committed": committed,
        "partial": bool(appeared) or (bool(to_commit) and config.git.auto_commit
                                      and not committed),
        "index_errors": index_errors,
    }
    if git_error:
        result["git_error"] = git_error
    if appeared:
        result["recovery"] = (
            "Some paths were taken between the preview and the write, so those "
            "records weren't written. Preview the same bundle again to place them."
        )
    elif result["partial"]:
        result["recovery"] = (
            "The files are written but not committed. Fix the git problem, then "
            "preview and apply the same bundle again: it commits them."
        )
    return result
