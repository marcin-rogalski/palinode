"""``palinode handoff`` — pass selected project memory to another store.

``export`` previews by default. ``--out FILE --expect DIGEST`` writes the
bundle, and only if the selection still matches the preview's digest. The
bundle is written on this machine; the store is only read.

``import FILE`` previews by default too. ``--apply --expect DIGEST`` writes the
previewed records into the store, and only if the plan still matches.
"""
from __future__ import annotations

import json
import os
from typing import Any

import click

from palinode.cli._api import api_client
from palinode.cli._format import emit_json
from palinode.cli.aliases import _FORMAT, _call, _json


@click.group(name="handoff")
def handoff() -> None:
    """Export selected project memory as a bundle file, or import one.

    See docs/HANDOFF.md.
    """


def _render_preview(data: dict[str, Any]) -> None:
    sel = data.get("selection") or {}
    records = data.get("records") or []
    chosen = [r for r in records if r.get("role") == "selected"]
    deps = [r for r in records if r.get("role") == "dependency"]
    click.echo(f"Handoff preview for project/{sel.get('project')}: "
               f"{len(chosen)} selected, {len(deps)} dependencies.")
    for record in records:
        fm = record.get("frontmatter") or {}
        state = record.get("state")
        tag = "dependency" if record.get("role") == "dependency" else "selected"
        line = f"  + {record['origin_path']}  [{fm.get('type', '?')}, {tag}, {state}]"
        if record.get("retired_reason"):
            line += f" retired: {record['retired_reason']}"
        click.echo(line)
        if record.get("dropped_fields"):
            click.echo(f"      fields not sent: {', '.join(record['dropped_fields'])}")
        for anchor in fm.get("sources") or []:
            if isinstance(anchor, dict):
                click.echo(f"      quotes {anchor.get('ref')}: \"{anchor.get('quote')}\"")
        for span in record.get("agent_directed") or []:
            click.echo(f"      addressed to agents: \"{span.strip()[:120]}\"")
    exclusions = data.get("exclusions") or []
    if exclusions:
        click.echo(f"Left out ({len(exclusions)}):")
        for row in exclusions:
            via = f" (via {row['via']})" if row.get("via") else ""
            click.echo(f"  - {row['path']}: {row['reason']}{via}")
    unresolved = data.get("unresolved") or []
    if unresolved:
        click.echo(f"Links the recipient can't follow ({len(unresolved)}):")
        for row in unresolved:
            click.echo(f"  ? {row['from']} {row['relation']} → {row['ref']}: {row['reason']}")
    for row in data.get("warnings") or []:
        click.echo(f"  ! {row['path']}: {row['warning']}")
    click.echo("")
    click.echo(data.get("notice", ""))
    click.echo("")
    click.echo(f"digest: {data.get('digest')}")
    click.echo("To write it: palinode handoff export ... --out FILE "
               f"--expect {data.get('digest')}")


@handoff.command(name="export")
@click.option("--project", required=True, help="Project slug the records are tagged to")
@click.option("--type", "types", multiple=True,
              help="Memory type to select (repeatable; default Decision and Insight)")
@click.option("--path", "paths", multiple=True,
              help="Export exactly this record instead of the default selection (repeatable)")
@click.option("--include", multiple=True,
              help="Carry this dependency even though the default rules leave it out")
@click.option("--since", default=None, help="Only records updated on or after this ISO date")
@click.option("--out", "out", type=click.Path(dir_okay=False), default=None,
              help="Write the bundle here (needs --expect); refuses to overwrite")
@click.option("--expect", "expect_digest", default=None,
              help="The digest from the preview you reviewed")
@_FORMAT
def export_cmd(project, types, paths, include, since, out, expect_digest, fmt) -> None:
    """Preview, then write, a handoff bundle for one project.

    Without --out this only previews: every record that would leave, the
    fields that don't, the records left out and why, and a digest. Pass that
    digest with --expect and a path with --out to write the file.
    """
    if bool(out) != bool(expect_digest):
        click.echo("Error: --out and --expect go together; preview first to get the digest.")
        raise SystemExit(2)
    if out and os.path.exists(out):
        click.echo(f"Error: {out} already exists; choose another path.")
        raise SystemExit(1)
    data = _call(
        api_client.handoff_export, project,
        types=list(types) or None, paths=list(paths) or None,
        include=list(include) or None, since=since, expect_digest=expect_digest,
    )
    if not out:
        if _json(fmt):
            emit_json(data)
        else:
            _render_preview(data)
        return
    bundle = data["bundle"]
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    manifest = bundle["manifest"]
    if _json(fmt):
        emit_json({"written": out, "bundle_id": manifest["bundle_id"],
                   "records": len(bundle["records"])})
        return
    click.echo(f"Wrote {out}: {len(bundle['records'])} records, bundle {manifest['bundle_id']}.")
    click.echo(manifest["notice"])


#: The API refuses request bodies over 5 MB; say so before sending one.
_MAX_BUNDLE_BYTES = 5 * 1024 * 1024

_ACTION_MARK = {"new": "+", "collision": "+", "duplicate": "=", "rejected": "x"}


def _render_import_preview(data: dict[str, Any]) -> None:
    proj = data.get("project") or {}
    counts = data.get("counts") or {}
    origin = data.get("origin") or {}
    mapping = (f"project/{proj.get('source')} → project/{proj.get('destination')}"
               if proj.get("source") != proj.get("destination")
               else f"project/{proj.get('destination')}")
    click.echo(f"Import preview for bundle {data.get('bundle_id')} "
               f"(exported {origin.get('exported_at')}), {mapping}:")
    click.echo(f"  {counts.get('new', 0)} new, {counts.get('collision', 0)} renamed, "
               f"{counts.get('duplicate', 0)} already here, "
               f"{counts.get('rejected', 0)} rejected.")
    for row in data.get("records") or []:
        action = row.get("action")
        mark = _ACTION_MARK.get(action, "?")
        fm = row.get("frontmatter") or {}
        if action == "rejected":
            click.echo(f"  {mark} {row.get('origin_path')}: rejected — {row.get('reason')}")
            continue
        if action == "duplicate":
            click.echo(f"  {mark} {row.get('origin_path')}: already here as {row.get('path')}")
            continue
        line = f"  {mark} {row['path']}  [{fm.get('type', '?')}, {row.get('role')}"
        line += f", {row.get('state')}]" if row.get("state") else "]"
        if action == "collision":
            line += f"  renamed: {row['origin_path']} is taken"
        click.echo(line)
        if row.get("stripped_fields"):
            click.echo(f"      fields ignored: {', '.join(row['stripped_fields'])}")
        for span in row.get("agent_directed") or []:
            click.echo(f"      addressed to agents: \"{span.strip()[:120]}\"")
    excluded = origin.get("exclusions") or {}
    if excluded:
        parts = ", ".join(f"{n} {reason}" for reason, n in excluded.items())
        click.echo(f"The sender left out: {parts}.")
    unresolved = data.get("unresolved") or []
    if unresolved:
        click.echo(f"Links that point at nothing in your store ({len(unresolved)}):")
        for row in unresolved:
            click.echo(f"  ? {row['from']} {row['relation']} → {row['ref']}")
    for row in data.get("warnings") or []:
        click.echo(f"  ! {row['path']}: {row['warning']}")
    click.echo("")
    click.echo(data.get("notice", ""))
    click.echo("")
    click.echo(f"digest: {data.get('digest')}")
    click.echo("To import it: palinode handoff import FILE ... --apply "
               f"--expect {data.get('digest')}")


@handoff.command(name="import")
@click.argument("bundle", metavar="FILE", type=click.Path(exists=True, dir_okay=False))
@click.option("--project", default=None,
              help="Your project for the records (default: the sender's, if you have it)")
@click.option("--apply", "do_apply", is_flag=True,
              help="Write the previewed records (needs --expect)")
@click.option("--expect", "expect_digest", default=None,
              help="The digest from the preview you reviewed")
@_FORMAT
def import_cmd(bundle, project, do_apply, expect_digest, fmt) -> None:
    """Preview, then apply, importing a handoff bundle file.

    Without --apply this only previews: where each record would land, what is
    already here, what is rejected and why, and a digest. Nothing in your
    store is overwritten or edited; imported records carry no authority.
    """
    if do_apply != bool(expect_digest):
        click.echo("Error: --apply and --expect go together; preview first to get the digest.")
        raise SystemExit(2)
    bundle_file = bundle  # the bundle param names the file; its contents are sent
    if os.path.getsize(bundle_file) > _MAX_BUNDLE_BYTES:
        click.echo(f"Error: {bundle_file} is larger than the API's 5 MB request limit; "
                   "ask the sender to split the export (--path or --since).")
        raise SystemExit(1)
    try:
        with open(bundle_file, encoding="utf-8") as handle:
            contents = json.load(handle)
    except (OSError, ValueError) as exc:
        click.echo(f"Error: {bundle_file} isn't a readable JSON bundle: {exc}")
        raise SystemExit(1) from None
    data = _call(api_client.handoff_import, contents, project=project,
                 expect_digest=expect_digest)
    if _json(fmt):
        emit_json(data)
        return
    if not do_apply:
        _render_import_preview(data)
        return
    click.echo(f"Imported {len(data.get('written') or [])} records from bundle "
               f"{data.get('bundle_id')} into project/{(data.get('project') or {}).get('destination')}.")
    for path in data.get("written") or []:
        click.echo(f"  + {path}")
    if data.get("skipped_duplicates"):
        click.echo(f"Already here, skipped: {len(data['skipped_duplicates'])}.")
    for row in data.get("rejected") or []:
        click.echo(f"  x {row['origin_path']}: {row['reason']}")
    if data.get("not_written"):
        click.echo(f"Not written, the path was taken after the preview: "
                   f"{', '.join(data['not_written'])}")
    if not data.get("committed"):
        click.echo(f"Not committed{': ' + data['git_error'] if data.get('git_error') else '.'}")
    if data.get("recovery"):
        click.echo(data["recovery"])
