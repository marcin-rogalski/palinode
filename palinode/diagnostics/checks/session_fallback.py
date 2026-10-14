"""
Check: session_floor_fallback

Warns when the current project holds a non-empty
``.claude/session-floor-fallback.jsonl``. Session-end hooks used to append a
capture there when their write failed, and nothing ever replayed or reported
it, so failed captures piled up unseen. No shipped hook writes the
file any more: the automatic session-end hook deliberately leaves no fallback,
because a failed write may be the API honouring a capture pause. A file that
exists is therefore a leftover holding sessions that never reached memory.

Read-only: stats and counts lines in the cwd project only.

Severity: warn (records waiting to be replayed). Pass otherwise.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from palinode.diagnostics.registry import register
from palinode.diagnostics.types import CheckResult, DoctorContext

_NAME = "session_floor_fallback"
FALLBACK_RELPATH = Path(".claude") / "session-floor-fallback.jsonl"


@register(tags=("fast",))
def session_floor_fallback(ctx: DoctorContext) -> CheckResult:
    """Warn on unreplayed session-end fallback records in the cwd project."""
    path = Path(os.getcwd()) / FALLBACK_RELPATH
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        mtime = path.stat().st_mtime
    except OSError:
        lines, mtime = [], 0.0
    records = [line for line in lines if line.strip()]

    if not records:
        return CheckResult(
            name=_NAME,
            severity="warn",
            passed=True,
            message="No unreplayed session-end fallback records in this project.",
        )

    last = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return CheckResult(
        name=_NAME,
        severity="warn",
        passed=False,
        message=(
            f"{path} holds {len(records)} session-end capture(s) that never "
            f"reached memory (file last written {last})."
        ),
        remediation=(
            "Replay each JSON record, then delete the file:\n"
            '  palinode session-end "<summary>" --trigger hook --source claude-code '
            "--project <slug> --cwd <cwd> --session-id <session_id>\n"
            "(or POST the record's fields to the API's /session-end). "
            "Delete records you no longer want instead of replaying them."
        ),
    )
