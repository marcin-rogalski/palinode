"""Operator routes for selective project-memory handoff.

  POST /handoff/export         preview a bundle, or build it for a previewed digest
  POST /handoff/import         preview what importing a bundle would write
  POST /handoff/import/apply   write a previewed import (capture-pause controlled)

There is deliberately no MCP twin: choosing what leaves a store, and what
enters it, is a person's decision, not an agent's (see ``palinode.core.parity``).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from palinode.core import handoff, handoff_import
from palinode.core.handoff import HandoffError
from palinode.core.path_guard import PathTraversalError

router = APIRouter()


class HandoffExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    types: list[str] | None = None
    paths: list[str] | None = None
    include: list[str] | None = None
    since: str | None = None
    #: Absent → preview. Present → the bundle, if the selection still hashes to it.
    expect_digest: str | None = None


@router.post("/handoff/export")
def handoff_export_api(req: HandoffExportRequest) -> dict[str, Any]:
    selection = {
        "types": req.types, "paths": req.paths, "include": req.include, "since": req.since,
    }
    try:
        if req.expect_digest is None:
            return {"preview": True, **handoff.plan_export(req.project, **selection)}
        return {"preview": False,
                "bundle": handoff.export_bundle(req.expect_digest, req.project, **selection)}
    except HandoffError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except PathTraversalError:
        raise HTTPException(status_code=403, detail="Invalid path") from None


class HandoffImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle: dict[str, Any]
    #: The recipient's project for the records. Absent → the sender's, if this store has it.
    project: str | None = None


class HandoffImportApplyRequest(HandoffImportRequest):
    expect_digest: str


@router.post("/handoff/import")
def handoff_import_preview_api(req: HandoffImportRequest) -> dict[str, Any]:
    try:
        return {"preview": True, **handoff_import.plan_import(req.bundle, project=req.project)}
    except HandoffError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None


@router.post("/handoff/import/apply")
def handoff_import_apply_api(req: HandoffImportApplyRequest) -> dict[str, Any]:
    try:
        return {"preview": False, **handoff_import.apply_import(
            req.bundle, req.expect_digest, project=req.project,
        )}
    except HandoffError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except PathTraversalError:
        raise HTTPException(status_code=403, detail="Invalid path") from None
