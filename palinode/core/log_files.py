"""File-backed logging that touches the filesystem only when a record is written.

``palinode.api.server`` and ``palinode.indexer.watcher`` attach an operations-log
handler at import time. A plain ``logging.FileHandler`` opens its file in the
constructor, and the directory had to exist first, so *importing* either module
created ``<memory_dir>/logs/operations.jsonl``. Anything that imported them for
another reason (a CLI subcommand, a test collecting a module, a linter plugin)
wrote into whatever store the process default resolved to.

``LazyDirFileHandler`` defers both: the parent directory is created and the file
opened on the first emitted record, never on construction.
"""
from __future__ import annotations

import logging
import os


class LazyDirFileHandler(logging.FileHandler):
    """A ``FileHandler`` that creates its parent directory on first write."""

    def __init__(self, filename: str, mode: str = "a", encoding: str | None = None) -> None:
        super().__init__(filename, mode=mode, encoding=encoding, delay=True)

    def _open(self):  # noqa: ANN202 — mirrors logging.FileHandler._open
        parent = os.path.dirname(self.baseFilename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return super()._open()
