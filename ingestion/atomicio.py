"""Atomic file writes: temp file in the same directory -> flush -> fsync -> os.replace.

A single audited implementation used by the ledger, the registry, and every
ingestion artifact so an interrupted process can never replace a good final
file with a half-written one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj, *, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(row) + "\n" for row in rows))
