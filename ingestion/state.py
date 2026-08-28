"""Ingestion ledger — a small, atomic JSON record of what has been ingested.

Keyed by ``filing_id`` (the 18-digit dashless accession number, globally
unique). Tracks per-stage status so a rerun resumes from the first incomplete
stage and a completed filing is skipped unless ``--force``.

No database. Writes are atomic: temp file -> fsync -> os.replace, so an
interrupted process never leaves a corrupt ledger.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ingestion.atomicio import atomic_write_json

logger = logging.getLogger("ingestion.state")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER_PATH = REPO_ROOT / "data" / "registry" / "ingestion_state.json"

# Ordered pipeline stages. "sparse_upserted" is soft: "skipped" or "failed"
# does not block completion (sparse is not in the production retrieval path).
STAGES = (
    "discovered",
    "downloaded",
    "cleaned",
    "chunked",
    "embedded",
    "dense_upserted",
    "sparse_upserted",
    "bm25_registered",
    "registry_updated",
    "complete",
)
SOFT_STAGES = frozenset({"sparse_upserted"})
_TERMINAL_OK = frozenset({"ok", "skipped"})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Ledger:
    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)
        self._data: dict = self._read()

    # ---------------------------------------------------------------- #
    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.error("ledger event=corrupt path=%s (starting empty)", self.path)
            return {}

    def _write(self) -> None:
        atomic_write_json(self.path, self._data, indent=2)

    # ---------------------------------------------------------------- #
    def get(self, filing_id: str) -> dict | None:
        return self._data.get(filing_id)

    def all(self) -> dict:
        return dict(self._data)

    def start_filing(self, filing) -> dict:
        """Create (or return existing) a ledger entry, stage=discovered."""
        existing = self._data.get(filing.filing_id)
        if existing is not None:
            return existing

        entry = {
            "ticker": filing.ticker,
            "company_name": filing.company_name,
            "cik": filing.cik,
            "filing_type": filing.filing_type,
            "fiscal_year": filing.fiscal_year,
            "accession_number": filing.accession_number,
            "filing_id": filing.filing_id,
            "filing_date": filing.filing_date,
            "report_date": filing.report_date,
            "primary_document": filing.primary_document,
            "primary_doc_sha256": None,
            "source_url": filing.source_url,
            "stage": "discovered",
            "chunk_count": None,
            "embedding_count": None,
            "dense_upserted": None,
            "artifacts": {},
            "hashes": {},
            "stages": {
                "discovered": {"status": "ok", "ts": _now()},
            },
            "last_error": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._data[filing.filing_id] = entry
        self._write()
        return entry

    def record_stage(
        self,
        filing_id: str,
        stage: str,
        *,
        status: str = "ok",
        error: str | None = None,
        duration_ms: float | None = None,
        **fields,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        entry = self._data[filing_id]
        stage_record = {"status": status, "ts": _now()}
        if duration_ms is not None:
            stage_record["duration_ms"] = round(duration_ms, 1)
        if error is not None:
            stage_record["error"] = str(error)[:500]
        entry.setdefault("stages", {})[stage] = stage_record

        # promote top-level convenience fields; "hashes" merges into a sub-dict
        for key, value in fields.items():
            if key == "hashes" and isinstance(value, dict):
                entry.setdefault("hashes", {}).update(value)
            else:
                entry[key] = value
        if status in _TERMINAL_OK:
            entry["stage"] = stage
            entry["last_error"] = None
        else:
            entry["last_error"] = {
                "stage": stage,
                "error_class": fields.get("error_class"),
                "message": str(error)[:500] if error else None,
                "ts": _now(),
            }
        entry["updated_at"] = _now()
        self._write()

    def mark_complete(self, filing_id: str) -> None:
        entry = self._data[filing_id]
        for stage in STAGES:
            if stage in ("complete",) or stage in SOFT_STAGES:
                continue
            status = entry.get("stages", {}).get(stage, {}).get("status")
            if status not in _TERMINAL_OK:
                raise RuntimeError(
                    f"cannot mark complete: stage {stage!r} status={status!r}"
                )
        entry.setdefault("stages", {})["complete"] = {"status": "ok", "ts": _now()}
        entry["stage"] = "complete"
        entry["last_error"] = None
        entry["updated_at"] = _now()
        self._write()

    # ---------------------------------------------------------------- #
    def is_complete(self, filing_id: str) -> bool:
        entry = self._data.get(filing_id)
        return bool(
            entry
            and entry.get("stages", {}).get("complete", {}).get("status") == "ok"
        )

    def last_successful_stage(self, filing_id: str) -> str | None:
        entry = self._data.get(filing_id)
        if not entry:
            return None
        done = [
            stage
            for stage in STAGES
            if entry.get("stages", {}).get(stage, {}).get("status") in _TERMINAL_OK
        ]
        return done[-1] if done else None

    def next_incomplete_stage(self, filing_id: str, *, skip_sparse: bool = False) -> str:
        entry = self._data.get(filing_id)
        stages = (entry or {}).get("stages", {})
        for stage in STAGES:
            if stage == "complete":
                return "complete"
            if skip_sparse and stage == "sparse_upserted":
                continue
            if stages.get(stage, {}).get("status") not in _TERMINAL_OK:
                return stage
        return "complete"
