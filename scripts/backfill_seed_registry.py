"""Phase 5: register the three seed 10-K filings in the registry + ledger.

``scripts/backfill_metadata.py`` adds fiscal-year / accession / cik metadata to
the seed CHUNKS (local JSONL + dense + sparse). This companion adds the seed
FILINGS to:

  - data/registry/companies.json   -> so ``registry.available_fiscal_years``
                                      knows AAPL FY2024 / MSFT FY2025 / NVDA
                                      FY2026 exist (API year validation)
  - data/registry/ingestion_state.json -> a synthetic "complete" ledger entry
                                      per accession, so ``ingest_company
                                      --years N`` treats the seed as already
                                      ingested and does not create a duplicate
                                      canonical copy.

Seed chunk ids stay ``{apple,microsoft,nvidia}_10k_N`` (no id/embedding change).
The ledger entry points ``artifacts.chunks`` at the legacy seed JSONL path.

    python -m scripts.backfill_seed_registry              # dry run
    python -m scripts.backfill_seed_registry --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import registry
from ingestion.state import Ledger

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"

# Same verified facts as scripts/backfill_metadata.py::SEED_FILINGS.
SEEDS = [
    {"prefix": "apple_10k", "ticker": "AAPL", "legal_name": "Apple Inc.",
     "cik": "0000320193", "accession_number": "0000320193-24-000123",
     "filing_date": "2024-11-01", "report_date": "2024-09-28", "fiscal_year": 2024},
    {"prefix": "microsoft_10k", "ticker": "MSFT", "legal_name": "MICROSOFT CORP",
     "cik": "0000789019", "accession_number": "0000950170-25-100235",
     "filing_date": "2025-07-30", "report_date": "2025-06-30", "fiscal_year": 2025},
    {"prefix": "nvidia_10k", "ticker": "NVDA", "legal_name": "NVIDIA CORP",
     "cik": "0001045810", "accession_number": "0001045810-26-000021",
     "filing_date": "2026-02-25", "report_date": "2026-01-25", "fiscal_year": 2026},
]


def _chunk_count(prefix: str) -> int:
    path = CHUNKS_DIR / f"{prefix}_chunks.jsonl"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    ledger = Ledger()
    for seed in SEEDS:
        n = _chunk_count(seed["prefix"])
        filing_id = seed["accession_number"].replace("-", "")
        seed_jsonl = f"data/chunks/{seed['prefix']}_chunks.jsonl"
        print(f"{seed['ticker']} FY{seed['fiscal_year']}  {seed['accession_number']}  "
              f"{n} chunks  ({seed_jsonl})")

        existing_reg = registry.get_company(seed["ticker"]) or {}
        has_reg = any(f.get("accession_number") == seed["accession_number"]
                      for f in existing_reg.get("filings", []))
        has_ledger = ledger.is_complete(filing_id)
        print(f"   registry recorded: {has_reg}   ledger complete: {has_ledger}")

        if not a.apply:
            continue

        registry.record_filing(seed["ticker"], {
            "filing_type": "10-K",
            "fiscal_year": seed["fiscal_year"],
            "accession_number": seed["accession_number"],
            "filing_date": seed["filing_date"],
            "report_date": seed["report_date"],
            "chunk_count": n,
        })

        if not has_ledger:
            entry = {
                "ticker": seed["ticker"], "company_name": seed["legal_name"],
                "cik": seed["cik"], "filing_type": "10-K",
                "fiscal_year": seed["fiscal_year"],
                "accession_number": seed["accession_number"],
                "filing_id": filing_id,
                "filing_date": seed["filing_date"], "report_date": seed["report_date"],
                "primary_document": seed["accession_number"] + " (seed)",
                "source_url": existing_reg.get("source_url", ""),
                "stage": "complete", "chunk_count": n, "embedding_count": n,
                "dense_upserted": n,
                "artifacts": {"chunks": seed_jsonl},
                "hashes": {},
                "stages": {s: {"status": "ok", "ts": "seed-backfill"} for s in (
                    "discovered", "downloaded", "cleaned", "chunked", "embedded",
                    "dense_upserted", "sparse_upserted", "bm25_registered",
                    "registry_updated", "complete")},
                "last_error": None,
                "created_at": "seed-backfill", "updated_at": "seed-backfill",
                "note": "seed filing; chunks use legacy ids "
                        f"{seed['prefix']}_N (no canonical migration)",
            }
            ledger._data[filing_id] = entry  # noqa: SLF001 - one-shot seed import
            ledger._write()  # noqa: SLF001

    if a.apply:
        registry.reload()
        print("\nAPPLIED. registry.available_fiscal_years:")
        for seed in SEEDS:
            print(f"  {seed['ticker']}: {registry.available_fiscal_years(seed['ticker'])}")
    else:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
