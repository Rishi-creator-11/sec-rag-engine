"""Verify a previously-ingested filing — local + remote consistency.

    python -m ingestion.verify_company --ticker AMZN [--full] [--json]

Checks (NO OpenAI / NO Cohere calls):

  LOCAL    registry has the company + this filing; ledger says complete;
           chunk JSONL exists and validates; chunk_count agrees with registry
  DENSE    canonical chunk vector ids exist in sec-rag-engine; metadata
           ticker / filing_id / fiscal_year are correct (sample, or --full)
  SPARSE   if the sparse stage succeeded, a sample of records exist in
           sec-rag-sparse
  SERVING  the BM25 loader sees this filing's chunks (count matches)

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ingestion import registry
from ingestion.sec_client import SecClient
from ingestion.stages import validate_chunks_artifact

load_dotenv()

logger = logging.getLogger("ingestion.verify")

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "data" / "registry" / "ingestion_state.json"
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
DENSE_INDEX = "sec-rag-engine"
SPARSE_INDEX = "sec-rag-sparse"
NAMESPACE = "__default__"
SAMPLE_SIZE = 6
# Dense read-after-write poll schedule (verify may run right after ingest).
DENSE_VERIFY_WAITS = (0.0, 2.0, 5.0, 10.0)


class Check:
    def __init__(self, name: str):
        self.name = name
        self.ok = True
        self.details: list[str] = []

    def fail(self, message: str) -> None:
        self.ok = False
        self.details.append(f"FAIL {message}")

    def note(self, message: str) -> None:
        self.details.append(message)


def _load_ledger_entry(filing_id: str) -> dict | None:
    if not LEDGER_PATH.exists():
        return None
    data = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return data.get(filing_id)


def _canonical_ids(entry: dict, chunk_file: Path | None = None) -> list[str]:
    """Chunk ids for this filing.

    Prefer the ids actually written in the chunk JSONL (the serving source of
    truth) — this covers seed filings, which keep legacy ``<prefix>_N`` ids
    rather than the canonical ``TICKER_FY_10-K_ACC_N`` form. Fall back to the
    canonical construction when the file is unavailable.
    """
    if chunk_file is not None and Path(chunk_file).exists():
        ids = []
        for line in Path(chunk_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                cid = json.loads(line).get("chunk_id")
                if cid:
                    ids.append(cid)
        if ids:
            return ids

    from retrieval.metadata import canonical_chunk_id

    count = entry.get("chunk_count") or 0
    return [
        canonical_chunk_id(
            ticker=entry["ticker"],
            fiscal_year=entry["fiscal_year"],
            filing_type=entry["filing_type"],
            accession_number=entry["accession_number"],
            chunk_index=i,
        )
        for i in range(count)
    ]


def verify(
    ticker: str,
    *,
    full: bool = False,
    client: SecClient | None = None,
    cik: str | None = None,
    fiscal_year: int | None = None,
) -> dict:
    ticker = ticker.strip().upper()
    checks: list[Check] = []

    # ---- discover filing_id (needs SEC, but no OpenAI/Cohere) ----------
    local = Check("LOCAL")
    checks.append(local)
    company = registry.get_company(ticker)
    if not company:
        local.fail(f"{ticker} not in registry")
        return _finish(ticker, checks)

    # A ticker whose registrant differs from its current CIK (a succession)
    # carries the filing CIK in registry lineage; use it so discovery does not
    # fall back to a successor entity that has not filed the 10-K. An explicit
    # cik= argument still wins.
    lineage = company.get("lineage") or {}
    cik_override = cik or (lineage.get("registrant_cik") if lineage.get("cik_override") else None)
    if cik_override:
        local.note(f"cik override in effect: registrant CIK {cik_override}")

    client = client or SecClient()
    filing = client.discover_latest_10k(
        ticker, cik=cik_override, fiscal_year=fiscal_year
    )
    filing_id = filing.filing_id
    if fiscal_year is not None:
        local.note(f"verifying FY{filing.fiscal_year} ({filing.accession_number})")

    reg_filing = next(
        (f for f in company.get("filings", [])
         if f.get("accession_number") == filing.accession_number),
        None,
    )
    if not reg_filing:
        local.fail(f"registry has no filing {filing.accession_number}")
    entry = _load_ledger_entry(filing_id)
    if not entry:
        local.fail(f"ledger has no entry for {filing_id}")
        return _finish(ticker, checks)
    if entry.get("stages", {}).get("complete", {}).get("status") != "ok":
        local.fail("ledger does not say complete")

    # Canonical path for pipeline filings; a seed filing keeps legacy chunk ids
    # and records its own path in the ledger (data/chunks/<prefix>_chunks.jsonl).
    ledger_chunks = (entry.get("artifacts") or {}).get("chunks")
    chunk_file = (
        (REPO_ROOT / ledger_chunks) if ledger_chunks
        else CHUNKS_DIR / ticker / f"{filing_id}_chunks.jsonl"
    )
    valid, reason = validate_chunks_artifact(
        chunk_file,
        expected_count=entry.get("chunk_count"),
        expected_sha256=entry.get("hashes", {}).get("chunks_sha256"),
    )
    if not valid:
        local.fail(f"chunk artifact: {reason}")
    else:
        local.note(f"chunk artifact valid ({entry.get('chunk_count')} chunks)")
    if reg_filing and reg_filing.get("chunk_count") != entry.get("chunk_count"):
        local.fail(
            f"registry chunk_count {reg_filing.get('chunk_count')} != "
            f"ledger {entry.get('chunk_count')}"
        )

    ids = _canonical_ids(entry, chunk_file)
    if not ids:
        local.fail("no canonical chunk ids (chunk_count missing)")
        return _finish(ticker, checks)

    sample = ids if full else (ids[:3] + ids[-3:])

    # ---- DENSE -------------------------------------------------------
    dense = Check("DENSE")
    checks.append(dense)
    try:
        from pinecone import Pinecone
        import os

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(DENSE_INDEX)
        to_fetch = ids if full else sample
        found: dict = {}
        # tolerate Pinecone read-after-write lag when verify runs right after ingest
        for wait in DENSE_VERIFY_WAITS:
            if wait:
                time.sleep(wait)
            pending = [i for i in to_fetch if i not in found]
            for start in range(0, len(pending), 100):
                fetched = index.fetch(ids=pending[start:start + 100], namespace=NAMESPACE)
                found.update(fetched.vectors or {})
            if all(i in found for i in to_fetch):
                break
        missing = [i for i in to_fetch if i not in found]
        if missing:
            dense.fail(f"{len(missing)} vectors missing (e.g. {missing[:3]})")
        else:
            dense.note(f"{len(found)} sampled vectors present")
        for vid, vector in found.items():
            meta = vector.metadata or {}
            if meta.get("ticker") != ticker:
                dense.fail(f"{vid}: ticker={meta.get('ticker')!r}")
            if meta.get("filing_id") not in (None, filing_id):
                dense.fail(f"{vid}: filing_id={meta.get('filing_id')!r}")
            if meta.get("fiscal_year") not in (None, entry["fiscal_year"]):
                dense.fail(f"{vid}: fiscal_year={meta.get('fiscal_year')!r}")
    except Exception as exc:  # noqa: BLE001
        dense.fail(f"{type(exc).__name__}: {exc}")

    # ---- SPARSE ----------------------------------------------------
    sparse = Check("SPARSE")
    checks.append(sparse)
    sparse_status = entry.get("stages", {}).get("sparse_upserted", {}).get("status")
    if sparse_status != "ok":
        sparse.note(f"sparse stage status={sparse_status!r}; skipping remote check")
    else:
        try:
            from pinecone import Pinecone
            import os

            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(SPARSE_INDEX)
            # Batch: fetch() puts ids in the request line, so a --full run over a
            # large filing (e.g. 400 chunks) overflows with a single call (431).
            present: set[str] = set()
            for start in range(0, len(sample), 50):
                fetched = index.fetch(ids=sample[start:start + 50], namespace=NAMESPACE)
                present.update((fetched.vectors or {}).keys())
            missing = [i for i in sample if i not in present]
            if missing:
                sparse.fail(f"{len(missing)} sparse records missing")
            else:
                sparse.note(f"{len(present)} sampled sparse records present")
        except Exception as exc:  # noqa: BLE001
            sparse.fail(f"{type(exc).__name__}: {exc}")

    # ---- SERVING (BM25 loader) -----------------------------------
    serving = Check("SERVING")
    checks.append(serving)
    try:
        from retrieval.bm25_search import load_chunks

        id_set = set(ids)
        rows = [
            r for r in load_chunks()
            if r.get("chunk_id") in id_set or r.get("filing_id") == filing_id
        ]
        if len(rows) != entry.get("chunk_count"):
            serving.fail(
                f"BM25 loader sees {len(rows)} chunks, expected {entry.get('chunk_count')}"
            )
        else:
            serving.note(f"BM25 loader sees all {len(rows)} chunks")
    except Exception as exc:  # noqa: BLE001
        serving.fail(f"{type(exc).__name__}: {exc}")

    return _finish(ticker, checks, filing_id=filing_id)


def _finish(ticker: str, checks: list[Check], filing_id: str | None = None) -> dict:
    passed = all(c.ok for c in checks)
    return {
        "ticker": ticker,
        "filing_id": filing_id,
        "result": "PASS" if passed else "FAIL",
        "checks": [
            {"name": c.name, "ok": c.ok, "details": c.details} for c in checks
        ],
    }


def _print_report(report: dict) -> None:
    print("=" * 70)
    print(f"VERIFY {report['ticker']}"
          + (f"  filing {report['filing_id']}" if report["filing_id"] else "")
          + f"  ->  {report['result']}")
    print("=" * 70)
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"  [{mark}] {check['name']}")
        for detail in check["details"]:
            print(f"         {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--full", action="store_true", help="fetch every vector, not a sample")
    parser.add_argument("--cik", default=None,
                        help="explicit registrant CIK (bypasses ticker->CIK "
                        "discovery); normally taken from registry lineage")
    parser.add_argument("--fiscal-year", type=int, default=None,
                        help="verify a specific fiscal year's 10-K (default: latest)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    report = verify(args.ticker, full=args.full, cik=args.cik,
                    fiscal_year=args.fiscal_year)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
