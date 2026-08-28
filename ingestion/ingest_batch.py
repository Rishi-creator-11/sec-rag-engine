"""Sequential batch ingestion driver for the controlled company pilot.

    python -m ingestion.ingest_batch --tickers GOOGL META
    python -m ingestion.ingest_batch --manifest data/registry/pilot10.json --batch 1
    python -m ingestion.ingest_batch --tickers GOOGL META --dry-run
    python -m ingestion.ingest_batch --tickers GOOGL META --verify

Thin wrapper over ``ingestion.ingest_company`` — no ingestion logic is
duplicated here. One company at a time (no concurrency, no SEC parallel
crawling). Each company is isolated: a failure in one does not touch another's
ledger/registry state. By default the batch STOPS on the first hard failure;
pass ``--continue-on-error`` to attempt the rest anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ingestion.ingest_company import IngestionError, ingest_company
from ingestion.sec_client import SecClientError
from ingestion.verify_company import verify

load_dotenv()

logger = logging.getLogger("ingestion.batch")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tickers_from_manifest(manifest_path: str, batch: int | None) -> list[str]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = data.get("companies", [])
    out = []
    for entry in entries:
        if entry.get("status") == "existing":
            continue
        if batch is not None and entry.get("batch") != batch:
            continue
        out.append(entry["ticker"].strip().upper())
    return out


def run_batch(
    tickers: list[str],
    *,
    dry_run: bool = False,
    force: bool = False,
    skip_sparse: bool = False,
    do_verify: bool = False,
    continue_on_error: bool = False,
) -> dict:
    results: list[dict] = []
    stopped = False

    for ticker in tickers:
        started = time.perf_counter()
        row: dict = {"ticker": ticker}
        try:
            outcome = ingest_company(
                ticker, dry_run=dry_run, force=force, skip_sparse=skip_sparse
            )
            row["ingest_status"] = outcome.get("status")
            if not dry_run:
                row["chunk_count"] = outcome.get("chunk_count")
                row["dense_upserted"] = outcome.get("dense_upserted")
                row["sparse_status"] = outcome.get("sparse_status")
            else:
                row["filing"] = outcome.get("filing")
                row["collision"] = outcome.get("collision")

            if do_verify and not dry_run:
                report = verify(ticker, full=True)
                row["verify"] = report["result"]
                row["verify_checks"] = {
                    c["name"]: c["ok"] for c in report["checks"]
                }
                if report["result"] != "PASS":
                    raise IngestionError(f"verification FAILED for {ticker}")

            row["ok"] = True
        except (SecClientError, IngestionError) as exc:
            row["ok"] = False
            row["error_class"] = type(exc).__name__
            row["error"] = str(exc)
            logger.error(
                "batch event=company_failed ticker=%s error_class=%s: %s",
                ticker, type(exc).__name__, exc,
            )
        except Exception as exc:  # noqa: BLE001 - never let one company crash the driver
            row["ok"] = False
            row["error_class"] = type(exc).__name__
            row["error"] = str(exc)
            logger.exception("batch event=company_failed_unexpected ticker=%s", ticker)

        row["elapsed_s"] = round(time.perf_counter() - started, 1)
        results.append(row)
        _print_company_row(row)

        if not row["ok"] and not continue_on_error:
            stopped = True
            logger.error("batch event=stopped after=%s (use --continue-on-error to override)",
                         ticker)
            break

    summary = {
        "requested": tickers,
        "attempted": [r["ticker"] for r in results],
        "succeeded": [r["ticker"] for r in results if r["ok"]],
        "failed": [r["ticker"] for r in results if not r["ok"]],
        "stopped_early": stopped,
        "results": results,
    }
    return summary


def _print_company_row(row: dict) -> None:
    mark = "OK  " if row["ok"] else "FAIL"
    line = f"  [{mark}] {row['ticker']:<6} {row.get('ingest_status', '-'):<16} {row['elapsed_s']:>6.1f}s"
    if not row["ok"]:
        line += f"   {row.get('error_class')}: {row.get('error')}"
    elif "chunk_count" in row:
        line += (f"   chunks={row['chunk_count']} dense={row.get('dense_upserted')}"
                 f" sparse={row.get('sparse_status')}"
                 + (f" verify={row['verify']}" if "verify" in row else ""))
    print(line)


def _print_summary(summary: dict, dry_run: bool) -> None:
    print("\n" + "=" * 72)
    print(f"BATCH {'DRY-RUN' if dry_run else 'RESULT'}")
    print("=" * 72)
    print(f"  requested : {summary['requested']}")
    print(f"  succeeded : {summary['succeeded']}")
    print(f"  failed    : {summary['failed']}")
    if summary["stopped_early"]:
        print("  STOPPED EARLY — fix the root cause and rerun (ingestion is resumable).")
    if dry_run:
        for row in summary["results"]:
            filing = row.get("filing")
            if filing:
                print(f"\n  {filing['ticker']}  {filing['company_name']}  CIK {filing['cik']}")
                print(f"    {filing['filing_type']} FY{filing['fiscal_year']}  "
                      f"accession {filing['accession_number']}  "
                      f"filed {filing['filing_date']}  period {filing['report_date']}")
                print(f"    {filing['source_url']}")
                print(f"    collision: {row.get('collision')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers", nargs="+")
    src.add_argument("--manifest")
    parser.add_argument("--batch", type=int, default=None,
                        help="with --manifest: only companies in this batch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="run verify_company --full after each company; FAIL stops the batch")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="attempt remaining companies even after a failure")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers]
    else:
        tickers = _tickers_from_manifest(args.manifest, args.batch)
        if not tickers:
            print(f"manifest {args.manifest} batch {args.batch}: no pending companies")
            return 0

    print(f"batch: {tickers}  (dry_run={args.dry_run} verify={args.verify} "
          f"continue_on_error={args.continue_on_error})")

    summary = run_batch(
        tickers,
        dry_run=args.dry_run,
        force=args.force,
        skip_sparse=args.skip_sparse,
        do_verify=args.verify,
        continue_on_error=args.continue_on_error,
    )
    _print_summary(summary, args.dry_run)

    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
