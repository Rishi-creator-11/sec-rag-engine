"""Safe, idempotent metadata backfill for the three seed filings.

Adds the identifiers needed for future ticker/year/filing filtering
(``cik``, ``accession_number``, ``filing_id``, ``fiscal_year``,
``report_date``, ``chunk_index``) to:

  A. local ``data/chunks/*.jsonl``
  B. the dense Pinecone index  (``sec-rag-engine``)
  C. the sparse Pinecone index (``sec-rag-sparse``)

WITHOUT re-embedding, without changing vector IDs, without deleting or
recreating anything.

Scope is hard-limited to the filings already in this repository:

    AAPL  Apple Inc.            FY2024 10-K
    MSFT  Microsoft Corporation FY2025 10-K
    NVDA  NVIDIA Corporation    FY2026 10-K

The seed facts below (accession numbers, real SEC filing dates, period-end
dates, CIKs) were confirmed against https://data.sec.gov/submissions/ on
2026-08-27. Accession numbers and CIKs are also cross-checked against the
canonical SEC URLs stored in ``data/raw/*.json``; the script aborts on any
mismatch rather than guessing.

Usage
-----
    python scripts/backfill_metadata.py                 # dry run (default)
    python scripts/backfill_metadata.py --inspect-remote # dry run + read Pinecone
    python scripts/backfill_metadata.py --apply-local --yes
    python scripts/backfill_metadata.py --apply-dense --apply-sparse --yes
    python scripts/backfill_metadata.py --apply-local --fix-filing-date --yes

``--fix-filing-date`` additionally overwrites the legacy ``filing_date`` value
(currently the period-end date) with the real SEC submission date. It is
OFF by default so a plain backfill does not change ``/ask`` output.

Running any apply mode twice converges to the same state.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from retrieval.metadata import (  # noqa: E402
    ChunkMetadata,
    accession_to_filing_id,
    derive_fiscal_year,
    format_accession,
    normalize_cik,
    parse_accession_from_sec_url,
    parse_cik_from_sec_url,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
RAW_DIR = REPO_ROOT / "data" / "raw"

DENSE_INDEX = "sec-rag-engine"
SPARSE_INDEX = "sec-rag-sparse"
PINECONE_NAMESPACE = "__default__"

# Fields this backfill adds. ``filing_date`` is handled separately and only
# touched with --fix-filing-date.
BACKFILL_FIELDS = (
    "cik",
    "accession_number",
    "filing_id",
    "fiscal_year",
    "report_date",
    "chunk_index",
)


# --------------------------------------------------------------------------- #
# Verified seed facts                                                         #
# --------------------------------------------------------------------------- #
SEED_FILINGS: dict[str, dict] = {
    "apple_10k": {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "accession_number": "0000320193-24-000123",
        "filing_type": "10-K",
        "filing_date": "2024-11-01",   # SEC submission date
        "report_date": "2024-09-28",   # fiscal period end
        "fiscal_year": 2024,
        "expected_chunks": 66,
    },
    "microsoft_10k": {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "cik": "0000789019",
        "accession_number": "0000950170-25-100235",
        "filing_type": "10-K",
        "filing_date": "2025-07-30",
        "report_date": "2025-06-30",
        "fiscal_year": 2025,
        "expected_chunks": 98,
    },
    "nvidia_10k": {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "cik": "0001045810",
        "accession_number": "0001045810-26-000021",
        "filing_type": "10-K",
        "filing_date": "2026-02-25",
        "report_date": "2026-01-25",
        "fiscal_year": 2026,
        "expected_chunks": 103,
    },
}


class BackfillAbort(RuntimeError):
    """Raised when a mapping is ambiguous or inconsistent."""


# --------------------------------------------------------------------------- #
# Loading + validation                                                        #
# --------------------------------------------------------------------------- #
def _chunk_file(prefix: str) -> pathlib.Path:
    return CHUNKS_DIR / f"{prefix}_chunks.jsonl"


def load_local_chunks() -> dict[str, list[dict]]:
    """Return ``{seed_prefix: [chunk_record, ...]}`` from the JSONL files."""
    by_prefix: dict[str, list[dict]] = {}
    for prefix in SEED_FILINGS:
        path = _chunk_file(prefix)
        if not path.exists():
            raise BackfillAbort(f"missing chunk file: {path}")
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise BackfillAbort(f"{path}:{line_number}: {exc}") from exc
        by_prefix[prefix] = records
    return by_prefix


def _seed_for_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    for prefix in SEED_FILINGS:
        marker = f"{prefix}_"
        if chunk_id.startswith(marker):
            suffix = chunk_id[len(marker):]
            if suffix.isdigit():
                return prefix, int(suffix)
    return None


def validate(by_prefix: dict[str, list[dict]]) -> list[str]:
    """Validate every local chunk against its seed facts.

    Returns a list of human-readable problems; empty means safe to proceed.
    """
    problems: list[str] = []
    total = 0

    # Cross-check seed facts against the canonical SEC URLs in data/raw/*.json.
    for prefix, seed in SEED_FILINGS.items():
        raw_path = RAW_DIR / f"{prefix}.json"
        if not raw_path.exists():
            problems.append(f"{prefix}: missing {raw_path}")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        seed["source_url"] = raw["source_url"]
        try:
            url_cik = parse_cik_from_sec_url(raw["source_url"])
            url_accession = parse_accession_from_sec_url(raw["source_url"])
        except ValueError as exc:
            problems.append(f"{prefix}: cannot parse SEC URL: {exc}")
            continue
        if url_cik != seed["cik"]:
            problems.append(
                f"{prefix}: CIK mismatch (URL {url_cik} vs seed {seed['cik']})"
            )
        if url_accession != seed["accession_number"]:
            problems.append(
                f"{prefix}: accession mismatch "
                f"(URL {url_accession} vs seed {seed['accession_number']})"
            )
        if raw.get("ticker") != seed["ticker"]:
            problems.append(f"{prefix}: ticker mismatch with {raw_path}")
        if raw.get("company") != seed["company_name"]:
            problems.append(f"{prefix}: company mismatch with {raw_path}")
        if raw.get("filing_type") != seed["filing_type"]:
            problems.append(f"{prefix}: filing_type mismatch with {raw_path}")
        # The raw JSON's "filing_date" is really the period-end date.
        if raw.get("filing_date") != seed["report_date"]:
            problems.append(
                f"{prefix}: raw filing_date {raw.get('filing_date')!r} != "
                f"expected report_date {seed['report_date']!r}"
            )
        if format_accession(seed["accession_number"]) != seed["accession_number"]:
            problems.append(f"{prefix}: seed accession not canonical")
        if accession_to_filing_id(seed["accession_number"]) != seed.get("filing_id", ""):
            seed["filing_id"] = accession_to_filing_id(seed["accession_number"])
        if normalize_cik(seed["cik"]) != seed["cik"]:
            problems.append(f"{prefix}: seed CIK not normalized")
        try:
            if derive_fiscal_year(seed["report_date"]) != seed["fiscal_year"]:
                problems.append(
                    f"{prefix}: derived fiscal_year "
                    f"{derive_fiscal_year(seed['report_date'])} != "
                    f"seed {seed['fiscal_year']}"
                )
        except ValueError as exc:
            problems.append(f"{prefix}: {exc}")

    # Per-chunk checks.
    for prefix, records in by_prefix.items():
        seed = SEED_FILINGS[prefix]
        seen_indices: set[int] = set()
        for record in records:
            total += 1
            chunk_id = record.get("chunk_id")
            if not chunk_id:
                problems.append(f"{prefix}: record without chunk_id")
                continue
            resolved = _seed_for_chunk_id(chunk_id)
            if resolved is None or resolved[0] != prefix:
                problems.append(f"{chunk_id}: does not map to seed {prefix}")
                continue
            _, index_from_id = resolved
            record_index = record.get("chunk_index")
            if record_index != index_from_id:
                problems.append(
                    f"{chunk_id}: chunk_index {record_index!r} != "
                    f"suffix {index_from_id}"
                )
            seen_indices.add(index_from_id)
            if record.get("ticker") != seed["ticker"]:
                problems.append(f"{chunk_id}: ticker {record.get('ticker')!r}")
            if record.get("filing_type") != seed["filing_type"]:
                problems.append(
                    f"{chunk_id}: filing_type {record.get('filing_type')!r}"
                )
            if record.get("company") != seed["company_name"]:
                problems.append(f"{chunk_id}: company {record.get('company')!r}")
            if record.get("source_url") != seed.get("source_url"):
                problems.append(f"{chunk_id}: source_url differs from seed")
            if record.get("filing_date") != seed["report_date"]:
                problems.append(
                    f"{chunk_id}: existing filing_date {record.get('filing_date')!r}"
                    f" != expected report_date {seed['report_date']!r}"
                )
            if not str(record.get("text", "")).strip():
                problems.append(f"{chunk_id}: empty text")

        # Contiguity + count.
        if len(records) != seed["expected_chunks"]:
            problems.append(
                f"{prefix}: {len(records)} chunks, expected "
                f"{seed['expected_chunks']}"
            )
        expected_indices = set(range(len(records)))
        if seen_indices != expected_indices:
            missing = sorted(expected_indices - seen_indices)
            extra = sorted(seen_indices - expected_indices)
            problems.append(
                f"{prefix}: non-contiguous chunk_index (missing={missing} "
                f"extra={extra})"
            )

        # The canonical schema must accept a real chunk from this filing.
        if records:
            try:
                ChunkMetadata.from_parts(
                    cik=seed["cik"],
                    accession_number=seed["accession_number"],
                    chunk_id=records[0]["chunk_id"],
                    chunk_index=records[0]["chunk_index"],
                    ticker=seed["ticker"],
                    filing_type=seed["filing_type"],
                    filing_date=seed["filing_date"],
                    report_date=seed["report_date"],
                    company_name=seed["company_name"],
                    source_url=seed["source_url"],
                    fiscal_year=seed["fiscal_year"],
                )
            except (ValueError, TypeError) as exc:
                problems.append(f"{prefix}: canonical schema rejects chunk 0: {exc}")

    if total != sum(s["expected_chunks"] for s in SEED_FILINGS.values()):
        problems.append(
            f"total chunk count {total} != "
            f"{sum(s['expected_chunks'] for s in SEED_FILINGS.values())}"
        )

    return problems


# --------------------------------------------------------------------------- #
# Change planning                                                             #
# --------------------------------------------------------------------------- #
def target_additions(seed: dict, chunk_index: int, *, fix_filing_date: bool) -> dict:
    additions = {
        "cik": seed["cik"],
        "accession_number": seed["accession_number"],
        "filing_id": seed["filing_id"],
        "fiscal_year": seed["fiscal_year"],
        "report_date": seed["report_date"],
        "chunk_index": chunk_index,
    }
    if fix_filing_date:
        additions["filing_date"] = seed["filing_date"]
    return additions


def diff_record(existing: dict, additions: dict) -> dict[str, tuple]:
    """Return ``{field: (old, new)}`` for fields that would change."""
    changes: dict[str, tuple] = {}
    for field, new_value in additions.items():
        old_value = existing.get(field, None)
        if old_value != new_value:
            changes[field] = (old_value, new_value)
    return changes


def plan(
    by_prefix: dict[str, list[dict]],
    *,
    fix_filing_date: bool,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for prefix, records in by_prefix.items():
        seed = SEED_FILINGS[prefix]
        per_field_changes: dict[str, int] = {}
        sample = None
        for record in records:
            additions = target_additions(
                seed, record["chunk_index"], fix_filing_date=fix_filing_date
            )
            changes = diff_record(record, additions)
            for field in changes:
                per_field_changes[field] = per_field_changes.get(field, 0) + 1
            if sample is None:
                sample = (record["chunk_id"], additions, changes)
        result[prefix] = {
            "seed": seed,
            "chunk_count": len(records),
            "per_field_changes": per_field_changes,
            "sample": sample,
        }
    return result


def print_plan(the_plan: dict[str, dict], *, fix_filing_date: bool) -> None:
    print("=" * 78)
    print("BACKFILL PLAN (dry run)")
    print("=" * 78)
    for prefix, info in the_plan.items():
        seed = info["seed"]
        print()
        print(
            f"{prefix}  [{seed['ticker']}]  FY{seed['fiscal_year']} "
            f"{seed['filing_type']}  accession {seed['accession_number']}"
        )
        print(f"  chunks: {info['chunk_count']}")
        print(f"  real SEC filing_date : {seed['filing_date']}")
        print(f"  report_date (period) : {seed['report_date']}")
        changes = info["per_field_changes"]
        if not changes:
            print("  no changes needed (already backfilled)")
        else:
            for field in ("cik", "accession_number", "filing_id", "fiscal_year",
                          "report_date", "chunk_index", "filing_date"):
                if field in changes:
                    print(f"    + {field:<17} -> set on {changes[field]} chunks")
        if not fix_filing_date:
            print(
                "    ~ filing_date       : left unchanged "
                f"(still {seed['report_date']!r}; pass --fix-filing-date to "
                f"set {seed['filing_date']!r})"
            )
        if info["sample"]:
            chunk_id, additions, sample_changes = info["sample"]
            print(f"  sample chunk {chunk_id}:")
            print(f"    additions: {json.dumps(additions)}")
            print(f"    changes  : {json.dumps({k: list(v) for k, v in sample_changes.items()})}")
    print()
    print("Affected vectors per index: dense 267, sparse 267 (IDs and embeddings unchanged).")


# --------------------------------------------------------------------------- #
# Apply: local JSONL                                                          #
# --------------------------------------------------------------------------- #
def apply_local(by_prefix: dict[str, list[dict]], *, fix_filing_date: bool) -> None:
    from retrieval.metadata import metadata_fields

    ordered = metadata_fields()
    for prefix, records in by_prefix.items():
        seed = SEED_FILINGS[prefix]
        path = _chunk_file(prefix)
        updated_lines: list[str] = []
        for record in records:
            additions = target_additions(
                seed, record["chunk_index"], fix_filing_date=fix_filing_date
            )
            merged = dict(record)
            merged.update(additions)
            # Stable key order: canonical fields first, then text, then extras.
            reordered = {}
            for key in ordered:
                if key in merged:
                    reordered[key] = merged[key]
            if "text" in merged:
                reordered["text"] = merged["text"]
            for key, value in merged.items():
                if key not in reordered:
                    reordered[key] = value
            updated_lines.append(json.dumps(reordered))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        print(f"  wrote {path} ({len(updated_lines)} chunks)")


# --------------------------------------------------------------------------- #
# Apply: Pinecone metadata (no re-embedding, same IDs)                        #
# --------------------------------------------------------------------------- #
def _pinecone_index(index_name: str):
    from dotenv import load_dotenv
    from pinecone import Pinecone

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise BackfillAbort("PINECONE_API_KEY is not set")
    return Pinecone(api_key=api_key).Index(index_name)


def inspect_remote(by_prefix: dict[str, list[dict]], *, fix_filing_date: bool) -> None:
    print()
    print("=" * 78)
    print("REMOTE INSPECTION (read-only)")
    print("=" * 78)
    for index_name in (DENSE_INDEX, SPARSE_INDEX):
        try:
            index = _pinecone_index(index_name)
        except BackfillAbort as exc:
            print(f"  {index_name}: skipped ({exc})")
            continue
        print(f"\n  {index_name} (namespace {PINECONE_NAMESPACE})")
        for prefix, records in by_prefix.items():
            seed = SEED_FILINGS[prefix]
            sample_id = records[0]["chunk_id"]
            fetched = index.fetch(ids=[sample_id], namespace=PINECONE_NAMESPACE)
            vector = (fetched.vectors or {}).get(sample_id)
            if vector is None:
                print(f"    {sample_id}: NOT FOUND")
                continue
            current = dict(vector.metadata or {})
            additions = target_additions(
                seed, records[0]["chunk_index"], fix_filing_date=fix_filing_date
            )
            missing = [k for k in additions if k not in current]
            differing = [
                k for k in additions if k in current and current[k] != additions[k]
            ]
            print(
                f"    {sample_id}: has {sorted(current)} | "
                f"missing {missing} | differing {differing}"
            )


def apply_pinecone(
    index_name: str,
    by_prefix: dict[str, list[dict]],
    *,
    fix_filing_date: bool,
    sleep_seconds: float = 0.05,
) -> None:
    index = _pinecone_index(index_name)
    total = 0
    for prefix, records in by_prefix.items():
        seed = SEED_FILINGS[prefix]
        for record in records:
            additions = target_additions(
                seed, record["chunk_index"], fix_filing_date=fix_filing_date
            )
            index.update(
                id=record["chunk_id"],
                set_metadata=additions,
                namespace=PINECONE_NAMESPACE,
            )
            total += 1
            if sleep_seconds:
                time.sleep(sleep_seconds)
        print(f"  {index_name}: updated {seed['ticker']} ({len(records)} vectors)")
    print(f"  {index_name}: {total} vectors updated total")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-local", action="store_true",
                        help="rewrite data/chunks/*.jsonl")
    parser.add_argument("--apply-dense", action="store_true",
                        help="update dense Pinecone metadata")
    parser.add_argument("--apply-sparse", action="store_true",
                        help="update sparse Pinecone metadata")
    parser.add_argument("--fix-filing-date", action="store_true",
                        help="also overwrite legacy filing_date with the real "
                             "SEC submission date (changes /ask output)")
    parser.add_argument("--inspect-remote", action="store_true",
                        help="read current Pinecone metadata and show the diff")
    parser.add_argument("--yes", action="store_true",
                        help="required confirmation for any --apply-* mode")
    args = parser.parse_args(argv)

    apply_modes = [args.apply_local, args.apply_dense, args.apply_sparse]
    is_apply = any(apply_modes)

    if is_apply and not args.yes:
        print("Refusing to apply without --yes. Re-run with --yes to confirm.")
        return 1

    try:
        by_prefix = load_local_chunks()
    except BackfillAbort as exc:
        print(f"ABORT: {exc}")
        return 2

    print(f"Loaded {sum(len(v) for v in by_prefix.values())} local chunks "
          f"from {len(by_prefix)} filings.")

    problems = validate(by_prefix)
    if problems:
        print("\nABORT: metadata mapping is not unambiguous:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    print("Validation: all chunks map unambiguously to a seed filing.\n")

    the_plan = plan(by_prefix, fix_filing_date=args.fix_filing_date)
    print_plan(the_plan, fix_filing_date=args.fix_filing_date)

    if args.inspect_remote:
        try:
            inspect_remote(by_prefix, fix_filing_date=args.fix_filing_date)
        except Exception as exc:  # noqa: BLE001 - inspection must never mutate
            print(f"  remote inspection failed: {exc!r}")

    if not is_apply:
        print()
        print("DRY RUN — nothing was written.")
        print("To apply:  python scripts/backfill_metadata.py "
              "--apply-local --apply-dense --apply-sparse --yes")
        return 0

    print()
    print("=" * 78)
    print("APPLYING CHANGES")
    print("=" * 78)
    if args.apply_local:
        print("\nlocal JSONL:")
        apply_local(by_prefix, fix_filing_date=args.fix_filing_date)
    if args.apply_dense:
        print("\ndense Pinecone:")
        apply_pinecone(DENSE_INDEX, by_prefix, fix_filing_date=args.fix_filing_date)
    if args.apply_sparse:
        print("\nsparse Pinecone:")
        apply_pinecone(SPARSE_INDEX, by_prefix, fix_filing_date=args.fix_filing_date)
    print("\nDone. Re-run this script to confirm it now reports no changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
