"""Automated single-filing SEC ingestion (latest 10-K only).

    python -m ingestion.ingest_company --ticker AMZN
    python -m ingestion.ingest_company --ticker AMZN --dry-run
    python -m ingestion.ingest_company --ticker AMZN --force
    python -m ingestion.ingest_company --ticker AMZN --skip-sparse --verbose

Pipeline (each stage checkpointed in data/registry/ingestion_state.json):

    discover -> download -> clean -> chunk -> embed -> dense_upsert
    -> sparse_upsert (soft) -> bm25_register -> registry_update -> complete

Idempotent (deterministic canonical chunk/vector IDs, accession-keyed ledger),
resumable (rerun continues from the first incomplete stage), SEC-compliant
(see ingestion.sec_client). No multi-year, no other filing types, no batch.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ingestion.atomicio import atomic_write_jsonl, atomic_write_text
from ingestion.chunk_records import build_chunk_records, validate_chunk_records
from ingestion.sec_client import DiscoveredFiling, SecClient, SecClientError
from ingestion.state import STAGES, Ledger
from ingestion import registry, stages as stage_graph
from retrieval import bm25_search
from retrieval.build_embeddings import EMBEDDING_DIMENSION, embed_records
from retrieval.filters import RetrievalFilter
from retrieval.metadata import (
    ChunkMetadata,
    canonical_chunk_id,
    validate_iso_date,
)

load_dotenv()

logger = logging.getLogger("ingestion.ingest")

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"

MIN_HTML_BYTES = 50_000
MIN_CLEAN_CHARS = 40_000
SEC_BLOCK_MARKERS = (
    "request rate threshold",
    "undeclared automated tool",
    "you have been blocked",
    "access denied",
)


class IngestionError(RuntimeError):
    """A fatal ingestion problem (validation failed, unrecoverable stage)."""


def log_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    """Structured-ish log line: ``<event> k=v k=v`` (no secrets, no payloads)."""
    parts = " ".join(
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    logger.log(level, "%s %s", event, parts)


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
def raw_dir(ticker: str, filing_id: str) -> Path:
    return RAW_DIR / ticker / filing_id


def html_path(ticker: str, filing_id: str) -> Path:
    return raw_dir(ticker, filing_id) / "filing.html"


def text_path(ticker: str, filing_id: str) -> Path:
    return raw_dir(ticker, filing_id) / "filing.txt"


def meta_path(ticker: str, filing_id: str) -> Path:
    return raw_dir(ticker, filing_id) / "metadata.json"


def chunks_path(ticker: str, filing_id: str) -> Path:
    return CHUNKS_DIR / ticker / f"{filing_id}_chunks.jsonl"


def embeddings_path(ticker: str, filing_id: str) -> Path:
    return EMBEDDINGS_DIR / ticker / f"{filing_id}_embeddings.jsonl"


def _rel(path: Path) -> str:
    """Repo-relative path for display/ledger; falls back to the full path."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _artifact_paths(filing: DiscoveredFiling) -> dict[str, Path]:
    """Absolute paths for every local artifact of this filing (for assess_filing)."""
    return {
        "raw_html": html_path(filing.ticker, filing.filing_id),
        "clean_text": text_path(filing.ticker, filing.filing_id),
        "metadata": meta_path(filing.ticker, filing.filing_id),
        "chunks": chunks_path(filing.ticker, filing.filing_id),
        "embeddings": embeddings_path(filing.ticker, filing.filing_id),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write_jsonl(path, rows)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def validate_filing_metadata(filing: DiscoveredFiling) -> None:
    problems: list[str] = []
    try:
        validate_iso_date(filing.filing_date, field="filing_date")
        validate_iso_date(filing.report_date, field="report_date")
        if filing.filing_date < filing.report_date:
            problems.append("filing_date precedes report_date")
    except ValueError as exc:
        problems.append(str(exc))

    if filing.filing_type != "10-K":
        problems.append(f"unexpected filing_type {filing.filing_type!r}")
    if len(filing.cik) != 10 or not filing.cik.isdigit():
        problems.append(f"cik not 10 digits: {filing.cik!r}")
    if filing.filing_id != filing.accession_number.replace("-", ""):
        problems.append("filing_id does not match accession_number")
    if not filing.source_url.startswith("https://www.sec.gov/Archives/edgar/data/"):
        problems.append(f"unexpected source_url: {filing.source_url}")
    if not filing.primary_document.lower().endswith((".htm", ".html")):
        problems.append(f"primary_document is not HTML: {filing.primary_document}")
    if not (1994 <= filing.fiscal_year <= 2100):
        problems.append(f"fiscal_year out of range: {filing.fiscal_year}")
    if not filing.company_name:
        problems.append("empty company_name")

    # Constructing ChunkMetadata for chunk 0 exercises the full canonical schema.
    try:
        ChunkMetadata.from_parts(
            cik=filing.cik,
            accession_number=filing.accession_number,
            chunk_id=canonical_chunk_id(
                ticker=filing.ticker,
                fiscal_year=filing.fiscal_year,
                filing_type=filing.filing_type,
                accession_number=filing.accession_number,
                chunk_index=0,
            ),
            chunk_index=0,
            ticker=filing.ticker,
            filing_type=filing.filing_type,
            filing_date=filing.filing_date,
            report_date=filing.report_date,
            company_name=filing.company_name,
            source_url=filing.source_url,
            fiscal_year=filing.fiscal_year,
        )
    except (ValueError, TypeError) as exc:
        problems.append(f"canonical schema rejects filing: {exc}")

    if problems:
        raise IngestionError("filing metadata validation failed: " + "; ".join(problems))


def validate_html_payload(text: str, content_type: str, filing: DiscoveredFiling) -> None:
    lowered = text[:5000].lower()
    if any(marker in lowered for marker in SEC_BLOCK_MARKERS):
        raise IngestionError("SEC returned a rate-limit / access-denied page")
    if len(text) < MIN_HTML_BYTES:
        raise IngestionError(
            f"filing HTML is only {len(text)} bytes (< {MIN_HTML_BYTES}); "
            "likely not the real document"
        )
    if content_type and "html" not in content_type.lower() and "xml" not in content_type.lower():
        raise IngestionError(f"unexpected content-type {content_type!r}")


def validate_clean_text(clean: str, filing: DiscoveredFiling) -> None:
    lowered = clean.lower()
    if any(marker in lowered[:5000] for marker in SEC_BLOCK_MARKERS):
        raise IngestionError("cleaned text looks like an SEC block page")
    if len(clean) < MIN_CLEAN_CHARS:
        raise IngestionError(
            f"cleaned text is only {len(clean)} chars (< {MIN_CLEAN_CHARS})"
        )
    if "securities and exchange commission" not in lowered:
        raise IngestionError("cleaned text missing 'Securities and Exchange Commission'")
    if "10-k" not in lowered and "annual report" not in lowered:
        raise IngestionError("cleaned text missing '10-K' / 'annual report' context")
    # Loose company-context check: at least one distinctive name token present.
    tokens = [
        token
        for token in filing.company_name.lower().replace(",", " ").split()
        if token not in {"inc", "inc.", "corp", "corp.", "corporation", "co", "com",
                         "the", "ltd", "plc", "holdings", "group"}
        and len(token) >= 3
    ]
    if tokens and not any(token in lowered for token in tokens):
        raise IngestionError(
            f"cleaned text does not mention any company token {tokens}"
        )


def validate_embeddings(embedded: list[dict], records: list[dict]) -> None:
    if len(embedded) != len(records):
        raise IngestionError(
            f"embedding count {len(embedded)} != chunk count {len(records)}"
        )
    for chunk, vector_row in zip(records, embedded):
        if chunk["chunk_id"] != vector_row["chunk_id"]:
            raise IngestionError("embedding chunk_id order mismatch")
        embedding = vector_row.get("embedding")
        if not embedding or len(embedding) != EMBEDDING_DIMENSION:
            raise IngestionError(
                f"{vector_row['chunk_id']}: bad embedding dimension"
            )


def assert_no_chunk_id_collision(records: list[dict], filing: DiscoveredFiling) -> None:
    new_ids = {record["chunk_id"] for record in records}
    own_file = chunks_path(filing.ticker, filing.filing_id)
    existing: set[str] = set()
    if CHUNKS_DIR.exists():
        for path in CHUNKS_DIR.rglob("*_chunks.jsonl"):
            if path == own_file:
                continue
            for row in _read_jsonl(path):
                existing.add(row.get("chunk_id"))
    clash = new_ids & existing
    if clash:
        raise IngestionError(f"chunk_id collision with existing corpus: {sorted(clash)[:5]}")


# --------------------------------------------------------------------------- #
# Stage context + machine                                                     #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Ctx:
    filing: DiscoveredFiling
    ledger: Ledger
    client: SecClient | None
    force: bool
    skip_sparse: bool
    stage_log: list[dict] = dataclasses.field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stage_download(ctx: Ctx) -> None:
    filing = ctx.filing
    response = ctx.client.download_document(filing.source_url)
    text = response.text
    validate_html_payload(text, response.headers.get("Content-Type", ""), filing)
    path = html_path(filing.ticker, filing.filing_id)
    _write_text(path, text)
    ctx.ledger.record_stage(
        filing.filing_id, "downloaded",
        bytes=len(text.encode("utf-8", errors="replace")),
        primary_doc_sha256=_sha256(text),
        artifacts={**ctx.ledger.get(filing.filing_id).get("artifacts", {}),
                   "raw_html": _rel(path)},
    )


def stage_clean(ctx: Ctx) -> None:
    from ingestion.sec_loader import clean_html

    filing = ctx.filing
    html = html_path(filing.ticker, filing.filing_id).read_text(encoding="utf-8")
    clean = clean_html(html)
    validate_clean_text(clean, filing)
    clean_file = text_path(filing.ticker, filing.filing_id)
    _write_text(clean_file, clean)
    _write_text(
        meta_path(filing.ticker, filing.filing_id),
        json.dumps(filing.to_dict(), indent=2),
    )
    ctx.ledger.record_stage(
        filing.filing_id, "cleaned",
        clean_chars=len(clean),
        hashes={"clean_text_sha256": _sha256_file(clean_file)},
        artifacts={**ctx.ledger.get(filing.filing_id).get("artifacts", {}),
                   "clean_text": _rel(clean_file),
                   "metadata": _rel(meta_path(filing.ticker, filing.filing_id))},
    )


def stage_chunk(ctx: Ctx) -> None:
    filing = ctx.filing
    clean = text_path(filing.ticker, filing.filing_id).read_text(encoding="utf-8")
    records = build_chunk_records(clean, filing)

    problems = validate_chunk_records(records, filing)
    if problems:
        raise IngestionError("chunk validation failed: " + "; ".join(problems[:8]))
    assert_no_chunk_id_collision(records, filing)

    chunk_file = chunks_path(filing.ticker, filing.filing_id)
    _write_jsonl(chunk_file, records)
    ctx.ledger.record_stage(
        filing.filing_id, "chunked",
        chunk_count=len(records),
        hashes={"chunks_sha256": _sha256_file(chunk_file)},
        artifacts={**ctx.ledger.get(filing.filing_id).get("artifacts", {}),
                   "chunks": _rel(chunk_file)},
    )


def stage_embed(ctx: Ctx) -> None:
    filing = ctx.filing
    records = _read_jsonl(chunks_path(filing.ticker, filing.filing_id))
    embedded = embed_records(records)
    validate_embeddings(embedded, records)
    embed_file = embeddings_path(filing.ticker, filing.filing_id)
    _write_jsonl(embed_file, embedded)
    ctx.ledger.record_stage(
        filing.filing_id, "embedded",
        embedding_count=len(embedded),
        hashes={"embeddings_sha256": _sha256_file(embed_file)},
        artifacts={**ctx.ledger.get(filing.filing_id).get("artifacts", {}),
                   "embeddings": _rel(embed_file)},
    )


def stage_dense(ctx: Ctx) -> None:
    from retrieval.pinecone_store import upsert_filing

    filing = ctx.filing
    embedded = _read_jsonl(embeddings_path(filing.ticker, filing.filing_id))
    result = upsert_filing(embedded, verify=True)
    ctx.ledger.record_stage(
        filing.filing_id, "dense_upserted",
        dense_upserted=result["upserted"],
        dense_verified=result["verified"],
    )


def stage_sparse(ctx: Ctx) -> None:
    filing = ctx.filing
    if ctx.skip_sparse:
        ctx.ledger.record_stage(filing.filing_id, "sparse_upserted", status="skipped")
        logger.warning("stage event=skipped ticker=%s stage=sparse_upserted reason=flag",
                       filing.ticker)
        return
    try:
        from retrieval.sparse_store import canary_check, upsert_filing_sparse

        canary = canary_check()
        if not canary.get("ok"):
            raise IngestionError(f"sparse canary failed: {canary}")
        records = _read_jsonl(chunks_path(filing.ticker, filing.filing_id))
        result = upsert_filing_sparse(records)
        ctx.ledger.record_stage(
            filing.filing_id, "sparse_upserted", status="ok",
            sparse_upserted=result["upserted"], sparse_canary=canary,
        )
    except Exception as exc:  # noqa: BLE001 - sparse is non-fatal for Phase 3
        logger.warning(
            "stage event=soft_fail ticker=%s stage=sparse_upserted error_class=%s: %s",
            filing.ticker, type(exc).__name__, exc,
        )
        ctx.ledger.record_stage(
            filing.filing_id, "sparse_upserted", status="failed",
            error=str(exc), error_class=type(exc).__name__,
        )


def stage_bm25(ctx: Ctx) -> None:
    filing = ctx.filing
    bm25_search.reload()
    hits = bm25_search.search(
        f"{filing.company_name} risk revenue business",
        top_k=25,
        filters=RetrievalFilter(tickers=(filing.ticker,)),
    )
    if not hits or any(hit["ticker"] != filing.ticker for hit in hits):
        raise IngestionError(
            f"BM25 did not pick up {filing.ticker} after reload "
            f"(hits={len(hits)})"
        )
    ctx.ledger.record_stage(
        filing.filing_id, "bm25_registered", bm25_hits=len(hits)
    )


def stage_registry(ctx: Ctx) -> None:
    filing = ctx.filing
    chunk_count = ctx.ledger.get(filing.filing_id).get("chunk_count")
    # legal_name is SEC-authoritative; display_name is left to curation.
    registry.upsert_company(
        filing.ticker, legal_name=filing.company_name, cik=filing.cik
    )
    registry.record_filing(
        filing.ticker,
        {
            "filing_type": filing.filing_type,
            "fiscal_year": filing.fiscal_year,
            "accession_number": filing.accession_number,
            "filing_date": filing.filing_date,
            "report_date": filing.report_date,
            "chunk_count": chunk_count,
        },
    )
    registry.reload()
    ctx.ledger.record_stage(filing.filing_id, "registry_updated")


_STAGE_FUNCS = {
    "downloaded": stage_download,
    "cleaned": stage_clean,
    "chunked": stage_chunk,
    "embedded": stage_embed,
    "dense_upserted": stage_dense,
    "sparse_upserted": stage_sparse,
    "bm25_registered": stage_bm25,
    "registry_updated": stage_registry,
}


# --------------------------------------------------------------------------- #
# Plan / dry-run                                                              #
# --------------------------------------------------------------------------- #
def build_plan(filing: DiscoveredFiling) -> dict:
    sample_ids = [
        canonical_chunk_id(
            ticker=filing.ticker,
            fiscal_year=filing.fiscal_year,
            filing_type=filing.filing_type,
            accession_number=filing.accession_number,
            chunk_index=index,
        )
        for index in (0, 1, 2)
    ]
    return {
        "filing": filing.to_dict(),
        "paths": {
            "raw_html": _rel(html_path(filing.ticker, filing.filing_id)),
            "clean_text": _rel(text_path(filing.ticker, filing.filing_id)),
            "metadata": _rel(meta_path(filing.ticker, filing.filing_id)),
            "chunks": _rel(chunks_path(filing.ticker, filing.filing_id)),
            "embeddings": _rel(embeddings_path(filing.ticker, filing.filing_id)),
        },
        "dense_index": "sec-rag-engine (namespace __default__)",
        "sparse_index": "sec-rag-sparse (namespace __default__)",
        "sample_vector_ids": sample_ids,
        "index_operations": "upsert only — no create, no recreate, no delete "
                            "(sparse canary self-cleans its synthetic id)",
        "seed_ids_touched": False,
    }


def _print_plan(plan: dict) -> None:
    filing = plan["filing"]
    print("=" * 74)
    print("DRY RUN — no writes, no embeddings, no upserts")
    print("=" * 74)
    for key in ("company_name", "ticker", "cik", "filing_type", "fiscal_year",
                "accession_number", "filing_id", "filing_date", "report_date",
                "primary_document", "source_url"):
        print(f"  {key:<18} {filing[key]}")
    print("\n  planned local paths:")
    for name, path in plan["paths"].items():
        print(f"    {name:<12} {path}")
    print(f"\n  dense index : {plan['dense_index']}")
    print(f"  sparse index: {plan['sparse_index']}")
    print(f"  index ops   : {plan['index_operations']}")
    print("  sample vector ids:")
    for vector_id in plan["sample_vector_ids"]:
        print(f"    {vector_id}")


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def ingest_company(
    ticker: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    skip_sparse: bool = False,
    client: SecClient | None = None,
    ledger: Ledger | None = None,
) -> dict:
    ticker = ticker.strip().upper()
    started = time.perf_counter()

    client = client or SecClient()
    logger.info("ingest event=discover ticker=%s", ticker)
    filing = client.discover_latest_10k(ticker)
    validate_filing_metadata(filing)
    logger.info(
        "ingest event=discovered ticker=%s filing_id=%s fiscal_year=%d",
        ticker, filing.filing_id, filing.fiscal_year,
    )

    plan = build_plan(filing)

    if dry_run:
        _print_plan(plan)
        # Collision check against the existing corpus using the canonical id
        # prefix (chunk_index does not affect whether a clash is possible).
        prefix = canonical_chunk_id(
            ticker=filing.ticker, fiscal_year=filing.fiscal_year,
            filing_type=filing.filing_type,
            accession_number=filing.accession_number, chunk_index=0,
        ).rsplit("_", 1)[0]
        own_file = chunks_path(filing.ticker, filing.filing_id)
        clash = []
        if CHUNKS_DIR.exists():
            for path in CHUNKS_DIR.rglob("*_chunks.jsonl"):
                if path == own_file:
                    continue
                for row in _read_jsonl(path):
                    cid = row.get("chunk_id", "")
                    if cid.rsplit("_", 1)[0] == prefix:
                        clash.append(cid)
        if clash:
            print(f"\n  chunk_id collision check: FAIL — clashes: {clash[:5]}")
        else:
            print("\n  chunk_id collision check: PASS (no clash with existing corpus)")
        return {"status": "dry_run", "filing": filing.to_dict(), "plan": plan,
                "collision": bool(clash)}

    ledger = ledger or Ledger()
    entry = ledger.get(filing.filing_id)
    artifact_paths = _artifact_paths(filing)

    if entry is not None and not force:
        assessment = stage_graph.assess_filing(
            entry, artifact_paths, skip_sparse=skip_sparse
        )
        for note in assessment.notes:
            log_event("assess", ticker=ticker, filing_id=filing.filing_id, note=repr(note))

        if assessment.complete and assessment.servable and assessment.resume_stage is None:
            log_event(
                "already_complete", ticker=ticker, filing_id=filing.filing_id,
                missing_rebuild=(",".join(assessment.missing_rebuild_artifacts) or None),
            )
            return {
                "status": "already_ingested",
                "ticker": ticker,
                "filing_id": filing.filing_id,
                "accession_number": filing.accession_number,
                "chunk_count": entry.get("chunk_count"),
                "missing_rebuild_artifacts": assessment.missing_rebuild_artifacts,
            }

        resume_from = assessment.resume_stage or "downloaded"
        if assessment.complete and not assessment.servable:
            log_event(
                "reingest_broken", level=logging.WARNING, ticker=ticker,
                filing_id=filing.filing_id, resume_from=resume_from,
                reason="serving artifact (chunks) missing or corrupt",
            )
        else:
            log_event(
                "resume", ticker=ticker, filing_id=filing.filing_id,
                resume_from=resume_from,
                invalid_stages=(",".join(assessment.invalid_stages) or None),
            )
    else:
        if entry is None:
            ledger.start_filing(filing)
        resume_from = "downloaded"
        log_event("resume", ticker=ticker, filing_id=filing.filing_id,
                  resume_from=resume_from, force=force)

    ctx = Ctx(filing=filing, ledger=ledger, client=client, force=force,
              skip_sparse=skip_sparse)

    resume_index = STAGES.index(resume_from) if resume_from in STAGES else len(STAGES)

    for stage in STAGES:
        if stage in ("discovered", "complete"):
            continue
        if STAGES.index(stage) < resume_index:
            ctx.stage_log.append({"stage": stage, "status": "skipped_done"})
            continue
        stage_started = time.perf_counter()
        try:
            _STAGE_FUNCS[stage](ctx)
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - stage_started) * 1000
            if stage != "sparse_upserted":  # sparse handles its own soft-fail
                ledger.record_stage(
                    filing.filing_id, stage, status="failed",
                    error=str(exc), error_class=type(exc).__name__,
                    duration_ms=duration_ms,
                )
            logger.error(
                "stage event=failed ticker=%s filing_id=%s stage=%s "
                "duration_ms=%.0f error_class=%s",
                ticker, filing.filing_id, stage, duration_ms, type(exc).__name__,
            )
            raise
        duration_ms = (time.perf_counter() - stage_started) * 1000
        stage_status = ledger.get(filing.filing_id)["stages"].get(stage, {}).get("status")
        ctx.stage_log.append({"stage": stage, "status": stage_status,
                              "duration_ms": round(duration_ms, 1)})
        logger.info(
            "stage event=complete ticker=%s filing_id=%s stage=%s "
            "duration_ms=%.0f status=%s",
            ticker, filing.filing_id, stage, duration_ms, stage_status,
        )

    ledger.mark_complete(filing.filing_id)
    final = ledger.get(filing.filing_id)
    total_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "ingest event=complete ticker=%s filing_id=%s chunk_count=%s total_ms=%.0f",
        ticker, filing.filing_id, final.get("chunk_count"), total_ms,
    )
    return {
        "status": "ingested",
        "ticker": ticker,
        "filing": filing.to_dict(),
        "chunk_count": final.get("chunk_count"),
        "embedding_count": final.get("embedding_count"),
        "dense_upserted": final.get("dense_upserted"),
        "sparse_status": final["stages"].get("sparse_upserted", {}).get("status"),
        "stages": ctx.stage_log,
        "total_ms": round(total_ms, 1),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _print_summary(result: dict) -> None:
    print("\n" + "=" * 74)
    print(f"INGESTION RESULT: {result['status'].upper()}")
    print("=" * 74)
    if result["status"] in ("ingested",):
        filing = result["filing"]
        print(f"  {filing['ticker']}  {filing['company_name']}  (CIK {filing['cik']})")
        print(f"  {filing['filing_type']} FY{filing['fiscal_year']}  "
              f"accession {filing['accession_number']}")
        print(f"  filing_date {filing['filing_date']}  report_date {filing['report_date']}")
        print(f"  chunks {result['chunk_count']}  embeddings {result['embedding_count']}  "
              f"dense_upserted {result['dense_upserted']}  sparse {result['sparse_status']}")
        print(f"  total {result['total_ms']:.0f} ms")
    elif result["status"] == "already_ingested":
        print(f"  {result['ticker']} accession {result['accession_number']} "
              f"already complete ({result['chunk_count']} chunks). Use --force to rebuild.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="after ingestion, run ingestion.verify_company")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    try:
        result = ingest_company(
            args.ticker,
            dry_run=args.dry_run,
            force=args.force,
            skip_sparse=args.skip_sparse,
        )
    except (SecClientError, IngestionError) as exc:
        logger.error("ingest event=aborted ticker=%s error_class=%s: %s",
                     args.ticker, type(exc).__name__, exc)
        print(f"\nABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if not args.dry_run:
        _print_summary(result)

    if args.verify and not args.dry_run:
        from ingestion.verify_company import _print_report, verify

        report = verify(args.ticker)
        _print_report(report)
        if report["result"] != "PASS":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
