"""Build canonical chunk records for a discovered filing.

Reuses the existing token chunker (``ingestion.chunker.chunk_text``: ~800
tokens, 120 overlap, cl100k_base — unchanged) and the Phase 1A canonical
metadata schema. Emits both ``company_name`` (canonical) and ``company``
(back-compat alias read by BM25 and dense retrieval).
"""

from __future__ import annotations

from ingestion.chunker import chunk_text
from ingestion.sec_client import DiscoveredFiling
from retrieval.metadata import ChunkMetadata, canonical_chunk_id


def build_chunk_records(clean_text: str, filing: DiscoveredFiling) -> list[dict]:
    texts = chunk_text(clean_text)
    records: list[dict] = []
    for index, text in enumerate(texts):
        chunk_id = canonical_chunk_id(
            ticker=filing.ticker,
            fiscal_year=filing.fiscal_year,
            filing_type=filing.filing_type,
            accession_number=filing.accession_number,
            chunk_index=index,
        )
        metadata = ChunkMetadata.from_parts(
            cik=filing.cik,
            accession_number=filing.accession_number,
            chunk_id=chunk_id,
            chunk_index=index,
            ticker=filing.ticker,
            filing_type=filing.filing_type,
            filing_date=filing.filing_date,
            report_date=filing.report_date,
            company_name=filing.company_name,
            source_url=filing.source_url,
            fiscal_year=filing.fiscal_year,
        )
        record = metadata.to_chunk_record(text)
        record["company"] = filing.company_name  # back-compat alias
        records.append(record)
    return records


def validate_chunk_records(records: list[dict], filing: DiscoveredFiling) -> list[str]:
    """Return a list of problems; empty means safe to embed/upsert."""
    problems: list[str] = []

    if not records:
        return ["no chunk records produced"]
    if len(records) < 5:
        problems.append(f"suspiciously few chunks: {len(records)}")

    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            problems.append(f"record {index}: missing chunk_id")
            continue
        if chunk_id in seen_ids:
            problems.append(f"duplicate chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)
        if record.get("chunk_index") != index:
            problems.append(
                f"{chunk_id}: chunk_index {record.get('chunk_index')!r} != {index}"
            )
        if not str(record.get("text", "")).strip():
            problems.append(f"{chunk_id}: empty text")
        if record.get("ticker") != filing.ticker:
            problems.append(f"{chunk_id}: ticker {record.get('ticker')!r}")
        if record.get("accession_number") != filing.accession_number:
            problems.append(f"{chunk_id}: accession mismatch")
        if record.get("filing_id") != filing.filing_id:
            problems.append(f"{chunk_id}: filing_id mismatch")
        if record.get("source_url") != filing.source_url:
            problems.append(f"{chunk_id}: source_url mismatch")

    indices = [record.get("chunk_index") for record in records]
    if indices != list(range(len(records))):
        problems.append("chunk_index is not contiguous from 0")

    return problems
