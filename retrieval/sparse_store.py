import argparse
import logging
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone

from retrieval.bm25_search import load_chunks


logger = logging.getLogger("ingestion.sparse_store")

INDEX_NAME = "sec-rag-sparse"
NAMESPACE = "__default__"
CANARY_ID = "__ingest_sparse_canary__"


def get_client() -> Pinecone:
    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing. Add it to your .env file."
        )

    return Pinecone(
        api_key=api_key
    )


def create_sparse_index() -> None:
    pc = get_client()

    if pc.has_index(INDEX_NAME):
        print(
            f"Index '{INDEX_NAME}' already exists."
        )
        return

    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "pinecone-sparse-english-v0",
            "field_map": {
                "text": "chunk_text",
            },
            "read_parameters": {
                "max_tokens_per_sequence": 2048,
            },
            "write_parameters": {
                "max_tokens_per_sequence": 2048,
            },
        },
    )

    print(
        f"Created sparse index '{INDEX_NAME}'."
    )


def chunk_to_record(
    chunk: dict,
) -> dict:
    return {
        "_id": chunk["chunk_id"],
        "chunk_text": chunk["text"],
        "chunk_id": chunk["chunk_id"],
        "company": chunk["company"],
        "ticker": chunk["ticker"],
        "filing_type": chunk["filing_type"],
        "filing_date": chunk["filing_date"],
        "source_url": chunk["source_url"],
    }


# --------------------------------------------------------------------------- #
# Phase 3: per-filing sparse ingestion.
#
# Sparse is NOT used in production retrieval. These helpers keep the sparse
# index in data parity with dense/BM25 when it is safe to do so, and fail
# soft (the caller marks the stage skipped/failed) rather than blocking a
# filing's dense + BM25 ingestion.
# --------------------------------------------------------------------------- #
_SPARSE_EXTRA_FIELDS = (
    "company_name",
    "cik",
    "fiscal_year",
    "accession_number",
    "filing_id",
    "report_date",
    "chunk_index",
)


def record_to_sparse(record: dict) -> dict:
    company_name = record.get("company_name") or record.get("company")
    payload = {
        "_id": record["chunk_id"],
        "chunk_text": record["text"],
        "chunk_id": record["chunk_id"],
        "company": record.get("company") or company_name,
        "ticker": record["ticker"],
        "filing_type": record["filing_type"],
        "filing_date": record["filing_date"],
        "source_url": record["source_url"],
    }
    for key in _SPARSE_EXTRA_FIELDS:
        if key in record and record[key] is not None:
            payload[key] = record[key]
    return payload


def canary_check(index_name: str = INDEX_NAME) -> dict:
    """Insert one synthetic record, read it back, then remove it.

    Proves the integrated sparse index still accepts a canonical ``_id`` and
    preserves metadata WITHOUT touching any real vector. The only delete this
    project performs is of this synthetic ``__ingest_sparse_canary__`` id.
    """
    index = get_client().Index(index_name)
    probe = {
        "_id": CANARY_ID,
        "chunk_text": "ingestion pipeline sparse canary probe document",
        "chunk_id": CANARY_ID,
        "company": "CANARY",
        "ticker": "CNRY",
        "filing_type": "10-K",
        "filing_date": "2000-01-01",
        "source_url": "https://www.sec.gov/canary",
        "canary": True,
    }
    result = {"insert_ok": False, "fetch_ok": False, "metadata_preserved": False,
              "cleaned_up": False, "error": None}
    try:
        index.upsert_records(namespace=NAMESPACE, records=[probe])
        result["insert_ok"] = True
        time.sleep(1.0)
        fetched = index.fetch(ids=[CANARY_ID], namespace=NAMESPACE)
        vector = (fetched.vectors or {}).get(CANARY_ID)
        result["fetch_ok"] = vector is not None
        if vector is not None:
            result["metadata_preserved"] = (
                (vector.metadata or {}).get("ticker") == "CNRY"
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            index.delete(ids=[CANARY_ID], namespace=NAMESPACE)
            result["cleaned_up"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = (result["error"] or "") + f" cleanup:{type(exc).__name__}"
    result["ok"] = bool(
        result["insert_ok"] and result["fetch_ok"] and result["metadata_preserved"]
    )
    return result


def upsert_filing_sparse(
    records: list[dict],
    *,
    index_name: str = INDEX_NAME,
    batch_size: int = 50,
) -> dict:
    if not records:
        return {"upserted": 0}
    index = get_client().Index(index_name)
    upserted = 0
    for start in range(0, len(records), batch_size):
        batch = [record_to_sparse(r) for r in records[start:start + batch_size]]
        index.upsert_records(namespace=NAMESPACE, records=batch)
        upserted += len(batch)
        logger.info("sparse_upsert event=batch count=%d total=%d", len(batch), upserted)
    return {"upserted": upserted}


def inspect_one_record() -> None:
    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks were loaded."
        )

    first_chunk = chunks[0]

    record = chunk_to_record(
        first_chunk
    )

    print("=" * 80)
    print("ORIGINAL CHUNK")
    print("=" * 80)

    print(
        f"chunk_id: "
        f"{first_chunk['chunk_id']}"
    )

    print(
        f"company: "
        f"{first_chunk['company']}"
    )

    print(
        f"ticker: "
        f"{first_chunk['ticker']}"
    )

    print(
        f"filing_type: "
        f"{first_chunk['filing_type']}"
    )

    print(
        f"filing_date: "
        f"{first_chunk['filing_date']}"
    )

    print(
        f"source_url: "
        f"{first_chunk['source_url']}"
    )

    print()

    print(
        first_chunk["text"][:1000]
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PINECONE SPARSE RECORD"
    )

    print("=" * 80)

    print(
        f"_id: {record['_id']}"
    )

    print(
        f"chunk_id: "
        f"{record['chunk_id']}"
    )

    print(
        f"company: "
        f"{record['company']}"
    )

    print(
        f"ticker: "
        f"{record['ticker']}"
    )

    print(
        f"filing_type: "
        f"{record['filing_type']}"
    )

    print(
        f"filing_date: "
        f"{record['filing_date']}"
    )

    print(
        f"source_url: "
        f"{record['source_url']}"
    )

    print()

    print(
        "chunk_text preview:"
    )

    print(
        record["chunk_text"][:1000]
    )


def upsert_one_record() -> None:
    pc = get_client()

    if not pc.has_index(INDEX_NAME):
        raise RuntimeError(
            f"Index '{INDEX_NAME}' does not exist. "
            "Run the create command first."
        )

    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks were loaded."
        )

    record = chunk_to_record(
        chunks[0]
    )

    index = pc.Index(
        INDEX_NAME
    )

    index.upsert_records(
        namespace=NAMESPACE,
        records=[
            record
        ],
    )

    print(
        f"Upserted one record into "
        f"'{INDEX_NAME}': "
        f"{record['_id']}"
    )


def upsert_all_records(
    batch_size: int = 50,
) -> None:
    pc = get_client()

    if not pc.has_index(INDEX_NAME):
        raise RuntimeError(
            f"Index '{INDEX_NAME}' does not exist. "
            "Run the create command first."
        )

    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks were loaded."
        )

    records = [
        chunk_to_record(chunk)
        for chunk in chunks
    ]

    index = pc.Index(
        INDEX_NAME
    )

    total = len(records)

    print(
        f"Uploading {total} records "
        f"in batches of {batch_size}..."
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        batch = records[
            start:end
        ]

        index.upsert_records(
            namespace=NAMESPACE,
            records=batch,
        )

        print(
            f"Uploaded {end}/{total}"
        )

    print(
        f"Finished uploading "
        f"{total} records."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create and manage "
            "the Pinecone sparse index."
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "create",
            "inspect",
            "upsert-one",
            "upsert-all",
        ],
        help="Action to run.",
    )

    args = parser.parse_args()

    if args.command == "create":
        create_sparse_index()

    elif args.command == "inspect":
        inspect_one_record()

    elif args.command == "upsert-one":
        upsert_one_record()

    elif args.command == "upsert-all":
        upsert_all_records()


if __name__ == "__main__":
    main()