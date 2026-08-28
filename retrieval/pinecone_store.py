import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

logger = logging.getLogger("ingestion.dense_store")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "sec-rag-engine"
DEFAULT_NAMESPACE = "__default__"
VECTOR_DIMENSION = 1536
BATCH_SIZE = 50
UPSERT_MAX_ATTEMPTS = 4
# Read-after-write poll schedule for the post-upsert verification (Pinecone
# serverless is eventually consistent). Patchable in tests.
VERIFY_POLL_WAITS = (1.0, 2.0, 4.0, 8.0, 15.0, 15.0)

pc = Pinecone(api_key=PINECONE_API_KEY)

# Canonical dense metadata keys. "company" is kept as a back-compat alias for
# "company_name" because retrieval/pinecone_search.py (and BM25) read "company".
_DENSE_STR_FIELDS = (
    "ticker",
    "cik",
    "filing_type",
    "accession_number",
    "filing_id",
    "filing_date",
    "report_date",
    "source_url",
)


def build_dense_metadata(record: dict) -> dict:
    company_name = record.get("company_name") or record.get("company")
    metadata = {
        "text": record["text"],
        "company": record.get("company") or company_name,
        "company_name": company_name,
        "fiscal_year": int(record["fiscal_year"]),
        "chunk_index": int(record["chunk_index"]),
    }
    for key in _DENSE_STR_FIELDS:
        metadata[key] = str(record[key])
    return metadata


def _iter_batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _upsert_with_retry(index, vectors: list[dict], namespace: str) -> None:
    for attempt in range(1, UPSERT_MAX_ATTEMPTS + 1):
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            return
        except Exception as exc:  # noqa: BLE001 - Pinecone raises varied types
            if attempt == UPSERT_MAX_ATTEMPTS:
                raise
            wait = min(2 ** attempt, 15)
            logger.warning(
                "dense_upsert event=retry attempt=%d/%d wait=%ds error_class=%s",
                attempt, UPSERT_MAX_ATTEMPTS, wait, type(exc).__name__,
            )
            time.sleep(wait)


def upsert_filing(
    records: list[dict],
    *,
    index_name: str = INDEX_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    batch_size: int = BATCH_SIZE,
    verify: bool = True,
) -> dict:
    """Upsert one filing's embedded chunk records into the existing dense index.

    - vector id == record["chunk_id"] (canonical, deterministic) so a rerun
      overwrites the same vectors instead of creating duplicates
    - the index is never created or recreated here
    - when ``verify`` is set, every chunk_id is fetched back and confirmed
    """
    if not records:
        return {"upserted": 0, "verified": 0, "missing": []}

    for record in records:
        embedding = record.get("embedding")
        if not embedding or len(embedding) != VECTOR_DIMENSION:
            raise ValueError(
                f"{record.get('chunk_id')}: embedding dim "
                f"{len(embedding) if embedding else 0} != {VECTOR_DIMENSION}"
            )

    index = pc.Index(index_name)
    upserted = 0
    for batch in _iter_batches(records, batch_size):
        vectors = [
            {
                "id": record["chunk_id"],
                "values": record["embedding"],
                "metadata": build_dense_metadata(record),
            }
            for record in batch
        ]
        _upsert_with_retry(index, vectors, namespace)
        upserted += len(vectors)
        logger.info(
            "dense_upsert event=batch index=%s namespace=%s count=%d total=%d",
            index_name, namespace, len(vectors), upserted,
        )

    result = {"upserted": upserted, "verified": 0, "missing": []}
    if verify:
        ids = [record["chunk_id"] for record in records]
        # Pinecone serverless is eventually consistent on read-after-write, so
        # poll with backoff rather than declaring vectors missing after one
        # short sleep. Total wait is bounded (~1+2+4+8+15+15 ≈ 45s worst case).
        waits = VERIFY_POLL_WAITS
        found: set[str] = set()
        missing = list(ids)
        for attempt, wait in enumerate(waits, start=1):
            time.sleep(wait)
            pending = [chunk_id for chunk_id in missing if chunk_id not in found]
            for batch in _iter_batches(pending, 100):
                fetched = index.fetch(ids=batch, namespace=namespace)
                found.update((fetched.vectors or {}).keys())
            missing = [chunk_id for chunk_id in ids if chunk_id not in found]
            if not missing:
                break
            logger.info(
                "dense_upsert event=verify_poll attempt=%d/%d found=%d/%d",
                attempt, len(waits), len(found), len(ids),
            )
        result["verified"] = len(found)
        result["missing"] = missing
        if missing:
            raise RuntimeError(
                f"dense upsert verification failed: {len(missing)} vectors missing "
                f"after {sum(waits):.0f}s of polling (e.g. {missing[:3]})"
            )
    return result


def create_index() -> None:
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Creating Pinecone index...")

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

        print("Index ready.")
    else:
        print("Index already exists.")


def load_embedded_chunks(
    path: str = "data/embeddings/sec_chunks.jsonl",
) -> list[dict]:
    chunks = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def upload_chunks(chunks: list[dict]) -> None:
    index = pc.Index(INDEX_NAME)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        vectors = []

        for chunk in batch:
            vectors.append({
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "company": chunk["company"],
                    "ticker": chunk["ticker"],
                    "filing_type": chunk["filing_type"],
                    "filing_date": chunk["filing_date"],
                    "source_url": chunk["source_url"],
                },
            })

        index.upsert(vectors=vectors)

        print(
            f"Uploaded "
            f"{min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)} vectors"
        )


if __name__ == "__main__":
    create_index()

    chunks = load_embedded_chunks()

    print(f"Loaded {len(chunks)} embedded chunks")

    upload_chunks(chunks)

    print("Finished uploading vectors.")

    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()

    print(stats)