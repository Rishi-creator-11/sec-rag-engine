import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logger = logging.getLogger("ingestion.embeddings")

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50


def load_chunks(input_dir: str) -> list[dict]:
    chunks = []

    for path in Path(input_dir).glob("*_chunks.jsonl"):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                chunks.append(json.loads(line))

    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]


def embed_records(
    records: list[dict],
    *,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Return copies of ``records`` with an ``embedding`` field added.

    Reuses ``embed_batch`` (same model, ``text-embedding-3-small``). Input
    records are not mutated. Order is preserved.
    """
    embedded: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        vectors = embed_batch([record["text"] for record in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding count {len(vectors)} != batch size {len(batch)}"
            )
        for record, vector in zip(batch, vectors):
            if len(vector) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"{record.get('chunk_id')}: embedding dim {len(vector)} "
                    f"!= {EMBEDDING_DIMENSION}"
                )
            new_record = dict(record)
            new_record["embedding"] = vector
            embedded.append(new_record)
        logger.info(
            "embeddings event=batch done=%d/%d",
            len(embedded), len(records),
        )
    return embedded


def add_embeddings(chunks: list[dict]) -> list[dict]:
    embedded_chunks = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        embeddings = embed_batch(texts)

        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
            embedded_chunks.append(chunk)

        print(
            f"Embedded "
            f"{min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    return embedded_chunks


def save_embeddings(
    chunks: list[dict],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk) + "\n"
            )


if __name__ == "__main__":
    chunks = load_chunks("data/chunks")

    print(f"Loaded {len(chunks)} chunks")

    embedded_chunks = add_embeddings(chunks)

    save_embeddings(
        embedded_chunks,
        "data/embeddings/sec_chunks.jsonl",
    )

    print(
        f"Saved {len(embedded_chunks)} embedded chunks"
    )