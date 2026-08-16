import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "sec-rag-engine"
VECTOR_DIMENSION = 1536
BATCH_SIZE = 50

pc = Pinecone(api_key=PINECONE_API_KEY)


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