import json
import math
from pathlib import Path

from retrieval.embedder import embed_text

EMBEDDINGS_PATH = "data/embeddings/sec_chunks.jsonl"


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(y * y for y in b)
    )

    return dot_product / (
        magnitude_a * magnitude_b
    )


def load_embedded_chunks(
    path: str = EMBEDDINGS_PATH,
) -> list[dict]:
    chunks = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            chunks.append(
                json.loads(line)
            )

    return chunks


def search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    query_embedding = embed_text(query)

    chunks = load_embedded_chunks()

    results = []

    for chunk in chunks:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        results.append({
            "score": score,
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "company": chunk["company"],
            "ticker": chunk["ticker"],
            "filing_type": chunk["filing_type"],
            "filing_date": chunk["filing_date"],
            "source_url": chunk["source_url"],
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]


if __name__ == "__main__":
    query = input("Enter your SEC question: ")

    results = search(
        query,
        top_k=5,
    )

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Company: {result['company']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Source: {result['source_url']}")
        print()
        print(result["text"][:1200])
        print()