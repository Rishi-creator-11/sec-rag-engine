import os

from dotenv import load_dotenv
from pinecone import Pinecone

from retrieval.embedder import embed_text
from retrieval.filters import RetrievalFilter

load_dotenv()

INDEX_NAME = "sec-rag-engine"

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index(INDEX_NAME)


def search(
    query: str,
    top_k: int = 5,
    filters: RetrievalFilter | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:

    # `query_embedding` lets a caller (e.g. scoped_search running one hybrid
    # call per ticker) embed the query once and reuse the vector across calls.
    # When it is None the behavior is unchanged: embed here.
    vector = query_embedding if query_embedding is not None else embed_text(query)

    query_kwargs = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    # An empty / absent filter leaves the request byte-for-byte identical to
    # the pre-Phase-1B behavior (no `filter` key sent to Pinecone).
    if filters is not None and not filters.is_empty():
        query_kwargs["filter"] = filters.to_pinecone_filter()

    response = index.query(**query_kwargs)

    results = []

    for match in response.matches:
        results.append({
            "chunk_id": match.id,
            "score": match.score,
            "text": match.metadata["text"],
            "company": match.metadata["company"],
            "ticker": match.metadata["ticker"],
            "filing_type": match.metadata["filing_type"],
            "filing_date": match.metadata["filing_date"],
            "source_url": match.metadata["source_url"],
        })

    return results


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