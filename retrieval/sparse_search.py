import os

from dotenv import load_dotenv
from pinecone import Pinecone


INDEX_NAME = "sec-rag-sparse"
NAMESPACE = "__default__"


load_dotenv()

# Lazy for the same reason as retrieval/pinecone_search.py: pc.Index()
# resolves the index host with a real network call, so building it at import
# time made every module that transitively imports this one require a live
# Pinecone credential just to collect, independent of whether sparse search
# is ever actually called (it's currently unused by the production hybrid
# path -- see retrieval/hybrid_search.py).
pc: Pinecone | None = None
index = None


def _ensure_index():
    global pc, index
    if index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(INDEX_NAME)
    return index


def search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    response = _ensure_index().search(
        namespace=NAMESPACE,
        query={
            "inputs": {
                "text": query,
            },
            "top_k": top_k,
        },
        fields=[
            "chunk_text",
            "chunk_id",
            "company",
            "ticker",
            "filing_type",
            "filing_date",
            "source_url",
        ],
    )

    results = []

    for hit in response.result.hits:
        fields = hit.fields

        results.append({
            "chunk_id": fields["chunk_id"],
            "score": hit.score,
            "text": fields["chunk_text"],
            "company": fields["company"],
            "ticker": fields["ticker"],
            "filing_type": fields["filing_type"],
            "filing_date": fields["filing_date"],
            "source_url": fields["source_url"],
        })

    return results


if __name__ == "__main__":
    query = input(
        "Enter your SEC question: "
    )

    results = search(
        query,
        top_k=5,
    )

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(
            f"Company: "
            f"{result['company']}"
        )
        print(
            f"Score: "
            f"{result['score']:.4f}"
        )
        print(
            f"Chunk: "
            f"{result['chunk_id']}"
        )
        print(
            f"Source: "
            f"{result['source_url']}"
        )
        print()
        print(
            result["text"][:1200]
        )
        print()