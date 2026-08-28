from retrieval.pinecone_search import search as dense_search
from retrieval.bm25_search import search as bm25_search
from retrieval.sparse_search import search as sparse_search
from retrieval.filters import RetrievalFilter


RRF_K = 60


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict]],
    top_k: int = 5,
    rrf_k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    if weights is None:
        weights = {
            name: 1.0
            for name in rankings
        }

    fused: dict[str, dict] = {}

    for retriever_name, results in rankings.items():
        weight = weights.get(
            retriever_name,
            1.0,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieved_by": [],
                    "ranks": {},
                }

            fused[chunk_id]["rrf_score"] += (
                weight / (rrf_k + rank)
            )

            fused[chunk_id]["retrieved_by"].append(
                retriever_name
            )

            fused[chunk_id]["ranks"][
                retriever_name
            ] = rank

    ranked = sorted(
        fused.values(),
        key=lambda result: (
            -result["rrf_score"],
            result["chunk_id"],
        ),
    )

    return ranked[:top_k]


def search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    use_bm25: bool = True,
    use_sparse: bool = False,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    sparse_weight: float = 1.0,
    filters: RetrievalFilter | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    rankings = {}
    weights = {}

    scope = filters if (filters is not None and not filters.is_empty()) else None

    dense_results = dense_search(
        query,
        top_k=candidate_k,
        filters=scope,
        query_embedding=query_embedding,
    )

    rankings["dense"] = dense_results
    weights["dense"] = dense_weight

    if use_bm25:
        rankings["bm25"] = bm25_search(
            query,
            top_k=candidate_k,
            filters=scope,
        )
        weights["bm25"] = bm25_weight

    if use_sparse:
        # Sparse retrieval stays disabled by default (use_sparse=False) and is
        # not filter-aware yet; scoped sparse retrieval is a later phase.
        rankings["sparse"] = sparse_search(
            query,
            top_k=candidate_k,
        )
        weights["sparse"] = sparse_weight

    return reciprocal_rank_fusion(
        rankings,
        top_k=top_k,
        weights=weights,
    )


if __name__ == "__main__":
    query = input(
        "Enter your SEC question: "
    )

    results = search(
        query,
        top_k=5,
        candidate_k=10,
        use_bm25=True,
        use_sparse=False,
    )

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Company: {result['company']}")
        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )
        print(
            f"Retrieved by: "
            f"{result['retrieved_by']}"
        )
        print(
            f"Ranks: "
            f"{result['ranks']}"
        )
        print()
        print(result["text"][:1200])
        print()