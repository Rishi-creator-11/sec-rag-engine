"""Cohere low-latency reranker for hybrid top-k candidates.

Retrieves nothing. It rescores a provided candidate list with
rerank-v4.0-fast. The API sees only the query and candidate text,
never retrieval ranks or RRF scores.

Results are sorted by rerank_score descending, with the original
hybrid order as the tie-breaker. Original hybrid fields are copied,
not mutated.
"""

from __future__ import annotations

import copy
import os
import time

import cohere
from cohere.errors import TooManyRequestsError
from dotenv import load_dotenv


load_dotenv()

MODEL = "rerank-v4.0-fast"


def rerank_enabled() -> bool:
    value = os.getenv("COHERE_RERANK_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def is_rate_limited(error: Exception) -> bool:
    if isinstance(error, TooManyRequestsError):
        return True

    status = getattr(error, "status_code", None)
    return status == 429


def get_client() -> cohere.ClientV2:
    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is missing. "
            "Set it in the environment or .env file."
        )

    return cohere.ClientV2(api_key=api_key)


def candidate_text(candidate: dict) -> str:
    return candidate.get("text") or ""


def result_index(item) -> int:
    if hasattr(item, "index"):
        return int(item.index)

    return int(item["index"])


def result_score(item) -> float:
    if hasattr(item, "relevance_score"):
        return float(item.relevance_score)

    return float(item["relevance_score"])


def validate_indices(
    indices: list[int],
    candidate_count: int,
) -> None:
    if len(indices) != candidate_count:
        raise ValueError(
            "Cohere returned "
            f"{len(indices)} results for "
            f"{candidate_count} candidates."
        )

    if len(indices) != len(set(indices)):
        raise ValueError("Cohere returned duplicate indices.")

    invalid = [
        index
        for index in indices
        if index < 0 or index >= candidate_count
    ]

    if invalid:
        raise ValueError(
            f"Cohere returned invalid indices: {invalid}"
        )

    expected = set(range(candidate_count))

    if set(indices) != expected:
        missing = sorted(expected - set(indices))
        extra = sorted(set(indices) - expected)
        raise ValueError(
            "Cohere index mismatch. "
            f"missing={missing} extra={extra}"
        )


def apply_scores(
    candidates: list[dict],
    scores_by_index: dict[int, float],
) -> list[dict]:
    ranked = []

    for hybrid_index, candidate in enumerate(candidates):
        result = copy.deepcopy(candidate)
        result["rerank_score"] = scores_by_index[hybrid_index]
        ranked.append(
            (
                scores_by_index[hybrid_index],
                hybrid_index,
                result,
            )
        )

    ranked.sort(key=lambda item: (-item[0], item[1]))
    reranked = [item[2] for item in ranked]

    if len(reranked) != len(candidates):
        raise ValueError("Reranked candidate count changed.")

    original_ids = [candidate["chunk_id"] for candidate in candidates]
    reranked_ids = [candidate["chunk_id"] for candidate in reranked]

    if set(original_ids) != set(reranked_ids):
        raise ValueError("Reranked chunk IDs do not match the input set.")

    if len(reranked_ids) != len(set(reranked_ids)):
        raise ValueError("Reranked results contain duplicate chunk IDs.")

    return reranked


def rerank_timed(
    query: str,
    candidates: list[dict],
    top_n: int | None = None,
) -> tuple[list[dict], float]:
    """Rerank candidates and return (results, cohere_latency_seconds)."""
    if not candidates:
        return [], 0.0

    documents = [candidate_text(candidate) for candidate in candidates]
    request_top_n = len(candidates) if top_n is None else top_n

    if request_top_n < 1:
        return [], 0.0

    client = get_client()
    start = time.perf_counter()
    response = client.rerank(
        model=MODEL,
        query=query,
        documents=documents,
        top_n=len(candidates),
    )
    latency = time.perf_counter() - start

    raw_results = getattr(response, "results", None)

    if raw_results is None:
        raise ValueError("Cohere response is missing results.")

    indices = [result_index(item) for item in raw_results]
    validate_indices(indices, len(candidates))

    scores_by_index = {
        result_index(item): result_score(item)
        for item in raw_results
    }
    ranked = apply_scores(candidates, scores_by_index)

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked, latency


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """Rerank hybrid candidates. Original result dicts are not mutated."""
    ranked, _latency = rerank_timed(
        query,
        candidates,
        top_n=top_n,
    )
    return ranked
