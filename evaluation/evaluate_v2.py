"""Evaluate Dense, BM25, Sparse, and Hybrid on Benchmark v2.

Loads evaluation/benchmark_v2.json and retrieves top 10 for every
question with no company/ticker filter. Hybrid uses the existing
search() defaults except top_k=10. Weights are not tuned.

Unsupported questions are excluded from Recall, Precision, MRR, and
Numeric Evidence. They are used only for false-positive reporting.
"""

from collections.abc import Callable
import json
import statistics
import time
from pathlib import Path

from retrieval.bm25_search import search as bm25_search
from retrieval.hybrid_search import search as hybrid_search
from retrieval.pinecone_search import search as dense_search
from retrieval.sparse_search import search as sparse_search


BENCHMARK_PATH = Path("evaluation/benchmark_v2.json")
JSON_PATH = Path(
    "evaluation/results/v2_retrieval_evaluation.json"
)
SUMMARY_PATH = Path(
    "evaluation/results/v2_retrieval_summary.txt"
)

TOP_K = 10
EXPECTED_QUESTIONS = 60

RETRIEVERS: dict[str, Callable[[str], list[dict]]] = {
    "dense": lambda query: dense_search(query, top_k=TOP_K),
    "bm25": lambda query: bm25_search(query, top_k=TOP_K),
    "sparse": lambda query: sparse_search(query, top_k=TOP_K),
    "hybrid": lambda query: hybrid_search(query, top_k=TOP_K),
}


def load_benchmark() -> list[dict]:
    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    if len(benchmark) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} questions, "
            f"got {len(benchmark)}"
        )

    return benchmark


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float | None:
    if not relevant_ids:
        return None

    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    hits = sum(chunk_id in relevant_ids for chunk_id in retrieved)
    return hits / k


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> tuple[float, int | None]:
    if not relevant_ids:
        return 0.0, None

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank, rank

    return 0.0, None


def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    return float(bool(set(retrieved_ids[:k]) & relevant_ids))


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = int(round(percentile_value * (len(ordered) - 1)))
    return ordered[index]


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None

    return statistics.mean(values)


def top_score(results: list[dict]) -> float | None:
    """Use the retriever's own top-hit score field if present."""
    if not results:
        return None

    result = results[0]

    if "score" in result and result["score"] is not None:
        return float(result["score"])

    if "rrf_score" in result and result["rrf_score"] is not None:
        return float(result["rrf_score"])

    return None


def is_unsupported(item: dict) -> bool:
    return (
        item.get("answer_type") == "unsupported"
        or len(item.get("relevant_chunks", [])) == 0
    )


def is_numeric(item: dict) -> bool:
    return (
        item.get("answer_type") == "numeric"
        and len(item.get("relevant_chunks", [])) > 0
    )


def evaluate_retriever(
    name: str,
    search_fn: Callable[[str], list[dict]],
    benchmark: list[dict],
) -> dict:
    print(f"Evaluating {name}...")

    questions = []

    for item in benchmark:
        relevant_ids = set(item.get("relevant_chunks", []))
        start = time.perf_counter()
        results = search_fn(item["question"])
        latency = time.perf_counter() - start

        retrieved_ids = [
            result["chunk_id"] for result in results[:TOP_K]
        ]
        rr, first_relevant_rank = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        questions.append(
            {
                "id": item["id"],
                "answer_type": item.get("answer_type"),
                "relevant_chunks": list(item.get("relevant_chunks", [])),
                "retrieved_chunk_ids_top10": retrieved_ids,
                "first_relevant_rank": first_relevant_rank,
                "recall_at_1": recall_at_k(retrieved_ids, relevant_ids, 1),
                "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, 3),
                "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
                "recall_at_10": recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    10,
                ),
                "precision_at_1": precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    1,
                ),
                "precision_at_3": precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    3,
                ),
                "precision_at_5": precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    5,
                ),
                "reciprocal_rank": rr,
                "latency_seconds": latency,
                "top_score": top_score(results),
                "unsupported": is_unsupported(item),
                "numeric": is_numeric(item),
            }
        )

        print(
            f"  {item['id']}: "
            f"first_relevant_rank={first_relevant_rank} "
            f"latency={latency:.3f}s"
        )

    eligible = [row for row in questions if not row["unsupported"]]
    numeric = [row for row in questions if row["numeric"]]
    unsupported = [row for row in questions if row["unsupported"]]
    latencies = [row["latency_seconds"] for row in questions]
    unsupported_scores = [
        row["top_score"]
        for row in unsupported
        if row["top_score"] is not None
    ]

    summary = {
        "retriever": name,
        "questions_evaluated": len(questions),
        "eligible_questions": len(eligible),
        "numeric_questions": len(numeric),
        "unsupported_questions_evaluated": len(unsupported),
        "recall_at_1": mean_or_none(
            [row["recall_at_1"] for row in eligible]
        ),
        "recall_at_3": mean_or_none(
            [row["recall_at_3"] for row in eligible]
        ),
        "recall_at_5": mean_or_none(
            [row["recall_at_5"] for row in eligible]
        ),
        "recall_at_10": mean_or_none(
            [row["recall_at_10"] for row in eligible]
        ),
        "precision_at_1": mean_or_none(
            [row["precision_at_1"] for row in eligible]
        ),
        "precision_at_3": mean_or_none(
            [row["precision_at_3"] for row in eligible]
        ),
        "precision_at_5": mean_or_none(
            [row["precision_at_5"] for row in eligible]
        ),
        "mrr": mean_or_none(
            [row["reciprocal_rank"] for row in eligible]
        ),
        "numeric_evidence_hit_at_5": mean_or_none(
            [
                hit_at_k(
                    row["retrieved_chunk_ids_top10"],
                    set(row["relevant_chunks"]),
                    5,
                )
                for row in numeric
            ]
        ),
        "numeric_evidence_hit_at_10": mean_or_none(
            [
                hit_at_k(
                    row["retrieved_chunk_ids_top10"],
                    set(row["relevant_chunks"]),
                    10,
                )
                for row in numeric
            ]
        ),
        "unsupported_false_positive_at_5": mean_or_none(
            [
                float(bool(row["retrieved_chunk_ids_top10"][:5]))
                for row in unsupported
            ]
        ),
        "unsupported_false_positive_at_10": mean_or_none(
            [
                float(bool(row["retrieved_chunk_ids_top10"][:10]))
                for row in unsupported
            ]
        ),
        "unsupported_average_top_score": mean_or_none(unsupported_scores),
        "mean_latency_seconds": mean_or_none(latencies),
        "median_latency_seconds": (
            statistics.median(latencies) if latencies else None
        ),
        "p95_latency_seconds": percentile(latencies, 0.95),
    }

    return {
        "summary": summary,
        "questions": questions,
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "   n/a"

    return f"{value:6.3f}"


def format_ms(value: float | None) -> str:
    if value is None:
        return "   n/a"

    return f"{value * 1000:7.1f}"


def build_table(summaries: list[dict]) -> str:
    headers = [
        f"{'retriever':<10}",
        f"{'R@1':>6}",
        f"{'R@3':>6}",
        f"{'R@5':>6}",
        f"{'R@10':>6}",
        f"{'P@1':>6}",
        f"{'P@3':>6}",
        f"{'P@5':>6}",
        f"{'MRR':>6}",
        f"{'NE@5':>6}",
        f"{'NE@10':>6}",
        f"{'UFP@5':>6}",
        f"{'UFP@10':>6}",
        f"{'mean ms':>8}",
        f"{'med ms':>8}",
        f"{'p95 ms':>8}",
    ]
    lines = [
        "BENCHMARK V2 RETRIEVAL COMPARISON",
        "Top-10, no company filter. Unsupported excluded from R/P/MRR/NE.",
        "NE = Numeric Evidence Hit. UFP = Unsupported false-positive.",
        "",
        " ".join(headers),
        "-" * 130,
    ]

    for summary in summaries:
        lines.append(
            " ".join(
                [
                    f"{summary['retriever']:<10}",
                    format_metric(summary["recall_at_1"]),
                    format_metric(summary["recall_at_3"]),
                    format_metric(summary["recall_at_5"]),
                    format_metric(summary["recall_at_10"]),
                    format_metric(summary["precision_at_1"]),
                    format_metric(summary["precision_at_3"]),
                    format_metric(summary["precision_at_5"]),
                    format_metric(summary["mrr"]),
                    format_metric(summary["numeric_evidence_hit_at_5"]),
                    format_metric(summary["numeric_evidence_hit_at_10"]),
                    format_metric(
                        summary["unsupported_false_positive_at_5"]
                    ),
                    format_metric(
                        summary["unsupported_false_positive_at_10"]
                    ),
                    format_ms(summary["mean_latency_seconds"]),
                    format_ms(summary["median_latency_seconds"]),
                    format_ms(summary["p95_latency_seconds"]),
                ]
            )
        )

    lines.append("")
    if summaries:
        first = summaries[0]
        lines.append(
            f"eligible questions: {first['eligible_questions']}"
        )
        lines.append(
            f"numeric questions: {first['numeric_questions']}"
        )
        lines.append(
            "unsupported questions evaluated: "
            f"{first['unsupported_questions_evaluated']}"
        )
        lines.append(
            "unsupported average top score is reported per retriever "
            "in the JSON using that retriever's own score field."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    benchmark = load_benchmark()
    retriever_results = {}
    summaries = []

    print(f"Evaluating {len(benchmark)} questions.\n")

    for name, search_fn in RETRIEVERS.items():
        result = evaluate_retriever(name, search_fn, benchmark)
        retriever_results[name] = result
        summaries.append(result["summary"])
        print()

    output = {
        "benchmark_path": str(BENCHMARK_PATH),
        "question_count": len(benchmark),
        "top_k": TOP_K,
        "retrievers": retriever_results,
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    table = build_table(summaries)

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(table)

    print()
    print(table)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
