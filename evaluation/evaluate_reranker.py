"""Compare hybrid top-10 vs hybrid top-10 + OpenAI reranker.

Uses evaluation/benchmark_v2.json. Hybrid candidates are retrieved once
per question and reused for both systems, so Recall@10 stays identical.

Unsupported questions with empty relevant_chunks are excluded from
Recall, Precision, MRR, and Numeric Evidence.
"""

import json
import statistics
import time
from pathlib import Path

from retrieval.hybrid_search import search as hybrid_search
from retrieval.reranker import rerank

from evaluation.evaluate_v2 import (
    hit_at_k,
    is_numeric,
    is_unsupported,
    load_benchmark,
    mean_or_none,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


BENCHMARK_PATH = Path("evaluation/benchmark_v2.json")
JSON_PATH = Path("evaluation/results/v2_reranker_evaluation.json")
SUMMARY_PATH = Path("evaluation/results/v2_reranker_summary.txt")

TOP_K = 10

METRICS = [
    ("recall_at_1", "R@1"),
    ("recall_at_3", "R@3"),
    ("recall_at_5", "R@5"),
    ("recall_at_10", "R@10"),
    ("precision_at_1", "P@1"),
    ("precision_at_3", "P@3"),
    ("precision_at_5", "P@5"),
    ("mrr", "MRR"),
    ("numeric_evidence_hit_at_5", "NE@5"),
    ("numeric_evidence_hit_at_10", "NE@10"),
]


def score_ranking(
    item: dict,
    results: list[dict],
    latency_seconds: float,
) -> dict:
    relevant_ids = set(item.get("relevant_chunks", []))
    retrieved_ids = [result["chunk_id"] for result in results[:TOP_K]]
    rr, first_relevant_rank = reciprocal_rank(retrieved_ids, relevant_ids)

    return {
        "id": item["id"],
        "answer_type": item.get("answer_type"),
        "relevant_chunks": list(item.get("relevant_chunks", [])),
        "retrieved_chunk_ids_top10": retrieved_ids,
        "first_relevant_rank": first_relevant_rank,
        "recall_at_1": recall_at_k(retrieved_ids, relevant_ids, 1),
        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "precision_at_1": precision_at_k(retrieved_ids, relevant_ids, 1),
        "precision_at_3": precision_at_k(retrieved_ids, relevant_ids, 3),
        "precision_at_5": precision_at_k(retrieved_ids, relevant_ids, 5),
        "reciprocal_rank": rr,
        "latency_seconds": latency_seconds,
        "unsupported": is_unsupported(item),
        "numeric": is_numeric(item),
    }


def summarize(name: str, questions: list[dict]) -> dict:
    eligible = [row for row in questions if not row["unsupported"]]
    numeric = [row for row in questions if row["numeric"]]
    latencies = [row["latency_seconds"] for row in questions]

    return {
        "system": name,
        "questions_evaluated": len(questions),
        "eligible_questions": len(eligible),
        "numeric_questions": len(numeric),
        "recall_at_1": mean_or_none([row["recall_at_1"] for row in eligible]),
        "recall_at_3": mean_or_none([row["recall_at_3"] for row in eligible]),
        "recall_at_5": mean_or_none([row["recall_at_5"] for row in eligible]),
        "recall_at_10": mean_or_none([row["recall_at_10"] for row in eligible]),
        "precision_at_1": mean_or_none(
            [row["precision_at_1"] for row in eligible]
        ),
        "precision_at_3": mean_or_none(
            [row["precision_at_3"] for row in eligible]
        ),
        "precision_at_5": mean_or_none(
            [row["precision_at_5"] for row in eligible]
        ),
        "mrr": mean_or_none([row["reciprocal_rank"] for row in eligible]),
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
        "mean_latency_seconds": mean_or_none(latencies),
        "median_latency_seconds": (
            statistics.median(latencies) if latencies else None
        ),
        "p95_latency_seconds": percentile(latencies, 0.95),
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "    n/a"

    return f"{value:8.3f}"


def improvement(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None

    return after - before


def build_table(hybrid: dict, reranked: dict) -> str:
    lines = [
        "BENCHMARK V2 RERANKER COMPARISON",
        "Same hybrid top-10 candidates, then gpt-5-mini rerank.",
        "Unsupported questions excluded from R/P/MRR/NE.",
        "Recall@10 should stay the same because the candidate set is unchanged.",
        "",
        f"{'metric':<10} {'Hybrid':>10} {'Hybrid+Rerank':>14} {'delta':>10}",
        "-" * 48,
    ]

    for key, label in METRICS:
        before = hybrid[key]
        after = reranked[key]
        delta = improvement(after, before)
        lines.append(
            f"{label:<10} "
            f"{format_metric(before)} "
            f"{format_metric(after)} "
            f"{format_metric(delta)}"
        )

    lines.extend(
        [
            "",
            "Reranker latency",
            f"  mean:   {format_metric(reranked['mean_latency_seconds'])} s",
            f"  median: {format_metric(reranked['median_latency_seconds'])} s",
            f"  p95:    {format_metric(reranked['p95_latency_seconds'])} s",
            "",
            f"eligible questions: {hybrid['eligible_questions']}",
            f"numeric questions: {hybrid['numeric_questions']}",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    benchmark = load_benchmark()
    hybrid_questions = []
    rerank_questions = []

    print(f"Evaluating reranker on {len(benchmark)} questions.\n")

    for item in benchmark:
        question_id = item["id"]
        question = item["question"]

        hybrid_start = time.perf_counter()
        hybrid_results = hybrid_search(question, top_k=TOP_K)
        hybrid_latency = time.perf_counter() - hybrid_start

        rerank_start = time.perf_counter()
        reranked_results = rerank(
            question,
            hybrid_results,
            question_id=question_id,
            answer_type=item.get("answer_type"),
        )
        rerank_latency = time.perf_counter() - rerank_start

        hybrid_questions.append(
            score_ranking(item, hybrid_results, hybrid_latency)
        )
        rerank_questions.append(
            score_ranking(item, reranked_results, rerank_latency)
        )

        print(
            f"{question_id}: "
            f"hybrid_first={hybrid_questions[-1]['first_relevant_rank']} "
            f"rerank_first={rerank_questions[-1]['first_relevant_rank']} "
            f"rerank_latency={rerank_latency:.3f}s"
        )

    hybrid_summary = summarize("hybrid", hybrid_questions)
    rerank_summary = summarize("hybrid_rerank", rerank_questions)

    output = {
        "benchmark_path": str(BENCHMARK_PATH),
        "question_count": len(benchmark),
        "top_k": TOP_K,
        "systems": {
            "hybrid": {
                "summary": hybrid_summary,
                "questions": hybrid_questions,
            },
            "hybrid_rerank": {
                "summary": rerank_summary,
                "questions": rerank_questions,
            },
        },
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    table = build_table(hybrid_summary, rerank_summary)

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(table)

    print()
    print(table)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
