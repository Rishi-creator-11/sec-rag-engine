"""Compare hybrid top-10 vs hybrid top-10 + Cohere rerank-v4.0-fast.

Uses evaluation/benchmark_v2.json. Hybrid candidates are retrieved once
per question and reused, so Recall@10 should stay identical.

Unsupported questions with empty relevant_chunks are excluded from
Recall, Precision, MRR, and Numeric Evidence.
"""

from copy import deepcopy
import json
import statistics
import time
from pathlib import Path

from cohere.errors import TooManyRequestsError

from retrieval.cohere_reranker import rerank_timed
from retrieval.hybrid_search import search as hybrid_search

from evaluation.evaluate_reranker import (
    METRICS,
    format_metric,
    improvement,
    score_ranking,
    summarize,
)
from evaluation.evaluate_v2 import (
    is_numeric,
    is_unsupported,
    load_benchmark,
    mean_or_none,
    percentile,
)


BENCHMARK_PATH = Path("evaluation/benchmark_v2.json")
GPT_RESULTS_PATH = Path(
    "evaluation/results/v2_reranker_evaluation.json"
)
CACHE_PATH = Path(
    "evaluation/results/cohere_reranker_cache.json"
)
JSON_PATH = Path(
    "evaluation/results/v2_cohere_reranker_evaluation.json"
)
SUMMARY_PATH = Path(
    "evaluation/results/v2_cohere_reranker_summary.txt"
)

TOP_K = 10
EXPECTED_QUESTIONS = 60
EXPECTED_ELIGIBLE = 55
EXPECTED_NUMERIC = 15
EXPECTED_UNSUPPORTED = 5
FLOAT_TOLERANCE = 1e-9
COHERE_MIN_CALL_INTERVAL_SECONDS = 6.5
RATE_LIMIT_RETRY_WAIT_SECONDS = 65
MAX_RATE_LIMIT_RETRIES = 3

QUALITY_TARGETS = [
    ("recall_at_10", "R@10", 0.90),
    ("precision_at_5", "P@5", 0.70),
    ("mrr", "MRR", 0.95),
    ("numeric_evidence_hit_at_5", "NE@5", 0.93),
]


def load_json(path: Path, default):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def reconstruct_from_cache(
    candidates: list[dict],
    cached: dict,
) -> list[dict]:
    by_chunk_id = {
        candidate["chunk_id"]: candidate
        for candidate in candidates
    }
    scores = cached["rerank_scores"]
    ranked = []

    for chunk_id in cached["ordered_chunk_ids"]:
        result = deepcopy(by_chunk_id[chunk_id])
        result["rerank_score"] = scores[chunk_id]
        ranked.append(result)

    return ranked


def cache_matches_candidates(
    cached: dict,
    candidates: list[dict],
) -> bool:
    cached_ids = cached.get("ordered_chunk_ids", [])
    current_ids = [candidate["chunk_id"] for candidate in candidates]

    return (
        len(cached_ids) == len(current_ids)
        and set(cached_ids) == set(current_ids)
        and len(cached_ids) == len(set(cached_ids))
        and set(cached.get("rerank_scores", {})) == set(current_ids)
    )


def same_chunk_set(left: list[dict], right: list[dict]) -> bool:
    left_ids = [candidate["chunk_id"] for candidate in left]
    right_ids = [candidate["chunk_id"] for candidate in right]

    return (
        len(left_ids) == len(right_ids)
        and set(left_ids) == set(right_ids)
        and len(left_ids) == len(set(left_ids))
        and len(right_ids) == len(set(right_ids))
    )


def count_groups(benchmark: list[dict]) -> dict:
    unsupported = [item for item in benchmark if is_unsupported(item)]
    eligible = [item for item in benchmark if not is_unsupported(item)]
    numeric = [item for item in benchmark if is_numeric(item)]

    return {
        "questions": len(benchmark),
        "eligible": len(eligible),
        "numeric": len(numeric),
        "unsupported": len(unsupported),
    }


def values_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right

    return abs(left - right) <= FLOAT_TOLERANCE


def wait_for_cohere_slot(last_call_monotonic: float | None) -> None:
    """Sleep only for unused trial-key pacing. Not included in API latency."""
    if last_call_monotonic is None:
        return

    elapsed = time.monotonic() - last_call_monotonic
    remaining = COHERE_MIN_CALL_INTERVAL_SECONDS - elapsed

    if remaining > 0:
        print(
            f"Rate-limit pacing: sleeping {remaining:.1f}s "
            f"(min interval {COHERE_MIN_CALL_INTERVAL_SECONDS}s)."
        )
        time.sleep(remaining)


def rerank_with_rate_limit(
    question: str,
    candidates: list[dict],
    last_call_monotonic: float | None,
) -> tuple[list[dict], float, float]:
    """Call Cohere with trial-key pacing and 429 retries.

    Returns (results, api_latency_seconds, last_call_monotonic).
    api_latency_seconds excludes pacing and retry sleeps.
    """
    last_error = None
    max_attempts = 1 + MAX_RATE_LIMIT_RETRIES

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            wait_for_cohere_slot(last_call_monotonic)

        try:
            results, api_latency = rerank_timed(question, candidates)
            return results, api_latency, time.monotonic()
        except TooManyRequestsError as error:
            last_error = error
            print(
                f"Cohere HTTP 429 TooManyRequestsError "
                f"(retry {attempt}/{MAX_RATE_LIMIT_RETRIES}). "
                f"Waiting {RATE_LIMIT_RETRY_WAIT_SECONDS}s, "
                "then retrying the same question."
            )

            if attempt == max_attempts:
                break

            time.sleep(RATE_LIMIT_RETRY_WAIT_SECONDS)

    raise RuntimeError(
        "Cohere rate limit persisted after "
        f"{MAX_RATE_LIMIT_RETRIES} retries."
    ) from last_error


def pass_fail(value: float | None, target: float) -> str:
    if value is None:
        return "n/a"

    return "PASS" if value >= target else "FAIL"


def build_comparison_table(
    hybrid: dict,
    gpt: dict | None,
    cohere: dict,
) -> str:
    lines = [
        "BENCHMARK V2 COHERE RERANKER COMPARISON",
        "Same hybrid top-10 candidates, then rerank-v4.0-fast.",
        "Unsupported questions excluded from R/P/MRR/NE.",
        "Recall@10 should stay the same because the candidate set is unchanged.",
        "",
        f"{'Metric':<10} {'Hybrid':>10} {'GPT Reranker':>14} {'Cohere Fast':>12}",
        "-" * 50,
    ]

    for key, label in METRICS:
        gpt_value = gpt[key] if gpt else None
        lines.append(
            f"{label:<10} "
            f"{format_metric(hybrid[key])} "
            f"{format_metric(gpt_value)} "
            f"{format_metric(cohere[key])}"
        )

    lines.extend(["", "Deltas: Cohere vs Hybrid"])

    for key, label in METRICS:
        delta = improvement(cohere[key], hybrid[key])
        lines.append(f"  {label:<8} {format_metric(delta)}")

    lines.extend(["", "Deltas: Cohere vs GPT Reranker"])

    for key, label in METRICS:
        gpt_value = gpt[key] if gpt else None
        delta = improvement(cohere[key], gpt_value)
        lines.append(f"  {label:<8} {format_metric(delta)}")

    return "\n".join(lines)


def build_latency_block(
    hybrid: dict,
    gpt: dict | None,
    cohere: dict,
    combined_latencies: list[float],
) -> str:
    gpt_median = gpt["median_latency_seconds"] if gpt else None
    gpt_p95 = gpt["p95_latency_seconds"] if gpt else None
    combined_median = (
        statistics.median(combined_latencies)
        if combined_latencies
        else None
    )

    return "\n".join(
        [
            "",
            "Latency",
            "  Hybrid retrieval median: "
            f"{format_metric(hybrid['median_latency_seconds'])} s",
            f"  GPT reranker median:      {format_metric(gpt_median)} s",
            f"  GPT reranker p95:        {format_metric(gpt_p95)} s",
            "  Cohere reranker mean:     "
            f"{format_metric(cohere['mean_latency_seconds'])} s",
            "  Cohere reranker median:   "
            f"{format_metric(cohere['median_latency_seconds'])} s",
            "  Cohere reranker p95:      "
            f"{format_metric(cohere['p95_latency_seconds'])} s",
            "  Hybrid + Cohere mean:    "
            f"{format_metric(mean_or_none(combined_latencies))} s",
            f"  Hybrid + Cohere median:  {format_metric(combined_median)} s",
            "  Hybrid + Cohere p95:     "
            f"{format_metric(percentile(combined_latencies, 0.95))} s",
        ]
    )


def build_quality_block(cohere: dict) -> str:
    lines = [
        "",
        "QUALITY TARGETS",
        "- R@10 >= 0.90",
        "- P@5 >= 0.70",
        "- MRR >= 0.95",
        "- NE@5 >= 0.93",
        "",
        "LATENCY TARGET",
        "- reranker p95 ideally < 2-3 seconds",
        "",
        "Cohere Fast vs targets",
    ]

    for key, label, target in QUALITY_TARGETS:
        value = cohere[key]
        lines.append(
            f"  {label} {format_metric(value)}  "
            f"target {target:.2f}  {pass_fail(value, target)}"
        )

    p95 = cohere["p95_latency_seconds"]
    latency_status = (
        "PASS" if p95 is not None and p95 < 3.0 else "FAIL"
    )
    lines.append(
        f"  p95 {format_metric(p95)} s  "
        f"target < 3.00  {latency_status}"
    )
    lines.append("")
    lines.append(
        "No winner is declared here. Compare the measured numbers above."
    )

    return "\n".join(lines)


def run_quality_checks(
    counts: dict,
    hybrid_summary: dict,
    cohere_summary: dict,
    set_mismatches: int,
) -> list[str]:
    warnings = []

    checks = [
        (counts["questions"] == EXPECTED_QUESTIONS, "60 questions"),
        (counts["eligible"] == EXPECTED_ELIGIBLE, "55 eligible questions"),
        (counts["numeric"] == EXPECTED_NUMERIC, "15 numeric questions"),
        (
            counts["unsupported"] == EXPECTED_UNSUPPORTED,
            "5 unsupported questions",
        ),
    ]

    for passed, label in checks:
        if not passed:
            warnings.append(f"Count check failed: expected {label}.")

    if set_mismatches:
        warnings.append(
            "Candidate-set invariant failed for "
            f"{set_mismatches} question(s)."
        )

    if not values_close(
        cohere_summary["recall_at_10"],
        hybrid_summary["recall_at_10"],
    ):
        warnings.append(
            "R@10 invariant failed: Cohere "
            f"{cohere_summary['recall_at_10']} vs hybrid "
            f"{hybrid_summary['recall_at_10']}."
        )

    if not values_close(
        cohere_summary["numeric_evidence_hit_at_10"],
        hybrid_summary["numeric_evidence_hit_at_10"],
    ):
        warnings.append(
            "NE@10 invariant failed: Cohere "
            f"{cohere_summary['numeric_evidence_hit_at_10']} vs hybrid "
            f"{hybrid_summary['numeric_evidence_hit_at_10']}."
        )

    return warnings


def main() -> None:
    benchmark = load_benchmark()
    counts = count_groups(benchmark)

    print(f"Evaluating Cohere reranker on {len(benchmark)} questions.\n")
    print(
        f"counts: questions={counts['questions']} "
        f"eligible={counts['eligible']} "
        f"numeric={counts['numeric']} "
        f"unsupported={counts['unsupported']}"
    )
    print()

    cache = load_json(CACHE_PATH, {})
    cached_count = sum(1 for item in benchmark if item["id"] in cache)
    remaining_calls = len(benchmark) - cached_count
    print(f"Cached questions: {cached_count}")
    print(f"Remaining Cohere calls: {remaining_calls}")
    print()

    hybrid_questions = []
    cohere_questions = []
    combined_latencies = []
    set_mismatches = 0
    last_cohere_call = None

    for item in benchmark:
        question_id = item["id"]
        question = item["question"]

        hybrid_start = time.perf_counter()
        hybrid_results = hybrid_search(question, top_k=TOP_K)
        hybrid_latency = time.perf_counter() - hybrid_start

        cached = cache.get(question_id)
        used_cache = False

        if cached and cache_matches_candidates(cached, hybrid_results):
            cohere_results = reconstruct_from_cache(
                hybrid_results,
                cached,
            )
            cohere_latency = float(cached["reranker_latency"])
            used_cache = True
        else:
            (
                cohere_results,
                cohere_latency,
                last_cohere_call,
            ) = rerank_with_rate_limit(
                question,
                hybrid_results,
                last_cohere_call,
            )
            cache[question_id] = {
                "question_id": question_id,
                "ordered_chunk_ids": [
                    candidate["chunk_id"]
                    for candidate in cohere_results
                ],
                "rerank_scores": {
                    candidate["chunk_id"]: candidate["rerank_score"]
                    for candidate in cohere_results
                },
                "reranker_latency": cohere_latency,
            }
            save_json(CACHE_PATH, cache)

        if not same_chunk_set(hybrid_results, cohere_results):
            set_mismatches += 1
            print(
                f"WARNING {question_id}: "
                "reranked chunk IDs differ from hybrid top-10."
            )

        hybrid_questions.append(
            score_ranking(item, hybrid_results, hybrid_latency)
        )
        cohere_row = score_ranking(
            item,
            cohere_results,
            cohere_latency,
        )
        cohere_row["hybrid_latency_seconds"] = hybrid_latency
        cohere_row["combined_latency_seconds"] = (
            hybrid_latency + cohere_latency
        )
        cohere_row["used_cache"] = used_cache
        cohere_questions.append(cohere_row)
        combined_latencies.append(hybrid_latency + cohere_latency)

        print(
            f"{question_id}: "
            f"hybrid_first={hybrid_questions[-1]['first_relevant_rank']} "
            f"cohere_first={cohere_questions[-1]['first_relevant_rank']} "
            f"cohere_latency={cohere_latency:.3f}s"
            f"{' cache' if used_cache else ''}"
        )

    hybrid_summary = summarize("hybrid", hybrid_questions)
    cohere_summary = summarize("hybrid_cohere", cohere_questions)

    gpt_payload = load_json(GPT_RESULTS_PATH, None)
    gpt_summary = None

    if gpt_payload:
        gpt_summary = (
            gpt_payload.get("systems", {})
            .get("hybrid_rerank", {})
            .get("summary")
        )

    warnings = run_quality_checks(
        counts,
        hybrid_summary,
        cohere_summary,
        set_mismatches,
    )

    table = build_comparison_table(
        hybrid_summary,
        gpt_summary,
        cohere_summary,
    )
    latency_block = build_latency_block(
        hybrid_summary,
        gpt_summary,
        cohere_summary,
        combined_latencies,
    )
    quality_block = build_quality_block(cohere_summary)
    warning_block = ""

    if warnings:
        warning_block = "\n\nWARNINGS\n" + "\n".join(
            f"- {warning}" for warning in warnings
        )

    summary_text = (
        table
        + latency_block
        + quality_block
        + warning_block
        + "\n"
    )

    output = {
        "benchmark_path": str(BENCHMARK_PATH),
        "question_count": len(benchmark),
        "top_k": TOP_K,
        "counts": counts,
        "warnings": warnings,
        "systems": {
            "hybrid": {
                "summary": hybrid_summary,
                "questions": hybrid_questions,
            },
            "hybrid_cohere": {
                "summary": cohere_summary,
                "questions": cohere_questions,
            },
        },
        "gpt_reranker_summary": gpt_summary,
    }

    save_json(JSON_PATH, output)

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(summary_text)

    print()
    print(summary_text)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved cache to {CACHE_PATH}")


if __name__ == "__main__":
    main()
