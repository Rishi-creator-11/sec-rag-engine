"""Compact production sanity test for the Hybrid + Cohere RAG path.

Selects 13 questions from evaluation/benchmark_v2.json:
5 numeric, 5 qualitative, and 3 unsupported, mixed across Apple,
Microsoft, and NVIDIA.

This run writes optimized results to a new file and compares them
with the previous baseline without overwriting it.
"""

import json
import re
import statistics
from pathlib import Path

from api.rag import ANSWER_MODEL, answer_question
from evaluation.evaluate_v2 import percentile


BENCHMARK_PATH = Path("evaluation/benchmark_v2.json")
BASELINE_JSON = Path("evaluation/results/final_rag_evaluation.json")
JSON_PATH = Path(
    "evaluation/results/final_rag_optimized_evaluation.json"
)
SUMMARY_PATH = Path(
    "evaluation/results/final_rag_optimized_summary.txt"
)

SELECTED_IDS = [
    "apple_total_net_sales_2024_01",
    "microsoft_total_revenue_2025_01",
    "nvidia_compute_networking_revenue_2026_01",
    "apple_iphone_sales_2024_01",
    "microsoft_operating_income_2025_01",
    "apple_product_categories_01",
    "microsoft_azure_competition_01",
    "nvidia_data_center_platform_01",
    "apple_seasonality_01",
    "nvidia_gaming_business_01",
    "apple_iphone17_revenue_2025_unsupported_01",
    "microsoft_fy2026_revenue_unsupported_01",
    "nvidia_fy2027_revenue_unsupported_01",
]

REFUSAL_MARKERS = (
    "do not contain enough information",
    "do not contain",
    "does not contain enough information",
    "does not contain",
    "not enough information",
    "insufficient information",
    "cannot answer",
    "cannot be answered",
)

SOURCE_HIT_TARGET = 0.90
REFUSAL_TARGET = 1.00
MEDIAN_LATENCY_TARGET_S = 3.0
P95_LATENCY_TARGET_S = 5.0


def load_json(path: Path, default=None):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_selected_questions() -> list[dict]:
    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    by_id = {item["id"]: item for item in benchmark}
    missing = [
        question_id
        for question_id in SELECTED_IDS
        if question_id not in by_id
    ]

    if missing:
        raise KeyError(f"Benchmark is missing selected ids: {missing}")

    return [by_id[question_id] for question_id in SELECTED_IDS]


def citation_count(answer: str) -> int:
    return len(re.findall(r"\[Source\s+\d+\]", answer or ""))


def refusal_observed(answer: str) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def source_hit(returned_ids: list[str], relevant_ids: list[str]) -> bool | None:
    if not relevant_ids:
        return None

    return bool(set(returned_ids) & set(relevant_ids))


def grounded_label(item: dict, refused: bool) -> str:
    if item.get("answer_type") == "unsupported":
        return "yes" if refused else "no"

    return "manual_review_required"


def mean(values: list[float]) -> float | None:
    if not values:
        return None

    return statistics.mean(values)


def median(values: list[float]) -> float | None:
    if not values:
        return None

    return statistics.median(values)


def ms_list(rows: list[dict], key: str) -> list[float]:
    values = []

    for row in rows:
        timings = row.get("timings") or {}
        value = timings.get(key)

        if value is not None:
            values.append(float(value) / 1000.0)

    return values


def collect_metric(rows: list[dict], key: str) -> list[float]:
    values = []

    for row in rows:
        timings = row.get("timings") or {}
        value = timings.get(key)

        if value is not None:
            values.append(float(value))

    return values


def format_value(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"

    return f"{value:.{digits}f}"


def gate(label: str, passed: bool, detail: str) -> str:
    status = "PASS" if passed else "FAIL"
    return f"{status}  {label}  ({detail})"


def summarize_run(rows: list[dict]) -> dict:
    supported = [row for row in rows if row["source_hit"] is not None]
    unsupported = [row for row in rows if row["refusal_expected"]]
    hits = [row for row in supported if row["source_hit"]]
    refusals = [row for row in unsupported if row["refusal_observed"]]
    totals = ms_list(rows, "total_ms")
    generation = ms_list(rows, "generation_ms")

    failures = []

    for row in supported:
        if not row["source_hit"]:
            failures.append(row["id"])

    for row in unsupported:
        if not row["refusal_observed"]:
            failures.append(row["id"])

    return {
        "questions_tested": len(rows),
        "source_hit_rate": (
            len(hits) / len(supported) if supported else None
        ),
        "unsupported_refusal_rate": (
            len(refusals) / len(unsupported) if unsupported else None
        ),
        "generation_mean": mean(generation),
        "generation_median": median(generation),
        "generation_p95": percentile(generation, 0.95) if generation else None,
        "total_mean": mean(totals),
        "total_median": median(totals),
        "total_p95": percentile(totals, 0.95) if totals else None,
        "avg_context_chunks": mean(collect_metric(rows, "context_chunks_used")),
        "avg_context_characters": mean(
            collect_metric(rows, "context_characters")
        ),
        "avg_answer_characters": mean(
            collect_metric(rows, "answer_characters")
        ),
        "failures": failures,
    }


def baseline_metrics(baseline: dict | None) -> dict:
    if not baseline:
        return {}

    questions = baseline.get("questions") or []
    if questions:
        return summarize_run(questions)

    return {
        "source_hit_rate": baseline.get("source_hit_rate"),
        "unsupported_refusal_rate": baseline.get("unsupported_refusal_rate"),
        "total_mean": baseline.get("mean_total_latency_seconds"),
        "total_median": baseline.get("median_total_latency_seconds"),
        "total_p95": baseline.get("p95_total_latency_seconds"),
        "generation_mean": None,
        "generation_median": None,
        "generation_p95": None,
        "avg_context_chunks": None,
        "avg_context_characters": None,
        "avg_answer_characters": None,
    }


def comparison_block(before: dict, after: dict) -> list[str]:
    rows = [
        ("source hit rate", "source_hit_rate", 3),
        ("unsupported refusal rate", "unsupported_refusal_rate", 3),
        ("generation mean", "generation_mean", 3),
        ("generation median", "generation_median", 3),
        ("generation p95", "generation_p95", 3),
        ("total mean", "total_mean", 3),
        ("total median", "total_median", 3),
        ("total p95", "total_p95", 3),
        ("average context chunks", "avg_context_chunks", 2),
        ("average context characters", "avg_context_characters", 1),
        ("average answer characters", "avg_answer_characters", 1),
    ]
    lines = [
        "BEFORE vs AFTER",
        f"{'metric':<28} {'before':>10} {'after':>10}",
        "-" * 50,
    ]

    for label, key, digits in rows:
        lines.append(
            f"{label:<28} "
            f"{format_value(before.get(key), digits):>10} "
            f"{format_value(after.get(key), digits):>10}"
        )

    return lines


def main() -> None:
    questions = load_selected_questions()
    rows = []

    print(f"production generation model: {ANSWER_MODEL}")
    print(f"Running optimized RAG sanity test on {len(questions)} questions.\n")

    for item in questions:
        print(f"{item['id']}: {item['question']}")
        response = answer_question(item["question"], top_k=5)

        returned_ids = [
            source["chunk_id"]
            for source in response.get("sources", [])
        ]
        relevant_ids = list(item.get("relevant_chunks", []))
        refused = refusal_observed(response.get("answer", ""))
        expected_refusal = item.get("answer_type") == "unsupported"
        hit = source_hit(returned_ids, relevant_ids)
        timings = response.get("timings", {})

        row = {
            "id": item["id"],
            "question": item["question"],
            "company": item.get("company"),
            "answer_type": item.get("answer_type"),
            "answer": response.get("answer"),
            "expected_relevant_chunks": relevant_ids,
            "returned_source_chunk_ids": returned_ids,
            "source_hit": hit,
            "answer_grounded": grounded_label(item, refused),
            "refusal_expected": expected_refusal,
            "refusal_observed": refused,
            "citation_count": citation_count(response.get("answer", "")),
            "reranker_fallback": response.get("reranker_fallback"),
            "timings": timings,
        }
        rows.append(row)

        print(
            f"  source_hit={hit} refusal={refused} "
            f"chunks={timings.get('context_chunks_used')} "
            f"generation_ms={timings.get('generation_ms')} "
            f"total_ms={timings.get('total_ms')}"
        )

    after = summarize_run(rows)
    before = baseline_metrics(load_json(BASELINE_JSON))

    source_hit_rate = after["source_hit_rate"]
    refusal_rate = after["unsupported_refusal_rate"]
    median_total = after["total_median"]
    p95_total = after["total_p95"]
    latency_missed = (
        median_total is None
        or p95_total is None
        or median_total >= MEDIAN_LATENCY_TARGET_S
        or p95_total >= P95_LATENCY_TARGET_S
    )

    gates = [
        gate(
            "source hit rate >= 0.90",
            source_hit_rate is not None and source_hit_rate >= SOURCE_HIT_TARGET,
            format_value(source_hit_rate),
        ),
        gate(
            "unsupported refusal rate == 1.00",
            refusal_rate is not None and refusal_rate == REFUSAL_TARGET,
            format_value(refusal_rate),
        ),
        gate(
            "median total latency < 3s",
            median_total is not None and median_total < MEDIAN_LATENCY_TARGET_S,
            f"{format_value(median_total)}s",
        ),
        gate(
            "p95 total latency < 5s",
            p95_total is not None and p95_total < P95_LATENCY_TARGET_S,
            f"{format_value(p95_total)}s",
        ),
    ]

    summary_lines = [
        "FINAL RAG OPTIMIZED SANITY TEST",
        f"production generation model: {ANSWER_MODEL}",
        "Same 13-question set. Generation-only optimizations; retrieval unchanged.",
        "",
        *comparison_block(before, after),
        "",
        f"failures by question id: {after['failures'] or 'none'}",
        "",
        "QUALITY GATES",
        *gates,
    ]

    if latency_missed:
        summary_lines.extend(
            [
                "",
                "Latency still misses the production target after context/prompt/output cuts.",
                "Do not keep tweaking this pass. Next controlled experiment: a faster generation model.",
            ]
        )

    summary_text = "\n".join(summary_lines) + "\n"

    output = {
        "benchmark_path": str(BENCHMARK_PATH),
        "baseline_path": str(BASELINE_JSON),
        "questions_tested": len(rows),
        "selected_ids": SELECTED_IDS,
        "source_hit_rate": after["source_hit_rate"],
        "unsupported_refusal_rate": after["unsupported_refusal_rate"],
        "mean_total_latency_seconds": after["total_mean"],
        "median_total_latency_seconds": after["total_median"],
        "p95_total_latency_seconds": after["total_p95"],
        "mean_generation_latency_seconds": after["generation_mean"],
        "median_generation_latency_seconds": after["generation_median"],
        "p95_generation_latency_seconds": after["generation_p95"],
        "avg_context_chunks": after["avg_context_chunks"],
        "avg_context_characters": after["avg_context_characters"],
        "avg_answer_characters": after["avg_answer_characters"],
        "failures": after["failures"],
        "gates": gates,
        "before": before,
        "after": after,
        "questions": rows,
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(summary_text)

    print()
    print(summary_text)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Baseline left unchanged at {BASELINE_JSON}")


if __name__ == "__main__":
    main()
