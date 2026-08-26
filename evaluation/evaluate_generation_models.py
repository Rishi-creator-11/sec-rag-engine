"""Compare answer-generation models on the frozen 13-question RAG set.

Retrieval, Cohere rerank, evidence selection, context compaction, and the
grounding prompt stay identical across models. Only the generation model
changes.

The production model in api/rag.py is not modified.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from openai import OpenAI

from api import rag
from retrieval.hybrid_search import search as hybrid_search
from evaluation.evaluate_final_rag import (
    P95_LATENCY_TARGET_S,
    REFUSAL_TARGET,
    SELECTED_IDS,
    SOURCE_HIT_TARGET,
    citation_count,
    format_value,
    gate,
    grounded_label,
    load_selected_questions,
    mean,
    median,
    refusal_observed,
    source_hit,
)
from evaluation.evaluate_v2 import percentile


JSON_PATH = Path("evaluation/results/generation_model_comparison.json")
SUMMARY_PATH = Path("evaluation/results/generation_model_comparison.txt")
COHERE_CACHE_PATH = Path(
    "evaluation/results/cohere_reranker_cache.json"
)
PRODUCTION_LATENCY_PATH = Path(
    "evaluation/results/final_rag_optimized_evaluation.json"
)

BASELINE_MODEL = rag.ANSWER_MODEL
CANDIDATE_MODELS = [
    BASELINE_MODEL,
    "gpt-5-nano",
    "gpt-5.6-terra",
]

OPTIONAL_MODELS = {"gpt-5.6-terra"}
MEDIAN_LATENCY_TARGET_S = 3.0

GPT_PARAM_ATTEMPTS = [
    {
        "reasoning": {"effort": "minimal"},
        "text": {"verbosity": "low"},
    },
    {
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    },
    {},
]


def load_json(path: Path, default=None):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_cohere_cache() -> dict:
    cache = load_json(COHERE_CACHE_PATH, None)

    if not isinstance(cache, dict):
        raise FileNotFoundError(
            f"Cohere reranker cache not found: {COHERE_CACHE_PATH}"
        )

    return cache


def load_production_retrieval_latency() -> dict[str, float]:
    """Hybrid+Cohere latency from the production optimized RAG run."""
    payload = load_json(PRODUCTION_LATENCY_PATH, None)

    if not payload:
        return {}

    latencies = {}

    for row in payload.get("questions", []):
        timings = row.get("timings") or {}
        hybrid_ms = timings.get("hybrid_ms", timings.get("hybrid_retrieval_ms"))
        rerank_ms = timings.get("rerank_ms")

        if hybrid_ms is None or rerank_ms is None:
            continue

        latencies[row["id"]] = float(hybrid_ms) + float(rerank_ms)

    return latencies


def hybrid_top10(question: str) -> list[dict]:
    return hybrid_search(
        question,
        top_k=rag.HYBRID_TOP_K,
        candidate_k=rag.CANDIDATE_K,
    )


def reconstruct_cohere_ranking(
    question_id: str,
    hybrid_results: list[dict],
    cache_entry: dict,
) -> list[dict]:
    hybrid_ids = [result["chunk_id"] for result in hybrid_results]
    ordered_ids = list(cache_entry.get("ordered_chunk_ids") or [])
    scores = cache_entry.get("rerank_scores") or {}

    problems = []

    if len(hybrid_ids) != len(set(hybrid_ids)):
        problems.append("hybrid top-10 contains duplicate chunk IDs")

    if len(ordered_ids) != len(set(ordered_ids)):
        problems.append("cached ranking contains duplicate chunk IDs")

    if not ordered_ids:
        problems.append("cached ranking is empty")

    if set(ordered_ids) != set(hybrid_ids):
        missing_in_hybrid = sorted(set(ordered_ids) - set(hybrid_ids))
        extra_in_hybrid = sorted(set(hybrid_ids) - set(ordered_ids))
        problems.append(
            "hybrid top-10 and cache candidate sets differ. "
            f"missing_in_hybrid={missing_in_hybrid} "
            f"extra_in_hybrid={extra_in_hybrid}"
        )

    missing_scores = [chunk_id for chunk_id in ordered_ids if chunk_id not in scores]

    if missing_scores:
        problems.append(f"cache missing rerank scores: {missing_scores}")

    if problems:
        raise ValueError(
            f"{question_id}: invalid Cohere cache reconstruction: "
            + "; ".join(problems)
        )

    by_id = {result["chunk_id"]: result for result in hybrid_results}
    ranked = []

    for chunk_id in ordered_ids:
        candidate = dict(by_id[chunk_id])
        candidate["rerank_score"] = scores[chunk_id]
        ranked.append(candidate)

    return ranked


def prepare_fixed_evidence(
    questions: list[dict],
    cache: dict,
) -> list[dict]:
    """Build one frozen evidence pack per question. Never calls Cohere."""
    missing = [
        item["id"]
        for item in questions
        if item["id"] not in cache
    ]
    errors = []
    prepared = []

    if missing:
        errors.extend(f"{question_id}: no Cohere cache entry" for question_id in missing)

    for item in questions:
        question_id = item["id"]

        if question_id not in cache:
            continue

        try:
            hybrid_results = hybrid_top10(item["question"])
            ranked = reconstruct_cohere_ranking(
                question_id,
                hybrid_results,
                cache[question_id],
            )
            evidence = rag.select_evidence(
                ranked,
                max_k=rag.DEFAULT_EVIDENCE_K,
            )
            compact_evidence = rag.prepare_evidence(evidence)
            context = rag.build_context(compact_evidence)
            returned_ids = [
                chunk["chunk_id"] for chunk in compact_evidence
            ]
            prepared.append(
                {
                    "item": item,
                    "hybrid_results": hybrid_results,
                    "ranked_ids": [
                        chunk["chunk_id"] for chunk in ranked
                    ],
                    "evidence": compact_evidence,
                    "context": context,
                    "returned_ids": returned_ids,
                    "context_characters": len(context),
                }
            )
        except Exception as error:
            errors.append(str(error))

    if errors:
        print("FIXED EVIDENCE PREPARATION FAILED")
        print("Do not call Cohere. Do not fall back to hybrid.")
        print("Missing or inconsistent questions:")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    return prepared


def is_unavailable_error(error: Exception) -> bool:
    text = str(error).lower()
    markers = (
        "model_not_found",
        "model not found",
        "does not exist",
        "invalid model",
        "not have access",
        "do not have access",
        "unknown model",
        "404",
    )
    status = getattr(error, "status_code", None)
    return status == 404 or any(marker in text for marker in markers)


def generate_with_model(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    extra_params: dict,
) -> tuple[str, float]:
    request = rag.build_generation_request(question, context)
    request["model"] = model
    request.update(extra_params)

    start = time.perf_counter()
    response = client.responses.create(**request)
    generation_ms = (time.perf_counter() - start) * 1000
    return response.output_text, generation_ms


def generate_for_model(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    working_params: dict | None,
) -> tuple[str, float, dict, bool]:
    attempts = []

    if working_params is not None:
        attempts.append(working_params)

    for extra in GPT_PARAM_ATTEMPTS:
        if extra != working_params:
            attempts.append(extra)

    last_error = None

    for extra in attempts:
        try:
            answer, generation_ms = generate_with_model(
                client,
                model,
                question,
                context,
                extra,
            )
            return answer, generation_ms, extra, False
        except Exception as error:
            last_error = error

            if is_unavailable_error(error):
                raise

            if extra is attempts[-1]:
                raise

    raise last_error


def score_answer(
    item: dict,
    answer: str,
    returned_ids: list[str],
    generation_ms: float,
    retrieval_ms: float,
) -> dict:
    relevant_ids = list(item.get("relevant_chunks", []))
    refused = refusal_observed(answer)
    expected_refusal = item.get("answer_type") == "unsupported"
    hit = source_hit(returned_ids, relevant_ids)
    citations = citation_count(answer)

    return {
        "id": item["id"],
        "question": item["question"],
        "answer_type": item.get("answer_type"),
        "answer": answer,
        "expected_relevant_chunks": relevant_ids,
        "returned_source_chunk_ids": returned_ids,
        "source_hit": hit,
        "answer_grounded": grounded_label(item, refused),
        "refusal_expected": expected_refusal,
        "refusal_observed": refused,
        "citation_count": citations,
        "citation_present": citations > 0,
        "generation_ms": round(generation_ms, 1),
        "retrieval_plus_rerank_ms": round(retrieval_ms, 1),
        "estimated_total_ms": round(retrieval_ms + generation_ms, 1),
        "answer_characters": len(answer or ""),
    }


def summarize_model(model: str, rows: list[dict]) -> dict:
    supported = [row for row in rows if row["source_hit"] is not None]
    unsupported = [row for row in rows if row["refusal_expected"]]
    hits = [row for row in supported if row["source_hit"]]
    refusals = [row for row in unsupported if row["refusal_observed"]]
    cited = [row for row in supported if row["citation_present"]]
    generation = [row["generation_ms"] / 1000.0 for row in rows]
    totals = [row["estimated_total_ms"] / 1000.0 for row in rows]

    failures = []
    format_failures = []

    for row in supported:
        if not row["source_hit"]:
            failures.append(row["id"])

        if not row["citation_present"]:
            format_failures.append(row["id"])

    for row in unsupported:
        if not row["refusal_observed"]:
            failures.append(row["id"])

    return {
        "model": model,
        "questions_tested": len(rows),
        "source_hit_rate": len(hits) / len(supported) if supported else None,
        "unsupported_refusal_rate": (
            len(refusals) / len(unsupported) if unsupported else None
        ),
        "citation_presence_rate": (
            len(cited) / len(supported) if supported else None
        ),
        "avg_answer_characters": mean(
            [float(row["answer_characters"]) for row in rows]
        ),
        "generation_mean": mean(generation),
        "generation_median": median(generation),
        "generation_p95": percentile(generation, 0.95) if generation else None,
        "total_mean": mean(totals),
        "total_median": median(totals),
        "total_p95": percentile(totals, 0.95) if totals else None,
        "failures": failures,
        "format_failures": format_failures,
    }


def target_lines(summary: dict) -> list[str]:
    source_hit_rate = summary["source_hit_rate"]
    refusal_rate = summary["unsupported_refusal_rate"]
    total_median = summary["total_median"]
    total_p95 = summary["total_p95"]

    return [
        gate(
            "source hit >= 0.90",
            source_hit_rate is not None and source_hit_rate >= SOURCE_HIT_TARGET,
            format_value(source_hit_rate),
        ),
        gate(
            "unsupported refusal == 1.00",
            refusal_rate is not None and refusal_rate == REFUSAL_TARGET,
            format_value(refusal_rate),
        ),
        gate(
            "total median < 3.0 s",
            total_median is not None and total_median < MEDIAN_LATENCY_TARGET_S,
            f"{format_value(total_median)}s",
        ),
        gate(
            "total p95 < 5.0 s",
            total_p95 is not None and total_p95 < P95_LATENCY_TARGET_S,
            f"{format_value(total_p95)}s",
        ),
    ]


def acceptable_faster_model(summary: dict, is_baseline: bool) -> bool:
    if is_baseline:
        return False

    source_ok = (
        summary["source_hit_rate"] is not None
        and summary["source_hit_rate"] >= SOURCE_HIT_TARGET
    )
    refusal_ok = summary["unsupported_refusal_rate"] == REFUSAL_TARGET
    citations_ok = (
        summary["citation_presence_rate"] is not None
        and summary["citation_presence_rate"] >= 0.90
    )
    format_ok = not summary["format_failures"]
    return source_ok and refusal_ok and citations_ok and format_ok


def print_table(summaries: list[dict]) -> list[str]:
    header = (
        f"{'Model':<22} {'SourceHit':>9} {'Refusal':>8} "
        f"{'GenMean':>8} {'GenMedian':>9} {'GenP95':>7} "
        f"{'TotalMedian':>11} {'TotalP95':>9}"
    )
    lines = [
        "GENERATION MODEL COMPARISON",
        f"Baseline production model: {BASELINE_MODEL}",
        "Same 13 questions, same cached Cohere evidence, same prompt.",
        "No live Cohere calls. Estimated totals use prior Hybrid+Cohere latency.",
        "",
        header,
        "-" * len(header),
    ]

    for summary in summaries:
        lines.append(
            f"{summary['model']:<22} "
            f"{format_value(summary['source_hit_rate']):>9} "
            f"{format_value(summary['unsupported_refusal_rate']):>8} "
            f"{format_value(summary['generation_mean']):>8} "
            f"{format_value(summary['generation_median']):>9} "
            f"{format_value(summary['generation_p95']):>7} "
            f"{format_value(summary['total_median']):>11} "
            f"{format_value(summary['total_p95']):>9}"
        )

    return lines


def main() -> None:
    questions = load_selected_questions()
    cache = load_cohere_cache()
    production_latency = load_production_retrieval_latency()
    missing_latency = [
        item["id"]
        for item in questions
        if item["id"] not in production_latency
    ]

    if missing_latency:
        print(
            "WARNING: missing production Hybrid+Cohere latency for "
            f"{missing_latency}. Estimated totals will omit those rows."
        )

    print(
        f"Preparing FIXED Cohere evidence for {len(questions)} questions.\n"
        "Hybrid top-10 is retrieved live. Cohere ranking is loaded from cache.\n"
        "No live Cohere API calls will be made.\n"
        f"Production generation model stays {BASELINE_MODEL} in api/rag.py.\n"
    )

    prepared = prepare_fixed_evidence(questions, cache)
    client = OpenAI()

    print("EVIDENCE VALIDATION")
    print(f"questions validated: {len(prepared)}")
    print("live Cohere calls: 0")
    print("fallbacks: 0")
    print()

    for bundle in prepared:
        item = bundle["item"]
        print(item["id"])
        print(f"final evidence chunk IDs: {bundle['returned_ids']}")
        print(f"context character count: {bundle['context_characters']}")
        print()

    model_results = []
    skipped = []
    working_params: dict[str, dict] = {}

    for model in CANDIDATE_MODELS:
        print(f"\nGenerating with {model}")
        rows = []
        extra = working_params.get(model)

        try:
            for bundle in prepared:
                item = bundle["item"]
                answer, generation_ms, extra, _ = generate_for_model(
                    client,
                    model,
                    item["question"],
                    bundle["context"],
                    extra,
                )
                extra = extra
                working_params[model] = extra
                row = score_answer(
                    item,
                    answer,
                    bundle["returned_ids"],
                    generation_ms,
                    production_latency.get(item["id"], 0.0),
                )
                rows.append(row)
                print(
                    f"  {item['id']}: gen_ms={generation_ms:.1f} "
                    f"total_ms={row['estimated_total_ms']:.1f} "
                    f"citations={row['citation_count']}"
                )
        except Exception as error:
            if model in OPTIONAL_MODELS and is_unavailable_error(error):
                print(
                    f"Skipping {model}: unavailable or no access ({error})."
                )
                skipped.append(
                    {
                        "model": model,
                        "reason": str(error),
                    }
                )
                continue

            raise

        summary = summarize_model(model, rows)
        model_results.append(
            {
                "summary": summary,
                "generation_params": working_params.get(model, {}),
                "questions": rows,
            }
        )

    summaries = [result["summary"] for result in model_results]
    table_lines = print_table(summaries)
    lines = list(table_lines)
    lines.extend(
        [
            "",
            "PRODUCTION TARGETS",
            "- source hit >= 0.90",
            "- unsupported refusal == 1.00",
            "- total median < 3.0 s",
            "- total p95 < 5.0 s",
            "",
        ]
    )

    for summary in summaries:
        lines.append(f"{summary['model']}")
        lines.extend(f"  {line}" for line in target_lines(summary))
        lines.append(
            f"  failures: {summary['failures'] or 'none'}"
        )
        lines.append(
            f"  citation presence: {format_value(summary['citation_presence_rate'])}"
        )
        lines.append("")

    lines.append("DECISION RULE")
    lines.append(
        "A faster model is acceptable only if source hit stays >= 0.90, "
        "unsupported refusal stays 1.00, citations remain present, and there "
        "are no obvious answer-format failures."
    )
    lines.append("Do not switch the production model automatically.")
    lines.append("")

    for summary in summaries:
        if summary["model"] == BASELINE_MODEL:
            continue

        ok = acceptable_faster_model(summary, is_baseline=False)
        lines.append(
            f"{summary['model']}: "
            f"{'meets quality bar for consideration' if ok else 'does not yet meet the quality bar'}"
        )

    if skipped:
        lines.append("")
        lines.append("SKIPPED MODELS")
        for item in skipped:
            lines.append(f"- {item['model']}: {item['reason']}")

    summary_text = "\n".join(lines) + "\n"

    output = {
        "baseline_production_model": BASELINE_MODEL,
        "models_configured": CANDIDATE_MODELS,
        "models_evaluated": [summary["model"] for summary in summaries],
        "skipped_models": skipped,
        "selected_ids": SELECTED_IDS,
        "max_output_tokens": rag.MAX_OUTPUT_TOKENS,
        "live_cohere_calls": False,
        "evidence": [
            {
                "id": bundle["item"]["id"],
                "evidence_chunk_ids": bundle["returned_ids"],
                "ranked_chunk_ids": bundle["ranked_ids"],
                "context_characters": bundle["context_characters"],
                "production_hybrid_cohere_ms": production_latency.get(
                    bundle["item"]["id"]
                ),
            }
            for bundle in prepared
        ],
        "results": model_results,
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
    print("Production model in api/rag.py was not changed.")


if __name__ == "__main__":
    main()
