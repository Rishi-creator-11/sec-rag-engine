"""Comparison retrieval evaluation — SCOPE COVERAGE, not relevance quality.

Phase 2 does not build relevance qrels for comparison questions (pooling +
LLM/human judging is substantial additional work and is deferred). This
evaluator measures only what ticker metadata can prove:

    scope_coverage@k        fraction of requested tickers with >= 1 final
                            evidence chunk (target 1.000)
    full_coverage_rate      fraction of questions with scope_coverage == 1.0
    min_evidence_per_scope  min, over requested tickers, of that ticker's
                            evidence count (mean across questions)
    cross_scope_leakage     fraction of final evidence chunks whose ticker is
                            outside the requested set (target 0.000)
    evidence_by_scope       per-question ticker -> count
    latency                 retrieval-only (plan_evidence): mean / p50 / p95

RELEVANCE QUALITY of comparison answers is explicitly OUT OF SCOPE here.
Add it later with the existing pooled-candidate + judging methodology
(build_v2_pool.py / auto_judge_v2_pool.py / review_auto_judgments.py).

Generation is not called, so this run spends no answer-model tokens.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from api.rag import plan_evidence  # noqa: E402
from evaluation.evaluate_v2 import percentile  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "evaluation" / "benchmark_comparisons.json"
JSON_PATH = REPO_ROOT / "evaluation" / "results" / "comparison_evaluation.json"
SUMMARY_PATH = REPO_ROOT / "evaluation" / "results" / "comparison_summary.txt"

TOP_K = 5


def load_benchmark() -> list[dict]:
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected benchmark shape in {BENCHMARK_PATH}")
    return data


def evaluate_question(item: dict) -> dict:
    requested = [t.upper() for t in item["tickers"]]

    start = time.perf_counter()
    plan = plan_evidence(item["question"], top_k=TOP_K, tickers=requested)
    latency_s = time.perf_counter() - start

    evidence = plan["evidence"]
    evidence_tickers = [
        str(chunk.get("ticker", "")).strip().upper() for chunk in evidence
    ]
    by_scope = plan["evidence_by_scope"]

    covered = [t for t in requested if by_scope.get(t, 0) >= 1]
    leaked = [t for t in evidence_tickers if t not in requested]

    return {
        "id": item["id"],
        "topic": item.get("topic"),
        "requested_tickers": requested,
        "comparison_mode": plan["comparison_mode"],
        "evidence_count": len(evidence),
        "evidence_by_scope": by_scope,
        "evidence_ticker_sequence": evidence_tickers,
        "scope_coverage": len(covered) / len(requested),
        "min_evidence_per_scope": min(by_scope.get(t, 0) for t in requested),
        "cross_scope_leakage": (len(leaked) / len(evidence)) if evidence else 0.0,
        "reranker_fallback": plan["reranker_fallback"],
        "reranker_fallback_reason": plan["reranker_fallback_reason"],
        "warnings": plan["warnings"],
        "latency_seconds": latency_s,
        "hybrid_ms": plan["hybrid_ms"],
        "rerank_ms": plan["rerank_ms"],
    }


def summarize(rows: list[dict]) -> dict:
    coverage = [r["scope_coverage"] for r in rows]
    leakage = [r["cross_scope_leakage"] for r in rows]
    min_per_scope = [r["min_evidence_per_scope"] for r in rows]
    latencies = [r["latency_seconds"] for r in rows]

    return {
        "questions": len(rows),
        "comparison_mode_questions": sum(1 for r in rows if r["comparison_mode"]),
        "scope_coverage_at_5": statistics.mean(coverage),
        "full_coverage_rate": statistics.mean(
            1.0 if c >= 1.0 else 0.0 for c in coverage
        ),
        "min_evidence_per_scope_mean": statistics.mean(min_per_scope),
        "min_evidence_per_scope_min": min(min_per_scope),
        "cross_scope_leakage": statistics.mean(leakage),
        "reranker_fallback_questions": sum(
            1 for r in rows if r["reranker_fallback"]
        ),
        "questions_with_warnings": sum(1 for r in rows if r["warnings"]),
        "latency_mean_s": statistics.mean(latencies),
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": percentile(latencies, 0.95),
    }


def build_summary_text(summary: dict, rows: list[dict]) -> str:
    lines = [
        "COMPARISON RETRIEVAL EVALUATION (scope coverage only)",
        f"benchmark: {BENCHMARK_PATH.name}   questions: {summary['questions']}   top_k: {TOP_K}",
        "Relevance quality is NOT judged here (deferred).",
        "",
        f"scope_coverage@5          {summary['scope_coverage_at_5']:.3f}   (target 1.000)",
        f"full_coverage_rate        {summary['full_coverage_rate']:.3f}",
        f"cross_scope_leakage       {summary['cross_scope_leakage']:.3f}   (target 0.000)",
        f"min_evidence_per_scope    mean {summary['min_evidence_per_scope_mean']:.2f}  "
        f"min {summary['min_evidence_per_scope_min']}",
        f"reranker_fallback         {summary['reranker_fallback_questions']}/{summary['questions']} questions",
        f"questions_with_warnings   {summary['questions_with_warnings']}",
        f"latency (plan_evidence)   mean {summary['latency_mean_s']:.2f}s  "
        f"p50 {summary['latency_p50_s']:.2f}s  p95 {summary['latency_p95_s']:.2f}s",
        "",
        "GATES",
        f"  scope_coverage@5 == 1.000   "
        f"{'PASS' if abs(summary['scope_coverage_at_5'] - 1.0) < 1e-9 else 'FAIL'}",
        f"  cross_scope_leakage == 0.0  "
        f"{'PASS' if abs(summary['cross_scope_leakage']) < 1e-9 else 'FAIL'}",
        "",
        "per-question:",
    ]
    for r in rows:
        flag = "" if r["scope_coverage"] >= 1.0 and r["cross_scope_leakage"] == 0.0 else "  <-- CHECK"
        lines.append(
            f"  {r['id']:<34} {'+'.join(r['requested_tickers']):<14} "
            f"by_scope={r['evidence_by_scope']} "
            f"cov={r['scope_coverage']:.2f} leak={r['cross_scope_leakage']:.2f} "
            f"{r['latency_seconds']:.2f}s{flag}"
        )
        if r["warnings"]:
            for warning in r["warnings"]:
                lines.append(f"      warning: {warning}")
    return "\n".join(lines) + "\n"


def main() -> int:
    benchmark = load_benchmark()
    print(f"Evaluating {len(benchmark)} comparison questions (retrieval only)...\n")

    rows = []
    for item in benchmark:
        row = evaluate_question(item)
        rows.append(row)
        print(
            f"  {row['id']:<34} cov={row['scope_coverage']:.2f} "
            f"leak={row['cross_scope_leakage']:.2f} "
            f"by_scope={row['evidence_by_scope']} {row['latency_seconds']:.2f}s"
        )

    summary = summarize(rows)
    output = {
        "benchmark": str(BENCHMARK_PATH),
        "top_k": TOP_K,
        "summary": summary,
        "questions": rows,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    text = build_summary_text(summary, rows)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
