"""Phase 5.5 Batch 1 multi-year structural evaluation — AAPL / MSFT / AMZN.

Same metric definitions as ``evaluation.evaluate_multiyear`` (the NVDA canary),
run against ``benchmark_multiyear_batch1.json`` and with an explicit
cross-company-leakage check added (the canary never mixed tickers).

    fiscal_year_filter_correctness   final evidence in the requested (ticker,fy) set. Target 1.000
    cross_year_leakage@k             top-k evidence with a fiscal_year not requested. Target 0.000
    cross_company_leakage@k          top-k evidence with a ticker not requested.   Target 0.000
    scope_coverage                   requested scopes that got >= 1 chunk.         Target 1.000 (comparisons)
    numeric_year_correctness         right number in a chunk OF THE RIGHT YEAR.    Target 1.000
    unsupported_year rejected        registry says the year is absent (-> API 422). Target 1.000

    python -m evaluation.evaluate_multiyear_batch1
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from api.rag import plan_evidence  # noqa: E402
from evaluation.evaluate_multiyear import (  # noqa: E402
    TOP_K,
    evaluate_supported,
    evaluate_unsupported_year,
)


def _digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")


def _numeric_year_correct(item: dict) -> bool | None:
    """Right number, right year — comparison-aware.

    Single-year question: the target digit-string must appear in a retrieved
    chunk whose fiscal_year == that year (identical to the NVDA canary check).

    Comparison question: a year's target counts if it appears in ANY retrieved
    in-scope chunk. A 10-K income statement always carries 2-3 years of
    comparatives, so the prior year's figure legitimately rides in the newer
    filing's chunk with its columns explicitly year-labelled. cross_year_leakage
    (measured separately, target 0.000) is the independent guard that no
    wrong-year number is presented as this year's.
    """
    hints = item.get("number_hint_by_year")
    if not hints:
        return None
    tickers = sorted({s["ticker"].upper() for s in item["scopes"]})
    years = sorted({int(s["fiscal_year"]) for s in item["scopes"]})
    comparison = len(item["scopes"]) >= 2
    plan = plan_evidence(item["question"], top_k=TOP_K, tickers=tickers,
                         fiscal_years=years)
    evidence = plan["evidence"]
    ok = True
    for year_str, hint in hints.items():
        target = _digits(hint)
        if comparison:
            hit = any(target and target in _digits(c.get("text", ""))
                      for c in evidence)
        else:
            hit = any(
                target and target in _digits(c.get("text", ""))
                and int(c.get("fiscal_year") or -1) == int(year_str)
                for c in evidence
            )
        ok = ok and hit
    return ok

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "evaluation" / "benchmark_multiyear_batch1.json"
OUT_JSON = REPO_ROOT / "evaluation" / "results" / "multiyear_batch1_evaluation.json"
OUT_TXT = REPO_ROOT / "evaluation" / "results" / "multiyear_batch1_summary.txt"


def _cross_company_leak(row: dict, item: dict) -> float:
    """Fraction of top-k evidence scopes whose ticker was not requested."""
    requested_tickers = {s["ticker"].upper() for s in item["scopes"]}
    top = row["evidence_scopes"][:TOP_K]
    if not top:
        return 0.0
    leaked = [s for s in top if s.split(":")[0] not in requested_tickers]
    return len(leaked) / len(top)


def main() -> int:
    items = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    supported = [i for i in items if i["answer_type"] != "unsupported_year"]
    unsupported = [i for i in items if i["answer_type"] == "unsupported_year"]

    rows = []
    for item in supported:
        r = evaluate_supported(item)
        r["cross_company_leakage_at_k"] = _cross_company_leak(r, item)
        # comparison-aware numeric-year check (see _numeric_year_correct)
        r["numeric_year_correct"] = _numeric_year_correct(item)
        rows.append(r)
    unsup_rows = [evaluate_unsupported_year(i) for i in unsupported]

    numeric_rows = [r for r in rows if r["numeric_year_correct"] is not None]
    comparison_rows = [r for r in rows if r["comparison_mode"]]

    summary = {
        "questions": len(items),
        "supported_questions": len(supported),
        "unsupported_year_questions": len(unsupported),
        "companies": sorted({i["ticker"] for i in items}),
        "fiscal_year_filter_correctness": statistics.mean(
            r["fiscal_year_filter_correctness"] for r in rows
        ),
        "cross_year_leakage_at_k": statistics.mean(
            r["cross_year_leakage_at_k"] for r in rows
        ),
        "cross_company_leakage_at_k": statistics.mean(
            r["cross_company_leakage_at_k"] for r in rows
        ),
        "scope_coverage_mean": statistics.mean(r["scope_coverage"] for r in rows),
        "comparison_scope_coverage_mean": (
            statistics.mean(r["scope_coverage"] for r in comparison_rows)
            if comparison_rows else None
        ),
        "numeric_year_correctness": (
            statistics.mean(1.0 if r["numeric_year_correct"] else 0.0
                            for r in numeric_rows)
            if numeric_rows else None
        ),
        "numeric_questions": len(numeric_rows),
        "unsupported_year_correctly_rejected": (
            statistics.mean(1.0 if r["correctly_unavailable"] else 0.0
                            for r in unsup_rows)
            if unsup_rows else None
        ),
        "reranker_fallback_questions": sum(1 for r in rows if r["reranker_fallback"]),
        "latency_mean_s": statistics.mean(r["latency_seconds"] for r in rows),
        "latency_p50_s": statistics.median(r["latency_seconds"] for r in rows),
    }

    passed = (
        abs(summary["fiscal_year_filter_correctness"] - 1.0) < 1e-9
        and abs(summary["cross_year_leakage_at_k"]) < 1e-9
        and abs(summary["cross_company_leakage_at_k"]) < 1e-9
        and (summary["comparison_scope_coverage_mean"] is None
             or abs(summary["comparison_scope_coverage_mean"] - 1.0) < 1e-9)
        and (summary["numeric_year_correctness"] in (None, 1.0))
        and (summary["unsupported_year_correctly_rejected"] in (None, 1.0))
    )
    summary["result"] = "PASS" if passed else "FAIL"

    OUT_JSON.write_text(json.dumps(
        {"summary": summary, "questions": rows, "unsupported_year": unsup_rows},
        indent=2), encoding="utf-8")

    lines = [
        "MULTI-YEAR (FISCAL-YEAR SCOPE) EVALUATION - Phase 5.5 Batch 1 (AAPL/MSFT/AMZN)",
        f"questions: {summary['questions']}  supported: {summary['supported_questions']}  "
        f"unsupported-year: {summary['unsupported_year_questions']}   top_k: {TOP_K}",
        "",
        f"fiscal_year_filter_correctness   {summary['fiscal_year_filter_correctness']:.3f}   (target 1.000)",
        f"cross_year_leakage@{TOP_K}             {summary['cross_year_leakage_at_k']:.3f}   (target 0.000)",
        f"cross_company_leakage@{TOP_K}          {summary['cross_company_leakage_at_k']:.3f}   (target 0.000)",
        f"scope_coverage (all)             {summary['scope_coverage_mean']:.3f}",
        f"scope_coverage (comparison Qs)   "
        + (f"{summary['comparison_scope_coverage_mean']:.3f}   (target 1.000)"
           if summary['comparison_scope_coverage_mean'] is not None else "n/a"),
        f"numeric_year_correctness         "
        + (f"{summary['numeric_year_correctness']:.3f}   ({summary['numeric_questions']} numeric Qs, right number right year)"
           if summary['numeric_year_correctness'] is not None else "n/a"),
        f"unsupported_year rejected        "
        + (f"{summary['unsupported_year_correctly_rejected']:.3f}   (target 1.000)"
           if summary['unsupported_year_correctly_rejected'] is not None else "n/a"),
        f"reranker_fallback               {summary['reranker_fallback_questions']}/{len(rows)}",
        f"latency  mean {summary['latency_mean_s']:.2f}s  p50 {summary['latency_p50_s']:.2f}s",
        "",
        f"GATE  {summary['result']}",
        "",
        "per-question:",
    ]
    for r in rows:
        flag = "" if (
            abs(r["fiscal_year_filter_correctness"] - 1.0) < 1e-9
            and r["cross_year_leakage_at_k"] == 0.0
            and r["cross_company_leakage_at_k"] == 0.0
        ) else "  <-- CHECK"
        num = ("" if r["numeric_year_correct"] is None
               else f" numeric_year_ok={r['numeric_year_correct']}")
        lines.append(
            f"  {r['id']:<40} scopes={r['requested_scopes']} "
            f"ev_by_scope={r['evidence_by_scope']} "
            f"yr_leak={r['cross_year_leakage_at_k']:.2f} co_leak={r['cross_company_leakage_at_k']:.2f}{num}{flag}"
        )
    for r in unsup_rows:
        lines.append(
            f"  {r['id']:<40} requested {r['requested']} available {r['available_years']} "
            f"-> correctly_unavailable={r['correctly_unavailable']}"
        )
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
