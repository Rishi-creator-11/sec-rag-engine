"""Phase 5.5 Batch 2 multi-year structural evaluation — GOOGL / META / JPM.

Same metric definitions as ``evaluation.evaluate_multiyear`` (the NVDA canary)
and ``evaluation.evaluate_multiyear_batch1``, run against
``benchmark_multiyear_batch2.json``.

    fiscal_year_filter_correctness   final evidence in the requested (ticker,fy) set. Target 1.000
    cross_year_leakage@k             top-k evidence with a fiscal_year not requested. Target 0.000
    cross_company_leakage@k          top-k evidence with a ticker not requested.   Target 0.000
    scope_coverage                   requested scopes that got >= 1 chunk.         Target 1.000 (comparisons)
    numeric_year_correctness         right number in a chunk OF THE RIGHT YEAR.    Target 1.000
    unsupported_year rejected        registry says the year is absent (-> API 422). Target 1.000

Unlike Batch 1, every per-question metric here (structural AND numeric) is
computed from a SINGLE ``plan_evidence`` retrieval. JPM's 10-K is ~400 chunks
and the Cohere trial key rate-limits hard, so a second retrieval inside the
numeric check (Batch 1's approach) is non-deterministic under fallback. One
retrieval per question keeps the numeric check consistent with the leakage
and coverage checks that are read from the same evidence set.

    python -m evaluation.evaluate_multiyear_batch2
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from api.rag import plan_evidence  # noqa: E402
from evaluation.evaluate_multiyear import TOP_K, evaluate_unsupported_year  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "evaluation" / "benchmark_multiyear_batch2.json"
OUT_JSON = REPO_ROOT / "evaluation" / "results" / "multiyear_batch2_evaluation.json"
OUT_TXT = REPO_ROOT / "evaluation" / "results" / "multiyear_batch2_summary.txt"


def _digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")


def _numeric_year_correct(item: dict, evidence: list[dict]) -> bool | None:
    """Right number, right year — comparison-aware, from the SAME retrieval.

    Single-year question: the target digit-string must appear in a retrieved
    chunk whose fiscal_year == that year (identical to the NVDA canary check).

    Comparison question: a year's target counts if it appears in ANY retrieved
    in-scope chunk. A 10-K income statement carries 2-3 years of comparatives,
    so the prior year's figure legitimately rides in the newer filing's chunk
    with its columns explicitly year-labelled. cross_year_leakage (measured
    separately, target 0.000) is the independent guard that no wrong-year
    number is presented as this year's.
    """
    hints = item.get("number_hint_by_year")
    if not hints:
        return None
    comparison = len(item["scopes"]) >= 2
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


def evaluate_supported(item: dict) -> dict:
    tickers = sorted({s["ticker"].upper() for s in item["scopes"]})
    years = sorted({int(s["fiscal_year"]) for s in item["scopes"]})
    requested = {(s["ticker"].upper(), int(s["fiscal_year"])) for s in item["scopes"]}
    requested_tickers = {t for t, _ in requested}

    start = time.perf_counter()
    plan = plan_evidence(item["question"], top_k=TOP_K, tickers=tickers,
                         fiscal_years=years)
    latency = time.perf_counter() - start

    evidence = plan["evidence"]
    ev_keys = [
        (str(c.get("ticker", "")).upper(),
         int(c["fiscal_year"]) if c.get("fiscal_year") not in (None, "") else None)
        for c in evidence
    ]
    in_scope = [k for k in ev_keys if k in requested]
    top_k_keys = ev_keys[:TOP_K]
    year_leaked = [k for k in top_k_keys if k not in requested]
    company_leaked = [k for k in top_k_keys if k[0] not in requested_tickers]

    covered = {k for k in ev_keys if k in requested}
    scope_coverage = len(covered) / len(requested) if requested else 1.0

    return {
        "id": item["id"],
        "answer_type": item["answer_type"],
        "requested_scopes": sorted(f"{t}:{y}" for t, y in requested),
        "evidence_scopes": [f"{t}:{y}" for t, y in ev_keys],
        "evidence_count": len(evidence),
        "fiscal_year_filter_correctness": (
            len(in_scope) / len(evidence) if evidence else 1.0
        ),
        "cross_year_leakage_at_k": (
            len(year_leaked) / len(top_k_keys) if top_k_keys else 0.0
        ),
        "cross_company_leakage_at_k": (
            len(company_leaked) / len(top_k_keys) if top_k_keys else 0.0
        ),
        "scope_coverage": scope_coverage,
        "evidence_by_scope": plan["evidence_by_scope"],
        "comparison_mode": plan["comparison_mode"],
        "numeric_year_correct": _numeric_year_correct(item, evidence),
        "reranker_fallback": plan["reranker_fallback"],
        "latency_seconds": latency,
        "warnings": plan["warnings"],
    }


def main() -> int:
    items = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    supported = [i for i in items if i["answer_type"] != "unsupported_year"]
    unsupported = [i for i in items if i["answer_type"] == "unsupported_year"]

    rows = [evaluate_supported(i) for i in supported]
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
        "MULTI-YEAR (FISCAL-YEAR SCOPE) EVALUATION - Phase 5.5 Batch 2 (GOOGL/META/JPM)",
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
