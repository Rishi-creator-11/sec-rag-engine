"""Multi-year (fiscal-year scope) retrieval evaluation for the NVDA canary.

Measures what filing metadata can prove — no relevance qrels:

    fiscal_year_filter_correctness   fraction of final evidence chunks whose
                                     (ticker, fiscal_year) is in the requested
                                     scope set. Target 1.000.
    cross_year_leakage@k             fraction of the top-k evidence chunks with a
                                     fiscal_year NOT requested. Target 0.000.
    scope_coverage                   fraction of requested scopes that got >= 1
                                     evidence chunk. Target 1.000 for comparison
                                     questions where both years exist.
    numeric_year_correctness         for numeric questions: the requested number
                                     appears in a retrieved chunk OF THE RIGHT
                                     YEAR (a correct number from the wrong year
                                     is a FAILURE). Lexical proxy, not a qrel.

Unsupported-year questions (answer_type "unsupported_year") are checked
separately: the API must reject them (422) rather than silently widen scope.

    python -m evaluation.evaluate_multiyear
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
from ingestion.registry import available_fiscal_years  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "evaluation" / "benchmark_multiyear.json"
OUT_JSON = REPO_ROOT / "evaluation" / "results" / "multiyear_evaluation.json"
OUT_TXT = REPO_ROOT / "evaluation" / "results" / "multiyear_summary.txt"

TOP_K = 5


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text or "")


def _scope_keys(item: dict) -> set[tuple[str, int]]:
    return {
        (s["ticker"].upper(), int(s["fiscal_year"]))
        for s in item["scopes"]
    }


def evaluate_supported(item: dict) -> dict:
    tickers = sorted({s["ticker"].upper() for s in item["scopes"]})
    years = sorted({int(s["fiscal_year"]) for s in item["scopes"]})
    requested = _scope_keys(item)

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
    leaked = [k for k in top_k_keys if k not in requested]

    covered_scopes = {k for k in ev_keys if k in requested}
    scope_coverage = len(covered_scopes) / len(requested) if requested else 1.0

    number_hits: dict[str, bool] = {}
    for year_str, hint in (item.get("number_hint_by_year") or {}).items():
        target = _digits(hint)
        number_hits[year_str] = any(
            target and target in _digits(c.get("text", ""))
            and int(c.get("fiscal_year") or -1) == int(year_str)
            for c in evidence
        )

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
            len(leaked) / len(top_k_keys) if top_k_keys else 0.0
        ),
        "scope_coverage": scope_coverage,
        "evidence_by_scope": plan["evidence_by_scope"],
        "comparison_mode": plan["comparison_mode"],
        "number_year_hits": number_hits,
        "numeric_year_correct": (
            all(number_hits.values()) if number_hits else None
        ),
        "reranker_fallback": plan["reranker_fallback"],
        "latency_seconds": latency,
        "warnings": plan["warnings"],
    }


def evaluate_unsupported_year(item: dict) -> dict:
    """The requested year must not be available; the API contract rejects it."""
    ticker = item["scopes"][0]["ticker"].upper()
    year = int(item["scopes"][0]["fiscal_year"])
    have = available_fiscal_years(ticker)
    year_available = year in have
    # plan_evidence itself does not validate (that is api/main.py); assert the
    # registry says the year is absent, which is what the 422 is built on.
    return {
        "id": item["id"],
        "answer_type": item["answer_type"],
        "requested": f"{ticker}:{year}",
        "available_years": have,
        "year_available": year_available,
        "correctly_unavailable": not year_available,
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
        "fiscal_year_filter_correctness": statistics.mean(
            r["fiscal_year_filter_correctness"] for r in rows
        ),
        "cross_year_leakage_at_k": statistics.mean(
            r["cross_year_leakage_at_k"] for r in rows
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
        and (summary["comparison_scope_coverage_mean"] is None
             or abs(summary["comparison_scope_coverage_mean"] - 1.0) < 1e-9)
        and (summary["unsupported_year_correctly_rejected"] in (None, 1.0))
    )
    summary["result"] = "PASS" if passed else "FAIL"

    OUT_JSON.write_text(json.dumps(
        {"summary": summary, "questions": rows, "unsupported_year": unsup_rows},
        indent=2), encoding="utf-8")

    lines = [
        "MULTI-YEAR (FISCAL-YEAR SCOPE) EVALUATION - NVDA canary",
        f"questions: {summary['questions']}  supported: {summary['supported_questions']}  "
        f"unsupported-year: {summary['unsupported_year_questions']}   top_k: {TOP_K}",
        "",
        f"fiscal_year_filter_correctness   {summary['fiscal_year_filter_correctness']:.3f}   (target 1.000)",
        f"cross_year_leakage@{TOP_K}             {summary['cross_year_leakage_at_k']:.3f}   (target 0.000)",
        f"scope_coverage (all)             {summary['scope_coverage_mean']:.3f}",
        f"scope_coverage (comparison Qs)   "
        + (f"{summary['comparison_scope_coverage_mean']:.3f}   (target 1.000)"
           if summary['comparison_scope_coverage_mean'] is not None else "n/a"),
        f"numeric_year_correctness         "
        + (f"{summary['numeric_year_correctness']:.3f}   (right number, right year)"
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
        ) else "  <-- CHECK"
        lines.append(
            f"  {r['id']:<42} scopes={r['requested_scopes']} "
            f"ev_by_scope={r['evidence_by_scope']} "
            f"leak={r['cross_year_leakage_at_k']:.2f}{flag}"
        )
    for r in unsup_rows:
        lines.append(
            f"  {r['id']:<42} requested {r['requested']} available {r['available_years']} "
            f"-> correctly_unavailable={r['correctly_unavailable']}"
        )
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"saved {OUT_JSON}")
    print(f"saved {OUT_TXT}")
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
