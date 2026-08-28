"""AMZN canary evaluation — scoped-retrieval sanity for the newly ingested filing.

This does NOT build relevance qrels (deferred, same as evaluate_comparison).
It measures what ticker metadata + light lexical checks can prove:

    filter_correctness    fraction of retrieved chunks with ticker == AMZN
                          (target 1.000)
    cross_company_leakage fraction with ticker != AMZN (target 0.000)
    scoped_retrieval      every supported question returns >= 1 AMZN chunk
    numeric_evidence      for numeric questions with a `number_hint`, whether a
                          retrieved chunk's text contains that number string
                          (a lexical proxy, NOT a judged qrel)

Separated clearly: ticker coverage (proven) vs relevance quality (not judged).
Generation is not called.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "evaluation" / "benchmark_amzn.json"
JSON_PATH = REPO_ROOT / "evaluation" / "results" / "amzn_evaluation.json"
SUMMARY_PATH = REPO_ROOT / "evaluation" / "results" / "amzn_summary.txt"

TICKER = "AMZN"
TOP_K = 5


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def evaluate_question(item: dict) -> dict:
    start = time.perf_counter()
    plan = plan_evidence(item["question"], top_k=TOP_K, tickers=[TICKER])
    latency_s = time.perf_counter() - start

    evidence = plan["evidence"]
    tickers = [str(c.get("ticker", "")).strip().upper() for c in evidence]
    in_scope = [t for t in tickers if t == TICKER]
    supported = item.get("answer_type") != "unsupported"

    number_hit = None
    if item.get("number_hint"):
        target = _digits(item["number_hint"])
        number_hit = any(target and target in _digits(c.get("text", "")) for c in evidence)

    return {
        "id": item["id"],
        "category": item.get("category"),
        "answer_type": item.get("answer_type"),
        "evidence_count": len(evidence),
        "evidence_tickers": tickers,
        "filter_correctness": (len(in_scope) / len(evidence)) if evidence else 1.0,
        "cross_company_leakage": (
            (len(evidence) - len(in_scope)) / len(evidence) if evidence else 0.0
        ),
        "has_amzn_evidence": bool(in_scope),
        "supported": supported,
        "number_hint": item.get("number_hint"),
        "number_evidence_present": number_hit,
        "reranker_fallback": plan["reranker_fallback"],
        "latency_seconds": latency_s,
    }


def main() -> int:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    rows = [evaluate_question(item) for item in benchmark]

    supported = [r for r in rows if r["supported"]]
    numeric_checked = [r for r in rows if r["number_evidence_present"] is not None]

    summary = {
        "questions": len(rows),
        "supported_questions": len(supported),
        "filter_correctness": statistics.mean(r["filter_correctness"] for r in rows),
        "cross_company_leakage": statistics.mean(r["cross_company_leakage"] for r in rows),
        "scoped_retrieval_rate": statistics.mean(
            1.0 if r["has_amzn_evidence"] else 0.0 for r in supported
        ),
        "numeric_evidence_present_rate": (
            statistics.mean(1.0 if r["number_evidence_present"] else 0.0
                            for r in numeric_checked)
            if numeric_checked else None
        ),
        "reranker_fallback_questions": sum(1 for r in rows if r["reranker_fallback"]),
        "latency_mean_s": statistics.mean(r["latency_seconds"] for r in rows),
        "latency_p50_s": statistics.median(r["latency_seconds"] for r in rows),
    }

    lines = [
        "AMZN CANARY EVALUATION (scoped-retrieval sanity; relevance NOT judged)",
        f"benchmark: {BENCHMARK_PATH.name}   questions: {summary['questions']}   top_k: {TOP_K}",
        "",
        f"filter_correctness            {summary['filter_correctness']:.3f}   (target 1.000)",
        f"cross_company_leakage         {summary['cross_company_leakage']:.3f}   (target 0.000)",
        f"scoped_retrieval_rate         {summary['scoped_retrieval_rate']:.3f}   "
        "(supported Qs with >=1 AMZN chunk)",
        f"numeric_evidence_present_rate {summary['numeric_evidence_present_rate']}   "
        "(lexical proxy over number_hint, not a qrel)",
        f"reranker_fallback             {summary['reranker_fallback_questions']}/{summary['questions']}",
        f"latency                       mean {summary['latency_mean_s']:.2f}s  "
        f"p50 {summary['latency_p50_s']:.2f}s",
        "",
        "GATES",
        f"  filter_correctness == 1.000   "
        f"{'PASS' if abs(summary['filter_correctness'] - 1.0) < 1e-9 else 'FAIL'}",
        f"  cross_company_leakage == 0.0  "
        f"{'PASS' if abs(summary['cross_company_leakage']) < 1e-9 else 'FAIL'}",
        f"  scoped_retrieval_rate == 1.0  "
        f"{'PASS' if abs(summary['scoped_retrieval_rate'] - 1.0) < 1e-9 else 'FAIL'}",
        "",
        "per-question:",
    ]
    for r in rows:
        extra = ""
        if r["number_evidence_present"] is not None:
            extra = f" number_evidence={'yes' if r['number_evidence_present'] else 'no'}"
        lines.append(
            f"  {r['id']:<40} {r['category']:<14} "
            f"amzn_evidence={r['evidence_count']} leak={r['cross_company_leakage']:.2f}"
            f"{extra}  {r['latency_seconds']:.2f}s"
        )

    text = "\n".join(lines) + "\n"
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps({"summary": summary, "questions": rows}, indent=2), encoding="utf-8"
    )
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved JSON to {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
