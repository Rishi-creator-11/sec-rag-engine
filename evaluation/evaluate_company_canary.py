"""Per-company canary evaluation — scoped-retrieval sanity for a pilot company.

    python -m evaluation.evaluate_company_canary --ticker GOOGL

Reads ``evaluation/benchmark_<ticker_lower>.json``. Measures only what ticker
metadata + light lexical checks can prove (relevance is NOT judged here):

    filter_correctness      fraction of retrieved chunks with ticker == <TICKER>
    cross_company_leakage   fraction with a different ticker (target 0.000)
    scoped_retrieval_rate   supported questions returning >= 1 <TICKER> chunk
    numeric_evidence        for numeric Qs with a `number_hint`, whether a
                            retrieved chunk contains that digit string (a
                            lexical proxy, NOT a judged qrel)

Generation is not called. This generalizes evaluate_amzn.py.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from api.rag import plan_evidence  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def evaluate_question(item: dict, ticker: str) -> dict:
    start = time.perf_counter()
    plan = plan_evidence(item["question"], top_k=TOP_K, tickers=[ticker])
    latency_s = time.perf_counter() - start

    evidence = plan["evidence"]
    tickers = [str(c.get("ticker", "")).strip().upper() for c in evidence]
    in_scope = [t for t in tickers if t == ticker]
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
        "has_company_evidence": bool(in_scope),
        "supported": supported,
        "number_hint": item.get("number_hint"),
        "number_evidence_present": number_hit,
        "reranker_fallback": plan["reranker_fallback"],
        "latency_seconds": latency_s,
    }


def run(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    bench_path = REPO_ROOT / "evaluation" / f"benchmark_{ticker.lower()}.json"
    if not bench_path.exists():
        raise SystemExit(f"no canary benchmark: {bench_path}")
    benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
    rows = [evaluate_question(item, ticker) for item in benchmark]

    supported = [r for r in rows if r["supported"]]
    numeric_checked = [r for r in rows if r["number_evidence_present"] is not None]

    summary = {
        "ticker": ticker,
        "questions": len(rows),
        "supported_questions": len(supported),
        "filter_correctness": statistics.mean(r["filter_correctness"] for r in rows),
        "cross_company_leakage": statistics.mean(r["cross_company_leakage"] for r in rows),
        "scoped_retrieval_rate": statistics.mean(
            1.0 if r["has_company_evidence"] else 0.0 for r in supported
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

    passed = (
        abs(summary["filter_correctness"] - 1.0) < 1e-9
        and abs(summary["cross_company_leakage"]) < 1e-9
        and abs(summary["scoped_retrieval_rate"] - 1.0) < 1e-9
    )
    summary["result"] = "PASS" if passed else "FAIL"

    out = {"summary": summary, "questions": rows}
    (REPO_ROOT / "evaluation" / "results" / f"{ticker.lower()}_canary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out


def _print(out: dict) -> None:
    s = out["summary"]
    print(f"CANARY {s['ticker']}  ({s['questions']} questions, top_k {TOP_K})  ->  {s['result']}")
    print(f"  filter_correctness            {s['filter_correctness']:.3f}   (must be 1.000)")
    print(f"  cross_company_leakage         {s['cross_company_leakage']:.3f}   (must be 0.000)")
    print(f"  scoped_retrieval_rate         {s['scoped_retrieval_rate']:.3f}   (supported Qs with company evidence)")
    print(f"  numeric_evidence_present_rate {s['numeric_evidence_present_rate']}   (lexical proxy, not a qrel)")
    print(f"  reranker_fallback             {s['reranker_fallback_questions']}/{s['questions']}")
    print(f"  latency  mean {s['latency_mean_s']:.2f}s  p50 {s['latency_p50_s']:.2f}s")
    for r in out["questions"]:
        extra = ""
        if r["number_evidence_present"] is not None:
            extra = f"  number_evidence={'yes' if r['number_evidence_present'] else 'no'}"
        print(f"    {r['id']:<40} {r['category']:<16} ev={r['evidence_count']} "
              f"leak={r['cross_company_leakage']:.2f}{extra}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args(argv)
    out = run(args.ticker)
    _print(out)
    return 0 if out["summary"]["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
