"""Scoped retrieval evaluation on Benchmark v3 (current 8-company corpus).

Benchmark v3 replaces the depth-10 / 3-company pooled qrels of v2 with a
deep (union of dense + bm25s + hybrid + sparse, depth 50), scoped, re-judged
pool on the current AAPL/AMZN/GOOGL/JPM/META/MSFT/NVDA/WMT corpus. See
evaluation/build_benchmark_v3.py, build_v3_pool.py, judge_v3_pool.py.

Every question is scoped to its own company (filters=RetrievalFilter(tickers=
(company,))), matching how production retrieves once a ticker is known. Production
retrieval depth (candidate_k=10) is unchanged; only the *judging* pool was deeper.

Metrics (supported questions only; unsupported reported separately):
    Recall@{1,3,5,10}, Precision@{1,3,5}, MRR, Numeric Evidence Hit@{5,10}

Structural gates (must hold on the 8-company corpus, benchmark-independent):
    filter_correctness            == 1.000
    cross_company_leakage@10      == 0.000

Retrieval-quality baseline: recorded empirically from this run. The v2 0.900
Recall@10 gate is NOT carried over (v2 qrels under-counted relevant chunks on
the grown corpus). The recommended v3 regression rule is emitted in the summary.

    python -m evaluation.evaluate_v3 --retriever all
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evaluation.evaluate_v2 import (  # noqa: E402
    hit_at_k,
    mean_or_none,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.evaluate_scoped import ticker_of_result, leakage_at_k  # noqa: E402
from retrieval.filters import RetrievalFilter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "evaluation" / "benchmark_v3.json"
JSON_PATH = REPO_ROOT / "evaluation" / "results" / "v3_retrieval_evaluation.json"
SUMMARY_PATH = REPO_ROOT / "evaluation" / "results" / "v3_retrieval_summary.txt"

TOP_K = 10
# Recommended v3 regression tolerance (see summary). Not a hard 0.900 gate.
REGRESSION_TOL = 0.02


def is_unsupported(item: dict) -> bool:
    return (
        item.get("answer_type") == "unsupported"
        or not item.get("relevant_chunks")
    )


def is_numeric(item: dict) -> bool:
    return (
        item.get("answer_type") in {"numeric", "mixed"}
        and bool(item.get("relevant_chunks"))
    )


def load_benchmark() -> list[dict]:
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected benchmark shape in {BENCHMARK_PATH}")
    return data


def get_retrievers():
    from retrieval.bm25_search import search as bm25_search
    from retrieval.hybrid_search import search as hybrid_search
    from retrieval.pinecone_search import search as dense_search
    from retrieval.embedder import embed_text

    def dense(q, filt):
        return dense_search(q, top_k=TOP_K, filters=filt,
                            query_embedding=embed_text(q))

    def bm25(q, filt):
        return bm25_search(q, top_k=TOP_K, filters=filt)

    def hybrid(q, filt):
        # Production mix: dense + bm25s, sparse disabled, candidate_k unchanged.
        return hybrid_search(q, top_k=TOP_K, candidate_k=TOP_K, filters=filt,
                             query_embedding=embed_text(q))

    return {"dense": dense, "bm25": bm25, "hybrid": hybrid}


def evaluate(name: str, search_fn, benchmark: list[dict], limit: int | None) -> dict:
    rows: list[dict] = []
    for item in (benchmark[:limit] if limit else benchmark):
        company = str(item["company"]).strip().upper()
        relevant = set(item.get("relevant_chunks", []))
        unsupported = is_unsupported(item)
        numeric = is_numeric(item)
        filt = RetrievalFilter(tickers=(company,))

        results = search_fn(item["question"], filt)[:TOP_K]
        ids = [r["chunk_id"] for r in results]
        tickers = [ticker_of_result(r) for r in results]
        rr, first_rank = reciprocal_rank(ids, relevant)

        rows.append({
            "id": item["id"],
            "company": company,
            "source": item.get("source"),
            "unsupported": unsupported,
            "numeric": numeric,
            "retrieved_chunk_ids": ids,
            "retrieved_tickers": tickers,
            "first_relevant_rank": first_rank,
            "recall_at_1": recall_at_k(ids, relevant, 1),
            "recall_at_3": recall_at_k(ids, relevant, 3),
            "recall_at_5": recall_at_k(ids, relevant, 5),
            "recall_at_10": recall_at_k(ids, relevant, 10),
            "precision_at_1": precision_at_k(ids, relevant, 1),
            "precision_at_3": precision_at_k(ids, relevant, 3),
            "precision_at_5": precision_at_k(ids, relevant, 5),
            "reciprocal_rank": rr,
            "numeric_hit_at_5": hit_at_k(ids, relevant, 5) if numeric else None,
            "numeric_hit_at_10": hit_at_k(ids, relevant, 10) if numeric else None,
            "returned_count": len(results),
            "filter_hits_top10": sum(1 for t in tickers if t == company),
            "leakage_at_1": leakage_at_k(tickers, {company}, 1),
            "leakage_at_3": leakage_at_k(tickers, {company}, 3),
            "leakage_at_5": leakage_at_k(tickers, {company}, 5),
            "leakage_at_10": leakage_at_k(tickers, {company}, 10),
            "unsupported_returned_top10": len(results) if unsupported else None,
        })

    supported = [r for r in rows if not r["unsupported"]]
    numeric = [r for r in rows if r["numeric"]]
    unsupported = [r for r in rows if r["unsupported"]]

    def _m(field, group):
        return mean_or_none([r[field] for r in group if r[field] is not None])

    correctness = [
        r["filter_hits_top10"] / r["returned_count"]
        for r in rows if r["returned_count"]
    ]

    summary = {
        "retriever": name,
        "questions": len(rows),
        "supported_questions": len(supported),
        "numeric_questions": len(numeric),
        "unsupported_questions": len(unsupported),
        "recall_at_1": _m("recall_at_1", supported),
        "recall_at_3": _m("recall_at_3", supported),
        "recall_at_5": _m("recall_at_5", supported),
        "recall_at_10": _m("recall_at_10", supported),
        "precision_at_1": _m("precision_at_1", supported),
        "precision_at_3": _m("precision_at_3", supported),
        "precision_at_5": _m("precision_at_5", supported),
        "mrr": _m("reciprocal_rank", supported),
        "numeric_evidence_hit_at_5": _m("numeric_hit_at_5", numeric),
        "numeric_evidence_hit_at_10": _m("numeric_hit_at_10", numeric),
        "filter_correctness": statistics.mean(correctness) if correctness else None,
        "cross_company_leakage_at_1": _m("leakage_at_1", rows),
        "cross_company_leakage_at_3": _m("leakage_at_3", rows),
        "cross_company_leakage_at_5": _m("leakage_at_5", rows),
        "cross_company_leakage_at_10": _m("leakage_at_10", rows),
        "mean_unsupported_returned_top10": _m("unsupported_returned_top10", unsupported),
    }
    return {"summary": summary, "questions": rows}


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:6.3f}"


def build_summary(results: dict[str, dict]) -> str:
    lines = [
        "SCOPED RETRIEVAL EVALUATION - Benchmark v3 (8-company corpus)",
        f"top_k: {TOP_K}   every question scoped to its own company",
        "Unsupported questions excluded from Recall/Precision/MRR/NumericEvidence.",
        "",
        f"{'retriever':<9} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} "
        f"{'P@1':>6} {'P@3':>6} {'P@5':>6} {'MRR':>6} {'NE@5':>6} {'NE@10':>6} "
        f"{'FiltOK':>7} {'Leak@10':>8}",
    ]
    lines.append("-" * len(lines[-1]))
    for name, payload in results.items():
        s = payload["summary"]
        lines.append(
            f"{name:<9} {_fmt(s['recall_at_1'])} {_fmt(s['recall_at_3'])} "
            f"{_fmt(s['recall_at_5'])} {_fmt(s['recall_at_10'])} "
            f"{_fmt(s['precision_at_1'])} {_fmt(s['precision_at_3'])} "
            f"{_fmt(s['precision_at_5'])} {_fmt(s['mrr'])} "
            f"{_fmt(s['numeric_evidence_hit_at_5'])} "
            f"{_fmt(s['numeric_evidence_hit_at_10'])} "
            f"{_fmt(s['filter_correctness']):>7} "
            f"{_fmt(s['cross_company_leakage_at_10']):>8}"
        )

    lines.append("")
    if "hybrid" in results:
        h = results["hybrid"]["summary"]
        fc = h["filter_correctness"] or 0.0
        lk = h["cross_company_leakage_at_10"] or 0.0
        lines += [
            "STRUCTURAL GATES (hybrid, benchmark-independent)",
            f"  filter_correctness == 1.000      {fc:.3f}   "
            f"{'PASS' if abs(fc - 1.0) < 1e-9 else 'FAIL'}",
            f"  cross_company_leakage@10 == 0     {lk:.3f}   "
            f"{'PASS' if abs(lk) < 1e-9 else 'FAIL'}",
            "",
            "V3 RETRIEVAL-QUALITY BASELINE (record; do not hard-gate at 0.900)",
            f"  hybrid Recall@10 = {h['recall_at_10']:.3f}   hybrid MRR = {h['mrr']:.3f}",
            f"  Recommended regression rule: future runs must not drop hybrid",
            f"  Recall@10 or MRR by more than {REGRESSION_TOL:.2f} vs this baseline,",
            f"  and must keep both structural gates exact.",
        ]
    lines.append("")
    lines.append("NE@5/NE@10 = numeric evidence hit rate (numeric + mixed questions).")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retriever", choices=["dense", "bm25", "hybrid", "all"],
                   default="all")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)

    if not BENCHMARK_PATH.exists():
        raise SystemExit(
            f"{BENCHMARK_PATH} not found - run assemble_benchmark_v3.py first"
        )

    benchmark = load_benchmark()
    names = ["dense", "bm25", "hybrid"] if a.retriever == "all" else [a.retriever]
    retrievers = get_retrievers()

    results: dict[str, dict] = {}
    for name in names:
        print(f"evaluating {name} ...")
        results[name] = evaluate(name, retrievers[name], benchmark, a.limit)

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(
        {"benchmark": str(BENCHMARK_PATH), "top_k": TOP_K,
         "summaries": {n: r["summary"] for n, r in results.items()},
         "questions": {n: r["questions"] for n, r in results.items()}},
        indent=2), encoding="utf-8")

    text = build_summary(results)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
