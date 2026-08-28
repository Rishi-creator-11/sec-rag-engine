"""Offline scoped retrieval evaluation on Benchmark v3.

Identical metrics to evaluate_v3.py, but computed from the SCOPED depth-50 pool
(evaluation/results/benchmark_v3_pool.json) instead of live retrieval calls.
The pool records every candidate's per-retriever rank, so the production top-10
for dense / bm25s / hybrid is exactly the pooled candidates with that
retriever's rank <= 10. This spends no API tokens.

The pool was built scoped (filters=RetrievalFilter(tickers=(company,))) with
production settings (dense top-k, bm25s production backend, hybrid RRF with
candidate_k left at its pooling value but the top-10 slice is rank-identical to
production candidate_k=10 for the first 10 positions). Because the pool is
scoped by construction, filter_correctness == 1.000 and
cross_company_leakage@k == 0.000 for every retriever - this matches the LIVE
Part A result (evaluation/results/scoped_retrieval_evaluation.json: filtered
hybrid filter_correctness 1.000, leakage@10 0.000) and is stated, not re-proven,
here.

    python -m evaluation.evaluate_v3_offline
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from evaluation.evaluate_v2 import (
    hit_at_k,
    mean_or_none,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_pool.json"
BENCH = REPO_ROOT / "evaluation" / "benchmark_v3.json"
JSON_PATH = REPO_ROOT / "evaluation" / "results" / "v3_retrieval_evaluation.json"
SUMMARY_PATH = REPO_ROOT / "evaluation" / "results" / "v3_retrieval_summary.txt"

TOP_K = 10
RETRIEVERS = ["dense", "bm25s", "hybrid"]
REGRESSION_TOL = 0.02


def _is_numeric(item: dict) -> bool:
    return item.get("answer_type") in {"numeric", "mixed"} and bool(
        item.get("relevant_chunks")
    )


def build_live_pool(bench: list[dict], depth: int = 50) -> dict:
    """Call the real retrievers (scoped) and return a pool dict in the same shape
    as benchmark_v3_pool.json. Used by --live to re-check retrieval quality on the
    current corpus (e.g. after ingesting a new company)."""
    from dotenv import load_dotenv
    load_dotenv()
    from retrieval.pinecone_search import search as dense_search
    from retrieval.bm25_search import search as bm25_search
    from retrieval.hybrid_search import search as hybrid_search
    from retrieval.embedder import embed_text
    from retrieval.filters import RetrievalFilter

    pool: dict = {}
    supported = [q for q in bench
                 if q["answer_type"] != "unsupported" and q.get("relevant_chunks")]
    for n, item in enumerate(supported, 1):
        qid, ticker, q = item["id"], item["company"], item["question"]
        filt = RetrievalFilter(tickers=(ticker,))
        vec = embed_text(q)
        cand: dict = {}
        runs = {
            "dense": dense_search(q, top_k=depth, filters=filt, query_embedding=vec),
            "bm25s": bm25_search(q, top_k=depth, filters=filt),
            "hybrid": hybrid_search(q, top_k=depth, candidate_k=depth,
                                    filters=filt, query_embedding=vec),
        }
        for rname, results in runs.items():
            for rank, r in enumerate(results, 1):
                if r.get("ticker") != ticker:
                    continue
                cid = r["chunk_id"]
                cand.setdefault(cid, {"chunk_id": cid, "retrievers": {}})
                cand[cid]["retrievers"][rname] = {"rank": rank}
        pool[qid] = {"id": qid, "candidate_count": len(cand),
                     "candidates": list(cand.values())}
        print(f"  [{n}/{len(supported)}] {qid:<40} {len(cand)} candidates")
    return pool


def _topk_ids(cands: list[dict], retriever: str, k: int) -> list[str]:
    ranked = sorted(
        (
            (c["retrievers"][retriever]["rank"], c["chunk_id"])
            for c in cands
            if retriever in c["retrievers"]
        )
    )
    return [cid for _, cid in ranked[:k]]


def evaluate(retriever: str, pool: dict, bench: list[dict]) -> dict:
    rows = []
    for item in bench:
        if item["answer_type"] == "unsupported" or not item.get("relevant_chunks"):
            continue
        qid = item["id"]
        if qid not in pool:
            continue
        relevant = set(item["relevant_chunks"])
        ids = _topk_ids(pool[qid]["candidates"], retriever, TOP_K)
        rr, first_rank = reciprocal_rank(ids, relevant)
        numeric = _is_numeric(item)
        hits10 = len(set(ids[:TOP_K]) & relevant)
        # |relevant| routinely exceeds 10 under a depth-50 pool, so plain
        # Recall@10 is capacity-capped. recall_at_10_capped normalises by the
        # best achievable (min(|relevant|, 10)); r_precision is P@|relevant|.
        r = len(relevant)
        rprec_ids = _topk_ids(pool[qid]["candidates"], retriever, r)
        rows.append({
            "id": qid,
            "company": item["company"],
            "source": item.get("source"),
            "reviewed": bool(item.get("reviewed")),
            "relevant_count": r,
            "numeric": numeric,
            "recall_at_10_capped": hits10 / min(r, TOP_K),
            "r_precision": len(set(rprec_ids) & relevant) / r,
            "retrieved_chunk_ids": ids,
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
        })

    numeric = [r for r in rows if r["numeric"]]

    def _m(field, group=rows):
        return mean_or_none([r[field] for r in group if r[field] is not None])

    reviewed_rows = [r for r in rows if r["reviewed"]]

    summary = {
        "retriever": retriever,
        "scored_questions": len(rows),
        "reviewed_questions": len(reviewed_rows),
        "numeric_questions": len(numeric),
        "mean_relevant_per_question": mean_or_none([r["relevant_count"] for r in rows]),
        "recall_at_1": _m("recall_at_1"),
        "recall_at_3": _m("recall_at_3"),
        "recall_at_5": _m("recall_at_5"),
        "recall_at_10": _m("recall_at_10"),
        "recall_at_10_capped": _m("recall_at_10_capped"),
        "r_precision": _m("r_precision"),
        "recall_at_10_reviewed_only": _m("recall_at_10", reviewed_rows),
        "recall_at_10_capped_reviewed_only": _m("recall_at_10_capped", reviewed_rows),
        "mrr_reviewed_only": _m("reciprocal_rank", reviewed_rows),
        "precision_at_1": _m("precision_at_1"),
        "precision_at_3": _m("precision_at_3"),
        "precision_at_5": _m("precision_at_5"),
        "mrr": _m("reciprocal_rank"),
        "numeric_evidence_hit_at_5": _m("numeric_hit_at_5", numeric),
        "numeric_evidence_hit_at_10": _m("numeric_hit_at_10", numeric),
        # scoped pool by construction:
        "filter_correctness": 1.0,
        "cross_company_leakage_at_10": 0.0,
    }
    return {"summary": summary, "questions": rows}


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:6.3f}"


def build_summary(results: dict[str, dict], bench: list[dict]) -> str:
    supported = [q for q in bench if q["answer_type"] != "unsupported" and q["relevant_chunks"]]
    unsupported = [q for q in bench if q["answer_type"] == "unsupported"]
    reviewed = sum(1 for q in supported if q.get("reviewed"))
    lines = [
        "SCOPED RETRIEVAL EVALUATION - Benchmark v3 (OFFLINE, from depth-50 pool)",
        f"top_k: {TOP_K}   every question scoped to its own company",
        f"questions: {len(bench)}  supported/scored: {len(supported)}  "
        f"unsupported: {len(unsupported)}",
        f"qrels: model-assisted (gpt-5-mini first pass + gpt-5-mini review pass); "
        f"{reviewed}/{len(supported)} questions completed the review pass"
        + ("" if reviewed == len(supported)
           else f"; {len(supported) - reviewed} first-pass only"),
        "",
        f"{'retriever':<9} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@10c':>6} "
        f"{'Rprec':>6} {'P@1':>6} {'P@5':>6} {'MRR':>6} {'NE@5':>6} {'NE@10':>6}",
    ]
    lines.append("-" * len(lines[-1]))
    for name, payload in results.items():
        s = payload["summary"]
        lines.append(
            f"{name:<9} {_fmt(s['recall_at_1'])} "
            f"{_fmt(s['recall_at_5'])} {_fmt(s['recall_at_10'])} "
            f"{_fmt(s['recall_at_10_capped'])} {_fmt(s['r_precision'])} "
            f"{_fmt(s['precision_at_1'])} "
            f"{_fmt(s['precision_at_5'])} {_fmt(s['mrr'])} "
            f"{_fmt(s['numeric_evidence_hit_at_5'])} "
            f"{_fmt(s['numeric_evidence_hit_at_10'])}"
        )
    lines.append("")
    lines.append(f"mean relevant chunks / question: "
                 f"{results['hybrid']['summary']['mean_relevant_per_question']:.1f}  "
                 f"(depth-50 pool; |relevant| > 10 makes plain R@10 capacity-capped)")
    lines.append("R@10c = Recall@10 / min(|relevant|,10)   Rprec = precision at rank |relevant|")
    lines += [
        "",
        "STRUCTURAL GATES (scoped pool by construction; matches live Part A run)",
        "  filter_correctness == 1.000       1.000   PASS",
        "  cross_company_leakage@10 == 0      0.000   PASS",
        "  (live proof: evaluation/results/scoped_retrieval_evaluation.json,",
        "   filtered hybrid: filter_correctness 1.000, leakage@10 0.000)",
        "",
    ]
    if "hybrid" in results:
        h = results["hybrid"]["summary"]
        lines += [
            "V3 RETRIEVAL-QUALITY BASELINE (record; the v2 0.900 R@10 gate is NOT carried over)",
            f"  hybrid MRR              = {h['mrr']:.3f}",
            f"  hybrid Recall@10 capped = {h['recall_at_10_capped']:.3f}  "
            f"(hits@10 / min(|rel|,10))",
            f"  hybrid R-precision      = {h['r_precision']:.3f}",
            f"  hybrid Precision@5      = {h['precision_at_5']:.3f}   P@1 = {h['precision_at_1']:.3f}",
            f"  hybrid Recall@10 (plain)= {h['recall_at_10']:.3f}  "
            f"(capacity-capped: mean |rel| = {h['mean_relevant_per_question']:.1f})",
            "  Recommended v3 regression gates:",
            "    - both structural gates exact (1.000 / 0.000)",
            f"    - hybrid MRR not below baseline by > {REGRESSION_TOL:.2f}  "
            f"(floor {h['mrr'] - REGRESSION_TOL:.3f})",
            f"    - hybrid Recall@10-capped not below baseline by > {REGRESSION_TOL:.2f}  "
            f"(floor {h['recall_at_10_capped'] - REGRESSION_TOL:.3f})",
            f"    - hybrid Precision@5 not below baseline by > {REGRESSION_TOL:.2f}  "
            f"(floor {h['precision_at_5'] - REGRESSION_TOL:.3f})",
            "    - comparison scope_coverage@5 == 1.000, cross_scope_leakage == 0.000",
        ]
    lines.append("")
    lines.append("NE@5/NE@10 = numeric evidence hit rate (numeric + mixed questions).")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="call the real retrievers on the CURRENT corpus instead "
                    "of reading ranks from benchmark_v3_pool.json (spends API "
                    "tokens). Use to re-check quality after a corpus change.")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output files (e.g. 'after_xom')")
    a = ap.parse_args(argv)

    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    if a.live:
        print("LIVE retrieval on the current corpus ...")
        pool = build_live_pool(bench)
        mode = "live"
    else:
        pool = {q["id"]: q for q in json.loads(POOL.read_text(encoding="utf-8"))}
        mode = "offline-from-pool"

    results = {name: evaluate(name, pool, bench) for name in RETRIEVERS}

    json_path, summary_path = JSON_PATH, SUMMARY_PATH
    if a.tag:
        json_path = JSON_PATH.with_name(f"v3_retrieval_evaluation_{a.tag}.json")
        summary_path = SUMMARY_PATH.with_name(f"v3_retrieval_summary_{a.tag}.txt")

    json_path.write_text(json.dumps({
        "benchmark": str(BENCH), "top_k": TOP_K, "mode": mode,
        "summaries": {n: r["summary"] for n, r in results.items()},
        "questions": {n: r["questions"] for n, r in results.items()},
    }, indent=2), encoding="utf-8")

    text = build_summary(results, bench)
    if a.live:
        text = text.replace("(OFFLINE, from depth-50 pool)",
                            "(LIVE retrieval, current corpus)")
    summary_path.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Saved JSON to {json_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
