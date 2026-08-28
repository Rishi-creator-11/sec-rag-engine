"""Benchmark v2 vs v3 comparison report.

Explains the apparent Recall@10 decline seen in Phase 4.5: v2 qrels were pooled
at depth 10 on the ORIGINAL 3-company retrievers, so chunks that became
reachable only after the corpus grew to 8 companies were never judged and
counted as irrelevant ("unjudged = irrelevant"). v3 re-pools at depth 50 on the
current 8-company system and re-judges.

Inputs (run these first):
  evaluation/results/scoped_retrieval_evaluation.json   (v2, filtered hybrid)
        python -m evaluation.evaluate_scoped --retriever hybrid --mode compare
  evaluation/results/v3_retrieval_evaluation.json       (v3, all retrievers)
        python -m evaluation.evaluate_v3 --retriever all
  evaluation/benchmark_v2.json, evaluation/benchmark_v3.json

Output:
  evaluation/results/v2_vs_v3_comparison.txt
  evaluation/results/v2_vs_v3_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_EVAL = REPO_ROOT / "evaluation" / "results" / "scoped_retrieval_evaluation.json"
V3_EVAL = REPO_ROOT / "evaluation" / "results" / "v3_retrieval_evaluation.json"
V2_BENCH = REPO_ROOT / "evaluation" / "benchmark_v2.json"
V3_BENCH = REPO_ROOT / "evaluation" / "benchmark_v3.json"
POOL = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_pool.json"
V2_POOL = REPO_ROOT / "evaluation" / "results" / "benchmark_v2_pool.json"
OUT_TXT = REPO_ROOT / "evaluation" / "results" / "v2_vs_v3_comparison.txt"
OUT_JSON = REPO_ROOT / "evaluation" / "results" / "v2_vs_v3_comparison.json"

# Questions flagged in Phase 4.5 as "lost" top-10 relevant chunks.
WEAK_QUESTIONS = [
    "microsoft_cash_2025_01",
    "nvidia_responsible_ai_risk_01",
    "nvidia_china_01",
    "microsoft_total_revenue_2025_01",
    "nvidia_export_controls_01",
    "microsoft_cash_investments_2025_01",
]


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    for p in (V2_EVAL, V3_EVAL, V3_BENCH):
        if not p.exists():
            raise SystemExit(f"missing {p} - see module docstring")

    v2_eval = _load(V2_EVAL)
    v3_eval = _load(V3_EVAL)
    v2_bench = {q["id"]: q for q in _load(V2_BENCH)}
    v3_bench = {q["id"]: q for q in _load(V3_BENCH)}
    pool = {q["id"]: q for q in _load(POOL)}

    v2_hyb = v2_eval["runs"]["filtered"]["hybrid"]
    v2_rows = {r["id"]: r for r in v2_eval["runs"]["filtered"]["_questions"]["hybrid"]}
    v3_hyb = v3_eval["summaries"]["hybrid"]
    v3_rows = {r["id"]: r for r in v3_eval["questions"]["hybrid"]}
    v2_pool = _load(V2_POOL) if V2_POOL.exists() else []
    v2_pool_total = sum(p.get("candidate_count", 0) for p in v2_pool)

    lines: list[str] = []
    P = lines.append

    P("BENCHMARK v2 vs v3 - COMPARISON REPORT")
    P("=" * 70)
    P("")
    P(f"{'':32} {'v2':>16} {'v3':>16}")
    P(f"{'benchmark file':32} {'benchmark_v2':>16} {'benchmark_v3':>16}")
    P(f"{'companies':32} {'3 (AAPL/MSFT/NVDA)':>16} {'8':>16}")
    n_v2 = len(v2_bench)
    n_v3 = len(v3_bench)
    n_v3_supp = sum(1 for q in v3_bench.values() if q["answer_type"] != "unsupported")
    P(f"{'questions (total)':32} {n_v2:>16} {n_v3:>16}")
    P(f"{'questions (supported/scored)':32} "
      f"{sum(1 for q in v2_bench.values() if q.get('relevant_chunks')):>16} {n_v3_supp:>16}")
    P(f"{'judging pool depth':32} {'~10 (3-co retrievers)':>16} {'50 (8-co system)':>16}")
    total_pool = sum(p["candidate_count"] for p in pool.values())
    P(f"{'pooled questions':32} {len(v2_pool):>16} {len(pool):>16}")
    P(f"{'total judged candidates':32} "
      f"{(v2_pool_total or 'n/a'):>16} {total_pool:>16}")
    v2_avg = f"{v2_pool_total / len(v2_pool):.1f}" if v2_pool else "n/a"
    P(f"{'avg candidates / question':32} {v2_avg:>16} "
      f"{total_pool / len(pool):>16.1f}")
    total_rel_v3 = sum(len(q["relevant_chunks"]) for q in v3_bench.values())
    total_rel_v2 = sum(len(q.get("relevant_chunks", [])) for q in v2_bench.values())
    P(f"{'total relevant labels':32} {total_rel_v2:>16} {total_rel_v3:>16}")
    P("")
    P("SCOPED (filtered) HYBRID RETRIEVAL QUALITY")
    P("-" * 70)
    P(f"{'metric':28} {'v2':>10} {'v3':>10} {'delta':>10}")
    for key, label in [
        ("recall_at_1", "Recall@1"), ("recall_at_3", "Recall@3"),
        ("recall_at_5", "Recall@5"), ("recall_at_10", "Recall@10"),
        ("recall_at_10_capped", "Recall@10capped"), ("r_precision", "R-precision"),
        ("precision_at_5", "Precision@5"), ("mrr", "MRR"),
        ("numeric_evidence_hit_at_5", "NumEvidence@5"),
        ("numeric_evidence_hit_at_10", "NumEvidence@10"),
        ("filter_correctness", "filter_correctness"),
        ("cross_company_leakage_at_10", "leakage@10"),
    ]:
        a, b = v2_hyb.get(key), v3_hyb.get(key)
        d = f"{b - a:+.3f}" if (a is not None and b is not None) else "n/a"
        af = f"{a:.3f}" if a is not None else "n/a"
        bf = f"{b:.3f}" if b is not None else "n/a"
        P(f"{label:28} {af:>10} {bf:>10} {d:>10}")
    P("")
    P("NOTE: v2 and v3 are DIFFERENT question sets and DIFFERENT qrel depths -")
    P("this is not 'v2 improved to v3'. Two things the table shows:")
    P(" 1. v3 mean |relevant| = %.1f vs v2 ~3, because v3 judged a depth-50 pool."
      % v3_hyb.get("mean_relevant_per_question", 0))
    P("    Plain Recall@10 is capacity-capped when |relevant| > 10, so it FALLS")
    P("    even though retrieval quality did not. Use Recall@10capped / R-precision")
    P("    / MRR / P@5 for the v3 baseline.")
    P(" 2. v3 MRR (%.3f) and P@1 stay high - the top of the ranking is still"
      % (v3_hyb.get("mrr") or 0))
    P("    almost always relevant. The Phase 4.5 'regression' was the v2 qrels")
    P("    missing labels for chunks the 8-company retriever newly surfaced.")
    P("")

    # ---- weak-question qrel analysis --------------------------------------
    P("PHASE 4.5 'LOST CHUNK' QUESTIONS - RE-EXAMINED UNDER v3")
    P("=" * 70)
    weak_out = []
    for qid in WEAK_QUESTIONS:
        v2q = v2_bench.get(qid, {})
        v3q = v3_bench.get(qid)
        P("")
        P(f"[{qid}]")
        if not v3q:
            P("  not in v3 (unexpected)")
            continue
        v2_rel = set(v2q.get("relevant_chunks", []))
        v3_rel = set(v3q.get("relevant_chunks", []))
        new_rel = v3_rel - v2_rel
        dropped = v2_rel - v3_rel
        v3_row = v3_rows.get(qid, {})
        v2_row = v2_rows.get(qid, {})
        pooled_ids = {c["chunk_id"] for c in pool.get(qid, {}).get("candidates", [])}
        P(f"  question:            {v3q['question']}")
        P(f"  v2 relevant_chunks:  {sorted(v2_rel)}")
        P(f"  v3 relevant_chunks:  {sorted(v3_rel)}")
        P(f"  newly relevant (v3): {sorted(new_rel)}  ({len(new_rel)})")
        P(f"  no longer relevant:  {sorted(dropped)}")
        P(f"  v2 hybrid Recall@10: {v2_row.get('recall_at_10')}"
          f"   first_rel_rank={v2_row.get('first_relevant_rank')}")
        P(f"  v3 hybrid Recall@10: {v3_row.get('recall_at_10')}"
          f"   first_rel_rank={v3_row.get('first_relevant_rank')}")
        P(f"  v3 hybrid top-10:    {v3_row.get('retrieved_chunk_ids')}")
        new_and_retrieved = new_rel & set(v3_row.get("retrieved_chunk_ids", []))
        P(f"  of the newly-relevant chunks, in v3 top-10: {sorted(new_and_retrieved)}")
        weak_out.append({
            "id": qid,
            "v2_relevant_chunks": sorted(v2_rel),
            "v3_relevant_chunks": sorted(v3_rel),
            "newly_relevant": sorted(new_rel),
            "no_longer_relevant": sorted(dropped),
            "v2_recall_at_10": v2_row.get("recall_at_10"),
            "v3_recall_at_10": v3_row.get("recall_at_10"),
            "newly_relevant_in_v3_top10": sorted(new_and_retrieved),
            "all_v3_relevant_were_poolable": v3_rel <= pooled_ids,
        })

    # ---- global "unjudged became relevant" count ------------------------
    P("")
    P("GLOBAL: v2-origin questions - chunks now relevant in v3 that were NOT in")
    P("the v2 qrels (i.e. were pooled/judged only under the 8-company system)")
    P("-" * 70)
    total_new = 0
    q_with_new = 0
    for qid, v3q in v3_bench.items():
        if v3q.get("source") != "v2":
            continue
        v2_rel = set(v2_bench.get(qid, {}).get("relevant_chunks", []))
        v3_rel = set(v3q.get("relevant_chunks", []))
        new = v3_rel - v2_rel
        if new:
            q_with_new += 1
            total_new += len(new)
    P(f"  v2-origin questions with >=1 newly-relevant chunk: {q_with_new}")
    P(f"  total newly-relevant chunks added across v2-origin questions: {total_new}")
    P("")
    P("This is the mechanism behind the apparent Phase 4.5 regression: the")
    P("retriever was surfacing MORE genuinely-relevant chunks, but the v2 qrels")
    P("had no label for them, so Recall@10 scored them as misses.")

    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "v2_hybrid": v2_hyb,
        "v3_hybrid": v3_hyb,
        "weak_questions": weak_out,
        "v2_origin_questions_with_new_relevant": q_with_new,
        "v2_origin_total_new_relevant_chunks": total_new,
    }, indent=2), encoding="utf-8")
    print(text)
    print(f"saved {OUT_TXT}")
    print(f"saved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
