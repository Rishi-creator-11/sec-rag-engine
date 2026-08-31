"""Old-vs-new benchmark comparison for the Phase 5.5 re-pool.

Runs ONE year-scoped live retrieval per supported question (dense / bm25s /
hybrid, exactly as evaluation.evaluate_v3_offline.build_live_pool) against the
CURRENT corpus, then scores that same retrieval two ways:

    A) against the frozen benchmark_v3.json qrels        (the stale baseline)
    B) against the fresh benchmark_v3_repool.json qrels  (re-anchored)

The gap between A and B is the stale-pool dilution. Deterministic given the
corpus (retrieval is called once and reused for both scorings).

    python -m evaluation.compare_v3_repool
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evaluation.evaluate_v3_offline import build_live_pool, _topk_ids, TOP_K  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "evaluation" / "benchmark_v3.json"
REPOOL = REPO / "evaluation" / "benchmark_v3_repool.json"
OUT = REPO / "evaluation" / "results" / "v3_repool_comparison.json"


def _rr(ids, rel):
    for i, cid in enumerate(ids, 1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def _score(pool, bench, retriever):
    rows = []
    for item in bench:
        if item["answer_type"] == "unsupported" or not item.get("relevant_chunks"):
            continue
        qid = item["id"]
        if qid not in pool:
            continue
        rel = set(item["relevant_chunks"])
        ids = _topk_ids(pool[qid]["candidates"], retriever, TOP_K)
        r = len(rel)
        rprec_ids = _topk_ids(pool[qid]["candidates"], retriever, r)
        hits10 = len(set(ids[:TOP_K]) & rel)
        rows.append({
            "mrr": _rr(ids, rel),
            "r10c": hits10 / min(r, TOP_K),
            "p1": 1.0 if ids[:1] and ids[0] in rel else 0.0,
            "p5": len(set(ids[:5]) & rel) / 5,
            "rprec": len(set(rprec_ids) & rel) / r,
            "relevant_count": r,
        })
    m = lambda k: statistics.mean(x[k] for x in rows)
    return {
        "questions": len(rows),
        "MRR": round(m("mrr"), 4), "R@10c": round(m("r10c"), 4),
        "P@1": round(m("p1"), 4), "P@5": round(m("p5"), 4),
        "Rprec": round(m("rprec"), 4),
        "mean_relevant": round(m("relevant_count"), 1),
    }


def main() -> int:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    repool = json.loads(REPOOL.read_text(encoding="utf-8"))

    # single live retrieval per question, scoped to the frozen qrel years
    # (build_live_pool reads relevant_chunks off the bench it is handed)
    print("building one year-scoped live retrieval per question (current corpus)...")
    pool = build_live_pool(frozen, depth=60)

    print("\n" + "=" * 68)
    for retr in ("dense", "bm25s", "hybrid"):
        a = _score(pool, frozen, retr)
        b = _score(pool, repool, retr)
        print(f"\n{retr.upper()}  (same retrieval, two qrel sets)")
        print(f"  {'metric':10} {'frozen v3':>10} {'repool':>10} {'delta':>9}")
        for k in ("MRR", "R@10c", "P@1", "P@5", "Rprec"):
            print(f"  {k:10} {a[k]:>10.4f} {b[k]:>10.4f} {b[k]-a[k]:>+9.4f}")
        print(f"  {'mean|rel|':10} {a['mean_relevant']:>10.1f} {b['mean_relevant']:>10.1f}")

    result = {
        "note": "same current-corpus retrieval scored against frozen vs re-pooled qrels",
        "frozen": {r: _score(pool, frozen, r) for r in ("dense", "bm25s", "hybrid")},
        "repool": {r: _score(pool, repool, r) for r in ("dense", "bm25s", "hybrid")},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
