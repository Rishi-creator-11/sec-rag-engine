"""Re-pool benchmark_v3 against the CURRENT multi-year corpus (Phase 5.5).

The frozen benchmark_v3 pool (evaluation/results/benchmark_v3_pool.json) was
built when every company had exactly ONE 10-K. The corpus is now 4,262 chunks
across 31 filings, and bm25s computes IDF over the whole corpus before scope
filtering, so within-scope lexical rankings drift as the corpus grows. That
drift (live hybrid MRR 0.903 -> 0.900 -> 0.897 across Batches 1-3) is a
stale-pool artifact, not a ranking regression — this script re-anchors it.

  * SAME question set (evaluation/benchmark_v3_questions.json), verbatim.
  * benchmark_v3.json is NEVER touched. Output is a NEW file,
    evaluation/benchmark_v3_repool.json.
  * Each question is scoped to the fiscal year(s) its EXISTING v3 qrels live in
    (evaluation.evaluate_v3_offline._qrel_years) — the benchmark is single-year
    per question by construction, and this keeps the pool comparable to the
    frozen one instead of diluting it with other-year chunks.
  * Pool depth 80 (was 50): the corpus tripled, so pool deeper than production
    candidate_k=10 by a wider margin. dense + bm25s + hybrid + sparse.
  * Unjudged pool chunks are NOT automatically non-relevant — only the judged
    head (JUDGE_HEAD) yields labels; everything else is simply unlabeled and is
    excluded from precision denominators the same way benchmark_v3 did it.

    python -m evaluation.build_v3_repool          # build the pool
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from retrieval.embedder import embed_text  # noqa: E402
from retrieval.pinecone_search import search as dense_search  # noqa: E402
from retrieval.bm25_search import search as bm25_search  # noqa: E402
from retrieval.hybrid_search import search as hybrid_search  # noqa: E402
from retrieval.filters import RetrievalFilter  # noqa: E402
from evaluation.evaluate_v3_offline import _qrel_years  # noqa: E402

try:
    from retrieval.sparse_search import search as sparse_search
    _HAVE_SPARSE = True
except Exception:  # noqa: BLE001
    _HAVE_SPARSE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "evaluation" / "benchmark_v3_questions.json"
FROZEN = REPO_ROOT / "evaluation" / "benchmark_v3.json"
OUTPUT = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_repool.json"

POOL_DEPTH = 80


def _merge(pool: dict, results: list[dict], retriever: str, ticker: str,
           years: list[int] | None) -> None:
    for rank, r in enumerate(results, start=1):
        if r.get("ticker") != ticker:
            continue
        if years is not None:
            fy = r.get("fiscal_year")
            if fy in (None, "") or int(fy) not in years:
                continue
        cid = r["chunk_id"]
        if cid not in pool:
            pool[cid] = {
                "chunk_id": cid, "ticker": r.get("ticker"),
                "company": r.get("company"), "fiscal_year": r.get("fiscal_year"),
                "text": r.get("text", ""), "retrievers": {},
            }
        pool[cid]["retrievers"][retriever] = {
            "rank": rank, "score": r.get("score", r.get("rrf_score")),
        }


def main() -> int:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    frozen = {q["id"]: q for q in json.loads(FROZEN.read_text(encoding="utf-8"))}
    supported = [q for q in questions if q.get("answer_type") != "unsupported"]
    print(f"re-pooling {len(supported)} supported questions at depth {POOL_DEPTH} "
          f"(year-scoped)  sparse={'yes' if _HAVE_SPARSE else 'no'}")

    out, t0 = [], time.time()
    for index, item in enumerate(supported, start=1):
        ticker = item["company"]
        q = item["question"]
        fz = frozen.get(item["id"], {})
        years = _qrel_years(fz.get("relevant_chunks", []))
        filt = RetrievalFilter(
            tickers=(ticker,),
            fiscal_years=tuple(years) if years else None,
        )
        vec = embed_text(q)

        pool: dict = {}
        _merge(pool, dense_search(q, top_k=POOL_DEPTH, filters=filt, query_embedding=vec),
               "dense", ticker, years)
        _merge(pool, bm25_search(q, top_k=POOL_DEPTH, filters=filt), "bm25s", ticker, years)
        _merge(pool, hybrid_search(q, top_k=POOL_DEPTH, candidate_k=POOL_DEPTH,
                                   filters=filt, query_embedding=vec),
               "hybrid", ticker, years)
        if _HAVE_SPARSE:
            try:
                _merge(pool, sparse_search(q, top_k=POOL_DEPTH * 3), "sparse", ticker, years)
            except Exception as exc:  # noqa: BLE001
                print(f"  sparse skipped for {item['id']}: {type(exc).__name__}")

        candidates = sorted(
            pool.values(),
            key=lambda c: (-len(c["retrievers"]),
                           min(s["rank"] for s in c["retrievers"].values()),
                           c["chunk_id"]),
        )
        out.append({
            "id": item["id"], "question": q, "company": ticker,
            "category": item.get("category"),
            "answer_type": item.get("answer_type"),
            "number_hint": fz.get("number_hint"),
            "scoped_years": years,
            "frozen_relevant_chunks": fz.get("relevant_chunks", []),
            "candidate_count": len(candidates),
            "candidates": candidates,
        })
        print(f"  [{index}/{len(supported)}] {item['id']:<40} yr={years} "
              f"{len(candidates)} candidates")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    total = sum(o["candidate_count"] for o in out)
    print(f"\nre-pooled {len(out)} questions, {total} candidates "
          f"(avg {total/len(out):.1f}/q) in {time.time()-t0:.0f}s -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
