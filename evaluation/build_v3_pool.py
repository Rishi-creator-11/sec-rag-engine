"""Build a DEEP, SCOPED candidate pool for benchmark_v3 (8-company corpus).

For every supported question, pool the top-N from each retriever, SCOPED to the
question's company (matching how production actually retrieves):

    dense   top 50   (Pinecone, filtered)
    bm25s   top 50   (production lexical backend, filtered)
    hybrid  top 50   (RRF of dense+bm25s, filtered)
    sparse  top 50   (Pinecone sparse, unfiltered then post-filtered) - pool
                     diversity only; sparse stays disabled in production

Pooling depth (50) is deliberately deeper than production candidate_k (10) so
that qrels are not limited to what production surfaces. Union + dedupe; record
which retriever(s) contributed each chunk and at what rank.

Input:  evaluation/benchmark_v3_questions.json
Output: evaluation/results/benchmark_v3_pool.json
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

try:
    from retrieval.sparse_search import search as sparse_search
    _HAVE_SPARSE = True
except Exception:  # noqa: BLE001
    _HAVE_SPARSE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "evaluation" / "benchmark_v3_questions.json"
OUTPUT = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_pool.json"

POOL_DEPTH = 50


def _merge(pool: dict, results: list[dict], retriever: str, ticker: str) -> None:
    for rank, r in enumerate(results, start=1):
        if r.get("ticker") != ticker:
            continue  # scoped pool: same-company only
        cid = r["chunk_id"]
        if cid not in pool:
            pool[cid] = {
                "chunk_id": cid,
                "ticker": r.get("ticker"),
                "company": r.get("company"),
                "filing_type": r.get("filing_type"),
                "text": r.get("text", ""),
                "retrievers": {},
            }
        pool[cid]["retrievers"][retriever] = {
            "rank": rank,
            "score": r.get("score", r.get("rrf_score")),
        }


def main() -> int:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    supported = [q for q in questions if q.get("answer_type") != "unsupported"]
    print(f"pooling {len(supported)} supported questions at depth {POOL_DEPTH} "
          f"(scoped)  sparse={'yes' if _HAVE_SPARSE else 'no'}")

    out = []
    for index, item in enumerate(supported, start=1):
        ticker = item["company"]
        q = item["question"]
        filt = RetrievalFilter(tickers=(ticker,))
        vec = embed_text(q)

        pool: dict = {}
        _merge(pool, dense_search(q, top_k=POOL_DEPTH, filters=filt, query_embedding=vec),
               "dense", ticker)
        _merge(pool, bm25_search(q, top_k=POOL_DEPTH, filters=filt), "bm25s", ticker)
        _merge(pool, hybrid_search(q, top_k=POOL_DEPTH, candidate_k=POOL_DEPTH,
                                   filters=filt, query_embedding=vec), "hybrid", ticker)
        if _HAVE_SPARSE:
            try:
                _merge(pool, sparse_search(q, top_k=POOL_DEPTH * 3), "sparse", ticker)
            except Exception as exc:  # noqa: BLE001
                print(f"  sparse skipped for {item['id']}: {type(exc).__name__}")

        candidates = sorted(
            pool.values(),
            key=lambda c: (
                -len(c["retrievers"]),
                min(s["rank"] for s in c["retrievers"].values()),
                c["chunk_id"],
            ),
        )
        out.append({
            "id": item["id"],
            "question": q,
            "company": ticker,
            "category": item.get("category"),
            "answer_type": item.get("answer_type"),
            "number_hint": item.get("number_hint"),
            "v2_relevant_chunks": item.get("v2_relevant_chunks", []),
            "candidate_count": len(candidates),
            "candidates": candidates,
        })
        print(f"  [{index}/{len(supported)}] {item['id']:<42} {len(candidates)} candidates")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    total = sum(o["candidate_count"] for o in out)
    print(f"\npooled {len(out)} questions, {total} total candidates "
          f"(avg {total/len(out):.1f}/question) -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
