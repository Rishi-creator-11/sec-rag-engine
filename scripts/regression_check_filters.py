"""Phase 1B regression guard: prove ``filters=None`` preserves current behavior.

Compares a captured baseline (produced by running this script with
``--capture`` on the pre-change code, via ``git stash``) against the current
code called with ``filters=None`` and with no filter argument at all.

Checks, per query, for dense / bm25 / hybrid:
  - identical chunk_id set
  - identical ordering
  - scores equal within tolerance

BM25 is pure-Python and deterministic, so its tolerance is 0. The Pinecone
dense index is an approximate (ANN) service whose returned scores vary by
~1e-4 between two identical consecutive calls (verified on the unmodified
code); dense/hybrid therefore use a small numeric tolerance while still
requiring identical IDs and ordering.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

SCRATCH = pathlib.Path(
    "/private/tmp/claude-501/-Users-rishiarora-sec-rag-engine/"
    "f730df29-7a59-4e2e-9ba4-6e353a4a41e1/scratchpad"
)
BASELINE_PATH = SCRATCH / "phase1b_regression_baseline.json"

QUERIES = [
    "What privacy risks does Apple face?",
    "What are Apple's major competitive risks?",
    "What does Apple say about Greater China?",
    "What does Microsoft say about artificial intelligence?",
    "What competitive risks does Microsoft face?",
    "What cybersecurity risks does Microsoft describe?",
    "What risks does NVIDIA face from export controls?",
    "What supply chain risks does NVIDIA face?",
    "How much revenue did Apple generate from iPhone 17 in fiscal 2025?",
    "What total revenue did NVIDIA report for fiscal 2027?",
]

DENSE_SCORE_TOL = 2e-3
BM25_SCORE_TOL = 0.0


def _score(row: dict) -> float:
    if row.get("score") is not None:
        return float(row["score"])
    return float(row.get("rrf_score", 0.0))


def capture() -> dict:
    from retrieval.bm25_search import search as bm25_search
    from retrieval.hybrid_search import search as hybrid_search
    from retrieval.pinecone_search import search as dense_search

    def run(fn, **kwargs):
        out = {}
        for query in QUERIES:
            rows = fn(query, **kwargs)
            out[query] = [[r["chunk_id"], _score(r)] for r in rows]
        return out

    return {
        "dense": run(dense_search, top_k=10),
        "bm25": run(bm25_search, top_k=10),
        "hybrid": run(hybrid_search, top_k=10, candidate_k=10),
    }


def capture_current() -> dict:
    """Capture current code via both filters=None and the no-arg default."""
    from retrieval.bm25_search import search as bm25_search
    from retrieval.filters import RetrievalFilter
    from retrieval.hybrid_search import search as hybrid_search
    from retrieval.pinecone_search import search as dense_search

    def run(fn, filt, **kwargs):
        out = {}
        for query in QUERIES:
            rows = fn(query, filters=filt, **kwargs) if filt is not None else fn(query, **kwargs)
            out[query] = [[r["chunk_id"], _score(r)] for r in rows]
        return out

    empty = RetrievalFilter()
    return {
        "dense_none": run(dense_search, None, top_k=10),
        "dense_empty": run(dense_search, empty, top_k=10),
        "bm25_none": run(bm25_search, None, top_k=10),
        "bm25_empty": run(bm25_search, empty, top_k=10),
        "hybrid_none": run(hybrid_search, None, top_k=10, candidate_k=10),
        "hybrid_empty": run(hybrid_search, empty, top_k=10, candidate_k=10),
    }


def compare(label: str, old: dict, new: dict, tol: float) -> int:
    problems = 0
    for query, old_rows in old.items():
        new_rows = new[query]
        old_ids = [cid for cid, _ in old_rows]
        new_ids = [cid for cid, _ in new_rows]
        if old_ids != new_ids:
            problems += 1
            print(f"  [{label}] IDS/ORDER differ for: {query}")
            print(f"      old: {old_ids}")
            print(f"      new: {new_ids}")
            continue
        worst = max(
            (abs(so - sn) for (_, so), (_, sn) in zip(old_rows, new_rows)),
            default=0.0,
        )
        if worst > tol:
            problems += 1
            print(f"  [{label}] score delta {worst:.2e} > tol {tol:.0e}: {query}")
    status = "IDENTICAL" if problems == 0 else f"{problems} PROBLEM(S)"
    print(f"{label:<28} {status}  ({len(old)} queries, max_tol={tol:.0e})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true",
                        help="write the baseline (run on pre-change code)")
    args = parser.parse_args(argv)

    if args.capture:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(capture(), indent=1), encoding="utf-8")
        print(f"baseline written to {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.exists():
        print(f"no baseline at {BASELINE_PATH}; run with --capture on pre-change code")
        return 2

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current = capture_current()

    total = 0
    total += compare("dense  filters=None", baseline["dense"], current["dense_none"], DENSE_SCORE_TOL)
    total += compare("dense  RetrievalFilter()", baseline["dense"], current["dense_empty"], DENSE_SCORE_TOL)
    total += compare("bm25   filters=None", baseline["bm25"], current["bm25_none"], BM25_SCORE_TOL)
    total += compare("bm25   RetrievalFilter()", baseline["bm25"], current["bm25_empty"], BM25_SCORE_TOL)
    total += compare("hybrid filters=None", baseline["hybrid"], current["hybrid_none"], DENSE_SCORE_TOL)
    total += compare("hybrid RetrievalFilter()", baseline["hybrid"], current["hybrid_empty"], DENSE_SCORE_TOL)

    print()
    print(f"REGRESSION: {'PASS' if total == 0 else f'FAIL ({total})'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
