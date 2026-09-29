"""Re-check benchmark_v3's retrieval quality against the CURRENT corpus.

Why this exists: benchmark_v3_repool.json's relevance judgments (which
chunk_id is relevant to which question) are a property of the filing text,
not of how many other companies are in the index. The Fortune-100 waves
only *added* companies -- they never touched AAPL/MSFT/etc.'s existing
chunks or re-ingested them, and chunk IDs are deterministic
({TICKER}_{FY}_{FILING_TYPE}_{ACCESSION}_{INDEX}, see README's Ingestion
section), so those judgments are still valid qrels today. What's stale is
the *retrieval ranking* math (BM25 IDF is computed over the whole corpus,
which is now ~10.6x bigger). So: reuse the existing judged qrels, re-run
real retrieval against the current corpus, and recompute the metrics --
without paying for a fresh LLM judging pass over the same questions.

This reuses evaluate_v3_offline.py's own build_live_pool/evaluate/
build_summary unchanged -- only the benchmark file passed in differs
(benchmark_v3_repool.json, the reviewed qrels, instead of the older frozen
benchmark_v3.json this module defaults to).

    python -m evaluation.evaluate_v3_current_corpus
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.evaluate_v3_offline import (
    RETRIEVERS,
    build_live_pool,
    build_summary,
    evaluate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "evaluation" / "benchmark_v3_repool.json"
JSON_PATH = REPO_ROOT / "evaluation" / "results" / "v3_retrieval_evaluation_current_corpus.json"
SUMMARY_PATH = REPO_ROOT / "evaluation" / "results" / "v3_retrieval_summary_current_corpus.txt"


def main() -> int:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    print(f"Loaded {len(bench)} questions (reviewed qrels) from {BENCH.name}")
    print("Running LIVE retrieval against the current corpus ...")
    pool = build_live_pool(bench)

    results = {name: evaluate(name, pool, bench) for name in RETRIEVERS}

    JSON_PATH.write_text(json.dumps({
        "benchmark": str(BENCH),
        "note": "qrels from benchmark_v3_repool.json (reviewed); retrieval is "
                "LIVE against the corpus as of this run -- see companies_count "
                "and chunk_count below for the exact corpus this was measured "
                "against.",
        "top_k": 10,
        "mode": "live-current-corpus",
        "summaries": {n: r["summary"] for n, r in results.items()},
        "questions": {n: r["questions"] for n, r in results.items()},
    }, indent=2), encoding="utf-8")

    text = build_summary(results, bench)
    text = text.replace(
        "(OFFLINE, from depth-50 pool)",
        "(LIVE retrieval, current corpus, reviewed qrels from benchmark_v3_repool.json)",
    )
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
