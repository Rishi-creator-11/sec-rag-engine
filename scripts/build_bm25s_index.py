"""Deterministic prebuild of the persisted bm25s lexical index for deployment.

    python -m scripts.build_bm25s_index            # build + verify
    python -m scripts.build_bm25s_index --check    # verify only (no write)

The deployed runtime (Vercel / AWS Lambda) has a READ-ONLY project filesystem,
so it can never rebuild the index under /var/task. Instead it loads a prebuilt,
persisted index bundled with the function. This script produces that bundle:

  1. load canonical chunks from data/chunks/**            (retrieval.bm25_search.load_chunks)
  2. build with the SAME production settings the runtime uses:
       method="lucene", k1=1.5, b=0.75, dtype="float64",
       tokenizer/stopwords = retrieval.bm25_search.tokenize
  3. persist to data/bm25s_index/ (7 files: 3 .npy + vocab + params +
     chunks.jsonl + corpus_version.json)
  4. verify:
       - load-after-build round-trips (fresh BM25SBackend.load)
       - corpus_version + document_count recorded and match
       - scoped-retrieval PARITY: for a fixed query set, the persisted index
         returns byte-identical top-k (ids + order) to a fresh in-memory build,
         both unfiltered and with a ticker+fiscal_year filter

Ranking is not touched. Commit the resulting data/bm25s_index/ — Vercel bundles
every tracked project file into the Python function by default, so the committed
index ships read-only and `retrieval.lexical_backend` loads it without rebuilding.

Run after any ingestion or metadata change, before deploying.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_search import load_chunks
from retrieval.filters import RetrievalFilter
from retrieval.lexical_backend import (
    BM25S_INDEX_DIR,
    BM25SBackend,
    build_persisted_bm25s,
    corpus_version,
)

PARITY_QUERIES = [
    "export control and China risk",
    "artificial intelligence risk factors",
    "total revenue for the fiscal year",
    "cybersecurity incident response and governance",
    "competition and pricing pressure",
    "supply chain and manufacturing constraints",
]


def _top(backend, query, k=10, filters=None):
    return [(r["chunk_id"], round(r["score"], 6))
            for r in backend.search(query, top_k=k, filters=filters)]


def verify(index_dir: Path) -> list[str]:
    problems: list[str] = []
    chunks = load_chunks()
    expected_version = corpus_version(chunks)

    vf = index_dir / "corpus_version.json"
    if not vf.exists():
        return [f"missing {vf}"]
    meta = json.loads(vf.read_text(encoding="utf-8"))
    if meta.get("corpus_version") != expected_version:
        problems.append(
            f"corpus_version mismatch: file {meta.get('corpus_version')} "
            f"!= corpus {expected_version}"
        )
    if meta.get("document_count") != len(chunks):
        problems.append(
            f"document_count mismatch: file {meta.get('document_count')} "
            f"!= corpus {len(chunks)}"
        )

    persisted = BM25SBackend.load(index_dir)
    fresh = BM25SBackend(chunks)
    if persisted.document_count != len(chunks):
        problems.append(
            f"persisted load has {persisted.document_count} docs, expected {len(chunks)}"
        )

    # ranking parity: persisted vs fresh in-memory build
    filt = RetrievalFilter(tickers=("NVDA",), fiscal_years=(2023,))
    for q in PARITY_QUERIES:
        for f, label in ((None, "unfiltered"), (filt, "NVDA:2023")):
            a = _top(persisted, q, filters=f)
            b = _top(fresh, q, filters=f)
            if a != b:
                problems.append(f"parity[{label}] mismatch for {q!r}: {a[:3]} vs {b[:3]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the existing index; do not rebuild")
    ap.add_argument("--index-dir", default=str(BM25S_INDEX_DIR))
    a = ap.parse_args(argv)
    index_dir = Path(a.index_dir)

    if not a.check:
        chunks = load_chunks()
        print(f"building bm25s index from {len(chunks)} canonical chunks ...")
        backend = build_persisted_bm25s(chunks, index_dir)
        meta = json.loads((index_dir / "corpus_version.json").read_text())
        size_mb = sum(f.stat().st_size for f in index_dir.iterdir()) / 1024 / 1024
        print(f"  wrote {index_dir}  ({size_mb:.2f} MB, {len(list(index_dir.iterdir()))} files)")
        print(f"  document_count = {backend.document_count}")
        print(f"  corpus_version = {meta['corpus_version']}")

    print("verifying (load round-trip + ranking parity) ...")
    problems = verify(index_dir)
    if problems:
        print("\nFAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK — persisted index loads read-only and ranks identically to a fresh build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
