"""Per-scope hybrid retrieval + union — candidate generation for comparisons.

This module does NOT rerank and does NOT select final evidence. Cohere
reranking and coverage-aware evidence selection stay in ``api/rag.py`` so the
serving architecture (one Cohere call, fallback handling) is unchanged.

What it does:
  1. run the EXISTING hybrid retriever once per scope
  2. reuse a single dense query embedding across all scopes
  3. tag each candidate with the scope(s) that surfaced it
  4. dedupe by chunk_id and return a capped, deterministically ordered union

For Phase 2 a "scope" is a ticker-only ``RetrievalFilter`` (one ticker each),
so scopes are disjoint and every chunk belongs to exactly one scope. The code
does not assume that, so later phases can pass richer scopes (e.g. a
``(ticker, fiscal_year)`` pair).
"""

from __future__ import annotations

import time

from retrieval.embedder import embed_text
from retrieval.filters import RetrievalFilter
from retrieval.hybrid_search import search as hybrid_search

DEFAULT_PER_SCOPE_K = 10
DEFAULT_CANDIDATE_K = 10
DEFAULT_UNION_CAP = 60


def scope_label(scope: RetrievalFilter) -> str:
    """A short, stable label for a scope.

    - single ticker, no other constraint  -> "NVDA"       (Phase 2 form)
    - single ticker + single fiscal year  -> "NVDA:2023"  (Phase 5 form)
    - anything else                       -> scope.describe()
    """
    if scope.tickers and len(scope.tickers) == 1 and not (
        scope.filing_types or scope.filing_ids
    ):
        if not scope.fiscal_years:
            return scope.tickers[0]
        if len(scope.fiscal_years) == 1:
            return f"{scope.tickers[0]}:{scope.fiscal_years[0]}"
    return scope.describe()


def scoped_search(
    query: str,
    scopes: list[RetrievalFilter],
    per_scope_k: int = DEFAULT_PER_SCOPE_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    union_cap: int = DEFAULT_UNION_CAP,
    query_embedding: list[float] | None = None,
) -> tuple[list[dict], dict[str, list[dict]], float]:
    """Retrieve candidates per scope and union them.

    Returns ``(union, per_scope_results, hybrid_ms)``:

    - ``union``: deduped candidate dicts in a deterministic order. Each entry
      gains ``"scopes"`` (labels that retrieved it) and ``"scope_rank"``
      (label -> best 1-based rank within that scope's hybrid list).
    - ``per_scope_results``: ``label -> that scope's hybrid list`` (pre-union),
      used for fallback coverage and diagnostics.
    - ``hybrid_ms``: wall time for embedding + all per-scope hybrid calls.
    """
    if not scopes:
        raise ValueError("scoped_search requires at least one scope")

    start = time.perf_counter()

    if query_embedding is None:
        query_embedding = embed_text(query)

    per_scope: dict[str, list[dict]] = {}
    union: dict[str, dict] = {}

    for scope in scopes:
        label = scope_label(scope)
        results = hybrid_search(
            query,
            top_k=per_scope_k,
            candidate_k=candidate_k,
            filters=scope,
            query_embedding=query_embedding,
        )
        per_scope[label] = results

        for rank, row in enumerate(results, start=1):
            chunk_id = row["chunk_id"]
            entry = union.get(chunk_id)
            if entry is None:
                entry = dict(row)
                entry["scopes"] = []
                entry["scope_rank"] = {}
                union[chunk_id] = entry
            if label not in entry["scopes"]:
                entry["scopes"].append(label)
            entry["scope_rank"][label] = min(
                entry["scope_rank"].get(label, rank), rank
            )

    # Deterministic union order: interleave scopes by best rank, then by the
    # retriever's own RRF score, then chunk_id. Cohere scores every document
    # regardless of input order; this ordering only makes the union stable and
    # keeps the reranker input balanced across companies.
    ordered = sorted(
        union.values(),
        key=lambda entry: (
            min(entry["scope_rank"].values()),
            -float(entry.get("rrf_score", 0.0) or 0.0),
            entry["chunk_id"],
        ),
    )

    hybrid_ms = (time.perf_counter() - start) * 1000
    return ordered[:union_cap], per_scope, hybrid_ms
