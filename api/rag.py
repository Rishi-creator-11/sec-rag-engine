import logging
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.cohere_reranker import (
    is_rate_limited,
    rerank_enabled,
    rerank_timed,
)
from retrieval.filters import RetrievalFilter
from retrieval.hybrid_search import search as hybrid_search
from retrieval.embedder import embed_text
from retrieval.scoped_search import scoped_search


load_dotenv()

logger = logging.getLogger(__name__)


class ScopeViolationError(RuntimeError):
    """Retrieval returned evidence outside the requested ticker scope, or
    coverage-aware selection dropped a company that had candidates.

    After Phase 1B/2 this must never happen; the check exists to fail loudly
    rather than serve wrong-company or one-sided evidence if a future change
    regresses filter plumbing or the selection logic.
    """

client = OpenAI()

ANSWER_MODEL = "gpt-5-nano"
HYBRID_TOP_K = 10
CANDIDATE_K = 10
DEFAULT_EVIDENCE_K = 5
MIN_EVIDENCE_K = 2
MAX_OUTPUT_TOKENS = 800

# Comparison (multi-ticker) retrieval.
UNION_CAP = 60          # hard cap on the deduped per-scope candidate union
MIN_PER_SCOPE = 1       # guaranteed final evidence chunks per requested ticker

# Trailing-chunk drop rule, documented from observed Cohere scores (~0.15-0.90):
# keep a lower-ranked chunk only if it is still reasonably close to the top hit.
SCORE_ABS_FLOOR = 0.40
SCORE_RELATIVE_FLOOR = 0.50
MAX_OVERLAP_CHARS = 800
MIN_OVERLAP_CHARS = 80

PAGE_MARK_RE = re.compile(
    r"(?:[A-Za-z0-9 .,&]+\s*\|\s*)?\d{4}\s*Form\s*10-K\s*\|\s*\d+",
    re.IGNORECASE,
)
BOILERPLATE_LINE_RE = re.compile(
    r"(?im)^(?:table of contents|form 10-k)\s*$",
)


def compact_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def strip_boilerplate(text: str) -> str:
    cleaned = PAGE_MARK_RE.sub(" ", text or "")
    lines = [
        line
        for line in cleaned.splitlines()
        if not BOILERPLATE_LINE_RE.match(line.strip())
    ]
    return "\n".join(lines)


def trim_overlap(previous: str, current: str) -> str:
    if not previous or not current:
        return current

    max_size = min(MAX_OVERLAP_CHARS, len(previous), len(current))

    for size in range(max_size, MIN_OVERLAP_CHARS - 1, -1):
        if previous[-size:] == current[:size]:
            return current[size:].lstrip()

    return current


def compress_chunk_text(text: str, previous_text: str = "") -> str:
    """Compact evidence text without summarizing or rewriting it."""
    cleaned = compact_whitespace(strip_boilerplate(text))

    if previous_text:
        cleaned = trim_overlap(previous_text, cleaned)

    return cleaned


def prepare_evidence(chunks: list[dict]) -> list[dict]:
    """Return copies with compacted text. Stored retrieval chunks stay unchanged."""
    prepared = []
    previous_text = ""

    for chunk in chunks:
        prepared_chunk = dict(chunk)
        prepared_chunk["text"] = compress_chunk_text(
            chunk.get("text", ""),
            previous_text,
        )
        previous_text = prepared_chunk["text"]
        prepared.append(prepared_chunk)

    return prepared


def select_evidence(
    ranked_chunks: list[dict],
    max_k: int = DEFAULT_EVIDENCE_K,
) -> list[dict]:
    """Keep the strongest Cohere hits; drop a weak tail when the gap is clear.

    Always keeps the top chunk. Keeps at least two chunks when available.
    If rerank scores are missing, keeps up to max_k unchanged.
    """
    if not ranked_chunks:
        return []

    limited = ranked_chunks[:max_k]
    scores = [chunk.get("rerank_score") for chunk in limited]

    if any(score is None for score in scores):
        return limited

    top_score = float(scores[0])
    selected = [limited[0]]

    for chunk, score in zip(limited[1:], scores[1:]):
        score_value = float(score)
        keep = (
            len(selected) < MIN_EVIDENCE_K
            or score_value >= SCORE_ABS_FLOOR
            or (
                top_score > 0
                and score_value >= top_score * SCORE_RELATIVE_FLOOR
            )
        )

        if not keep:
            break

        selected.append(chunk)

    return selected


def select_evidence_with_coverage(
    ranked_chunks: list[dict],
    requested_tickers: list[str],
    evidence_k: int,
    min_per_scope: int = MIN_PER_SCOPE,
) -> list[dict]:
    """Coverage-aware evidence selection for comparison mode.

    ``ranked_chunks`` are candidates in final rank order (Cohere order, or the
    per-scope union order on fallback).

    1. Reserve up to ``min_per_scope`` chunks for every requested ticker that
       has at least one candidate, taking that ticker's highest-ranked chunks.
    2. Fill the remaining slots (up to ``evidence_k``) strictly by global rank
       order — no per-company cap, so quality can favor one company
       (e.g. AAPL 4 / MSFT 1 is fine; AAPL 5 / MSFT 0 is not).
    3. Return the selection in global rank order for stable presentation.

    ``evidence_k`` is expected to already be >= number of requested tickers
    (the caller auto-raises it), so the reserved chunks always fit.
    """
    if not ranked_chunks:
        return []

    rank_of = {chunk["chunk_id"]: index for index, chunk in enumerate(ranked_chunks)}

    by_ticker: dict[str, list[dict]] = {}
    for chunk in ranked_chunks:
        ticker = str(chunk.get("ticker", "")).strip().upper()
        by_ticker.setdefault(ticker, []).append(chunk)

    selected_ids: set[str] = set()
    selected: list[dict] = []

    for ticker in requested_tickers:
        for chunk in by_ticker.get(ticker, [])[:min_per_scope]:
            if chunk["chunk_id"] not in selected_ids:
                selected_ids.add(chunk["chunk_id"])
                selected.append(chunk)

    for chunk in ranked_chunks:
        if len(selected) >= evidence_k:
            break
        if chunk["chunk_id"] not in selected_ids:
            selected_ids.add(chunk["chunk_id"])
            selected.append(chunk)

    selected.sort(key=lambda chunk: rank_of.get(chunk["chunk_id"], len(ranked_chunks)))
    return selected


def evidence_by_scope(
    evidence: list[dict],
    requested_tickers: list[str],
) -> dict[str, int]:
    """Count final evidence chunks per requested ticker (0 included)."""
    counts = {ticker: 0 for ticker in requested_tickers}
    for chunk in evidence:
        ticker = str(chunk.get("ticker", "")).strip().upper()
        if ticker in counts:
            counts[ticker] += 1
    return counts


def build_context(results: list[dict]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"SOURCE {index}\n"
            f"Company: {result.get('company')}\n"
            f"Ticker: {result.get('ticker')}\n"
            f"Filing: {result.get('filing_type')} "
            f"{result.get('filing_date')}\n"
            f"Chunk ID: {result.get('chunk_id')}\n"
            f"{result.get('text', '')}"
        )

    return "\n\n".join(context_parts)


def build_generation_request(
    question: str,
    context: str,
    comparison_tickers: list[str] | None = None,
) -> dict:
    instructions = (
        'Answer only from the SEC excerpts below. No outside knowledge.\n'
        'Do not invent numbers or facts. Cite a claim with [Source N] only when that\n'
        'source supports it; use the fewest citations needed.\n'
        'If the excerpts lack the answer, reply exactly:\n'
        '"The provided SEC filing excerpts do not contain enough information to answer this question."\n'
        'Keep answers short: 1-3 sentences for numbers; one short paragraph or short bullets otherwise.'
    )

    if comparison_tickers:
        instructions += (
            "\n\nThe user requested a comparison across these companies: "
            + ", ".join(comparison_tickers)
            + ".\nAddress each requested company explicitly, using only that "
            "company's own excerpts. If the excerpts do not contain enough "
            "information for a requested company, say so plainly for that "
            "company. Never infer or transfer facts for one company from "
            "another company's excerpts."
        )

    prompt = f"""{instructions}

Question: {question}

Excerpts:
{context}
"""
    return {
        "model": ANSWER_MODEL,
        "input": prompt,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    }


def build_sources(
    evidence: list[dict],
    hybrid_results: list[dict],
) -> list[dict]:
    hybrid_rank_by_id = {
        result["chunk_id"]: index
        for index, result in enumerate(hybrid_results, start=1)
    }

    sources = []

    for result in evidence:
        chunk_id = result["chunk_id"]
        sources.append(
            {
                "chunk_id": chunk_id,
                "company": result.get("company"),
                "ticker": result.get("ticker"),
                "filing_type": result.get("filing_type"),
                "date": result.get("filing_date"),
                "filing_date": result.get("filing_date"),
                "source_url": result.get("source_url"),
                "rerank_score": result.get("rerank_score"),
                "hybrid_rank": hybrid_rank_by_id.get(chunk_id),
                "retrieval_score": result.get("rrf_score", result.get("score")),
                "text": result.get("text", ""),
            }
        )

    return sources


def rerank_once(
    question: str,
    candidates: list[dict],
) -> tuple[list[dict], bool, str | None, float]:
    """Run the existing Cohere reranker exactly once over ``candidates``.

    Returns ``(ranked, fallback, reason, rerank_ms)``. On disabled reranking or
    any Cohere error the candidates are returned unchanged (for comparison mode
    that is the per-scope union, never a joint search) with ``fallback=True``.
    """
    if not rerank_enabled():
        logger.warning("COHERE_RERANK_ENABLED is false; skipping Cohere rerank")
        return candidates, True, "disabled", 0.0

    rerank_start = time.perf_counter()
    try:
        reranked, rerank_seconds = rerank_timed(question, candidates)
        return reranked, False, None, rerank_seconds * 1000
    except Exception as error:
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        if is_rate_limited(error):
            logger.warning(
                "Cohere rerank rate-limited (429); falling back to hybrid candidates"
            )
            reason = "rate_limited"
        else:
            logger.exception(
                "Cohere rerank failed; falling back to hybrid candidates"
            )
            reason = "api_error"
        return candidates, True, reason, rerank_ms


def retrieve_evidence(
    question: str,
    evidence_k: int,
    filters: RetrievalFilter | None = None,
) -> tuple[list[dict], list[dict], bool, str | None, float, float]:
    hybrid_start = time.perf_counter()
    hybrid_results = hybrid_search(
        question,
        top_k=HYBRID_TOP_K,
        candidate_k=CANDIDATE_K,
        filters=filters,
    )
    hybrid_ms = (time.perf_counter() - hybrid_start) * 1000

    ranked, fallback, reason, rerank_ms = rerank_once(question, hybrid_results)

    # select_evidence returns ranked[:evidence_k] verbatim when rerank scores
    # are absent, so the fallback path is identical to the pre-Phase-2
    # `hybrid_results[:evidence_k]`.
    evidence = select_evidence(ranked, max_k=evidence_k)

    return hybrid_results, evidence, fallback, reason, hybrid_ms, rerank_ms


def retrieve_evidence_comparison(
    question: str,
    evidence_k: int,
    requested_tickers: list[str],
) -> dict:
    """Comparison path: per-ticker hybrid -> union -> ONE Cohere rerank ->
    coverage-aware selection. Returns a dict with everything needed to build
    the response (no generation)."""
    scopes = [RetrievalFilter(tickers=(ticker,)) for ticker in requested_tickers]

    union, per_scope, hybrid_ms = scoped_search(
        question,
        scopes,
        per_scope_k=HYBRID_TOP_K,
        candidate_k=CANDIDATE_K,
        union_cap=UNION_CAP,
    )

    ranked, fallback, reason, rerank_ms = rerank_once(question, union)

    tickers_with_candidates = [
        ticker for ticker in requested_tickers if per_scope.get(ticker)
    ]
    evidence = select_evidence_with_coverage(
        ranked,
        requested_tickers,
        evidence_k,
        min_per_scope=MIN_PER_SCOPE,
    )

    return {
        "union": union,
        "per_scope": per_scope,
        "evidence": evidence,
        "reranker_fallback": fallback,
        "reranker_fallback_reason": reason,
        "hybrid_ms": hybrid_ms,
        "rerank_ms": rerank_ms,
        "tickers_with_candidates": tickers_with_candidates,
    }


def generate_answer(
    question: str,
    evidence: list[dict],
    comparison_tickers: list[str] | None = None,
) -> tuple[str, str, float]:
    context = build_context(evidence)
    request = build_generation_request(question, context, comparison_tickers)

    generation_start = time.perf_counter()
    response = client.responses.create(**request)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    return response.output_text, context, generation_ms


def normalize_requested_tickers(
    tickers: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Trim/upper/dedupe (order-preserving). The API already does this; repeated
    here so direct callers of answer_question/plan_evidence are safe too."""
    if not tickers:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        ticker = str(raw).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def build_scope_filter(
    tickers: list[str] | tuple[str, ...] | None,
) -> RetrievalFilter | None:
    """Turn a 0/1-ticker request into a RetrievalFilter (or None for global).

    Used for the global and single-company paths. Two or more tickers are
    handled by the comparison path (per-company quota retrieval), not here.
    """
    if not tickers:
        return None
    scope = RetrievalFilter(tickers=tuple(tickers))
    return None if scope.is_empty() else scope


def assert_coverage(
    evidence: list[dict],
    tickers_with_candidates: list[str],
    *,
    where: str,
) -> None:
    """Every requested ticker that HAD candidates must appear in final evidence."""
    if not tickers_with_candidates:
        return
    present = {
        str(chunk.get("ticker", "")).strip().upper() for chunk in evidence
    }
    missing = [ticker for ticker in tickers_with_candidates if ticker not in present]
    if missing:
        message = (
            f"{where}: coverage-aware selection dropped requested tickers "
            f"{missing} that had candidates"
        )
        logger.error(message)
        raise ScopeViolationError(message)


def assert_scope(
    rows: list[dict],
    scope: RetrievalFilter | None,
    *,
    where: str,
) -> None:
    """Fail loudly if any row falls outside a requested ticker scope."""
    if scope is None or scope.tickers is None:
        return
    allowed = set(scope.tickers)
    offenders = sorted(
        {
            str(row.get("ticker")).strip().upper()
            for row in rows
            if str(row.get("ticker", "")).strip().upper() not in allowed
        }
    )
    if offenders:
        message = (
            f"{where}: retrieval returned tickers {offenders} outside the "
            f"requested scope {sorted(allowed)}"
        )
        logger.error(message)
        raise ScopeViolationError(message)


def plan_evidence(
    question: str,
    top_k: int = DEFAULT_EVIDENCE_K,
    tickers: list[str] | None = None,
) -> dict:
    """Retrieve + rerank + select evidence and run scope assertions.

    Everything up to (but not including) answer generation. Exposed as a seam
    so evaluation can measure retrieval/coverage without spending generation
    tokens.

    Routing:
      0 tickers  -> global path            (unchanged)
      1 ticker   -> single-company filter  (unchanged, Phase 1C)
      2+ tickers -> comparison path        (per-company quota + coverage)
    """
    requested = normalize_requested_tickers(tickers)
    comparison_mode = len(requested) >= 2

    base_k = max(1, min(top_k, HYBRID_TOP_K))

    if comparison_mode:
        # Auto-raise so each requested ticker can get >= MIN_PER_SCOPE chunks,
        # capped at HYBRID_TOP_K. (The API caps ticker count at MAX_TICKERS=10.)
        evidence_k = min(max(base_k, len(requested) * MIN_PER_SCOPE), HYBRID_TOP_K)

        result = retrieve_evidence_comparison(question, evidence_k, requested)
        union = result["union"]
        evidence = result["evidence"]
        tickers_with_candidates = result["tickers_with_candidates"]

        combined_scope = RetrievalFilter(tickers=tuple(requested))
        assert_scope(union, combined_scope, where="comparison_candidates")
        assert_scope(evidence, combined_scope, where="comparison_evidence")
        assert_coverage(
            evidence, tickers_with_candidates, where="comparison_evidence"
        )

        warnings = [
            f"No relevant evidence was found for {ticker} in the requested scope."
            for ticker in requested
            if ticker not in tickers_with_candidates
        ]

        return {
            "evidence": evidence,
            "hybrid_results": union,
            "reranker_fallback": result["reranker_fallback"],
            "reranker_fallback_reason": result["reranker_fallback_reason"],
            "hybrid_ms": result["hybrid_ms"],
            "rerank_ms": result["rerank_ms"],
            "comparison_mode": True,
            "requested_tickers": requested,
            "comparison_tickers": requested,
            "evidence_by_scope": evidence_by_scope(evidence, requested),
            "warnings": warnings,
        }

    # Global / single-company path (unchanged behavior).
    evidence_k = base_k
    scope = build_scope_filter(requested)

    (
        hybrid_results,
        evidence,
        reranker_fallback,
        reranker_fallback_reason,
        hybrid_ms,
        rerank_ms,
    ) = retrieve_evidence(question, evidence_k, scope)

    assert_scope(hybrid_results, scope, where="hybrid_candidates")
    assert_scope(evidence, scope, where="evidence")

    return {
        "evidence": evidence,
        "hybrid_results": hybrid_results,
        "reranker_fallback": reranker_fallback,
        "reranker_fallback_reason": reranker_fallback_reason,
        "hybrid_ms": hybrid_ms,
        "rerank_ms": rerank_ms,
        "comparison_mode": False,
        "requested_tickers": requested,
        "comparison_tickers": None,
        "evidence_by_scope": evidence_by_scope(evidence, requested) if requested else {},
        "warnings": [],
    }


def answer_question(
    question: str,
    top_k: int = DEFAULT_EVIDENCE_K,
    tickers: list[str] | None = None,
) -> dict:
    total_start = time.perf_counter()

    plan = plan_evidence(question, top_k, tickers)

    compact_evidence = prepare_evidence(plan["evidence"])
    answer, context, generation_ms = generate_answer(
        question,
        compact_evidence,
        plan["comparison_tickers"],
    )
    total_ms = (time.perf_counter() - total_start) * 1000

    requested = plan["requested_tickers"]

    return {
        "question": question,
        "answer": answer,
        "sources": build_sources(plan["evidence"], plan["hybrid_results"]),
        "generation_model": ANSWER_MODEL,
        "reranker_fallback": plan["reranker_fallback"],
        "reranker_fallback_reason": plan["reranker_fallback_reason"],
        "search_scope": {
            "global_search": len(requested) == 0,
            "comparison_mode": plan["comparison_mode"],
            "tickers": requested or None,
            "evidence_by_scope": plan["evidence_by_scope"],
            "warnings": plan["warnings"],
        },
        "timings": {
            "hybrid_ms": round(plan["hybrid_ms"], 1),
            "hybrid_retrieval_ms": round(plan["hybrid_ms"], 1),
            "rerank_ms": round(plan["rerank_ms"], 1),
            "generation_ms": round(generation_ms, 1),
            "total_ms": round(total_ms, 1),
            "context_chunks_used": len(compact_evidence),
            "context_characters": len(context),
            "answer_characters": len(answer or ""),
        },
    }


if __name__ == "__main__":
    question = input("Ask an SEC question: ")

    result = answer_question(question)

    print("\nANSWER\n")
    print(result["answer"])

    print("\nSOURCES\n")

    for source in result["sources"]:
        print(
            f"{source['chunk_id']} | "
            f"{source['company']} | "
            f"hybrid_rank={source['hybrid_rank']} | "
            f"rerank_score={source['rerank_score']}"
        )

    print("\nTIMINGS\n")
    print(result["timings"])
    print("reranker_fallback:", result["reranker_fallback"])
    print("reranker_fallback_reason:", result["reranker_fallback_reason"])
