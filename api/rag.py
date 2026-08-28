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
from retrieval.scope import Scope, expand_scopes
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


def _as_scopes(scopes) -> list[Scope]:
    """Coerce a list of Scope | "TICKER" | "TICKER:YEAR" into Scope objects."""
    out: list[Scope] = []
    for item in scopes:
        if isinstance(item, Scope):
            out.append(item)
        elif isinstance(item, str) and ":" in item:
            ticker, year = item.split(":", 1)
            out.append(Scope(ticker, int(year)))
        else:
            out.append(Scope(str(item)))
    return out


def _chunk_scope_label(chunk: dict, scopes: list[Scope]) -> str | None:
    for scope in _as_scopes(scopes):
        if scope.matches(chunk):
            return scope.label
    return None


def select_evidence_with_coverage(
    ranked_chunks: list[dict],
    scopes: list[Scope],
    evidence_k: int,
    min_per_scope: int = MIN_PER_SCOPE,
) -> list[dict]:
    """Coverage-aware evidence selection for comparison mode.

    ``ranked_chunks`` are candidates in final rank order (Cohere order, or the
    per-scope union order on fallback). A *scope* is a ``(ticker, fiscal_year?)``
    pair — this handles company comparison, year comparison, and company+year
    comparison with one code path.

    1. Reserve up to ``min_per_scope`` chunks for every requested scope that has
       at least one candidate, taking that scope's highest-ranked chunks.
    2. Fill the remaining slots (up to ``evidence_k``) strictly by global rank
       order — no per-scope cap, so quality can favor one side
       (e.g. FY2023 4 / FY2025 1 is fine; FY2023 5 / FY2025 0 is not).
    3. Return the selection in global rank order for stable presentation.
    """
    if not ranked_chunks:
        return []

    rank_of = {chunk["chunk_id"]: index for index, chunk in enumerate(ranked_chunks)}

    scopes = _as_scopes(scopes)
    by_scope: dict[str, list[dict]] = {}
    for chunk in ranked_chunks:
        label = _chunk_scope_label(chunk, scopes)
        if label is not None:
            by_scope.setdefault(label, []).append(chunk)

    selected_ids: set[str] = set()
    selected: list[dict] = []

    for scope in scopes:
        for chunk in by_scope.get(scope.label, [])[:min_per_scope]:
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
    scopes: list[Scope],
) -> dict[str, int]:
    """Count final evidence chunks per requested scope (0 included).

    Keys are scope labels: ``"AAPL"`` for a ticker-only scope (unchanged
    company-comparison contract), ``"NVDA:2023"`` for a ticker+year scope.
    """
    scopes = _as_scopes(scopes)
    counts = {scope.label: 0 for scope in scopes}
    for chunk in evidence:
        label = _chunk_scope_label(chunk, scopes)
        if label in counts:
            counts[label] += 1
    return counts


def build_context(results: list[dict]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        fiscal_year = result.get("fiscal_year")
        fy_line = f"Fiscal year: FY{fiscal_year}\n" if fiscal_year else ""
        context_parts.append(
            f"SOURCE {index}\n"
            f"Company: {result.get('company')}\n"
            f"Ticker: {result.get('ticker')}\n"
            f"{fy_line}"
            f"Filing: {result.get('filing_type')} "
            f"{result.get('filing_date')}\n"
            f"Chunk ID: {result.get('chunk_id')}\n"
            f"{result.get('text', '')}"
        )

    return "\n\n".join(context_parts)


def build_generation_request(
    question: str,
    context: str,
    comparison_scopes: list[str] | None = None,
) -> dict:
    instructions = (
        'Answer only from the SEC excerpts below. No outside knowledge.\n'
        'Do not invent numbers or facts. Cite a claim with [Source N] only when that\n'
        'source supports it; use the fewest citations needed.\n'
        'If the excerpts lack the answer, reply exactly:\n'
        '"The provided SEC filing excerpts do not contain enough information to answer this question."\n'
        'Keep answers short: 1-3 sentences for numbers; one short paragraph or short bullets otherwise.'
    )

    if comparison_scopes:
        year_comparison = any(":" in label for label in comparison_scopes)
        if year_comparison:
            instructions += (
                "\n\nThe user requested a comparison across these filing scopes: "
                + ", ".join(comparison_scopes)
                + " (TICKER:FISCAL_YEAR).\n"
                "- Discuss each requested fiscal year explicitly, in its own section "
                "(e.g. 'FY2023:' then 'FY2025:'), then a final 'What changed:' section.\n"
                "- Use only evidence from that year's own filing. Never transfer a "
                "fact from one year's excerpts to another year.\n"
                "- Distinguish a change in disclosure LANGUAGE from a change in the "
                "underlying real-world fact.\n"
                "- Do not say something increased or decreased unless excerpts from "
                "both years support the direction.\n"
                "- If a year's excerpts lack the answer, say so plainly for that year."
            )
        else:
            instructions += (
                "\n\nThe user requested a comparison across these companies: "
                + ", ".join(comparison_scopes)
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
                "fiscal_year": result.get("fiscal_year"),
                "accession_number": result.get("accession_number"),
                "date": result.get("filing_date"),
                "filing_date": result.get("filing_date"),
                "report_date": result.get("report_date"),
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
    scopes: list[Scope],
) -> dict:
    """Comparison path: per-scope hybrid -> union -> ONE Cohere rerank ->
    coverage-aware selection. A scope is a ``(ticker, fiscal_year?)`` pair, so
    this serves company, year, and company+year comparisons identically.
    Returns a dict with everything needed to build the response (no generation).
    """
    scope_filters = [scope.to_filter() for scope in scopes]

    union, per_scope, hybrid_ms = scoped_search(
        question,
        scope_filters,
        per_scope_k=HYBRID_TOP_K,
        candidate_k=CANDIDATE_K,
        union_cap=UNION_CAP,
    )

    ranked, fallback, reason, rerank_ms = rerank_once(question, union)

    scopes_with_candidates = [
        scope for scope in scopes if per_scope.get(scope.label)
    ]
    evidence = select_evidence_with_coverage(
        ranked,
        scopes,
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
        "scopes_with_candidates": scopes_with_candidates,
    }


def generate_answer(
    question: str,
    evidence: list[dict],
    comparison_scopes: list[str] | None = None,
) -> tuple[str, str, float]:
    context = build_context(evidence)
    request = build_generation_request(question, context, comparison_scopes)

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
    scopes_with_candidates: list[Scope],
    *,
    where: str,
) -> None:
    """Every requested scope that HAD candidates must appear in final evidence."""
    if not scopes_with_candidates:
        return
    present = {
        _chunk_scope_label(chunk, scopes_with_candidates) for chunk in evidence
    }
    missing = [s.label for s in scopes_with_candidates if s.label not in present]
    if missing:
        message = (
            f"{where}: coverage-aware selection dropped requested scopes "
            f"{missing} that had candidates"
        )
        logger.error(message)
        raise ScopeViolationError(message)


def assert_scopes(
    rows: list[dict],
    scopes: list[Scope],
    *,
    where: str,
) -> None:
    """Fail loudly if any row falls outside every requested scope.

    Enforces ticker AND fiscal_year together: a chunk from NVDA FY2024 is a
    violation when only NVDA:2023 and NVDA:2025 were requested.
    """
    offenders = sorted(
        {
            f'{row.get("ticker")}:{row.get("fiscal_year")}'
            for row in rows
            if _chunk_scope_label(row, scopes) is None
        }
    )
    if offenders:
        message = (
            f"{where}: retrieval returned {offenders} outside the requested "
            f"scopes {[s.label for s in scopes]}"
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


def _normalize_years(fiscal_years) -> list[int]:
    if not fiscal_years:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for raw in fiscal_years:
        try:
            year = int(raw)
        except (TypeError, ValueError):
            continue
        if year not in seen:
            seen.add(year)
            out.append(year)
    return out


def plan_evidence(
    question: str,
    top_k: int = DEFAULT_EVIDENCE_K,
    tickers: list[str] | None = None,
    fiscal_years: list[int] | None = None,
) -> dict:
    """Retrieve + rerank + select evidence and run scope assertions.

    Everything up to (but not including) answer generation. Exposed as a seam
    so evaluation can measure retrieval/coverage without spending generation
    tokens.

    A *scope* is a ``(ticker, fiscal_year?)`` pair; a request expands to a list
    of scopes (retrieval.scope.expand_scopes). Routing is purely by scope count:

      0 scopes   -> global path                    (unchanged)
      1 scope    -> single ticker[/year] filter    (Phase 1C / Phase 5 year)
      2+ scopes  -> comparison path (company, year, or company+year)
    """
    requested = normalize_requested_tickers(tickers)
    years = _normalize_years(fiscal_years)
    scopes = expand_scopes(requested, years) if requested else []
    comparison_mode = len(scopes) >= 2

    base_k = max(1, min(top_k, HYBRID_TOP_K))

    if comparison_mode:
        # Auto-raise so each requested scope can get >= MIN_PER_SCOPE chunks,
        # capped at HYBRID_TOP_K.
        evidence_k = min(max(base_k, len(scopes) * MIN_PER_SCOPE), HYBRID_TOP_K)

        result = retrieve_evidence_comparison(question, evidence_k, scopes)
        union = result["union"]
        evidence = result["evidence"]
        scopes_with_candidates = result["scopes_with_candidates"]

        assert_scopes(union, scopes, where="comparison_candidates")
        assert_scopes(evidence, scopes, where="comparison_evidence")
        assert_coverage(evidence, scopes_with_candidates, where="comparison_evidence")

        covered = {s.label for s in scopes_with_candidates}
        warnings = [
            f"No relevant evidence was found for {s.label} in the requested scope."
            for s in scopes
            if s.label not in covered
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
            "fiscal_years": years,
            "scopes": [s.label for s in scopes],
            "comparison_tickers": requested,
            "comparison_scopes": [s.label for s in scopes],
            "evidence_by_scope": evidence_by_scope(evidence, scopes),
            "warnings": warnings,
        }

    # Global / single-scope path.
    evidence_k = base_k
    scope = scopes[0].to_filter() if scopes else None

    (
        hybrid_results,
        evidence,
        reranker_fallback,
        reranker_fallback_reason,
        hybrid_ms,
        rerank_ms,
    ) = retrieve_evidence(question, evidence_k, scope)

    if scopes:
        assert_scopes(hybrid_results, scopes, where="hybrid_candidates")
        assert_scopes(evidence, scopes, where="evidence")

    return {
        "evidence": evidence,
        "hybrid_results": hybrid_results,
        "reranker_fallback": reranker_fallback,
        "reranker_fallback_reason": reranker_fallback_reason,
        "hybrid_ms": hybrid_ms,
        "rerank_ms": rerank_ms,
        "comparison_mode": False,
        "requested_tickers": requested,
        "fiscal_years": years,
        "scopes": [s.label for s in scopes],
        "comparison_tickers": None,
        "comparison_scopes": None,
        "evidence_by_scope": evidence_by_scope(evidence, scopes) if scopes else {},
        "warnings": [],
    }


def answer_question(
    question: str,
    top_k: int = DEFAULT_EVIDENCE_K,
    tickers: list[str] | None = None,
    fiscal_years: list[int] | None = None,
) -> dict:
    total_start = time.perf_counter()

    plan = plan_evidence(question, top_k, tickers, fiscal_years)

    compact_evidence = prepare_evidence(plan["evidence"])
    answer, context, generation_ms = generate_answer(
        question,
        compact_evidence,
        plan["comparison_scopes"],
    )
    total_ms = (time.perf_counter() - total_start) * 1000

    requested = plan["requested_tickers"]
    years = plan["fiscal_years"]

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
            "fiscal_years": years or None,
            "scopes": plan["scopes"] or None,
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
