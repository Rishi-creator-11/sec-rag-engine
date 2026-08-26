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
from retrieval.hybrid_search import search as hybrid_search


load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI()

ANSWER_MODEL = "gpt-5-nano"
HYBRID_TOP_K = 10
CANDIDATE_K = 10
DEFAULT_EVIDENCE_K = 5
MIN_EVIDENCE_K = 2
MAX_OUTPUT_TOKENS = 800

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


def build_generation_request(question: str, context: str) -> dict:
    prompt = f"""Answer only from the SEC excerpts below. No outside knowledge.
Do not invent numbers or facts. Cite a claim with [Source N] only when that
source supports it; use the fewest citations needed.
If the excerpts lack the answer, reply exactly:
"The provided SEC filing excerpts do not contain enough information to answer this question."
Keep answers short: 1-3 sentences for numbers; one short paragraph or short bullets otherwise.

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


def retrieve_evidence(
    question: str,
    evidence_k: int,
) -> tuple[list[dict], list[dict], bool, str | None, float, float]:
    hybrid_start = time.perf_counter()
    hybrid_results = hybrid_search(
        question,
        top_k=HYBRID_TOP_K,
        candidate_k=CANDIDATE_K,
    )
    hybrid_ms = (time.perf_counter() - hybrid_start) * 1000

    if not rerank_enabled():
        logger.warning(
            "COHERE_RERANK_ENABLED is false; using hybrid top-%s",
            evidence_k,
        )
        return (
            hybrid_results,
            hybrid_results[:evidence_k],
            True,
            "disabled",
            hybrid_ms,
            0.0,
        )

    rerank_start = time.perf_counter()

    try:
        reranked, rerank_seconds = rerank_timed(
            question,
            hybrid_results,
        )
        rerank_ms = rerank_seconds * 1000
        evidence = select_evidence(reranked, max_k=evidence_k)
        return (
            hybrid_results,
            evidence,
            False,
            None,
            hybrid_ms,
            rerank_ms,
        )
    except Exception as error:
        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        if is_rate_limited(error):
            logger.warning(
                "Cohere rerank rate-limited (429); "
                "falling back to hybrid top-%s",
                evidence_k,
            )
            reason = "rate_limited"
        else:
            logger.exception(
                "Cohere rerank failed; falling back to hybrid top-%s",
                evidence_k,
            )
            reason = "api_error"

        return (
            hybrid_results,
            hybrid_results[:evidence_k],
            True,
            reason,
            hybrid_ms,
            rerank_ms,
        )


def generate_answer(question: str, evidence: list[dict]) -> tuple[str, str, float]:
    context = build_context(evidence)
    request = build_generation_request(question, context)

    generation_start = time.perf_counter()
    response = client.responses.create(**request)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    return response.output_text, context, generation_ms


def answer_question(
    question: str,
    top_k: int = DEFAULT_EVIDENCE_K,
) -> dict:
    total_start = time.perf_counter()
    evidence_k = max(1, min(top_k, HYBRID_TOP_K))

    (
        hybrid_results,
        evidence,
        reranker_fallback,
        reranker_fallback_reason,
        hybrid_ms,
        rerank_ms,
    ) = retrieve_evidence(question, evidence_k)

    compact_evidence = prepare_evidence(evidence)
    answer, context, generation_ms = generate_answer(
        question,
        compact_evidence,
    )
    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "question": question,
        "answer": answer,
        "sources": build_sources(evidence, hybrid_results),
        "generation_model": ANSWER_MODEL,
        "reranker_fallback": reranker_fallback,
        "reranker_fallback_reason": reranker_fallback_reason,
        "timings": {
            "hybrid_ms": round(hybrid_ms, 1),
            "hybrid_retrieval_ms": round(hybrid_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
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
