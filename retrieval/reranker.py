"""OpenAI relevance reranker for hybrid top-k candidates.

Retrieves nothing. It only rescores a provided candidate list using
gpt-5-mini. The model sees short aliases and chunk text, never RRF
scores or retriever ranks.

Scores
------
0  not relevant
1  weak / topic overlap
2  useful supporting evidence
3  directly answer-bearing evidence

Results are sorted by rerank_score descending, with the original
hybrid order as the tie-breaker. Original hybrid scores and ranks are
copied, not mutated.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "gpt-5-mini"
MAX_ATTEMPTS = 5
CACHE_PATH = Path("evaluation/results/reranker_cache.json")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}

    with CACHE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2)


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def extract_json(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        stripped,
        flags=re.DOTALL,
    )

    if fenced:
        stripped = fenced.group(1)

    return json.loads(stripped)


def alias_for(index: int) -> str:
    return f"C{index:02d}"


def build_alias_maps(
    candidates: list[dict],
) -> tuple[dict[str, dict], list[dict]]:
    alias_to_candidate = {}
    model_candidates = []

    for index, candidate in enumerate(candidates, start=1):
        alias = alias_for(index)
        alias_to_candidate[alias] = candidate
        model_candidates.append(
            {
                "candidate_id": alias,
                "text": clean_text(candidate.get("text", "")),
            }
        )

    return alias_to_candidate, model_candidates


def validate_scores(
    payload: dict,
    expected_aliases: set[str],
) -> dict[str, dict]:
    if not isinstance(payload, dict):
        raise ValueError("Response is not a JSON object.")

    scores = payload.get("scores")

    if not isinstance(scores, list):
        raise ValueError("scores must be a list.")

    returned = []
    by_alias = {}

    for item in scores:
        if not isinstance(item, dict):
            raise ValueError("Each score must be an object.")

        alias = item.get("candidate_id")

        if not isinstance(alias, str) or not alias:
            raise ValueError("Each score needs a candidate_id.")

        score = item.get("score")

        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError(f"{alias}: score must be an integer.")

        if score not in {0, 1, 2, 3}:
            raise ValueError(f"{alias}: score must be 0, 1, 2, or 3.")

        reason = item.get("reason", "")

        if not isinstance(reason, str):
            raise ValueError(f"{alias}: reason must be a string.")

        returned.append(alias)
        by_alias[alias] = {
            "candidate_id": alias,
            "score": score,
            "reason": reason,
        }

    if len(returned) != len(set(returned)):
        raise ValueError("Duplicate candidate_id in scores.")

    missing = expected_aliases - set(returned)
    extra = set(returned) - expected_aliases

    if missing or extra:
        raise ValueError(
            "candidate_id mismatch. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    return by_alias


def build_prompt(
    question: str,
    model_candidates: list[dict],
    answer_type: str | None = None,
) -> str:
    candidate_json = json.dumps(
        model_candidates,
        indent=2,
        ensure_ascii=False,
    )
    question_type = answer_type or "unspecified"
    last_alias = alias_for(len(model_candidates))

    return f"""You are reranking SEC 10-K retrieval candidates.

Score every candidate for relevance to the question. Do not use outside
knowledge. Do not consider how a candidate was retrieved. Retriever names,
ranks, and scores are omitted on purpose.

Each candidate is identified only by a short alias such as C01, C02, C03.
Use those aliases exactly. Do not invent aliases. Do not rename them.

Scoring:
- 0 = not relevant
- 1 = weak / topic overlap only
- 2 = useful supporting evidence
- 3 = directly answer-bearing evidence

Numeric questions: assign 3 only if the chunk contains the requested number
or evidence directly needed to derive the number.
Qualitative questions: assign 3 when the chunk substantively answers the
question. Mere topical overlap is not a 3.

Question type: {question_type}
Question: {question}
Candidate count: {len(model_candidates)}

Candidates:
{candidate_json}

Return JSON only, with this exact shape:
{{
  "scores": [
    {{
      "candidate_id": "C01",
      "score": 3,
      "reason": "brief reason"
    }}
  ]
}}

Requirements:
- Return every candidate_id from C01 through {last_alias} exactly once.
- score must be an integer: 0, 1, 2, or 3.
- Keep reasons brief.
"""


def cache_matches_candidates(
    cached: dict,
    candidates: list[dict],
) -> bool:
    cached_ids = set(cached.get("scores_by_chunk_id", {}))
    current_ids = {candidate["chunk_id"] for candidate in candidates}
    return cached_ids == current_ids


def score_with_model(
    question: str,
    candidates: list[dict],
    answer_type: str | None = None,
) -> dict[str, dict]:
    alias_to_candidate, model_candidates = build_alias_maps(candidates)
    expected_aliases = set(alias_to_candidate)
    prompt = build_prompt(question, model_candidates, answer_type)
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.responses.create(
            model=MODEL,
            input=prompt,
        )
        raw = response.output_text

        try:
            payload = extract_json(raw)
            alias_scores = validate_scores(payload, expected_aliases)
            by_chunk = {}

            for alias, scored in alias_scores.items():
                chunk_id = alias_to_candidate[alias]["chunk_id"]
                by_chunk[chunk_id] = {
                    "score": scored["score"],
                    "reason": scored["reason"],
                }

            return by_chunk
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            last_error = error
            print(
                "    malformed rerank response "
                f"(attempt {attempt}/{MAX_ATTEMPTS}): {error}"
            )

    raise RuntimeError(
        f"Failed to rerank after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def apply_scores(
    candidates: list[dict],
    scores_by_chunk_id: dict[str, dict],
) -> list[dict]:
    reranked = []

    for hybrid_index, candidate in enumerate(candidates):
        scored = scores_by_chunk_id[candidate["chunk_id"]]
        result = copy.deepcopy(candidate)
        result["rerank_score"] = scored["score"]
        result["rerank_reason"] = scored["reason"]
        result["hybrid_rank"] = hybrid_index + 1
        reranked.append((scored["score"], hybrid_index, result))

    reranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in reranked]


def rerank(
    question: str,
    candidates: list[dict],
    question_id: str | None = None,
    answer_type: str | None = None,
) -> list[dict]:
    """Rerank hybrid candidates. Original result dicts are not mutated."""
    if not candidates:
        return []

    cache = load_cache() if question_id else {}
    cached = cache.get(question_id) if question_id else None

    if (
        question_id
        and cached
        and cache_matches_candidates(cached, candidates)
    ):
        return apply_scores(candidates, cached["scores_by_chunk_id"])

    scores_by_chunk_id = score_with_model(
        question,
        candidates,
        answer_type,
    )

    if question_id:
        cache[question_id] = {
            "question": question,
            "scores_by_chunk_id": scores_by_chunk_id,
        }
        save_cache(cache)

    return apply_scores(candidates, scores_by_chunk_id)
