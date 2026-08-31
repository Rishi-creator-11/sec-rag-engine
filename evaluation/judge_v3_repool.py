"""LLM-assisted relevance judging for the benchmark_v3 RE-POOL.

Same methodology as benchmark_v2 (evaluation/auto_judge_v2_pool.py):
  - one gpt-5-mini call per question, all candidates in a single request
  - candidates shown only as aliases C01.. (no retriever / rank / score)
  - per-candidate {relevant: bool, confidence: high|medium|low, reason}
  - incremental save so a run can resume

    python -m evaluation.judge_v3_pool                 # first pass
    python -m evaluation.judge_v3_pool --review        # 2nd pass: re-judge only
                                                       #   medium/low-confidence
                                                       #   items with a stronger model

NO human judgments are produced here. Report accordingly.

Input:  evaluation/results/benchmark_v3_repool.json
Output: evaluation/results/benchmark_v3_repool_judgments.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_repool.json"
OUT = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_repool_judgments.json"

MODEL = "gpt-5-mini"                 # 1st pass: all candidates in one call
REVIEW_MODEL = "gpt-5-mini"          # 2nd pass: higher effort, uncertain + positive items
MAX_ATTEMPTS = 6
VALID_CONF = {"high", "medium", "low"}

# The pool union is pre-sorted by (retriever consensus desc, best rank asc), so
# the head is the strongest evidence. Judging the top-N keeps prompts tractable
# and still covers far deeper than production candidate_k=10. Chunks that appear
# only once, only in one retriever, only near rank 50, are neither judged nor
# production-reachable.
JUDGE_HEAD = 60
FIRST_PASS_EFFORT = "minimal"
REVIEW_EFFORT = "low"      # 2nd pass: gpt-5-mini/low re-checks positives + all
                          # medium/low-confidence candidates (recall + precision)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _clean(t: str) -> str:
    return " ".join((t or "").split())


def _extract_json(text: str) -> dict:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.DOTALL)
    if m:
        s = m.group(1)
    return json.loads(s)


def _aliases(item: dict, head: int | None = None) -> tuple[dict[str, str], list[dict]]:
    cands = item["candidates"]
    if head is not None:
        cands = cands[:head]
    alias_to_chunk, model_cands = {}, []
    for i, c in enumerate(cands, start=1):
        a = f"C{i:03d}"
        alias_to_chunk[a] = c["chunk_id"]
        model_cands.append({"candidate_id": a, "text": _clean(c.get("text", ""))})
    return alias_to_chunk, model_cands


def _prompt(item: dict, cands: list[dict]) -> str:
    return f"""You are labeling relevance for an SEC 10-K retrieval benchmark.

Judge every candidate against the question. Do not use outside knowledge. Do
not consider how a candidate was retrieved.

Relevance rules:
- Numeric question: relevant only if the candidate contains the requested number
  or evidence directly necessary to derive/identify it.
- Qualitative question: relevant only if it substantively answers the question.
  Merely mentioning the topic is NOT enough.
- Multi-part question: relevant if it directly supports at least one substantive part.

Question id: {item['id']}
Question: {item['question']}
Company: {item['company']}
Category: {item.get('category')}
Answer type: {item.get('answer_type')}
Candidate count: {len(cands)}

Candidates:
{json.dumps(cands, ensure_ascii=False)}

Return JSON only:
{{"judgments":[{{"candidate_id":"C001","relevant":true,"confidence":"high","reason":"brief"}}]}}

- Return every candidate_id from C001 through C{len(cands):03d} exactly once.
- confidence is one of: high, medium, low.
- reason: at most 8 words. No line breaks.
"""


def _validate(payload: dict, expected: set[str]) -> dict[str, dict]:
    js = payload.get("judgments")
    if not isinstance(js, list):
        raise ValueError("judgments must be a list")
    by_alias = {}
    for it in js:
        a = it.get("candidate_id")
        if not isinstance(a, str) or not a:
            raise ValueError("missing candidate_id")
        if not isinstance(it.get("relevant"), bool):
            raise ValueError(f"{a}: relevant must be bool")
        if it.get("confidence") not in VALID_CONF:
            raise ValueError(f"{a}: bad confidence")
        by_alias[a] = {
            "relevant": it["relevant"],
            "confidence": it["confidence"],
            "reason": str(it.get("reason", ""))[:300],
        }
    missing = expected - set(by_alias)
    extra = set(by_alias) - expected
    if missing or extra:
        raise ValueError(f"alias mismatch missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    return by_alias


def judge_question(
    item: dict, model: str = MODEL, *, head: int | None = JUDGE_HEAD,
    effort: str = FIRST_PASS_EFFORT,
) -> dict:
    alias_to_chunk, cands = _aliases(item, head=head)
    expected = set(alias_to_chunk)
    prompt = _prompt(item, cands)
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resp = client.responses.create(
            model=model, input=prompt, reasoning={"effort": effort},
            max_output_tokens=16000,
        )
        try:
            by_alias = _validate(_extract_json(resp.output_text), expected)
            judgments = sorted(
                (
                    {"chunk_id": alias_to_chunk[a], **v}
                    for a, v in by_alias.items()
                ),
                key=lambda j: j["chunk_id"],
            )
            return {
                "relevant_chunks": sorted(
                    j["chunk_id"] for j in judgments if j["relevant"]
                ),
                "judgments": judgments,
            }
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last = exc
            print(f"    malformed (attempt {attempt}/{MAX_ATTEMPTS}): {exc}")
    raise RuntimeError(f"judge failed for {item['id']}: {last}")


def _load_out() -> dict:
    if OUT.exists():
        return {r["id"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
    return {}


def _save(records: dict, pool: list[dict]) -> None:
    OUT.write_text(
        json.dumps([records[i["id"]] for i in pool if i["id"] in records], indent=2),
        encoding="utf-8",
    )


def first_pass() -> None:
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    done = _load_out()
    for idx, item in enumerate(pool, start=1):
        if item["id"] in done:
            continue
        print(f"[{idx}/{len(pool)}] {item['id']} ({item['candidate_count']} candidates)")
        res = judge_question(item)
        done[item["id"]] = {
            "id": item["id"], "question": item["question"], "company": item["company"],
            "category": item.get("category"), "answer_type": item.get("answer_type"),
            "number_hint": item.get("number_hint"),
            "v2_relevant_chunks": item.get("v2_relevant_chunks", []),
            "candidate_count": item["candidate_count"],
            **res,
            "reviewed": False,
        }
        _save(done, pool)
        rel = len(res["relevant_chunks"])
        conf = [j["confidence"] for j in res["judgments"] if j["relevant"]]
        print(f"    relevant={rel}  conf(relevant)={conf}")


def review_pass() -> None:
    """Second pass with the stronger model (gpt-5-mini, medium effort).

    Re-judges (a) every candidate the fast pass marked RELEVANT (precision
    check - the fast model over-labels) and (b) every medium/low-confidence
    candidate (recall check). High-confidence negatives are left as-is.
    """
    pool = {i["id"]: i for i in json.loads(POOL.read_text(encoding="utf-8"))}
    done = _load_out()
    changed = 0
    already = sum(1 for rec in done.values() if rec.get("reviewed"))
    todo = [rid for rid, rec in done.items() if not rec.get("reviewed")]
    print(f"review pass: {already} questions already reviewed (skipped), "
          f"{len(todo)} remaining")
    for rid in todo:
        rec = done[rid]
        if rec.get("reviewed"):          # defensive: never re-judge a done question
            continue
        uncertain = [
            j for j in rec["judgments"]
            if j["confidence"] in {"medium", "low"} or j["relevant"]
        ]
        if not uncertain:
            rec["reviewed"] = True
            continue
        item = pool[rid]
        cand_by_id = {c["chunk_id"]: c for c in item["candidates"]}
        subset = {
            "id": rid, "question": rec["question"], "company": rec["company"],
            "category": rec.get("category"), "answer_type": rec.get("answer_type"),
            "candidates": [cand_by_id[j["chunk_id"]] for j in uncertain
                           if j["chunk_id"] in cand_by_id],
        }
        if not subset["candidates"]:
            rec["reviewed"] = True
            continue
        print(f"review {rid}: {len(subset['candidates'])} uncertain")
        res = judge_question(subset, model=REVIEW_MODEL, head=None, effort=REVIEW_EFFORT)
        new_by_id = {j["chunk_id"]: j for j in res["judgments"]}
        for j in rec["judgments"]:
            if j["chunk_id"] in new_by_id:
                nj = new_by_id[j["chunk_id"]]
                if nj["relevant"] != j["relevant"]:
                    changed += 1
                j.update(relevant=nj["relevant"], confidence=nj["confidence"],
                         reason=nj["reason"], review_flipped=(nj["relevant"] != j["relevant"]))
        rec["relevant_chunks"] = sorted(
            j["chunk_id"] for j in rec["judgments"] if j["relevant"]
        )
        rec["reviewed"] = True
        _save(done, [pool[k] for k in pool if k in done])

    total_reviewed = sum(1 for rec in done.values() if rec.get("reviewed"))
    cumulative_flips = sum(
        1 for rec in done.values() for j in rec["judgments"]
        if j.get("review_flipped")
    )
    print(f"\nreview pass: {changed} judgments flipped this run; "
          f"{cumulative_flips} flipped cumulatively; "
          f"{total_reviewed}/{len(done)} questions reviewed")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", action="store_true")
    a = p.parse_args(argv)
    if a.review:
        review_pass()
    else:
        first_pass()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
