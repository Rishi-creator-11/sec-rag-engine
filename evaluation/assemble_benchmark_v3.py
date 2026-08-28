"""Assemble the final evaluation/benchmark_v3.json.

Combines:
  - evaluation/benchmark_v3_questions.json       (all 125 questions, no qrels)
  - evaluation/results/benchmark_v3_auto_judgments.json
        (LLM-assisted first pass + stronger-model review pass on the
         medium/low-confidence candidates; see judge_v3_pool.py.
         These are MODEL-ASSISTED judgments, not human judgments.)

Output rows keep the v2 schema (id, question, company, category, answer_type,
relevant_chunks) plus provenance fields:
    source                 v2 | canary | extra
    v2_relevant_chunks     original v2 qrels (empty for non-v2 questions)
    judged_candidate_count number of pooled candidates that were judged
    judgment_method        "model-assisted (gpt-5-nano first pass, gpt-5-mini
                            review of medium/low-confidence)"
    reviewed               whether the review pass touched this question

Unsupported questions get relevant_chunks: [] and are kept for the
"does the system correctly retrieve nothing useful" check, excluded from scoring.

benchmark_v2.json is NOT modified. XOM/UNH are not present.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "evaluation" / "benchmark_v3_questions.json"
JUDGMENTS = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_auto_judgments.json"
OUT = REPO_ROOT / "evaluation" / "benchmark_v3.json"

JUDGMENT_METHOD = (
    "model-assisted (gpt-5-nano first pass over the full pool; "
    "gpt-5-mini review pass re-judging medium/low-confidence candidates)"
)


def main() -> int:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    judged = {r["id"]: r for r in json.loads(JUDGMENTS.read_text(encoding="utf-8"))}

    out: list[dict] = []
    missing: list[str] = []
    for q in questions:
        qid = q["id"]
        unsupported = q.get("answer_type") == "unsupported"
        row = {
            "id": qid,
            "question": q["question"],
            "company": q["company"],
            "category": q.get("category", "general"),
            "answer_type": q.get("answer_type", "qualitative"),
            "relevant_chunks": [],
            "source": q.get("source", "v2"),
            "v2_relevant_chunks": q.get("v2_relevant_chunks", []),
            "number_hint": q.get("number_hint"),
        }
        if unsupported:
            row["judgment_method"] = "n/a (unsupported - expected empty)"
            out.append(row)
            continue

        j = judged.get(qid)
        if j is None:
            missing.append(qid)
            continue
        rel = sorted(j.get("relevant_chunks", []))
        conf = {c: 0 for c in ("high", "medium", "low")}
        for jd in j.get("judgments", []):
            if jd.get("relevant"):
                c = jd.get("confidence", "low")
                conf[c] = conf.get(c, 0) + 1
        row["relevant_chunks"] = rel
        row["judged_candidate_count"] = j.get("candidate_count")
        row["relevant_confidence_breakdown"] = conf
        row["judgment_method"] = JUDGMENT_METHOD
        row["reviewed"] = bool(j.get("reviewed"))
        out.append(row)

    if missing:
        raise SystemExit(
            f"{len(missing)} supported questions have no judgment yet: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            "Run: python -m evaluation.judge_v3_pool  (then --review)"
        )

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    from collections import Counter
    supported = [r for r in out if r["answer_type"] != "unsupported"]
    with_qrels = [r for r in supported if r["relevant_chunks"]]
    empty = [r for r in supported if not r["relevant_chunks"]]
    print(f"benchmark_v3.json: {len(out)} questions "
          f"({len(supported)} supported, {len(out) - len(supported)} unsupported)")
    print(f"  supported with >=1 relevant chunk: {len(with_qrels)}")
    if empty:
        print(f"  WARNING supported but 0 relevant chunks: "
              f"{[r['id'] for r in empty]}")
    print(f"  by company: {dict(sorted(Counter(r['company'] for r in out).items()))}")
    print(f"  total relevant labels: "
          f"{sum(len(r['relevant_chunks']) for r in out)}")
    print(f"  mean relevant/question (supported w/qrels): "
          f"{sum(len(r['relevant_chunks']) for r in with_qrels) / max(len(with_qrels), 1):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
