"""Assemble evaluation/benchmark_v3_repool.json from the re-pool judgments.

Same schema and model-assisted methodology as evaluation/assemble_benchmark_v3.py,
but sourced from the CURRENT-corpus re-pool (build_v3_repool.py + judge_v3_repool.py,
gpt-5-mini first pass minimal effort + gpt-5-mini review pass low effort over
medium/low-confidence + positive candidates).

benchmark_v3.json is NEVER touched. Output is benchmark_v3_repool.json.
These are MODEL-ASSISTED judgments, not human labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "evaluation" / "benchmark_v3_questions.json"
FROZEN = REPO_ROOT / "evaluation" / "benchmark_v3.json"
JUDGMENTS = REPO_ROOT / "evaluation" / "results" / "benchmark_v3_repool_judgments.json"
OUT = REPO_ROOT / "evaluation" / "benchmark_v3_repool.json"

METHOD = ("model-assisted (gpt-5-mini first pass minimal effort over the "
          "year-scoped depth-80 re-pool; gpt-5-mini review pass low effort "
          "re-judging every positive + every medium/low-confidence candidate)")


def main() -> int:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    frozen = {r["id"]: r for r in json.loads(FROZEN.read_text(encoding="utf-8"))}
    judged = {r["id"]: r for r in json.loads(JUDGMENTS.read_text(encoding="utf-8"))}

    out, missing = [], []
    for q in questions:
        qid = q["id"]
        fz = frozen.get(qid, {})
        row = {
            "id": qid, "question": q["question"], "company": q["company"],
            "category": q.get("category", "general"),
            "answer_type": q.get("answer_type", "qualitative"),
            "relevant_chunks": [],
            "source": q.get("source", "v2"),
            "number_hint": fz.get("number_hint"),
            "frozen_relevant_chunks": fz.get("relevant_chunks", []),
        }
        if q.get("answer_type") == "unsupported":
            row["judgment_method"] = "n/a (unsupported - expected empty)"
            out.append(row)
            continue
        j = judged.get(qid)
        if j is None:
            missing.append(qid)
            continue
        conf = {c: 0 for c in ("high", "medium", "low")}
        for jd in j.get("judgments", []):
            if jd.get("relevant"):
                conf[jd.get("confidence", "low")] = conf.get(jd.get("confidence", "low"), 0) + 1
        row["relevant_chunks"] = sorted(j.get("relevant_chunks", []))
        row["judged_candidate_count"] = j.get("candidate_count")
        row["relevant_confidence_breakdown"] = conf
        row["judgment_method"] = METHOD
        row["reviewed"] = bool(j.get("reviewed"))
        out.append(row)

    if missing:
        raise SystemExit(f"{len(missing)} supported questions unjudged: {missing[:10]}")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    supported = [r for r in out if r["answer_type"] != "unsupported"]
    withq = [r for r in supported if r["relevant_chunks"]]
    empty = [r for r in supported if not r["relevant_chunks"]]
    total_new = sum(len(r["relevant_chunks"]) for r in supported)
    total_old = sum(len(r.get("frozen_relevant_chunks", [])) for r in supported)
    reviewed = sum(1 for r in supported if r.get("reviewed"))
    conf_tot = Counter()
    for r in supported:
        for k, v in (r.get("relevant_confidence_breakdown") or {}).items():
            conf_tot[k] += v
    print(f"benchmark_v3_repool.json: {len(out)} questions "
          f"({len(supported)} supported, {len(out)-len(supported)} unsupported)")
    print(f"  reviewed: {reviewed}/{len(supported)}")
    print(f"  supported with >=1 relevant chunk: {len(withq)}   (0-qrel: {[r['id'] for r in empty]})")
    print(f"  relevant labels  old(frozen) {total_old}  ->  new(repool) {total_new}   "
          f"(expansion {total_new-total_old:+d}, {100*(total_new/total_old-1):+.1f}%)")
    print(f"  new mean relevant/question: {total_new/max(len(withq),1):.1f}   (old {total_old/len(supported):.1f})")
    print(f"  confidence of relevant labels: {dict(conf_tot)}")
    # label churn vs frozen
    kept = added = dropped = 0
    for r in supported:
        new, old = set(r["relevant_chunks"]), set(r.get("frozen_relevant_chunks", []))
        kept += len(new & old); added += len(new - old); dropped += len(old - new)
    print(f"  vs frozen qrels:  kept {kept}  added {added}  dropped {dropped} "
          f"(dropped = old-year/stale-pool chunks no longer surfaced or judged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
