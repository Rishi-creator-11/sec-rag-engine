"""P0 numeric-grounding probe — reproduce, instrument, classify.

Runs benchmark_numeric_probe_v1.json (6 companies x direct/loose/comparison/
anchored) N times each against the live pipeline and records, per run:

  - hybrid top-k (chunk_id, rank, rrf_score)
  - reranked order + rerank_score (or "fallback")
  - selected evidence chunk_ids + fiscal_year + whether the chunk text holds
    the expected value / the wrong-year value
  - [Source N] citations present in the answer
  - the final answer
  - value_ok / wrong_year_value / year_attribution per requested year
  - a failure classification

Classification (only assigned when a run is wrong):
  retrieval        : expected-value chunk never retrieved (not in hybrid top-k)
  reranking        : expected-value chunk retrieved but reranked out of evidence_k
  evidence_select  : expected-value chunk in reranked top but not selected
  repeated_table   : selected evidence contains the wrong-year value in a
                     different chunk/table than the expected-year value
  generation       : expected-value chunk IS selected & cited, but the answer
                     states the wrong number for the requested year

    python -m evaluation.run_numeric_probe_v1 [--runs 3] [--tag before]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from api.rag import (  # noqa: E402
    answer_question, plan_evidence, HYBRID_TOP_K, CANDIDATE_K,
)
from retrieval.hybrid_search import search as hybrid_search  # noqa: E402
from retrieval.embedder import embed_text  # noqa: E402
from retrieval.filters import RetrievalFilter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "evaluation" / "benchmark_numeric_probe_v1.json"


def _digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")


def _value_present(answer: str, value: str) -> bool:
    """Exact millions digit-string, OR a faithful billions rounding in context.

    "$57,048 million" -> a user answer may say "$57.0 billion" / "57.0 billion"
    / "$57 billion". Only accept the rounded form when 'billion' is nearby, so
    a bare "570" elsewhere never matches.
    """
    if _digits(value) in _digits(answer):
        return True
    try:
        b = int(_digits(value)) / 1000.0
    except ValueError:
        return False
    pats = [rf"{b:.1f}\s*billion", rf"{b:.2f}\s*billion", rf"{round(b)}\s*billion",
            rf"\${b:.1f}\s*billion", rf"\${round(b)}\s*billion"]
    return any(re.search(p, answer) for p in pats)


def _cited(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[[Ss]ource\s+(\d+)\]", answer)})


def _year_attribution_ok(answer: str, year: str, value: str, wrong: str | None) -> bool:
    """Is `value` associated with `year` in the answer text (and not `wrong`)?"""
    a = answer
    v, w = _digits(value), _digits(wrong or "")
    # window around each mention of the year
    for m in re.finditer(re.escape(year[-2:]) + r"\b|" + re.escape(year), a):
        seg = a[max(0, m.start() - 160): m.end() + 160]
        if _value_present(seg, value):
            return True
    # comparison answers section by 'FYxxxx:' headers
    for chunk in re.split(r"(?=FY?\s?20\d\d[:\s])", a):
        if year in chunk or year[-2:] in chunk:
            if _value_present(chunk, value):
                return True
    # fallback: value present anywhere and wrong value absent
    return _value_present(a, value) and (not w or w not in _digits(a))


def run_probe(item: dict) -> dict:
    tk = item["ticker"]
    years = item["fiscal_years"]
    q = item["question"]
    comparison = len(years) >= 2

    plan = plan_evidence(q, top_k=5, tickers=[tk], fiscal_years=years)
    resp = answer_question(q, top_k=5, tickers=[tk], fiscal_years=years)
    answer = resp["answer"]
    sources = resp.get("sources", [])
    sc = resp.get("search_scope", {})

    # hybrid top-k for the (scoped) query — single-scope only for clean instrumentation
    hybrid = []
    if not comparison:
        vec = embed_text(q)
        hy = hybrid_search(q, top_k=HYBRID_TOP_K, candidate_k=CANDIDATE_K,
                           filters=RetrievalFilter(tickers=(tk,), fiscal_years=tuple(years)),
                           query_embedding=vec)
        hybrid = [{"chunk_id": r["chunk_id"], "fy": r.get("fiscal_year"),
                   "rrf": round(r.get("rrf_score", r.get("score", 0)) or 0, 5),
                   "has_expected": any(_digits(v) in _digits(r.get("text", ""))
                                       for v in item["expect"].values()),
                   "has_wrong": _digits(item.get("wrong_year_value", "")) in _digits(r.get("text", "")) if item.get("wrong_year_value") else False}
                  for r in hy]

    ev = plan["evidence"]
    ev_rows = []
    for i, c in enumerate(ev, 1):
        txt = c.get("text", "")
        ev_rows.append({
            "src": i, "chunk_id": c["chunk_id"], "fy": c.get("fiscal_year"),
            "rerank_score": round(c["rerank_score"], 4) if c.get("rerank_score") is not None else None,
            "has_expected": {yr: (_digits(v) in _digits(txt)) for yr, v in item["expect"].items()},
            "has_wrong": (_digits(item["wrong_year_value"]) in _digits(txt)) if item.get("wrong_year_value") else None,
        })

    # per-year value / attribution
    per_year = {}
    for yr, val in item["expect"].items():
        value_in_answer = _value_present(answer, val)
        attr_ok = _year_attribution_ok(answer, yr, val, item.get("wrong_year_value"))
        per_year[yr] = {"value_in_answer": value_in_answer, "attribution_ok": attr_ok}
    wrong_in_answer = bool(item.get("wrong_year_value")) and \
        _digits(item["wrong_year_value"]) in _digits(answer) and \
        not all(_digits(v) in _digits(answer) for v in item["expect"].values())

    ok = all(pr["attribution_ok"] for pr in per_year.values()) and not wrong_in_answer

    # classification
    cls = None
    if not ok:
        exp_vals = {_digits(v) for v in item["expect"].values()}
        ev_has_expected = any(any(hx for hx in r["has_expected"].values()) for r in ev_rows)
        ev_ids_with_expected = {r["chunk_id"] for r in ev_rows if any(r["has_expected"].values())}
        ev_ids_with_wrong = {r["chunk_id"] for r in ev_rows if r["has_wrong"]}
        cited = set(_cited(answer))
        cited_chunk_ids = {ev_rows[n - 1]["chunk_id"] for n in cited if 1 <= n <= len(ev_rows)}
        hybrid_has_expected = any(h["has_expected"] for h in hybrid) if hybrid else None

        if hybrid_has_expected is False:
            cls = "retrieval"
        elif not ev_has_expected:
            cls = "reranking"
        elif ev_ids_with_wrong and (ev_ids_with_wrong - ev_ids_with_expected):
            cls = "repeated_table"
        elif ev_ids_with_expected and (cited_chunk_ids & ev_ids_with_expected):
            cls = "generation"
        elif ev_has_expected:
            cls = "evidence_select_or_generation"
        else:
            cls = "unclassified"

    return {
        "id": item["id"], "ticker": tk, "kind": item["kind"], "years": years,
        "question": q, "answer": answer.strip(),
        "scopes": sc.get("scopes"), "evidence_by_scope": sc.get("evidence_by_scope"),
        "reranker_fallback": sc.get("reranker_fallback"),
        "hybrid_topk": hybrid,
        "evidence": ev_rows,
        "citations": _cited(answer),
        "source_years": [s.get("fiscal_year") for s in sources],
        "per_year": per_year,
        "wrong_year_value_in_answer": wrong_in_answer,
        "ok": ok,
        "classification": cls,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args(argv)

    items = json.loads(PROBE.read_text(encoding="utf-8"))
    out = []
    fail_by_kind = {}
    fallback_ct = 0
    total = 0
    for item in items:
        runs = [run_probe(item) for _ in range(a.runs)]
        n_ok = sum(1 for r in runs if r["ok"])
        total += len(runs)
        fallback_ct += sum(1 for r in runs if r["reranker_fallback"])
        classifications = [r["classification"] for r in runs if not r["ok"]]
        fail_by_kind.setdefault(item["kind"], [0, 0])
        fail_by_kind[item["kind"]][0] += (a.runs - n_ok)
        fail_by_kind[item["kind"]][1] += a.runs
        flag = "" if n_ok == a.runs else f"  <-- {a.runs - n_ok}/{a.runs} FAIL {classifications}"
        print(f"{item['id']:<18} {n_ok}/{a.runs} ok{flag}")
        out.append({"item": item, "n_ok": n_ok, "runs": runs})

    OUT = REPO / "evaluation" / "results" / f"numeric_probe_v1_{a.tag}.json"
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    n_fail = total - sum(o["n_ok"] for o in out)
    print(f"\n=== {a.tag} ===  {total} runs, {n_fail} failing ({n_fail/total:.1%})  "
          f"reranker_fallback {fallback_ct}/{total}")
    print("by kind (fails/total):", {k: f"{v[0]}/{v[1]}" for k, v in fail_by_kind.items()})
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
