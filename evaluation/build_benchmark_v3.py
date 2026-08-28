"""Assemble benchmark_v3 question set for the current 8-company corpus.

Sources:
  - all 60 benchmark_v2 questions (AAPL/MSFT/NVDA) — still valid, re-pooled/re-judged
  - the per-company canary sets for AMZN/GOOGL/META/JPM/WMT (10 each)
  - a handful of extra questions per new company (defined below)

Output: evaluation/benchmark_v3_questions.json  (NO relevant_chunks yet;
those come from build_v3_pool.py + judge_v3_pool.py). benchmark_v2 is NOT
modified. XOM/UNH are not included.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V2 = REPO_ROOT / "evaluation" / "benchmark_v2.json"
OUT = REPO_ROOT / "evaluation" / "benchmark_v3_questions.json"

NEW_COMPANY_CANARIES = {
    "AMZN": REPO_ROOT / "evaluation" / "benchmark_amzn.json",
    "GOOGL": REPO_ROOT / "evaluation" / "benchmark_googl.json",
    "META": REPO_ROOT / "evaluation" / "benchmark_meta.json",
    "JPM": REPO_ROOT / "evaluation" / "benchmark_jpm.json",
    "WMT": REPO_ROOT / "evaluation" / "benchmark_wmt.json",
}

EXTRA = [
    # AMZN
    {"id": "amzn_advertising_01", "company": "AMZN", "category": "business",
     "answer_type": "qualitative",
     "question": "How does Amazon describe its advertising services business?"},
    {"id": "amzn_climate_regulation_01", "company": "AMZN", "category": "regulation",
     "answer_type": "qualitative",
     "question": "What environmental, climate, or sustainability related risks does Amazon disclose?"},
    {"id": "amzn_key_person_01", "company": "AMZN", "category": "risk",
     "answer_type": "qualitative",
     "question": "What does Amazon say about dependence on key personnel and talent retention?"},
    # GOOGL
    {"id": "googl_traffic_acquisition_01", "company": "GOOGL", "category": "financial",
     "answer_type": "qualitative",
     "question": "What does Alphabet say about traffic acquisition costs?"},
    {"id": "googl_other_bets_01", "company": "GOOGL", "category": "business",
     "answer_type": "qualitative",
     "question": "How does Alphabet describe the Other Bets segment and its financial performance?"},
    {"id": "googl_capex_01", "company": "GOOGL", "category": "financial",
     "answer_type": "qualitative",
     "question": "What does Alphabet say about capital expenditures and investment in technical infrastructure?"},
    # META
    {"id": "meta_capex_ai_01", "company": "META", "category": "financial",
     "answer_type": "qualitative",
     "question": "What does Meta say about capital expenditures for AI and data center infrastructure?"},
    {"id": "meta_class_structure_01", "company": "META", "category": "governance",
     "answer_type": "qualitative",
     "question": "How does Meta describe its dual-class share structure and voting control?"},
    {"id": "meta_litigation_01", "company": "META", "category": "regulation",
     "answer_type": "qualitative",
     "question": "What material legal proceedings and litigation does Meta describe?"},
    # JPM
    {"id": "jpm_deposits_01", "company": "JPM", "category": "financial",
     "answer_type": "qualitative",
     "question": "What does JPMorgan Chase disclose about its deposit base and funding?"},
    {"id": "jpm_liquidity_01", "company": "JPM", "category": "risk",
     "answer_type": "qualitative",
     "question": "How does JPMorgan Chase describe liquidity risk and its liquidity coverage ratio?"},
    {"id": "jpm_climate_risk_01", "company": "JPM", "category": "regulation",
     "answer_type": "qualitative",
     "question": "What does JPMorgan Chase say about climate-related financial risk?"},
    # WMT
    {"id": "wmt_membership_income_01", "company": "WMT", "category": "financial",
     "answer_type": "qualitative",
     "question": "What does Walmart say about membership and other income, including Sam's Club and Walmart+?"},
    {"id": "wmt_associates_labor_01", "company": "WMT", "category": "risk",
     "answer_type": "qualitative",
     "question": "What does Walmart describe about labor, associates, and wage-related risks?"},
    {"id": "wmt_advertising_01", "company": "WMT", "category": "business",
     "answer_type": "qualitative",
     "question": "How does Walmart describe its advertising business (Walmart Connect)?"},
]


def _norm_v2(q: dict) -> dict:
    return {
        "id": q["id"],
        "question": q["question"],
        "company": q["company"],
        "category": q.get("category", "general"),
        "answer_type": q.get("answer_type", "qualitative"),
        "source": "v2",
        "v2_relevant_chunks": q.get("relevant_chunks", []),
    }


def _norm_canary(q: dict, ticker: str) -> dict:
    return {
        "id": q["id"],
        "question": q["question"],
        "company": ticker,
        "category": q.get("category", "general"),
        "answer_type": q.get("answer_type", "qualitative"),
        "source": "canary",
        "number_hint": q.get("number_hint"),
    }


def main() -> int:
    out: list[dict] = []
    seen: set[str] = set()

    for q in json.loads(V2.read_text(encoding="utf-8")):
        out.append(_norm_v2(q))
        seen.add(q["id"])

    for ticker, path in NEW_COMPANY_CANARIES.items():
        for q in json.loads(path.read_text(encoding="utf-8")):
            if q["id"] in seen:
                continue
            out.append(_norm_canary(q, ticker))
            seen.add(q["id"])

    for q in EXTRA:
        if q["id"] in seen:
            continue
        q = {**q, "source": "extra"}
        q.setdefault("number_hint", None)
        out.append(q)
        seen.add(q["id"])

    from collections import Counter
    by_company = Counter(q["company"] for q in out)
    by_type = Counter(q["answer_type"] for q in out)

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"benchmark_v3_questions.json: {len(out)} questions")
    print(f"  by company: {dict(sorted(by_company.items()))}")
    print(f"  by answer_type: {dict(by_type)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
