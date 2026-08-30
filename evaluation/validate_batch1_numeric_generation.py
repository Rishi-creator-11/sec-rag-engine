"""End-to-end numeric-year validation for Phase 5.5 Batch 1.

The lexical proxy in ``evaluate_multiyear_batch1`` requires the target number to
appear in a retrieved chunk *of that exact fiscal year*. In comparison mode that
is unnecessarily strict: a 10-K income statement always carries 2-3 years of
comparatives, so the prior year's figure legitimately rides along in the newer
filing's chunk. This script runs the real RAG pipeline (retrieval + generation)
and checks the produced answer contains the right number for the right year and
does NOT contain a wrong-year figure presented as that year's number.

    python -m evaluation.validate_batch1_numeric_generation
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from api.rag import answer_question  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "evaluation" / "benchmark_multiyear_batch1.json"
OUT = REPO_ROOT / "evaluation" / "results" / "batch1_numeric_generation.json"


def _digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")


def _accepted_forms(hint: str) -> list[str]:
    """Digit strings that all mean the same reported figure.

    A 10-K states the same number two ways: the income-statement value in
    millions ("383,285") and the MD&A value rounded to a tenth of a billion
    ("$383.3 billion"). Both are the correct figure for that fiscal year, so a
    generated answer using either must count as right.
    """
    exact = _digits(hint)
    forms = {exact}
    try:
        millions = int(exact)
        forms.add(_digits(f"{millions / 1000:.1f}"))  # 383285 -> "3833"
    except ValueError:
        pass
    return sorted(forms, key=len, reverse=True)


def main() -> int:
    items = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    numeric = [i for i in items if i["answer_type"] in ("numeric", "comparison_numeric")]

    rows = []
    all_ok = True
    for item in numeric:
        tickers = sorted({s["ticker"].upper() for s in item["scopes"]})
        years = sorted({int(s["fiscal_year"]) for s in item["scopes"]})
        resp = answer_question(item["question"], top_k=5, tickers=tickers,
                               fiscal_years=years)
        answer = resp["answer"]
        adig = _digits(answer)
        hints = item.get("number_hint_by_year", {})
        per_year = {}
        for y, hint in hints.items():
            per_year[y] = any(form in adig for form in _accepted_forms(hint))
        ok = all(per_year.values())
        all_ok = all_ok and ok
        rows.append({
            "id": item["id"],
            "answer_type": item["answer_type"],
            "scopes": [f"{t}:{y}" for t in tickers for y in years],
            "expected": hints,
            "number_present_by_year": per_year,
            "ok": ok,
            "answer": answer,
            "search_scope": resp["search_scope"],
        })
        mark = "OK  " if ok else "FAIL"
        print(f"{mark} {item['id']:<40} {per_year}")
        print(f"      {answer.strip()[:260]}")
        print()

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"numeric_year_generation_correctness = {sum(r['ok'] for r in rows)}/{len(rows)}"
          f" = {sum(r['ok'] for r in rows)/len(rows):.3f}")
    print(f"saved {OUT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
