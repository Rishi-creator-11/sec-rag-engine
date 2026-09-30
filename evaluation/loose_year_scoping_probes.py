"""Supplemental regression suite: loose single-year numeric questions.

NOT part of benchmark_v3_repool (the 125-question retrieval-quality
benchmark behind the published 0.86 MRR / 0.66 Recall@10 / 0.60 Precision@5
figures). This suite exists because that benchmark is 100% anchored/
structured-scope phrasing and cannot see the failure mode this file probes:
a ticker scope with NO structured `fiscal_years`, where the only signal for
which year is wanted is in the question text itself.

Exercises `api.answer_question` in-process against real backends (Pinecone /
Cohere / OpenAI) -- no mocking. Requires the same .env credentials as local
dev. Prints a full transcript per trial and a summary table at the end.

Usage:
    python -m evaluation.loose_year_scoping_probes
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from api.rag import answer_question

REFUSAL_TEXT = (
    "The provided SEC filing excerpts do not contain enough information "
    "to answer this question."
)

# (ticker, year, ground-truth question) -- ground truth is fetched live via
# the STRUCTURED path (fiscal_years=[year]), which is the already-verified
# 1.000/0.000 scope-correct path. Loose-path answers are compared against it.
GROUND_TRUTH_QUESTION = "What was {company} total revenue?"

# XOM's own structured (already-scoped, hard-filtered) path is NOT a stable
# ground-truth oracle: its FY2025 filing's reconciliation table restates
# FY2024's comparative figure inline in the same FY2025-tagged chunk, so
# even a correctly year-scoped query can sample either number. Verified
# figure, from direct chunk inspection in the original investigation
# (evaluation/SCALE_READINESS_REPORT.md): chunk 94, "Total consolidated
# revenues and other income" = 332,238 (FY2025). Used as a fixed oracle
# instead of a live re-query for this one ticker only.
FIXED_GROUND_TRUTH = {
    ("XOM", 2025): 332238.0,
}

CASES = [
    # (ticker, company, year, ground-truth metric question, loose-phrasing questions)
    # The ground-truth question and loose questions must ask about the SAME
    # metric, or "WRONG NUMERIC ANSWER" conflates a real error with an
    # apples-to-oranges comparison bug in this script.
    (
        "XOM", "ExxonMobil", 2025, "total revenue",
        [
            "What was ExxonMobil's revenue in FY2025?",
            "What was XOM's revenue for fiscal year 2025?",
            "How much revenue did ExxonMobil report in FY2025?",
            "What revenue did XOM report for FY 2025?",
            "What was ExxonMobil's total revenue in 2025?",
            "What was ExxonMobil's revenue in FY2025?",
            "What was XOM's revenue for fiscal year 2025?",
            "How much revenue did ExxonMobil report in FY2025?",
            "What revenue did XOM report for FY 2025?",
            "What was ExxonMobil's total revenue in 2025?",
        ],
    ),
    (
        # Generalization target: same segment-heavy-filer risk profile the
        # investigation flagged as latent (energy, restated segment tables).
        "CVX", "Chevron", 2025, "total revenue",
        [
            "What was Chevron's revenue in FY2025?",
            "How much revenue did CVX report for fiscal year 2025?",
            "What was Chevron's total revenue in 2025?",
        ],
    ),
    (
        # Control: column-labeled filer, should already be unaffected either
        # way -- confirms the fix doesn't regress a company that was fine.
        "AAPL", "Apple", 2025, "total net sales",
        [
            "What was Apple's revenue in FY2025?",
            "How much revenue did AAPL report for fiscal year 2025?",
            "What was Apple's total net sales in 2025?",
        ],
    ),
    (
        "JPM", "JPMorgan", 2025, "net income",
        [
            "What was JPMorgan's net income in FY2025?",
            "How much net income did JPM report for fiscal year 2025?",
        ],
    ),
]


_NUMBER_RE = re.compile(r"\$?\s?([\d,]+(?:\.\d+)?)\s*(billion|million)?", re.IGNORECASE)


def _looks_like_a_year(raw: str, unit: str | None) -> bool:
    """True for a bare "FY2025" / "2025" style token, not a dollar figure.

    A real headline figure in this domain always either carries a
    thousands separator/decimal ("332,238", "$182.4 billion") or an
    explicit unit word. A plain 4-digit 20xx token with neither is almost
    always the fiscal year being restated in the sentence, not the answer.
    """
    if unit or "," in raw or "." in raw:
        return False
    if len(raw) != 4:
        return False
    try:
        value = int(raw)
    except ValueError:
        return False
    return 2000 <= value <= 2099


def _extract_headline_number(text: str) -> float | None:
    """Best-effort: first dollar figure in the answer, normalized to millions."""
    for match in _NUMBER_RE.finditer(text or ""):
        raw, unit = match.groups()
        if "," not in raw and "." not in raw and len(raw) < 4:
            continue  # skip bare small ints ("5 evidence chunks", "[Source 4]")
        if _looks_like_a_year(raw, unit):
            continue  # skip "FY2025" / bare "2025" -- not the figure itself
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit and unit.lower() == "billion":
            value *= 1000
        return value
    return None


@dataclass
class Trial:
    ticker: str
    year: int
    question: str
    answer: str
    source_years: list[int | None]
    fiscal_year_inferred: bool
    classification: str
    headline_number: float | None = None
    ground_truth_number: float | None = None


def classify(answer: str, source_years: list[int | None], year: int,
             ground_truth: float | None) -> str:
    """CORRECT / REFUSAL / WRONG YEAR ATTRIBUTION / WRONG NUMERIC ANSWER / UNVERIFIED.

    WRONG YEAR ATTRIBUTION is specifically "the answer's number came from a
    chunk tagged with a different fiscal year than requested" -- the exact
    failure mode this fix targets. It is distinct from a numerically wrong
    answer whose evidence was still correctly year-scoped (a real but
    different, pre-existing problem: multiple figures for the same year
    inside one correctly-scoped chunk, e.g. segment vs. consolidated
    totals -- out of scope for this fix, tracked separately).
    """
    if answer.strip() == REFUSAL_TEXT:
        return "REFUSAL"
    number = _extract_headline_number(answer)
    if number is None:
        return "REFUSAL"  # non-numeric non-refusal text; treat as no answer given
    if wrong_year_evidence(source_years, year):
        return "WRONG YEAR ATTRIBUTION"
    if ground_truth is None:
        return "UNVERIFIED"  # no reliable ground truth this run; not scored either way
    if abs(number - ground_truth) / ground_truth < 0.005:
        return "CORRECT"
    return "WRONG NUMERIC ANSWER"


def mixed_year(source_years: list[int | None]) -> bool:
    distinct = {y for y in source_years if y is not None}
    return len(distinct) > 1


def wrong_year_evidence(source_years: list[int | None], year: int) -> bool:
    return any(y is not None and y != year for y in source_years)


def run() -> list[Trial]:
    trials: list[Trial] = []

    for ticker, company, year, metric, questions in CASES:
        if (ticker, year) in FIXED_GROUND_TRUTH:
            ground_truth = FIXED_GROUND_TRUTH[(ticker, year)]
            print(f"=== {ticker} FY{year} ground truth (fixed oracle) ===")
            print(f"{ground_truth} (from prior chunk-level verification, "
                  f"not re-queried -- see FIXED_GROUND_TRUTH comment)\n")
        else:
            # 3x majority vote over the already-verified structured path,
            # rather than trusting a single sample -- some filers restate a
            # comparative prior-year figure inline in a correctly-scoped
            # chunk, which can make even the structured path non-deterministic.
            samples = []
            for _ in range(3):
                gt_result = answer_question(
                    f"What was {company}'s {metric}?",
                    tickers=[ticker],
                    fiscal_years=[year],
                )
                samples.append(_extract_headline_number(gt_result["answer"]))
                time.sleep(0.3)
            counts: dict[float | None, int] = {}
            for s in samples:
                counts[s] = counts.get(s, 0) + 1
            ground_truth = max(counts, key=counts.get)
            print(f"=== {ticker} FY{year} ground truth (structured path, 3x majority vote) ===")
            print(f"samples: {samples}")
            print(f"majority: {ground_truth}")
            if len(set(samples)) > 1:
                print("NOTE: structured-path samples disagreed -- non-determinism "
                      "in the ground truth itself, not just the loose path.")
            print()

        for question in questions:
            result = answer_question(question, tickers=[ticker])
            source_years = [s.get("fiscal_year") for s in result["sources"]]
            classification = classify(result["answer"], source_years, year, ground_truth)
            trial = Trial(
                ticker=ticker,
                year=year,
                question=question,
                answer=result["answer"],
                source_years=source_years,
                fiscal_year_inferred=result["search_scope"]["fiscal_year_inferred"],
                classification=classification,
                headline_number=_extract_headline_number(result["answer"]),
                ground_truth_number=ground_truth,
            )
            trials.append(trial)

            print(f"--- {ticker} | {question!r} ---")
            print(f"answer: {trial.answer[:200]}")
            print(f"source fiscal_years: {trial.source_years}")
            print(f"fiscal_year_inferred: {trial.fiscal_year_inferred}")
            print(f"classification: {trial.classification}  "
                  f"mixed_year_evidence={mixed_year(source_years)}  "
                  f"wrong_year_evidence={wrong_year_evidence(source_years, year)}")
            print()
            time.sleep(0.5)

    return trials


def summarize(trials: list[Trial]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    by_class: dict[str, int] = {}
    mixed_count = 0
    wrong_year_evidence_count = 0
    inferred_count = 0

    for t in trials:
        by_class[t.classification] = by_class.get(t.classification, 0) + 1
        if mixed_year(t.source_years):
            mixed_count += 1
        if wrong_year_evidence(t.source_years, t.year):
            wrong_year_evidence_count += 1
        if t.fiscal_year_inferred:
            inferred_count += 1

    print(f"Total trials: {len(trials)}")
    print(f"fiscal_year_inferred=True: {inferred_count}/{len(trials)}")
    print(f"MIXED-YEAR EVIDENCE: {mixed_count}/{len(trials)}")
    print(f"WRONG-YEAR EVIDENCE (any non-matching-year chunk): {wrong_year_evidence_count}/{len(trials)}")
    print()
    for label in ("CORRECT", "REFUSAL", "WRONG YEAR ATTRIBUTION",
                  "WRONG NUMERIC ANSWER", "UNVERIFIED"):
        print(f"{label}: {by_class.get(label, 0)}/{len(trials)}")

    print("\nBy ticker:")
    for ticker in sorted({t.ticker for t in trials}):
        subset = [t for t in trials if t.ticker == ticker]
        sub_mixed = sum(1 for t in subset if mixed_year(t.source_years))
        sub_correct = sum(1 for t in subset if t.classification == "CORRECT")
        print(f"  {ticker}: {len(subset)} trials, "
              f"{sub_correct} correct, {sub_mixed} mixed-year-evidence")


if __name__ == "__main__":
    all_trials = run()
    summarize(all_trials)
