"""Scoped retrieval evaluation on Benchmark v2.

Phase 1A goal: measure, on the existing 60-question benchmark, how much
cross-company evidence leaks into results today (UNFILTERED), and stand up the
harness that will prove ticker filtering removes that leakage once Phase 1B
wires :class:`retrieval.filters.RetrievalFilter` into the retrievers.

This module never modifies ``benchmark_v2.json`` or its ``relevant_chunks``.
The expected ticker for each question is derived, not stored:

  1. from the ``*_10k_*`` prefix of every entry in ``relevant_chunks``
     (the most direct signal), else
  2. from the benchmark ``company`` field (already a ticker), else
  3. from company names found in the question text.

Metrics
-------
Existing retrieval quality (unsupported questions excluded, as in evaluate_v2):
    Recall@{1,3,5,10}, Precision@{1,3,5}, MRR, Numeric Evidence Hit@{5,10}

New scope metrics (computed for every question with a derivable expected
ticker set):
    filter_correctness        fraction of returned chunks whose ticker is in
                              the expected set (headline; == 1 - leakage@10)
    cross_company_leakage@k   mean over questions of
                              (returned chunks in top-k with an unexpected
                               ticker) / k        for k in {1,3,5,10}

Targets once filtering is active:
    filter_correctness == 1.000
    cross_company_leakage@k == 0.000
    filtered Hybrid Recall@10 >= 0.90   (must not fall below the unfiltered
                                         baseline this script records now)

Modes
-----
    --mode unfiltered   (default) run retrievers as production does today
    --mode filtered     pass a RetrievalFilter; requires Phase 1B plumbing and
                        exits cleanly with a message if it is not present yet
    --mode compare      run both and diff (filtered half pending Phase 1B)

Filtered metrics are never synthesized. If the retriever cannot accept a
filter, the filtered run is skipped and reported as skipped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evaluation.evaluate_v2 import (  # noqa: E402
    hit_at_k,
    is_numeric,
    is_unsupported,
    mean_or_none,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.filters import RetrievalFilter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "evaluation" / "benchmark_v2.json"
JSON_PATH = REPO_ROOT / "evaluation" / "results" / "scoped_retrieval_evaluation.json"
SUMMARY_PATH = REPO_ROOT / "evaluation" / "results" / "scoped_retrieval_summary.txt"

TOP_K = 10
RECALL_GATE = 0.90

# Seed-filing chunk-id prefix -> ticker. Canonical future IDs already start with
# the ticker, so they are handled by the generic parser below.
PREFIX_TO_TICKER = {
    "apple_10k": "AAPL",
    "microsoft_10k": "MSFT",
    "nvidia_10k": "NVDA",
}
NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "aapl": "AAPL",
    "msft": "MSFT",
    "nvda": "NVDA",
}
KNOWN_TICKERS = {"AAPL", "MSFT", "NVDA"}


class FilterNotPlumbed(RuntimeError):
    """Raised when a retriever cannot yet accept a RetrievalFilter."""


# --------------------------------------------------------------------------- #
# Ticker derivation                                                           #
# --------------------------------------------------------------------------- #
_SEED_FISCAL_YEAR = {"apple_10k": 2024, "microsoft_10k": 2025, "nvidia_10k": 2026}


def _qrel_fiscal_years(relevant_chunks: list[str]) -> list[int]:
    """Fiscal years the qrels live in (seed prefix map + canonical TICKER_FY_...)."""
    years: set[int] = set()
    for cid in relevant_chunks:
        for prefix, year in _SEED_FISCAL_YEAR.items():
            if cid.startswith(prefix + "_"):
                years.add(year)
                break
        else:
            parts = cid.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                years.add(int(parts[1]))
    return sorted(years)


def ticker_from_chunk_id(chunk_id: str) -> str | None:
    for prefix, ticker in PREFIX_TO_TICKER.items():
        if chunk_id.startswith(f"{prefix}_"):
            return ticker
    # Canonical form: TICKER_YEAR_TYPE_ACCESSION_INDEX
    head = chunk_id.split("_", 1)[0].upper()
    if head in KNOWN_TICKERS or (head.isalpha() and 1 <= len(head) <= 5):
        return head
    return None


def ticker_of_result(result: dict) -> str | None:
    value = result.get("ticker")
    if value:
        return str(value).strip().upper()
    return ticker_from_chunk_id(result.get("chunk_id", ""))


def derive_expected_tickers(item: dict) -> tuple[set[str], list[str]]:
    """Return ``(expected_tickers, warnings)`` for one benchmark item."""
    warnings: list[str] = []
    from_chunks = {
        t
        for cid in item.get("relevant_chunks", [])
        if (t := ticker_from_chunk_id(cid)) is not None
    }

    company = str(item.get("company", "")).strip()
    from_company: set[str] = set()
    if company:
        mapped = NAME_TO_TICKER.get(company.lower())
        if mapped:
            from_company.add(mapped)
        elif company.upper() in KNOWN_TICKERS:
            from_company.add(company.upper())

    lowered = item.get("question", "").lower()
    from_question = {
        ticker for name, ticker in NAME_TO_TICKER.items() if name in lowered
    }

    if from_chunks:
        expected = from_chunks
        if from_company and not from_company <= expected:
            warnings.append(
                f"{item['id']}: company {sorted(from_company)} not in "
                f"relevant-chunk tickers {sorted(expected)}"
            )
    elif from_company:
        expected = from_company
    else:
        expected = from_question
        if not expected:
            warnings.append(f"{item['id']}: no expected ticker could be derived")

    return expected, warnings


# --------------------------------------------------------------------------- #
# Retrievers                                                                  #
# --------------------------------------------------------------------------- #
def _call(search_fn, question, filt, **kwargs):
    """Call a retriever, adding ``filters=`` only when a non-empty filter is given."""
    if filt is None or filt.is_empty():
        return search_fn(question, **kwargs)
    try:
        return search_fn(question, filters=filt, **kwargs)
    except TypeError as exc:
        if "filters" in str(exc):
            raise FilterNotPlumbed(getattr(search_fn, "__module__", "retriever"))
        raise


def get_retrievers():
    from retrieval.bm25_search import search as bm25_search
    from retrieval.hybrid_search import search as hybrid_search
    from retrieval.pinecone_search import search as dense_search

    return {
        "dense": lambda q, filt: _call(dense_search, q, filt, top_k=TOP_K),
        "bm25": lambda q, filt: _call(bm25_search, q, filt, top_k=TOP_K),
        # Production mix: dense + BM25. Sparse stays disabled (use_sparse=False).
        "hybrid": lambda q, filt: _call(
            hybrid_search, q, filt, top_k=TOP_K, candidate_k=TOP_K
        ),
    }


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def load_benchmark() -> list[dict]:
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected benchmark shape in {BENCHMARK_PATH}")
    return data


def leakage_at_k(tickers: list[str | None], expected: set[str], k: int) -> float:
    window = tickers[:k]
    if not window:
        return 0.0
    unexpected = sum(1 for t in window if t is None or t not in expected)
    return unexpected / k


def evaluate(retriever_name: str, mode: str, limit: int | None) -> dict | None:
    benchmark = load_benchmark()
    if limit is not None:
        benchmark = benchmark[:limit]

    retrievers = get_retrievers()
    if retriever_name not in retrievers:
        raise SystemExit(f"unknown retriever {retriever_name!r}")
    search_fn = retrievers[retriever_name]
    filtered = mode == "filtered"

    rows: list[dict] = []
    warnings: list[str] = []

    for item in benchmark:
        expected, item_warnings = derive_expected_tickers(item)
        warnings.extend(item_warnings)
        relevant = set(item.get("relevant_chunks", []))
        # Scope to the fiscal year(s) the qrels live in. benchmark_v2 predates
        # multi-year ingestion; once a company has several fiscal years in the
        # corpus a ticker-only filter dilutes the (single-year) qrels. Deriving
        # the year from the qrel chunk ids keeps the historical benchmark a fair
        # reference. (It is NOT a production gate — see PHASE_4.5_REPORT.md.)
        years = _qrel_fiscal_years(item.get("relevant_chunks", []))
        filt = (
            RetrievalFilter(
                tickers=tuple(sorted(expected)),
                fiscal_years=tuple(years) if years else None,
            )
            if (filtered and expected)
            else None
        )

        try:
            results = search_fn(item["question"], filt)
        except FilterNotPlumbed as exc:
            print(
                f"  filtered retrieval unavailable: {retriever_name} does not "
                f"accept a RetrievalFilter yet ({exc}).\n"
                "  Phase 1B wires filters into the retrievers; re-run "
                "`--mode filtered` after that."
            )
            return None

        retrieved = results[:TOP_K]
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        retrieved_tickers = [ticker_of_result(r) for r in retrieved]
        rr, first_rank = reciprocal_rank(retrieved_ids, relevant)

        rows.append(
            {
                "id": item["id"],
                "expected_tickers": sorted(expected),
                "unsupported": is_unsupported(item),
                "numeric": is_numeric(item),
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_tickers": retrieved_tickers,
                "first_relevant_rank": first_rank,
                "recall_at_1": recall_at_k(retrieved_ids, relevant, 1),
                "recall_at_3": recall_at_k(retrieved_ids, relevant, 3),
                "recall_at_5": recall_at_k(retrieved_ids, relevant, 5),
                "recall_at_10": recall_at_k(retrieved_ids, relevant, 10),
                "precision_at_1": precision_at_k(retrieved_ids, relevant, 1),
                "precision_at_3": precision_at_k(retrieved_ids, relevant, 3),
                "precision_at_5": precision_at_k(retrieved_ids, relevant, 5),
                "reciprocal_rank": rr,
                "numeric_hit_at_5": (
                    hit_at_k(retrieved_ids, relevant, 5)
                    if is_numeric(item)
                    else None
                ),
                "numeric_hit_at_10": (
                    hit_at_k(retrieved_ids, relevant, 10)
                    if is_numeric(item)
                    else None
                ),
                "scoped": bool(expected),
                "filter_hits_top10": sum(
                    1 for t in retrieved_tickers if t in expected
                ),
                "returned_count": len(retrieved),
                "leakage_at_1": leakage_at_k(retrieved_tickers, expected, 1)
                if expected
                else None,
                "leakage_at_3": leakage_at_k(retrieved_tickers, expected, 3)
                if expected
                else None,
                "leakage_at_5": leakage_at_k(retrieved_tickers, expected, 5)
                if expected
                else None,
                "leakage_at_10": leakage_at_k(retrieved_tickers, expected, 10)
                if expected
                else None,
            }
        )

    eligible = [r for r in rows if not r["unsupported"]]
    numeric = [r for r in rows if r["numeric"]]
    scoped = [r for r in rows if r["scoped"]]
    scoped_supported = [r for r in scoped if not r["unsupported"]]

    def _mean(field, group):
        return mean_or_none([r[field] for r in group if r[field] is not None])

    correctness_values = [
        r["filter_hits_top10"] / r["returned_count"]
        for r in scoped
        if r["returned_count"]
    ]

    summary = {
        "retriever": retriever_name,
        "mode": mode,
        "questions": len(rows),
        "eligible_questions": len(eligible),
        "numeric_questions": len(numeric),
        "scoped_questions": len(scoped),
        "recall_at_1": _mean("recall_at_1", eligible),
        "recall_at_3": _mean("recall_at_3", eligible),
        "recall_at_5": _mean("recall_at_5", eligible),
        "recall_at_10": _mean("recall_at_10", eligible),
        "precision_at_1": _mean("precision_at_1", eligible),
        "precision_at_3": _mean("precision_at_3", eligible),
        "precision_at_5": _mean("precision_at_5", eligible),
        "mrr": _mean("reciprocal_rank", eligible),
        "numeric_evidence_hit_at_5": _mean("numeric_hit_at_5", numeric),
        "numeric_evidence_hit_at_10": _mean("numeric_hit_at_10", numeric),
        "filter_correctness": (
            statistics.mean(correctness_values) if correctness_values else None
        ),
        "cross_company_leakage_at_1": _mean("leakage_at_1", scoped),
        "cross_company_leakage_at_3": _mean("leakage_at_3", scoped),
        "cross_company_leakage_at_5": _mean("leakage_at_5", scoped),
        "cross_company_leakage_at_10": _mean("leakage_at_10", scoped),
        "cross_company_leakage_at_10_supported_only": _mean(
            "leakage_at_10", scoped_supported
        ),
        "warnings": sorted(set(warnings)),
    }
    return {"summary": summary, "questions": rows}


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def _fmt(value) -> str:
    return "  n/a" if value is None else f"{value:6.3f}"


def build_summary_text(results: dict[str, dict], mode: str) -> str:
    lines = [
        "SCOPED RETRIEVAL EVALUATION (Benchmark v2)",
        f"mode: {mode}   top_k: {TOP_K}",
        "Expected ticker derived from relevant_chunks / company field.",
        "Unsupported questions excluded from Recall/Precision/MRR/NumericEvidence.",
        "",
    ]
    header = (
        f"{'retriever':<9} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} "
        f"{'P@1':>6} {'P@3':>6} {'P@5':>6} {'MRR':>6} {'NE@5':>6} "
        f"{'FiltOK':>7} {'Leak@1':>7} {'Leak@3':>7} {'Leak@5':>7} {'Leak@10':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, payload in results.items():
        s = payload["summary"]
        lines.append(
            f"{name:<9} {_fmt(s['recall_at_1'])} {_fmt(s['recall_at_3'])} "
            f"{_fmt(s['recall_at_5'])} {_fmt(s['recall_at_10'])} "
            f"{_fmt(s['precision_at_1'])} {_fmt(s['precision_at_3'])} "
            f"{_fmt(s['precision_at_5'])} {_fmt(s['mrr'])} "
            f"{_fmt(s['numeric_evidence_hit_at_5'])} "
            f"{_fmt(s['filter_correctness']):>7} "
            f"{_fmt(s['cross_company_leakage_at_1']):>7} "
            f"{_fmt(s['cross_company_leakage_at_3']):>7} "
            f"{_fmt(s['cross_company_leakage_at_5']):>7} "
            f"{_fmt(s['cross_company_leakage_at_10']):>8}"
        )

    lines.append("")
    lines.append("FiltOK = filter_correctness (target 1.000 once filtering is on)")
    lines.append("Leak@k = cross_company_leakage@k (target 0.000 once filtering is on)")
    lines.append("")

    if "hybrid" in results:
        hybrid = results["hybrid"]["summary"]
        recall10 = hybrid["recall_at_10"]
        if mode == "filtered" and recall10 is not None:
            verdict = "PASS" if recall10 >= RECALL_GATE else "FAIL"
            lines.append(
                f"GATE  filtered Hybrid Recall@10 >= {RECALL_GATE:.2f}  "
                f"({recall10:.3f})  {verdict}"
            )
        elif recall10 is not None:
            lines.append(
                f"BASELINE  unfiltered Hybrid Recall@10 = {recall10:.3f}  "
                f"(the filtered run in Phase 1B must stay >= "
                f"max({RECALL_GATE:.2f}, this))"
            )
            lines.append(
                "BASELINE  unfiltered Hybrid cross_company_leakage@10 = "
                f"{hybrid['cross_company_leakage_at_10']:.3f}  "
                "(this is the leakage Phase 1B filtering must drive to 0)"
            )

    warnings = sorted(
        {w for payload in results.values() for w in payload["summary"]["warnings"]}
    )
    if warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in warnings)

    return "\n".join(lines) + "\n"


def build_compare_text(runs: dict[str, dict]) -> str:
    """Side-by-side unfiltered vs filtered, with a per-metric delta for each retriever."""
    metrics = [
        ("recall_at_1", "R@1"),
        ("recall_at_3", "R@3"),
        ("recall_at_5", "R@5"),
        ("recall_at_10", "R@10"),
        ("precision_at_5", "P@5"),
        ("mrr", "MRR"),
        ("numeric_evidence_hit_at_5", "NE@5"),
        ("filter_correctness", "FiltOK"),
        ("cross_company_leakage_at_1", "Leak@1"),
        ("cross_company_leakage_at_3", "Leak@3"),
        ("cross_company_leakage_at_5", "Leak@5"),
        ("cross_company_leakage_at_10", "Leak@10"),
    ]
    lines = ["SCOPED RETRIEVAL: UNFILTERED vs FILTERED (Benchmark v2)", ""]
    for name in ("dense", "bm25", "hybrid"):
        if name not in runs.get("unfiltered", {}) or name not in runs.get("filtered", {}):
            continue
        u = runs["unfiltered"][name]
        f = runs["filtered"][name]
        lines.append(f"[{name}]")
        lines.append(f"  {'metric':<10} {'unfiltered':>12} {'filtered':>12} {'delta':>10}")
        for key, label in metrics:
            uv, fv = u.get(key), f.get(key)
            if uv is None and fv is None:
                continue
            delta = (
                f"{fv - uv:+.3f}" if (uv is not None and fv is not None) else "  n/a"
            )
            lines.append(
                f"  {label:<10} {_fmt(uv):>12} {_fmt(fv):>12} {delta:>10}"
            )
        lines.append("")

    if "hybrid" in runs.get("filtered", {}):
        r10 = runs["filtered"]["hybrid"]["recall_at_10"]
        fc = runs["filtered"]["hybrid"]["filter_correctness"]
        lk = runs["filtered"]["hybrid"]["cross_company_leakage_at_10"]
        lines.append("PRIMARY GATES (filtered hybrid)")
        lines.append(
            f"  filter_correctness == 1.000   {fc:.3f}   "
            f"{'PASS' if abs(fc - 1.0) < 1e-9 else 'FAIL'}"
        )
        lines.append(
            f"  cross_company_leakage@10 == 0  {lk:.3f}   "
            f"{'PASS' if abs(lk) < 1e-9 else 'FAIL'}"
        )
        lines.append(
            f"  Recall@10 >= 0.90              {r10:.3f}   "
            f"{'PASS' if r10 >= RECALL_GATE else 'FAIL'}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid", "all"],
        default="hybrid",
    )
    parser.add_argument(
        "--mode",
        choices=["unfiltered", "filtered", "compare"],
        default="unfiltered",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    modes = ["unfiltered", "filtered"] if args.mode == "compare" else [args.mode]
    names = (
        ["dense", "bm25", "hybrid"]
        if args.retriever == "all"
        else [args.retriever]
    )

    output: dict = {"benchmark": str(BENCHMARK_PATH), "top_k": TOP_K, "runs": {}}

    for mode in modes:
        results: dict[str, dict] = {}
        print(f"\n=== mode: {mode} ===")
        for name in names:
            print(f"evaluating {name}...")
            payload = evaluate(name, mode, args.limit)
            if payload is None:
                continue
            results[name] = payload
        if not results:
            continue
        output["runs"][mode] = {n: p["summary"] for n, p in results.items()}
        output["runs"][mode]["_questions"] = {
            n: p["questions"] for n, p in results.items()
        }
        text = build_summary_text(results, mode)
        print()
        print(text)

    if not output["runs"]:
        print("\nNo runs produced results; existing result files left untouched.")
        return 0

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    summaries_only = {
        mode: {n: s for n, s in run.items() if not n.startswith("_")}
        for mode, run in output["runs"].items()
    }
    if len(summaries_only) == 2:
        summary_text = build_compare_text(summaries_only)
    else:
        last_mode = list(summaries_only)[-1]
        summary_text = build_summary_text(
            {n: {"summary": s} for n, s in summaries_only[last_mode].items()},
            last_mode,
        )
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    print()
    print(summary_text)
    print(f"Saved JSON to {JSON_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
