"""Track how corpus growth affects retrieval quality — append-only history.

    python -m evaluation.track_corpus_growth            # print the history table
    python -m evaluation.track_corpus_growth --append   # run the scoped eval,
                                                        # append a new row

Each row records company_count, chunk_count, filtered/unfiltered Recall@10,
filtered MRR, filter_correctness and leakage. This is the signal for deciding
when the pure-Python BM25 backend must be upgraded (Phase 6): filter
correctness / leakage are structural and stay perfect, but filtered Recall@10
drifts as BM25's global statistics change with every added company.

Historical rows are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "evaluation" / "corpus_growth.json"

_COLUMNS = [
    ("company_count", "companies", 9),
    ("chunk_count", "chunks", 8),
    ("filtered_recall_at_10", "filt R@10", 10),
    ("unfiltered_recall_at_10", "unfilt R@10", 12),
    ("filtered_mrr", "filt MRR", 9),
    ("filter_correctness", "filt OK", 8),
    ("cross_company_leakage_at_10", "leak@10", 8),
]


def _load() -> dict:
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def print_history(doc: dict) -> None:
    print("CORPUS GROWTH HISTORY")
    print(doc.get("note", ""))
    print()
    header = f"{'phase':<10}" + "".join(f"{label:>{w}}" for _, label, w in _COLUMNS)
    print(header)
    print("-" * len(header))
    for row in doc["history"]:
        line = f"{str(row.get('phase','?')):<10}"
        for key, _label, width in _COLUMNS:
            value = row.get(key)
            if isinstance(value, float):
                line += f"{value:>{width}.4f}"
            else:
                line += f"{str(value):>{width}}"
        print(line)
    print()
    print("GATE: filtered Recall@10 must stay >= 0.90; filter_correctness == 1.0;")
    print("      cross_company_leakage@10 == 0.0. When filtered R@10 nears 0.90,")
    print("      upgrade the BM25 backend (Phase 6).")


def append_row(phase: str) -> dict:
    from ingestion import registry
    from retrieval.bm25_search import load_chunks
    from evaluation.evaluate_scoped import evaluate

    unfiltered = evaluate("hybrid", "unfiltered", None)["summary"]
    filtered = evaluate("hybrid", "filtered", None)
    if filtered is None:
        raise SystemExit("filtered retrieval unavailable — cannot append a row")
    filtered = filtered["summary"]

    row = {
        "phase": phase,
        "recorded": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "company_count": len(registry.known_tickers()),
        "companies": sorted(registry.known_tickers()),
        "chunk_count": len(load_chunks()),
        "filtered_recall_at_10": round(filtered["recall_at_10"], 4),
        "unfiltered_recall_at_10": round(unfiltered["recall_at_10"], 4),
        "filtered_mrr": round(filtered["mrr"], 4),
        "filter_correctness": round(filtered["filter_correctness"], 4),
        "cross_company_leakage_at_10": round(filtered["cross_company_leakage_at_10"], 4),
    }

    doc = _load()
    doc["history"].append(row)
    HISTORY_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--phase", default="unlabeled")
    args = parser.parse_args(argv)

    if args.append:
        from dotenv import load_dotenv

        load_dotenv()
        row = append_row(args.phase)
        print("appended:", json.dumps(row, indent=2))

    print_history(_load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
