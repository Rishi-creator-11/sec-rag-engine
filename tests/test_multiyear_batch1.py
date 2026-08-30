"""Phase 5.5 Batch 1: AAPL / MSFT / AMZN multi-year corpus + benchmark shape.

Registry / benchmark assertions only — no network, no retrieval calls.
"""

import json
import unittest
from pathlib import Path

from ingestion.registry import available_fiscal_years, get_company
from evaluation.validate_batch1_numeric_generation import _accepted_forms

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "evaluation" / "benchmark_multiyear_batch1.json"

# The exact filings Batch 1 targets (SEC discovery, 2026-08-29).
EXPECTED = {
    "AAPL": {
        2025: "0000320193-25-000079",
        2024: "0000320193-24-000123",  # legacy seed, kept
        2023: "0000320193-23-000106",
    },
    "MSFT": {
        2026: "0001193125-26-323660",
        2025: "0000950170-25-100235",  # legacy seed, kept
        2024: "0000950170-24-087843",
    },
    "AMZN": {
        2025: "0001018724-26-000004",  # pre-existing canonical, kept
        2024: "0001018724-25-000004",
        2023: "0001018724-24-000008",
    },
}


class RegistryStateTests(unittest.TestCase):
    def test_three_years_per_company_newest_first(self):
        for ticker, years in EXPECTED.items():
            entry = get_company(ticker)
            self.assertIsNotNone(entry, ticker)
            fys = [f["fiscal_year"] for f in entry["filings"]]
            self.assertEqual(fys, sorted(years, reverse=True), ticker)

    def test_accession_identity_matches_sec_discovery(self):
        for ticker, years in EXPECTED.items():
            by_year = {
                f["fiscal_year"]: f["accession_number"]
                for f in get_company(ticker)["filings"]
            }
            self.assertEqual(by_year, years, ticker)

    def test_no_duplicate_accessions(self):
        for ticker in EXPECTED:
            accs = [f["accession_number"] for f in get_company(ticker)["filings"]]
            self.assertEqual(len(accs), len(set(accs)), ticker)

    def test_available_fiscal_years(self):
        self.assertEqual(available_fiscal_years("AAPL"), [2025, 2024, 2023])
        self.assertEqual(available_fiscal_years("MSFT"), [2026, 2025, 2024])
        self.assertEqual(available_fiscal_years("AMZN"), [2025, 2024, 2023])

    def test_report_dates_are_not_calendar_year_end(self):
        # AAPL fiscal year ends late September, MSFT late June — the pipeline must
        # not assume Dec-31.
        aapl = {f["fiscal_year"]: f["report_date"]
                for f in get_company("AAPL")["filings"]}
        self.assertEqual(aapl[2025], "2025-09-27")
        msft = {f["fiscal_year"]: f["report_date"]
                for f in get_company("MSFT")["filings"]}
        self.assertEqual(msft[2026], "2026-06-30")


class BenchmarkShapeTests(unittest.TestCase):
    def setUp(self):
        self.items = json.loads(BENCH.read_text(encoding="utf-8"))

    def test_ids_unique(self):
        ids = [i["id"] for i in self.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scopes_reference_available_years(self):
        for it in self.items:
            if it["answer_type"] == "unsupported_year":
                for s in it["scopes"]:
                    self.assertNotIn(
                        int(s["fiscal_year"]),
                        available_fiscal_years(s["ticker"].upper()), it["id"],
                    )
            else:
                for s in it["scopes"]:
                    self.assertIn(
                        int(s["fiscal_year"]),
                        available_fiscal_years(s["ticker"].upper()), it["id"],
                    )

    def test_every_company_has_latest_oldest_and_comparison(self):
        for ticker, years in EXPECTED.items():
            mine = [i for i in self.items if i["ticker"] == ticker]
            single = {int(i["scopes"][0]["fiscal_year"])
                      for i in mine if len(i["scopes"]) == 1
                      and i["answer_type"] != "unsupported_year"}
            self.assertIn(max(years), single, f"{ticker} latest-year question")
            self.assertIn(min(years), single, f"{ticker} oldest-year question")
            self.assertTrue(
                any(len(i["scopes"]) >= 2 for i in mine),
                f"{ticker} year-comparison question",
            )

    def test_at_least_one_numeric_comparison_per_company(self):
        for ticker in EXPECTED:
            mine = [i for i in self.items if i["ticker"] == ticker]
            self.assertTrue(
                any(i["answer_type"] == "comparison_numeric" for i in mine),
                ticker,
            )

    def test_numeric_hints_are_digit_strings(self):
        for it in self.items:
            for hint in (it.get("number_hint_by_year") or {}).values():
                self.assertRegex(hint.replace(",", ""), r"^\d+$", it["id"])


class AcceptedNumberFormsTests(unittest.TestCase):
    def test_exact_and_rounded_billion_forms(self):
        forms = _accepted_forms("383,285")
        self.assertIn("383285", forms)   # income-statement millions
        self.assertIn("3833", forms)     # MD&A "$383.3 billion"

    def test_non_numeric_hint_is_safe(self):
        self.assertEqual(_accepted_forms("n/a"), [""])


if __name__ == "__main__":
    unittest.main()
