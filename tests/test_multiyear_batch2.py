"""Phase 5.5 Batch 2: GOOGL / META / JPM multi-year corpus + benchmark shape.

Registry / benchmark assertions only — no network, no retrieval calls.
"""

import json
import unittest
from pathlib import Path

from ingestion.registry import available_fiscal_years, get_company
from evaluation.validate_batch2_numeric_generation import _accepted_forms

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "evaluation" / "benchmark_multiyear_batch2.json"

# The exact filings Batch 2 targets (SEC discovery, 2026-08-30).
EXPECTED = {
    "GOOGL": {
        2025: "0001652044-26-000018",  # pre-existing canonical (Phase 4), kept
        2024: "0001652044-25-000014",
        2023: "0001652044-24-000022",
    },
    "META": {
        2025: "0001628280-26-003942",  # pre-existing canonical (Phase 4), kept
        2024: "0001326801-25-000017",
        2023: "0001326801-24-000012",
    },
    "JPM": {
        2025: "0001628280-26-008131",  # pre-existing canonical (Phase 4), kept
        2024: "0000019617-25-000270",
        2023: "0000019617-24-000225",
    },
}

# report_date (fiscal-period end) and filing_date from SEC discovery.
EXPECTED_DATES = {
    "0001652044-25-000014": ("2024-12-31", "2025-02-05"),
    "0001652044-24-000022": ("2023-12-31", "2024-01-31"),
    "0001326801-25-000017": ("2024-12-31", "2025-01-30"),
    "0001326801-24-000012": ("2023-12-31", "2024-02-02"),
    "0000019617-25-000270": ("2024-12-31", "2025-02-14"),
    "0000019617-24-000225": ("2023-12-31", "2024-02-16"),
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
        self.assertEqual(available_fiscal_years("GOOGL"), [2025, 2024, 2023])
        self.assertEqual(available_fiscal_years("META"), [2025, 2024, 2023])
        self.assertEqual(available_fiscal_years("JPM"), [2025, 2024, 2023])

    def test_report_and_filing_dates_match_sec(self):
        for ticker in EXPECTED:
            for f in get_company(ticker)["filings"]:
                exp = EXPECTED_DATES.get(f["accession_number"])
                if exp is None:  # pre-existing Phase-4 filing, not this batch
                    continue
                self.assertEqual(
                    (f["report_date"], f["filing_date"]), exp, f["accession_number"]
                )


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

    def test_at_least_one_unsupported_year_per_company(self):
        for ticker in EXPECTED:
            mine = [i for i in self.items if i["ticker"] == ticker]
            self.assertTrue(
                any(i["answer_type"] == "unsupported_year" for i in mine), ticker
            )

    def test_numeric_hints_are_digit_strings(self):
        for it in self.items:
            for hint in (it.get("number_hint_by_year") or {}).values():
                self.assertRegex(hint.replace(",", ""), r"^\d+$", it["id"])


class AcceptedNumberFormsTests(unittest.TestCase):
    def test_exact_and_rounded_billion_forms(self):
        forms = _accepted_forms("402,836")
        self.assertIn("402836", forms)   # income-statement millions
        self.assertIn("4028", forms)     # MD&A "$402.8 billion"

    def test_non_numeric_hint_is_safe(self):
        self.assertEqual(_accepted_forms("n/a"), [""])


if __name__ == "__main__":
    unittest.main()
