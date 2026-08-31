"""Phase 5.5 Batch 3: WMT / UNH / XOM multi-year corpus + benchmark shape.

Registry / benchmark assertions only — no network, no retrieval calls.
Includes XOM ticker->registrant lineage checks (historical EXXON MOBIL CORP,
CIK 0000034088, vs the 2026 successor CIK 0002115436).
"""

import json
import unittest
from pathlib import Path

from ingestion.registry import available_fiscal_years, get_company
from evaluation.validate_batch3_numeric_generation import _accepted_forms

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "evaluation" / "benchmark_multiyear_batch3.json"

# The exact filings Batch 3 targets (SEC discovery, 2026-08-31).
EXPECTED = {
    "WMT": {
        2026: "0000104169-26-000055",  # pre-existing canonical (Phase 4), kept
        2025: "0000104169-25-000021",
        2024: "0000104169-24-000056",
    },
    "UNH": {
        2025: "0000731766-26-000062",  # pre-existing canonical (Phase 4 Batch 3), kept
        2024: "0000731766-25-000063",
        2023: "0000731766-24-000081",
    },
    "XOM": {
        2025: "0000034088-26-000045",  # pre-existing, CIK-override filing, kept
        2024: "0000034088-25-000010",
        2023: "0000034088-24-000018",
    },
}

# report_date (fiscal-period end) and filing_date from SEC discovery.
EXPECTED_DATES = {
    "0000104169-25-000021": ("2025-01-31", "2025-03-14"),
    "0000104169-24-000056": ("2024-01-31", "2024-03-15"),
    "0000731766-25-000063": ("2024-12-31", "2025-02-27"),
    "0000731766-24-000081": ("2023-12-31", "2024-02-28"),
    "0000034088-25-000010": ("2024-12-31", "2025-02-19"),
    "0000034088-24-000018": ("2023-12-31", "2024-02-28"),
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
        self.assertEqual(available_fiscal_years("WMT"), [2026, 2025, 2024])
        self.assertEqual(available_fiscal_years("UNH"), [2025, 2024, 2023])
        self.assertEqual(available_fiscal_years("XOM"), [2025, 2024, 2023])

    def test_wmt_fiscal_year_is_end_january_not_calendar(self):
        wmt = {f["fiscal_year"]: f["report_date"] for f in get_company("WMT")["filings"]}
        self.assertEqual(wmt[2026], "2026-01-31")
        self.assertEqual(wmt[2025], "2025-01-31")
        self.assertEqual(wmt[2024], "2024-01-31")

    def test_report_and_filing_dates_match_sec(self):
        for ticker in EXPECTED:
            for f in get_company(ticker)["filings"]:
                exp = EXPECTED_DATES.get(f["accession_number"])
                if exp is None:
                    continue
                self.assertEqual(
                    (f["report_date"], f["filing_date"]), exp, f["accession_number"]
                )


class XomLineageTests(unittest.TestCase):
    def test_registry_keeps_historical_registrant(self):
        xom = get_company("XOM")
        self.assertEqual(xom["legal_name"], "EXXON MOBIL CORP")
        self.assertEqual(xom["cik"], "0000034088")

    def test_lineage_block_structured_fields(self):
        lin = get_company("XOM").get("lineage") or {}
        self.assertIs(lin.get("cik_override"), True)
        self.assertEqual(lin.get("registrant_cik"), "0000034088")
        self.assertEqual(lin.get("registrant_legal_name"), "EXXON MOBIL CORP")
        self.assertEqual(lin.get("successor_cik"), "0002115436")
        self.assertEqual(lin.get("successor_legal_name"), "ExxonMobil Holdings Corporation")
        self.assertEqual(lin.get("successor_effective_date"), "2026-07-01")

    def test_all_xom_filings_under_historical_accession_prefix(self):
        for f in get_company("XOM")["filings"]:
            self.assertTrue(
                f["accession_number"].startswith("0000034088-"),
                f["accession_number"],
            )

    def test_new_xom_chunk_artifacts_carry_historical_cik(self):
        for fid in ("000003408825000010", "000003408824000018"):
            p = REPO / "data" / "chunks" / "XOM" / f"{fid}_chunks.jsonl"
            rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec.get("cik"), "0000034088")
            self.assertEqual(
                str(rec.get("company") or rec.get("company_name")).upper(),
                "EXXON MOBIL CORP",
            )
            self.assertIn("/data/34088/", rec.get("source_url", ""))
            self.assertNotIn("2115436", json.dumps(rec))


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
                any(i["answer_type"] == "comparison_numeric" for i in mine), ticker
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
        forms = _accepted_forms("713,163")
        self.assertIn("713163", forms)
        self.assertIn("7132", forms)  # MD&A "$713.2 billion"

    def test_non_numeric_hint_is_safe(self):
        self.assertEqual(_accepted_forms("n/a"), [""])


if __name__ == "__main__":
    unittest.main()
