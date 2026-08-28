"""Phase 3/3.5: company registry writer — legal_name/display_name, atomic, no dups."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion import registry

SEED = {
    "schema_version": 2,
    "companies": [
        {"ticker": "AAPL", "legal_name": "Apple Inc.",
         "display_name": "Apple Inc.", "cik": "0000320193"},
        {"ticker": "MSFT", "legal_name": "MICROSOFT CORP",
         "display_name": "Microsoft Corporation", "cik": "0000789019"},
    ],
}

AMZN_FILING = {
    "filing_type": "10-K",
    "fiscal_year": 2025,
    "accession_number": "0001018724-26-000004",
    "filing_date": "2026-02-06",
    "report_date": "2025-12-31",
    "chunk_count": 108,
}


class RegistryWriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "companies.json"
        self.path.write_text(json.dumps(SEED, indent=2), encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)
        self._patch = patch.object(registry, "REGISTRY_PATH", self.path)
        self._patch.start()
        registry.reload()
        self.addCleanup(self._patch.stop)
        self.addCleanup(registry.reload)

    def _raw(self):
        return json.loads(self.path.read_text())

    def _amzn(self):
        return next(c for c in self._raw()["companies"] if c["ticker"] == "AMZN")

    def test_add_company_stores_sec_legal_name(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        self.assertTrue(registry.is_known_ticker("amzn"))
        self.assertEqual([c["ticker"] for c in self._raw()["companies"]],
                         ["AAPL", "AMZN", "MSFT"])  # sorted
        entry = registry.get_company("AMZN")
        self.assertEqual(entry["legal_name"], "AMAZON COM INC")
        self.assertIsNone(entry["display_name"])
        self.assertEqual(entry["name"], "AMAZON COM INC")  # falls back to legal_name

    def test_curated_display_name_wins_and_survives_reingest(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724",
                                display_name="Amazon.com, Inc.")
        self.assertEqual(registry.get_company("AMZN")["name"], "Amazon.com, Inc.")
        # a later ingestion pass supplies only the SEC legal name
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        entry = registry.get_company("AMZN")
        self.assertEqual(entry["legal_name"], "AMAZON COM INC")
        self.assertEqual(entry["display_name"], "Amazon.com, Inc.")  # not clobbered
        self.assertEqual(entry["name"], "Amazon.com, Inc.")

    def test_upsert_company_idempotent(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        before = self.path.read_text()
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        self.assertEqual(self.path.read_text(), before)

    def test_add_filing_and_no_duplicate_on_rerun(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        registry.record_filing("AMZN", AMZN_FILING)
        registry.record_filing("AMZN", AMZN_FILING)
        self.assertEqual(len(self._amzn()["filings"]), 1)
        self.assertEqual(self._amzn()["filings"][0]["fiscal_year"], 2025)

    def test_record_filing_updates_changed_chunk_count(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        registry.record_filing("AMZN", AMZN_FILING)
        registry.record_filing("AMZN", {**AMZN_FILING, "chunk_count": 120})
        self.assertEqual(len(self._amzn()["filings"]), 1)
        self.assertEqual(self._amzn()["filings"][0]["chunk_count"], 120)

    def test_record_filing_missing_fields_raises(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        with self.assertRaises(ValueError):
            registry.record_filing("AMZN", {"filing_type": "10-K"})

    def test_record_filing_unregistered_company_raises(self):
        with self.assertRaises(ValueError):
            registry.record_filing("ZZZZ", AMZN_FILING)

    def test_existing_company_filings_preserved_on_display_update(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        registry.record_filing("AMZN", AMZN_FILING)
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724",
                                display_name="Amazon.com, Inc.")
        self.assertEqual(len(self._amzn()["filings"]), 1)
        self.assertEqual(self._amzn()["display_name"], "Amazon.com, Inc.")

    def test_atomic_write_no_tmp_left(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        self.assertEqual(list(self.path.parent.glob(".companies*tmp*")), [])

    def test_list_companies_has_naming_and_identity_fields(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC", cik="0001018724")
        entry = registry.get_company("AMZN")
        self.assertEqual(set(entry) >= {"ticker", "cik", "legal_name", "display_name",
                                        "name", "filings"}, True)

    def test_old_schema_name_field_still_loads(self):
        self.path.write_text(json.dumps({
            "companies": [{"ticker": "AAPL", "name": "Apple Inc.", "cik": "0000320193"}]
        }), encoding="utf-8")
        registry.reload()
        entry = registry.get_company("AAPL")
        self.assertEqual(entry["legal_name"], "Apple Inc.")
        self.assertEqual(entry["name"], "Apple Inc.")


if __name__ == "__main__":
    unittest.main()
