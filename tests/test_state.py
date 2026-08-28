"""Phase 3: ingestion ledger — atomic writes, resume, complete-skip, force."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ingestion.state import Ledger


def _filing(filing_id="000101872426000004"):
    return SimpleNamespace(
        ticker="AMZN",
        company_name="AMAZON COM INC",
        cik="0001018724",
        filing_type="10-K",
        fiscal_year=2025,
        accession_number="0001018724-26-000004",
        filing_id=filing_id,
        filing_date="2026-02-06",
        report_date="2025-12-31",
        primary_document="amzn-20251231.htm",
        source_url="https://www.sec.gov/Archives/edgar/data/1018724/"
        "000101872426000004/amzn-20251231.htm",
    )


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ingestion_state.json"
        self.addCleanup(self._tmp.cleanup)

    def _ledger(self):
        return Ledger(self.path)

    def test_start_filing_is_idempotent(self):
        led = self._ledger()
        led.start_filing(_filing())
        led2 = self._ledger()
        led2.start_filing(_filing())
        data = json.loads(self.path.read_text())
        self.assertEqual(len(data), 1)
        self.assertEqual(data["000101872426000004"]["stage"], "discovered")

    def test_atomic_write_leaves_no_tmp(self):
        led = self._ledger()
        led.start_filing(_filing())
        led.record_stage("000101872426000004", "downloaded", bytes=123)
        leftovers = list(self.path.parent.glob(".ingestion_state.*.tmp"))
        self.assertEqual(leftovers, [])
        self.assertTrue(self.path.exists())

    def test_next_incomplete_stage_walks_in_order(self):
        led = self._ledger()
        led.start_filing(_filing())
        self.assertEqual(led.next_incomplete_stage("000101872426000004"), "downloaded")
        led.record_stage("000101872426000004", "downloaded")
        led.record_stage("000101872426000004", "cleaned")
        self.assertEqual(led.next_incomplete_stage("000101872426000004"), "chunked")

    def test_skip_sparse_advances_past_sparse(self):
        led = self._ledger()
        led.start_filing(_filing())
        for stage in ("downloaded", "cleaned", "chunked", "embedded", "dense_upserted"):
            led.record_stage("000101872426000004", stage)
        self.assertEqual(
            led.next_incomplete_stage("000101872426000004", skip_sparse=True),
            "bm25_registered",
        )

    def test_failed_stage_recorded_and_not_advanced(self):
        led = self._ledger()
        led.start_filing(_filing())
        led.record_stage("000101872426000004", "downloaded")
        led.record_stage(
            "000101872426000004", "cleaned", status="failed",
            error="too small", error_class="IngestionError",
        )
        self.assertEqual(led.last_successful_stage("000101872426000004"), "downloaded")
        self.assertEqual(led.next_incomplete_stage("000101872426000004"), "cleaned")
        entry = self._ledger().get("000101872426000004")
        self.assertEqual(entry["last_error"]["stage"], "cleaned")

    def test_mark_complete_requires_all_hard_stages(self):
        led = self._ledger()
        led.start_filing(_filing())
        led.record_stage("000101872426000004", "downloaded")
        with self.assertRaises(RuntimeError):
            led.mark_complete("000101872426000004")

    def test_mark_complete_allows_soft_sparse_failed(self):
        led = self._ledger()
        led.start_filing(_filing())
        for stage in ("downloaded", "cleaned", "chunked", "embedded", "dense_upserted"):
            led.record_stage("000101872426000004", stage)
        led.record_stage("000101872426000004", "sparse_upserted", status="failed",
                         error="canary down")
        led.record_stage("000101872426000004", "bm25_registered")
        led.record_stage("000101872426000004", "registry_updated")
        led.mark_complete("000101872426000004")
        self.assertTrue(self._ledger().is_complete("000101872426000004"))

    def test_persistence_across_instances(self):
        led = self._ledger()
        led.start_filing(_filing())
        led.record_stage("000101872426000004", "downloaded", bytes=999)
        fresh = self._ledger()
        self.assertEqual(fresh.get("000101872426000004")["bytes"], 999)


if __name__ == "__main__":
    unittest.main()
