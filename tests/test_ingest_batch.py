"""Phase 4: batch ingestion driver — sequencing, isolation, stop-on-failure."""

import unittest
from unittest.mock import patch

from ingestion import ingest_batch as ib
from ingestion.ingest_company import IngestionError


class BatchDriverTests(unittest.TestCase):
    def test_stops_on_first_hard_failure_by_default(self):
        calls = []

        def fake_ingest(ticker, **kw):
            calls.append(ticker)
            if ticker == "META":
                raise IngestionError("boom META")
            return {"status": "ingested", "chunk_count": 10, "dense_upserted": 10,
                    "sparse_status": "ok"}

        with patch("ingestion.ingest_batch.ingest_company", fake_ingest):
            summary = ib.run_batch(["GOOGL", "META", "JPM"])

        self.assertEqual(calls, ["GOOGL", "META"])          # JPM never attempted
        self.assertEqual(summary["succeeded"], ["GOOGL"])
        self.assertEqual(summary["failed"], ["META"])
        self.assertTrue(summary["stopped_early"])

    def test_continue_on_error_attempts_all(self):
        def fake_ingest(ticker, **kw):
            if ticker == "META":
                raise IngestionError("boom META")
            return {"status": "ingested", "chunk_count": 10, "dense_upserted": 10,
                    "sparse_status": "ok"}

        with patch("ingestion.ingest_batch.ingest_company", fake_ingest):
            summary = ib.run_batch(["GOOGL", "META", "JPM"], continue_on_error=True)

        self.assertEqual(summary["attempted"], ["GOOGL", "META", "JPM"])
        self.assertEqual(summary["succeeded"], ["GOOGL", "JPM"])
        self.assertFalse(summary["stopped_early"])

    def test_one_company_failure_does_not_touch_another(self):
        seen_kwargs = []

        def fake_ingest(ticker, **kw):
            seen_kwargs.append((ticker, kw))
            if ticker == "GOOGL":
                raise IngestionError("boom")
            return {"status": "ingested"}

        with patch("ingestion.ingest_batch.ingest_company", fake_ingest):
            ib.run_batch(["GOOGL", "META"], continue_on_error=True)

        # each call is an independent ingest_company invocation
        self.assertEqual([t for t, _ in seen_kwargs], ["GOOGL", "META"])

    def test_verify_failure_stops_batch(self):
        def fake_ingest(ticker, **kw):
            return {"status": "ingested", "chunk_count": 5, "dense_upserted": 5,
                    "sparse_status": "ok"}

        def fake_verify(ticker, **kw):
            return {"result": "FAIL" if ticker == "GOOGL" else "PASS",
                    "checks": [{"name": "DENSE", "ok": ticker != "GOOGL"}]}

        with patch("ingestion.ingest_batch.ingest_company", fake_ingest), \
             patch("ingestion.ingest_batch.verify", fake_verify):
            summary = ib.run_batch(["GOOGL", "META"], do_verify=True)

        self.assertEqual(summary["failed"], ["GOOGL"])
        self.assertTrue(summary["stopped_early"])

    def test_dry_run_passes_flag_through(self):
        captured = []

        def fake_ingest(ticker, **kw):
            captured.append(kw.get("dry_run"))
            return {"status": "dry_run", "filing": {"ticker": ticker}, "collision": False}

        with patch("ingestion.ingest_batch.ingest_company", fake_ingest):
            ib.run_batch(["GOOGL", "META"], dry_run=True)

        self.assertEqual(captured, [True, True])

    def test_manifest_batch_selection(self):
        import json
        import tempfile
        from pathlib import Path

        manifest = {
            "companies": [
                {"ticker": "AAPL", "batch": 0, "status": "existing"},
                {"ticker": "GOOGL", "batch": 1, "status": "pending"},
                {"ticker": "META", "batch": 1, "status": "pending"},
                {"ticker": "JPM", "batch": 2, "status": "pending"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(ib._tickers_from_manifest(str(path), 1), ["GOOGL", "META"])
            self.assertEqual(ib._tickers_from_manifest(str(path), 2), ["JPM"])
            self.assertEqual(
                ib._tickers_from_manifest(str(path), None), ["GOOGL", "META", "JPM"]
            )


if __name__ == "__main__":
    unittest.main()
