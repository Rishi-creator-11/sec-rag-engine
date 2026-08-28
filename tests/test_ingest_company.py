"""Phase 3: ingestion orchestration — dry-run, paths, validation, resume, idempotency.

External-touching stages (dense/sparse/bm25/registry) are replaced with ledger
stubs; download/clean/chunk/embed run for real against a temp tree with a
mocked embedding call.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ingestion import ingest_company as ic
from ingestion.sec_client import DiscoveredFiling
from ingestion.state import Ledger

FILING = DiscoveredFiling(
    company_name="AMAZON COM INC",
    ticker="AMZN",
    cik="0001018724",
    filing_type="10-K",
    fiscal_year=2025,
    accession_number="0001018724-26-000004",
    filing_id="000101872426000004",
    filing_date="2026-02-06",
    report_date="2025-12-31",
    primary_document="amzn-20251231.htm",
    source_url="https://www.sec.gov/Archives/edgar/data/1018724/"
    "000101872426000004/amzn-20251231.htm",
)

_PARAGRAPH = (
    "Amazon faces intense competition across retail cloud and advertising and "
    "describes risks related to AWS operations logistics and regulation. "
)
BIG_HTML = (
    "<html><body><p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>"
    "<p>FORM 10-K ANNUAL REPORT</p><p>AMAZON COM INC</p>"
    + "".join(f"<p>{_PARAGRAPH}</p>" for _ in range(900))
    + "</body></html>"
)


class FakeResp:
    def __init__(self, text, headers=None):
        self.text = text
        self.headers = headers or {"Content-Type": "text/html"}
        self.status_code = 200


class FakeClient:
    def __init__(self, filing=FILING, html=BIG_HTML):
        self._filing = filing
        self._html = html
        self.downloads = 0

    def discover_latest_10k(self, ticker, **kwargs):
        return self._filing

    def download_document(self, url):
        self.downloads += 1
        return FakeResp(self._html)


def _stub_dense(ctx):
    n = ctx.ledger.get(ctx.filing.filing_id).get("chunk_count")
    ctx.ledger.record_stage(ctx.filing.filing_id, "dense_upserted",
                            dense_upserted=n, dense_verified=n)


def _stub_sparse(ctx):
    ctx.ledger.record_stage(ctx.filing.filing_id, "sparse_upserted", status="skipped")


def _stub_bm25(ctx):
    ctx.ledger.record_stage(ctx.filing.filing_id, "bm25_registered", bm25_hits=7)


def _stub_registry(ctx):
    ctx.ledger.record_stage(ctx.filing.filing_id, "registry_updated")


STUBS = {
    "dense_upserted": _stub_dense,
    "sparse_upserted": _stub_sparse,
    "bm25_registered": _stub_bm25,
    "registry_updated": _stub_registry,
}


class IngestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ledger_path = root / "ingestion_state.json"

        patches = [
            patch.object(ic, "RAW_DIR", root / "raw"),
            patch.object(ic, "CHUNKS_DIR", root / "chunks"),
            patch.object(ic, "EMBEDDINGS_DIR", root / "embeddings"),
            patch("retrieval.build_embeddings.embed_batch",
                  side_effect=lambda texts: [[0.01] * 1536 for _ in texts]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        merged = dict(ic._STAGE_FUNCS)
        merged.update(STUBS)
        p = patch.object(ic, "_STAGE_FUNCS", merged)
        p.start()
        self.addCleanup(p.stop)

    def run_ingest(self, **kw):
        return ic.ingest_company(
            "AMZN", client=FakeClient(), ledger=Ledger(self.ledger_path), **kw
        )


class DryRunTests(IngestBase):
    def _dry_run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return ic.ingest_company("AMZN", client=FakeClient(), dry_run=True)

    def test_dry_run_writes_nothing(self):
        result = self._dry_run()
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(self.ledger_path.exists())
        self.assertFalse((Path(self._tmp.name) / "raw").exists())
        self.assertFalse(result["collision"])

    def test_dry_run_plan_has_canonical_ids_and_paths(self):
        result = self._dry_run()
        plan = result["plan"]
        self.assertEqual(
            plan["sample_vector_ids"][0],
            "AMZN_2025_10-K_000101872426000004_0",
        )
        self.assertIn("AMZN/000101872426000004", plan["paths"]["chunks"])
        self.assertFalse(plan["seed_ids_touched"])


class HappyPathTests(IngestBase):
    def test_full_pipeline_completes(self):
        result = self.run_ingest()
        self.assertEqual(result["status"], "ingested")
        self.assertGreater(result["chunk_count"], 5)
        self.assertEqual(result["chunk_count"], result["embedding_count"])
        led = Ledger(self.ledger_path)
        self.assertTrue(led.is_complete(FILING.filing_id))

    def test_canonical_paths_and_ids_on_disk(self):
        self.run_ingest()
        root = Path(self._tmp.name)
        chunks_file = root / "chunks" / "AMZN" / "000101872426000004_chunks.jsonl"
        self.assertTrue(chunks_file.exists())
        first = chunks_file.read_text().splitlines()[0]
        self.assertIn('"AMZN_2025_10-K_000101872426000004_0"', first)
        self.assertIn('"company_name": "AMAZON COM INC"', first)
        self.assertIn('"company": "AMAZON COM INC"', first)  # back-compat alias

    def test_deterministic_ids_across_runs(self):
        r1 = self.run_ingest()
        root = Path(self._tmp.name)
        cf = root / "chunks" / "AMZN" / "000101872426000004_chunks.jsonl"
        ids1 = [line for line in cf.read_text().splitlines()]
        r2 = self.run_ingest(force=True)
        ids2 = [line for line in cf.read_text().splitlines()]
        self.assertEqual(ids1, ids2)
        self.assertEqual(r1["chunk_count"], r2["chunk_count"])

    def test_idempotent_rerun_skips(self):
        self.run_ingest()
        with patch("retrieval.build_embeddings.embed_batch") as mock_embed:
            result = self.run_ingest()
        self.assertEqual(result["status"], "already_ingested")
        mock_embed.assert_not_called()  # no re-embedding

    def test_force_rebuilds(self):
        self.run_ingest()
        result = self.run_ingest(force=True)
        self.assertEqual(result["status"], "ingested")


class ValidationTests(IngestBase):
    def test_tiny_html_rejected(self):
        with self.assertRaises(ic.IngestionError):
            ic.ingest_company("AMZN",
                              client=FakeClient(html="<html>too small</html>"),
                              ledger=Ledger(self.ledger_path))

    def test_block_page_rejected(self):
        blocked = "<html><body>Request Rate Threshold Exceeded</body></html>" + "x" * 60000
        with self.assertRaises(ic.IngestionError):
            ic.ingest_company("AMZN", client=FakeClient(html=blocked),
                              ledger=Ledger(self.ledger_path))

    def test_wrong_company_text_rejected(self):
        html = (
            "<html><body><p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>"
            "<p>FORM 10-K ANNUAL REPORT</p>"
            + "".join("<p>Microsoft Azure cloud competition risk factors here.</p>"
                      for _ in range(900))
            + "</body></html>"
        )
        with self.assertRaises(ic.IngestionError):
            ic.ingest_company("AMZN", client=FakeClient(html=html),
                              ledger=Ledger(self.ledger_path))

    def test_metadata_validation_rejects_bad_filing(self):
        bad = SimpleNamespace(**{**FILING.to_dict(), "filing_date": "2024-01-01"})
        with self.assertRaises(ic.IngestionError):
            ic.validate_filing_metadata(bad)


class ResumeTests(IngestBase):
    def _fail_once(self, stage_name):
        real = ic._STAGE_FUNCS[stage_name]
        state = {"failed": False}

        def wrapper(ctx):
            if not state["failed"]:
                state["failed"] = True
                raise ic.IngestionError(f"boom in {stage_name}")
            return real(ctx)

        merged = dict(ic._STAGE_FUNCS)
        merged[stage_name] = wrapper
        return patch.object(ic, "_STAGE_FUNCS", merged), state

    def _resume_scenario(self, stage_name):
        ledger = Ledger(self.ledger_path)
        p, _state = self._fail_once(stage_name)
        p.start()
        try:
            with self.assertRaises(ic.IngestionError):
                ic.ingest_company("AMZN", client=FakeClient(), ledger=ledger)
        finally:
            p.stop()

        led = Ledger(self.ledger_path)
        self.assertEqual(led.next_incomplete_stage(FILING.filing_id), stage_name)
        self.assertFalse(led.is_complete(FILING.filing_id))

        # rerun: converges to complete, earlier stages not repeated
        client = FakeClient()
        result = ic.ingest_company("AMZN", client=client, ledger=Ledger(self.ledger_path))
        self.assertEqual(result["status"], "ingested")
        self.assertTrue(Ledger(self.ledger_path).is_complete(FILING.filing_id))
        return client, result

    def test_resume_after_chunk_failure(self):
        client, result = self._resume_scenario("chunked")
        self.assertEqual(client.downloads, 0)  # download stage already done, not repeated

    def test_resume_after_embed_failure(self):
        with patch("retrieval.build_embeddings.embed_batch",
                   side_effect=lambda texts: [[0.02] * 1536 for _ in texts]):
            client, result = self._resume_scenario("embedded")
        self.assertEqual(result["chunk_count"], result["embedding_count"])

    def test_resume_after_dense_failure(self):
        client, result = self._resume_scenario("dense_upserted")
        self.assertEqual(result["status"], "ingested")

    def test_failed_stage_persists_error_class(self):
        ledger = Ledger(self.ledger_path)
        p, _ = self._fail_once("chunked")
        p.start()
        try:
            with self.assertRaises(ic.IngestionError):
                ic.ingest_company("AMZN", client=FakeClient(), ledger=ledger)
        finally:
            p.stop()
        entry = Ledger(self.ledger_path).get(FILING.filing_id)
        self.assertEqual(entry["last_error"]["stage"], "chunked")
        self.assertEqual(entry["stages"]["chunked"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
