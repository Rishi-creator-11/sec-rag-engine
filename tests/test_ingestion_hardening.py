"""Phase 3.5: ledger/artifact consistency, stage invalidation, verification,
atomic writes, display naming."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ingestion import ingest_company as ic
from ingestion import stages as sg
from ingestion.atomicio import atomic_write_text
from ingestion.state import Ledger
from tests.test_ingest_company import BIG_HTML, FILING, FakeClient, STUBS


class ResumeArtifactBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ledger_path = self.root / "ingestion_state.json"

        for target, value in [
            ("RAW_DIR", self.root / "raw"),
            ("CHUNKS_DIR", self.root / "chunks"),
            ("EMBEDDINGS_DIR", self.root / "embeddings"),
        ]:
            p = patch.object(ic, target, value)
            p.start()
            self.addCleanup(p.stop)

        p = patch("retrieval.build_embeddings.embed_batch",
                  side_effect=lambda texts: [[0.01] * 1536 for _ in texts])
        p.start()
        self.addCleanup(p.stop)

        self.executed: list[str] = []
        merged = dict(ic._STAGE_FUNCS)
        merged.update(STUBS)
        for name, fn in list(merged.items()):
            merged[name] = self._recorder(name, fn)
        p = patch.object(ic, "_STAGE_FUNCS", merged)
        p.start()
        self.addCleanup(p.stop)

    def _recorder(self, name, fn):
        def wrapped(ctx):
            self.executed.append(name)
            return fn(ctx)
        return wrapped

    def run_ingest(self, **kw):
        self.executed.clear()
        return ic.ingest_company("AMZN", client=FakeClient(),
                                 ledger=Ledger(self.ledger_path), **kw)

    def artifact(self, key):
        return ic._artifact_paths(FILING)[key]


class MissingArtifactResumeTests(ResumeArtifactBase):
    def test_complete_filing_missing_embeddings_is_noop(self):
        # embeddings.jsonl is a REBUILD artifact: dense vectors are already in
        # Pinecone, so a plain rerun of a complete+servable filing does not
        # waste an embed call to recreate a gitignored file. Use --force.
        self.run_ingest()
        self.assertTrue(Ledger(self.ledger_path).is_complete(FILING.filing_id))
        self.artifact("embeddings").unlink()

        result = self.run_ingest()
        self.assertEqual(result["status"], "already_ingested")
        self.assertEqual(self.executed, [])
        self.assertIn("embeddings", result["missing_rebuild_artifacts"])

    def test_incomplete_filing_missing_embeddings_re_embeds_only(self):
        # fail the run at dense_upserted, then delete embeddings.jsonl
        real_dense = ic._STAGE_FUNCS["dense_upserted"]
        state = {"failed": False}

        def flaky_dense(ctx):
            if not state["failed"]:
                state["failed"] = True
                raise ic.IngestionError("boom dense")
            return real_dense(ctx)

        merged = dict(ic._STAGE_FUNCS)
        merged["dense_upserted"] = self._recorder("dense_upserted", flaky_dense)
        with patch.object(ic, "_STAGE_FUNCS", merged):
            with self.assertRaises(ic.IngestionError):
                ic.ingest_company("AMZN", client=FakeClient(),
                                  ledger=Ledger(self.ledger_path))
            self.artifact("embeddings").unlink()  # gone before resume
            self.executed.clear()
            result = ic.ingest_company("AMZN", client=FakeClient(),
                                       ledger=Ledger(self.ledger_path))

        self.assertEqual(result["status"], "ingested")
        self.assertIn("embedded", self.executed)      # re-embedded
        self.assertIn("dense_upserted", self.executed)
        self.assertNotIn("downloaded", self.executed)
        self.assertNotIn("chunked", self.executed)
        self.assertTrue(self.artifact("embeddings").exists())

    def test_missing_chunks_re_chunks_not_re_downloads(self):
        self.run_ingest()
        self.artifact("chunks").unlink()

        result = self.run_ingest()
        self.assertEqual(result["status"], "ingested")
        self.assertIn("chunked", self.executed)
        self.assertIn("embedded", self.executed)
        self.assertNotIn("downloaded", self.executed)
        self.assertNotIn("cleaned", self.executed)

    def test_missing_clean_text_re_cleans_not_re_downloads(self):
        self.run_ingest()
        self.artifact("clean_text").unlink()
        self.artifact("chunks").unlink()  # force chunked to need clean text

        result = self.run_ingest()
        self.assertEqual(result["status"], "ingested")
        self.assertIn("cleaned", self.executed)
        self.assertIn("chunked", self.executed)
        self.assertNotIn("downloaded", self.executed)

    def test_missing_clean_and_html_re_downloads(self):
        self.run_ingest()
        self.artifact("clean_text").unlink()
        self.artifact("chunks").unlink()
        self.artifact("raw_html").unlink()

        result = self.run_ingest()
        self.assertEqual(result["status"], "ingested")
        self.assertIn("downloaded", self.executed)
        self.assertIn("cleaned", self.executed)
        self.assertIn("chunked", self.executed)

    def test_complete_filing_missing_only_rebuild_artifacts_stays_done(self):
        self.run_ingest()
        self.artifact("raw_html").unlink()
        self.artifact("clean_text").unlink()
        self.artifact("embeddings").unlink()
        # chunks (serving artifact) still present

        result = self.run_ingest()
        self.assertEqual(result["status"], "already_ingested")
        self.assertEqual(self.executed, [])  # nothing re-ran
        self.assertIn("raw_html", result["missing_rebuild_artifacts"])
        self.assertIn("embeddings", result["missing_rebuild_artifacts"])

    def test_deterministic_ids_after_artifact_repair(self):
        self.run_ingest()
        first = self.artifact("chunks").read_text()
        self.artifact("chunks").unlink()
        self.artifact("embeddings").unlink()
        self.run_ingest()
        self.assertEqual(self.artifact("chunks").read_text(), first)


class CorruptArtifactTests(ResumeArtifactBase):
    def test_corrupt_chunk_jsonl_invalidates_and_rebuilds(self):
        self.run_ingest()
        self.artifact("chunks").write_text("{ not json\n", encoding="utf-8")

        result = self.run_ingest()
        self.assertEqual(result["status"], "ingested")
        self.assertIn("chunked", self.executed)

    def test_chunk_hash_mismatch_invalidates(self):
        self.run_ingest()
        good = self.artifact("chunks").read_text().splitlines()
        # tamper: keep it valid JSONL but change content so sha256 differs
        rows = [json.loads(x) for x in good]
        rows[0]["text"] = rows[0]["text"] + " tampered"
        self.artifact("chunks").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        result = self.run_ingest()
        self.assertIn("chunked", self.executed)  # hash mismatch forced re-chunk

    def test_wrong_embedding_dimension_rejected_by_validator(self):
        ok, reason = sg.validate_embeddings_artifact.__wrapped__ if hasattr(
            sg.validate_embeddings_artifact, "__wrapped__") else (None, None)
        tmp = Path(self._tmp.name) / "bad_emb.jsonl"
        tmp.write_text(json.dumps({"chunk_id": "x", "embedding": [0.1, 0.2]}) + "\n")
        valid, reason = sg.validate_embeddings_artifact(tmp)
        self.assertFalse(valid)
        self.assertIn("dim", reason)

    def test_wrong_embedding_count_rejected(self):
        tmp = Path(self._tmp.name) / "few_emb.jsonl"
        tmp.write_text(json.dumps({"chunk_id": "x", "embedding": [0.0] * 1536}) + "\n")
        valid, reason = sg.validate_embeddings_artifact(tmp, expected_count=5)
        self.assertFalse(valid)
        self.assertIn("count", reason)


class StageGraphTests(unittest.TestCase):
    def _entry(self, **stage_status):
        stages = {"discovered": {"status": "ok"}}
        for name, status in stage_status.items():
            stages[name] = {"status": status}
        return {"stages": stages, "chunk_count": 3, "hashes": {}}

    def _paths(self, tmp, present):
        names = {"raw_html": "filing.html", "clean_text": "filing.txt",
                 "metadata": "metadata.json", "chunks": "c.jsonl", "embeddings": "e.jsonl"}
        out = {}
        for key, fname in names.items():
            p = Path(tmp) / fname
            out[key] = p
            if key in present:
                if key == "raw_html":
                    p.write_text("x" * 60000)
                elif key == "clean_text":
                    p.write_text("y" * 50000)
                elif key == "metadata":
                    p.write_text(json.dumps({"accession_number": "a"}))
                elif key == "chunks":
                    p.write_text("".join(
                        json.dumps({"chunk_id": f"T_{i}", "chunk_index": i, "text": "z"})
                        + "\n" for i in range(3)))
                elif key == "embeddings":
                    p.write_text("".join(
                        json.dumps({"chunk_id": f"T_{i}", "embedding": [0.0] * 1536})
                        + "\n" for i in range(3)))
        return out

    def test_downstream_invalidation_cascade(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(downloaded="ok", cleaned="ok", chunked="ok",
                                embedded="ok", dense_upserted="ok",
                                sparse_upserted="ok", bm25_registered="ok",
                                registry_updated="ok", complete="ok")
            # chunks missing -> chunked invalid -> embedded/dense/... invalid
            paths = self._paths(tmp, present={"raw_html", "clean_text", "metadata",
                                              "embeddings"})
            a = sg.assess_filing(entry, paths)
            self.assertFalse(a.servable)
            self.assertIn("chunked", a.invalid_stages)
            self.assertIn("embedded", a.invalid_stages)
            self.assertIn("dense_upserted", a.invalid_stages)
            self.assertEqual(a.resume_stage, "chunked")

    def test_sparse_soft_failure_not_a_resume_blocker_when_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(downloaded="ok", cleaned="ok", chunked="ok",
                                embedded="ok", dense_upserted="ok",
                                sparse_upserted="failed", bm25_registered="ok",
                                registry_updated="ok", complete="ok")
            paths = self._paths(tmp, present={"raw_html", "clean_text", "metadata",
                                              "chunks", "embeddings"})
            a = sg.assess_filing(entry, paths, skip_sparse=True)
            self.assertIsNone(a.resume_stage)
            self.assertTrue(a.servable)

    def test_embeddings_missing_backs_up_to_chunked_if_chunks_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._entry(downloaded="ok", cleaned="ok", chunked="ok",
                                embedded="ok")
            paths = self._paths(tmp, present={"raw_html", "clean_text", "metadata"})
            a = sg.assess_filing(entry, paths)
            self.assertEqual(a.resume_stage, "chunked")


class VerificationTests(unittest.TestCase):
    """ingestion.verify_company with Pinecone + BM25 mocked; no OpenAI/Cohere."""

    def setUp(self):
        from ingestion import verify_company as vc
        self.vc = vc
        self.filing = FILING
        p = patch.object(vc, "DENSE_VERIFY_WAITS", (0.0,))  # no real sleeps in tests
        p.start(); self.addCleanup(p.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        # ledger
        self.ledger_path = root / "ingestion_state.json"
        self.ledger_path.write_text(json.dumps({FILING.filing_id: {
            "ticker": "AMZN", "fiscal_year": 2025, "filing_type": "10-K",
            "accession_number": FILING.accession_number,
            "chunk_count": 3, "hashes": {},
            "stages": {"complete": {"status": "ok"},
                       "sparse_upserted": {"status": "ok"}},
        }}))
        p = patch.object(vc, "LEDGER_PATH", self.ledger_path)
        p.start(); self.addCleanup(p.stop)

        # chunk file
        self.chunks_dir = root / "chunks"
        (self.chunks_dir / "AMZN").mkdir(parents=True)
        cf = self.chunks_dir / "AMZN" / f"{FILING.filing_id}_chunks.jsonl"
        cf.write_text("".join(
            json.dumps({"chunk_id": f"AMZN_2025_10-K_{FILING.filing_id}_{i}",
                        "chunk_index": i, "text": "body", "ticker": "AMZN",
                        "filing_id": FILING.filing_id}) + "\n" for i in range(3)))
        p = patch.object(vc, "CHUNKS_DIR", self.chunks_dir)
        p.start(); self.addCleanup(p.stop)

        # registry
        from ingestion import registry
        self.reg_path = root / "companies.json"
        self.reg_path.write_text(json.dumps({"companies": [{
            "ticker": "AMZN", "legal_name": "AMAZON COM INC", "cik": "0001018724",
            "filings": [{"filing_type": "10-K", "fiscal_year": 2025,
                         "accession_number": FILING.accession_number,
                         "filing_date": "2026-02-06", "report_date": "2025-12-31",
                         "chunk_count": 3}]}]}))
        p = patch.object(registry, "REGISTRY_PATH", self.reg_path)
        p.start(); self.addCleanup(p.stop)
        registry.reload(); self.addCleanup(registry.reload)

        # SEC discovery
        p = patch.object(vc.SecClient, "discover_latest_10k",
                         lambda self, t, **kw: FILING)
        p.start(); self.addCleanup(p.stop)

        # BM25 loader sees the 3 chunks
        p = patch("retrieval.bm25_search.load_chunks", return_value=[
            {"chunk_id": f"AMZN_2025_10-K_{FILING.filing_id}_{i}",
             "filing_id": FILING.filing_id} for i in range(3)])
        p.start(); self.addCleanup(p.stop)

    def _mock_pinecone(self, present_ids, meta_ticker="AMZN", meta_fy=2025):
        class Idx:
            def fetch(self, ids, namespace=None):
                vectors = {
                    i: SimpleNamespace(metadata={
                        "ticker": meta_ticker, "filing_id": FILING.filing_id,
                        "fiscal_year": meta_fy})
                    for i in ids if i in present_ids
                }
                return SimpleNamespace(vectors=vectors)

        class PC:
            def __init__(self, api_key=None): pass
            def Index(self, name): return Idx()

        return patch("pinecone.Pinecone", PC)

    def _run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            report = self.vc.verify("AMZN", client=self.vc.SecClient(
                user_agent="x test@example.com"))
        return report

    def test_healthy_filing_passes(self):
        all_ids = [f"AMZN_2025_10-K_{FILING.filing_id}_{i}" for i in range(3)]
        with self._mock_pinecone(set(all_ids)):
            report = self._run()
        self.assertEqual(report["result"], "PASS", report)

    def test_missing_dense_vector_fails(self):
        with self._mock_pinecone(set()):  # nothing present
            report = self._run()
        self.assertEqual(report["result"], "FAIL")
        self.assertFalse(next(c for c in report["checks"] if c["name"] == "DENSE")["ok"])

    def test_wrong_ticker_metadata_fails(self):
        all_ids = [f"AMZN_2025_10-K_{FILING.filing_id}_{i}" for i in range(3)]
        with self._mock_pinecone(set(all_ids), meta_ticker="MSFT"):
            report = self._run()
        self.assertEqual(report["result"], "FAIL")

    def test_bad_registry_count_fails(self):
        data = json.loads(self.reg_path.read_text())
        data["companies"][0]["filings"][0]["chunk_count"] = 999
        self.reg_path.write_text(json.dumps(data))
        from ingestion import registry
        registry.reload()
        all_ids = [f"AMZN_2025_10-K_{FILING.filing_id}_{i}" for i in range(3)]
        with self._mock_pinecone(set(all_ids)):
            report = self._run()
        self.assertEqual(report["result"], "FAIL")

    def test_no_openai_or_cohere_import_used(self):
        all_ids = [f"AMZN_2025_10-K_{FILING.filing_id}_{i}" for i in range(3)]
        with patch("openai.OpenAI", side_effect=AssertionError("no OpenAI in verify")), \
             self._mock_pinecone(set(all_ids)):
            report = self._run()
        self.assertEqual(report["result"], "PASS")


class AtomicWriteTests(unittest.TestCase):
    def test_interrupted_write_keeps_good_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.jsonl"
            atomic_write_text(path, "GOOD CONTENT\n")

            real_replace = __import__("os").replace

            def boom(src, dst):
                raise KeyboardInterrupt("interrupted mid-replace")

            with patch("ingestion.atomicio.os.replace", boom):
                with self.assertRaises(KeyboardInterrupt):
                    atomic_write_text(path, "PARTIAL BAD")

            self.assertEqual(path.read_text(), "GOOD CONTENT\n")
            self.assertEqual(list(Path(tmp).glob(".artifact*tmp*")), [])  # cleaned up


class DisplayNameEndpointTests(unittest.TestCase):
    def test_companies_endpoint_returns_display_names(self):
        from fastapi.testclient import TestClient
        from api.main import app

        companies = TestClient(app).get("/companies").json()["companies"]
        by_ticker = {c["ticker"]: c["name"] for c in companies}
        self.assertEqual(by_ticker.get("AMZN"), "Amazon.com, Inc.")   # curated
        self.assertEqual(by_ticker.get("MSFT"), "Microsoft Corporation")
        for entry in companies:
            self.assertEqual(set(entry), {"ticker", "name"})  # no legal_name leaked

    def test_health_has_additive_diagnostics(self):
        from fastapi.testclient import TestClient
        from api.main import app

        body = TestClient(app).get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("companies", body)
        self.assertIn("bm25_documents", body)


if __name__ == "__main__":
    unittest.main()


class SparseRateLimitTests(unittest.TestCase):
    def test_upsert_filing_sparse_retries_on_429(self):
        from retrieval import sparse_store as ss

        calls = {"n": 0}

        class Idx:
            def upsert_records(self, namespace=None, records=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("[429 RESOURCE_EXHAUSTED] max tokens per minute")

        class Client:
            def Index(self, name):
                return Idx()

        with patch.object(ss, "get_client", lambda: Client()), \
             patch.object(ss, "SPARSE_RATE_LIMIT_SLEEP", 0.0), \
             patch.object(ss, "SPARSE_INTER_BATCH_SLEEP", 0.0):
            result = ss.upsert_filing_sparse(
                [{"chunk_id": f"X_{i}", "text": "t", "ticker": "X", "company_name": "X",
                  "company": "X", "filing_type": "10-K", "filing_date": "2025-01-01",
                  "report_date": "2024-12-31", "source_url": "https://www.sec.gov/x",
                  "cik": "0000000001", "fiscal_year": 2024, "accession_number": "a",
                  "filing_id": "b", "chunk_index": i} for i in range(3)],
                batch_size=2,
            )
        self.assertEqual(result["upserted"], 3)
        self.assertEqual(calls["n"], 3)  # batch1 fail+retry, batch2 ok

    def test_upsert_filing_sparse_reraises_non_rate_limit(self):
        from retrieval import sparse_store as ss

        class Idx:
            def upsert_records(self, namespace=None, records=None):
                raise RuntimeError("some other pinecone error")

        class Client:
            def Index(self, name):
                return Idx()

        with patch.object(ss, "get_client", lambda: Client()):
            with self.assertRaises(RuntimeError):
                ss.upsert_filing_sparse(
                    [{"chunk_id": "X_0", "text": "t", "ticker": "X", "company_name": "X",
                      "company": "X", "filing_type": "10-K", "filing_date": "2025-01-01",
                      "report_date": "2024-12-31", "source_url": "https://www.sec.gov/x",
                      "cik": "0000000001", "fiscal_year": 2024, "accession_number": "a",
                      "filing_id": "b", "chunk_index": 0}],
                )
