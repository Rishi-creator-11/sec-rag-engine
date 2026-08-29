"""Deploy fix: Vercel read-only bm25s, content-aware corpus_version,
GET /companies/{ticker}/filings."""

import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval import lexical_backend as lb


REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_INDEX = REPO_ROOT / "data" / "bm25s_index"


def _chunk(cid, ticker="NVDA", fy=2023, filing_id="000104581023000017"):
    return {"chunk_id": cid, "ticker": ticker, "fiscal_year": fy,
            "filing_id": filing_id, "text": "body"}


# --------------------------------------------------------------------------- #
class CorpusVersionTests(unittest.TestCase):
    def test_schema_prefix_and_determinism(self):
        cs = [_chunk("a", fy=2023), _chunk("b", fy=2024)]
        v1 = lb.corpus_version(cs)
        v2 = lb.corpus_version(list(reversed(cs)))
        self.assertTrue(v1.startswith("v2:"))
        self.assertEqual(v1, v2)  # order-independent

    def test_changes_when_fiscal_year_changes(self):
        base = lb.corpus_version([_chunk("a", fy=2023)])
        moved = lb.corpus_version([_chunk("a", fy=2025)])
        self.assertNotEqual(base, moved)

    def test_changes_when_filing_id_changes(self):
        base = lb.corpus_version([_chunk("a", filing_id="111")])
        other = lb.corpus_version([_chunk("a", filing_id="222")])
        self.assertNotEqual(base, other)

    def test_changes_when_ticker_changes(self):
        self.assertNotEqual(
            lb.corpus_version([_chunk("a", ticker="NVDA")]),
            lb.corpus_version([_chunk("a", ticker="AAPL")]),
        )

    def test_accession_number_falls_back_for_filing_id(self):
        a = lb.corpus_version([{"chunk_id": "a", "ticker": "X", "fiscal_year": 1,
                                "accession_number": "0001-23-000017"}])
        b = lb.corpus_version([{"chunk_id": "a", "ticker": "X", "fiscal_year": 1,
                                "filing_id": "000123000017"}])
        self.assertEqual(a, b)  # dashes stripped, same identity

    def test_ignores_transient_fields(self):
        a = lb.corpus_version([_chunk("a")])
        c = dict(_chunk("a"))
        c["filing_date"] = "2099-01-01"
        c["score"] = 0.42
        c["text"] = "completely different text"
        self.assertEqual(a, lb.corpus_version([c]))

    def test_old_chunk_id_only_index_is_detected_stale(self):
        # a v1-style hash (chunk-id only) must not equal the v2 fingerprint
        import hashlib
        cs = [_chunk("a"), _chunk("b", cid="b")] if False else [_chunk("a")]
        old = hashlib.sha256("a".encode()).hexdigest()  # no schema, chunk_id only
        self.assertNotEqual(old, lb.corpus_version(cs))


# --------------------------------------------------------------------------- #
class ReadOnlyRuntimeDetectionTests(unittest.TestCase):
    def test_vercel_env_detected(self):
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=False):
            self.assertTrue(lb.is_read_only_runtime())

    def test_lambda_env_detected(self):
        with patch.dict(os.environ, {"AWS_LAMBDA_FUNCTION_NAME": "f"}, clear=False):
            self.assertTrue(lb.is_read_only_runtime())

    def test_explicit_flag_on_and_off(self):
        with patch.dict(os.environ, {"SEC_RAG_READ_ONLY_FS": "1"}, clear=False):
            self.assertTrue(lb.is_read_only_runtime())
        with patch.dict(os.environ,
                        {"VERCEL": "1", "SEC_RAG_FORCE_WRITABLE_FS": "1"}, clear=False):
            self.assertFalse(lb.is_read_only_runtime())

    def test_default_local_is_writable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(lb.is_read_only_runtime())


# --------------------------------------------------------------------------- #
@unittest.skipUnless(REAL_INDEX.exists(), "prebuilt bm25s index not present")
class ReadOnlyLexicalBackendTests(unittest.TestCase):
    """Uses the real prebuilt index (must be current — run build_bm25s_index)."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.ro = Path(self._tmp) / "bm25s_index"
        shutil.copytree(REAL_INDEX, self.ro)
        # make the copy genuinely read-only, like /var/task
        for root, _dirs, files in os.walk(self.ro):
            for f in files:
                os.chmod(os.path.join(root, f), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            os.chmod(root, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
        self.addCleanup(self._restore_and_rm)

    def _restore_and_rm(self):
        for root, _dirs, files in os.walk(self.ro):
            os.chmod(root, 0o755)
            for f in files:
                os.chmod(os.path.join(root, f), 0o644)
        shutil.rmtree(self._tmp)

    def test_loads_read_only_without_writing(self):
        before = sorted(p.name for p in self.ro.iterdir())
        backend = lb.load_readonly_bm25s(self.ro)
        self.assertGreater(backend.document_count, 1000)
        self.assertEqual(before, sorted(p.name for p in self.ro.iterdir()))  # nothing written
        rows = backend.search("export control china", top_k=3)
        self.assertTrue(rows and all("chunk_id" in r for r in rows))

    def test_get_lexical_backend_vercel_mode_never_rebuilds(self):
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=False):
            with patch.object(lb, "BM25S_INDEX_DIR", self.ro):
                with patch.object(lb, "build_persisted_bm25s",
                                  side_effect=AssertionError("rebuild attempted!")):
                    backend = lb.get_lexical_backend()
                    self.assertEqual(type(backend).__name__, "BM25SBackend")
                    with self.assertRaises(lb.LexicalBackendError):
                        lb.get_lexical_backend(force_rebuild=True)

    def test_missing_bundle_fails_clearly(self):
        with self.assertRaises(lb.LexicalBackendError) as cm:
            lb.load_readonly_bm25s(Path(self._tmp) / "does_not_exist")
        self.assertIn("no bundled bm25s index", str(cm.exception))

    def test_stale_corpus_version_fails_clearly(self):
        # rewrite the version file in a fresh writable copy
        w = Path(self._tmp) / "writable"
        shutil.copytree(REAL_INDEX, w)
        vf = w / "corpus_version.json"
        meta = json.loads(vf.read_text())
        meta["corpus_version"] = "v2:deadbeef"
        vf.write_text(json.dumps(meta))
        with self.assertRaises(lb.LexicalBackendError) as cm:
            lb.load_readonly_bm25s(w)
        self.assertIn("stale", str(cm.exception))

    def test_ranking_identical_readonly_vs_fresh(self):
        ro = lb.load_readonly_bm25s(self.ro)
        fresh = lb.BM25SBackend(lb.load_chunks())
        for q in ("china export controls", "cybersecurity governance",
                  "total revenue fiscal year"):
            a = [(r["chunk_id"], round(r["score"], 5)) for r in ro.search(q, top_k=10)]
            b = [(r["chunk_id"], round(r["score"], 5)) for r in fresh.search(q, top_k=10)]
            self.assertEqual(a, b, q)


# --------------------------------------------------------------------------- #
class LocalModeStillRebuildsTests(unittest.TestCase):
    def test_local_missing_index_rebuilds(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        missing = Path(tmp) / "idx"
        chunks = [_chunk(f"x{i}", filing_id="f") for i in range(20)]
        with patch.dict(os.environ, {}, clear=True):  # no VERCEL
            with patch.object(lb, "BM25S_INDEX_DIR", missing):
                built = lb.load_or_build_bm25s(missing, chunks)
        self.assertEqual(built.document_count, 20)
        self.assertTrue((missing / "corpus_version.json").exists())


# --------------------------------------------------------------------------- #
class CompanyFilingsEndpointTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from api.main import app
        self.client = TestClient(app)

    def test_nvda_returns_four_years_newest_first(self):
        r = self.client.get("/companies/NVDA/filings")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["ticker"], "NVDA")
        years = [f["fiscal_year"] for f in body["filings"]]
        self.assertEqual(years, sorted(years, reverse=True))
        self.assertEqual(set(years), {2026, 2025, 2024, 2023})
        self.assertEqual(body["available_fiscal_years"], [2026, 2025, 2024, 2023])
        for f in body["filings"]:
            self.assertEqual(set(f), {"filing_type", "fiscal_year", "report_date",
                                      "filing_date", "accession_number", "chunk_count"})
            self.assertNotIn("text", f)

    def test_lowercase_ticker_normalized(self):
        self.assertEqual(self.client.get("/companies/nvda/filings").status_code, 200)

    def test_aapl_current_registered_filing(self):
        body = self.client.get("/companies/AAPL/filings").json()
        self.assertIn(2024, [f["fiscal_year"] for f in body["filings"]])  # seed FY2024

    def test_xom_lineage_is_factual(self):
        body = self.client.get("/companies/XOM/filings").json()
        self.assertEqual(body["status_code"] if "status_code" in body else 200, 200)
        lin = body.get("registrant_lineage")
        self.assertIsNotNone(lin)
        self.assertEqual(lin["filing_registrant_cik"], "0000034088")
        self.assertEqual(lin["current_successor_cik"], "0002115436")
        self.assertNotEqual(lin["filing_registrant_cik"], lin["current_successor_cik"])
        # the FY2025 filing is present and attributed to the historical registrant
        self.assertIn(2025, [f["fiscal_year"] for f in body["filings"]])

    def test_company_without_lineage_has_no_block(self):
        body = self.client.get("/companies/NVDA/filings").json()
        self.assertNotIn("registrant_lineage", body)

    def test_unknown_ticker_404(self):
        r = self.client.get("/companies/ZZZZ/filings")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["detail"]["error"], "unknown_ticker")

    def test_companies_endpoint_unchanged(self):
        body = self.client.get("/companies").json()
        self.assertIn("companies", body)
        for c in body["companies"]:
            self.assertEqual(set(c), {"ticker", "name"})  # still lightweight


if __name__ == "__main__":
    unittest.main()
