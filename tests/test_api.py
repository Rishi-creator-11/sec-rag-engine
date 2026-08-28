"""Phase 1C: API + RAG-layer tests for ticker filtering.

External services (OpenAI, Cohere, Pinecone) are never called: retrieval and
generation are patched at the ``api.rag`` seam. These tests exercise request
normalization, registry validation, filter construction, the scope-safety
gate, the additive ``search_scope`` field, and ``GET /companies``.
"""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import MAX_TICKERS, app
from api.rag import ScopeViolationError, assert_scope, build_scope_filter
from retrieval.filters import RetrievalFilter

_COMPANY_NAME = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}
_LEGACY_RESPONSE_FIELDS = {
    "question",
    "answer",
    "sources",
    "generation_model",
    "reranker_fallback",
    "reranker_fallback_reason",
    "timings",
}


def _rows(tickers, n=3):
    rows = []
    for i in range(n):
        ticker = tickers[i % len(tickers)]
        rows.append(
            {
                "chunk_id": f"{ticker.lower()}_10k_{i}",
                "ticker": ticker,
                "company": _COMPANY_NAME[ticker],
                "filing_type": "10-K",
                "filing_date": "2024-09-28",
                "source_url": "https://www.sec.gov/x",
                "text": f"evidence {i} for {ticker}",
                "rerank_score": 0.9 - i * 0.1,
                "rrf_score": 0.5 - i * 0.05,
            }
        )
    return rows


class ApiAskTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.seen = {}

        def fake_retrieve(question, evidence_k, filters=None):
            self.seen["filters"] = filters
            self.seen["question"] = question
            tickers = (
                list(filters.tickers)
                if (filters is not None and filters.tickers)
                else ["AAPL", "MSFT", "NVDA"]
            )
            rows = _rows(tickers, n=3)
            return rows, rows[:evidence_k], False, None, 1.0, 1.0

        def fake_generate(question, evidence, comparison_tickers=None):
            self.seen["comparison_tickers"] = comparison_tickers
            return "canned answer [Source 1]", "CTX", 1.0

        def fake_comparison(question, evidence_k, requested_tickers):
            self.seen["comparison_requested"] = list(requested_tickers)
            self.seen["comparison_evidence_k"] = evidence_k
            rows = _rows(requested_tickers, n=max(evidence_k, len(requested_tickers)))
            per_scope = {t: [r for r in rows if r["ticker"] == t] for t in requested_tickers}
            return {
                "union": rows,
                "per_scope": per_scope,
                "evidence": rows[:evidence_k],
                "reranker_fallback": False,
                "reranker_fallback_reason": None,
                "hybrid_ms": 1.0,
                "rerank_ms": 1.0,
                "tickers_with_candidates": [t for t in requested_tickers if per_scope[t]],
            }

        self._patches = [
            patch("api.rag.retrieve_evidence", fake_retrieve),
            patch("api.rag.retrieve_evidence_comparison", fake_comparison),
            patch("api.rag.generate_answer", fake_generate),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    # A ---------------------------------------------------------------
    def test_legacy_request_no_tickers(self):
        resp = self.client.post("/ask", json={"question": "hello world", "top_k": 5})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(self.seen["filters"])
        self.assertEqual(
            body["search_scope"],
            {
                "global_search": True,
                "comparison_mode": False,
                "tickers": None,
                "evidence_by_scope": {},
                "warnings": [],
            },
        )
        self.assertIsNone(self.seen["comparison_tickers"])
        self.assertTrue(_LEGACY_RESPONSE_FIELDS.issubset(body))

    def test_legacy_request_without_top_k(self):
        resp = self.client.post("/ask", json={"question": "hello world"})
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(self.seen["filters"])

    # B ---------------------------------------------------------------
    def test_lowercase_ticker_normalized(self):
        resp = self.client.post(
            "/ask", json={"question": "hello world", "tickers": ["aapl"]}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.seen["filters"], RetrievalFilter(tickers=("AAPL",)))
        self.assertEqual(resp.json()["search_scope"]["tickers"], ["AAPL"])

    # C ---------------------------------------------------------------
    def test_single_ticker_scopes_sources(self):
        resp = self.client.post(
            "/ask", json={"question": "hello world", "tickers": ["AAPL"]}
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(self.seen["filters"].tickers, ("AAPL",))
        self.assertTrue(body["sources"])
        self.assertTrue(all(s["ticker"] == "AAPL" for s in body["sources"]))
        self.assertEqual(
            body["search_scope"],
            {
                "global_search": False,
                "comparison_mode": False,
                "tickers": ["AAPL"],
                "evidence_by_scope": {"AAPL": len(body["sources"])},
                "warnings": [],
            },
        )
        self.assertIsNone(self.seen["comparison_tickers"])  # single = Phase 1C path

    # D ---------------------------------------------------------------
    def test_duplicate_tickers_collapsed(self):
        resp = self.client.post(
            "/ask",
            json={"question": "hello world", "tickers": ["AAPL", "aapl", " AAPL "]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.seen["filters"].tickers, ("AAPL",))
        self.assertEqual(resp.json()["search_scope"]["tickers"], ["AAPL"])

    # E ---------------------------------------------------------------
    def test_unknown_ticker_returns_422(self):
        resp = self.client.post(
            "/ask", json={"question": "hello world", "tickers": ["XYZ"]}
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(
            resp.json()["detail"],
            {"error": "unknown_tickers", "unknown_tickers": ["XYZ"]},
        )
        self.assertNotIn("filters", self.seen)  # retrieval never ran

    def test_mixed_known_and_unknown_ticker_returns_422(self):
        resp = self.client.post(
            "/ask",
            json={"question": "hello world", "tickers": ["AAPL", "XYZ", "nope"]},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(
            resp.json()["detail"],
            {"error": "unknown_tickers", "unknown_tickers": ["XYZ", "NOPE"]},
        )

    def test_unknown_ticker_does_not_fall_back_to_global(self):
        resp = self.client.post(
            "/ask", json={"question": "hello world", "tickers": ["XYZ"]}
        )
        self.assertEqual(resp.status_code, 422)

    # F ---------------------------------------------------------------
    def test_empty_ticker_list_is_global(self):
        resp = self.client.post(
            "/ask", json={"question": "hello world", "tickers": []}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(self.seen["filters"])
        self.assertTrue(resp.json()["search_scope"]["global_search"])

    def test_null_ticker_is_global(self):
        resp = self.client.post(
            "/ask", json={"question": "hello world", "tickers": None}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(self.seen["filters"])

    # G ---------------------------------------------------------------
    def test_multi_ticker_enters_comparison_path(self):
        # Phase 2: 2+ tickers route through retrieve_evidence_comparison
        # (per-company quota), NOT the single joint-filter path.
        resp = self.client.post(
            "/ask", json={"question": "hello world", "tickers": ["aapl", "MSFT"]}
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertNotIn("filters", self.seen)  # single-scope path not used
        self.assertEqual(self.seen["comparison_requested"], ["AAPL", "MSFT"])
        self.assertEqual(self.seen["comparison_tickers"], ["AAPL", "MSFT"])
        self.assertTrue(all(s["ticker"] in {"AAPL", "MSFT"} for s in body["sources"]))
        scope = body["search_scope"]
        self.assertTrue(scope["comparison_mode"])
        self.assertFalse(scope["global_search"])
        self.assertEqual(scope["tickers"], ["AAPL", "MSFT"])
        # test L: evidence_by_scope matches the actual returned sources
        actual = {"AAPL": 0, "MSFT": 0}
        for source in body["sources"]:
            actual[source["ticker"]] += 1
        self.assertEqual(scope["evidence_by_scope"], actual)
        self.assertEqual(sum(scope["evidence_by_scope"].values()), len(body["sources"]))

    # extra guards ---------------------------------------------------
    def test_too_many_tickers_rejected(self):
        resp = self.client.post(
            "/ask",
            json={
                "question": "hello world",
                "tickers": [f"T{i}" for i in range(MAX_TICKERS + 1)],
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_scope_violation_is_internal_error(self):
        def leaky_retrieve(question, evidence_k, filters=None):
            rows = _rows(["AAPL", "NVDA"], n=3)  # NVDA leaks past an AAPL filter
            return rows, rows[:evidence_k], False, None, 1.0, 1.0

        with patch("api.rag.retrieve_evidence", leaky_retrieve):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/ask", json={"question": "hello world", "tickers": ["AAPL"]}
            )
        self.assertEqual(resp.status_code, 500)


class CompaniesEndpointTests(unittest.TestCase):
    def test_lists_registry_companies_sorted(self):
        client = TestClient(app)
        resp = client.get("/companies")
        self.assertEqual(resp.status_code, 200)
        companies = resp.json()["companies"]
        tickers = [c["ticker"] for c in companies]
        # the three original seed companies are always registered
        self.assertLessEqual({"AAPL", "MSFT", "NVDA"}, set(tickers))
        self.assertEqual(tickers, sorted(tickers))  # deterministic order

    def test_no_ingestion_state_leaked(self):
        client = TestClient(app)
        for entry in client.get("/companies").json()["companies"]:
            self.assertEqual(set(entry), {"ticker", "name"})  # no cik / filings


class ScopeHelperTests(unittest.TestCase):
    def test_build_scope_filter_none_and_empty(self):
        self.assertIsNone(build_scope_filter(None))
        self.assertIsNone(build_scope_filter([]))
        self.assertIsNone(build_scope_filter([""]))

    def test_build_scope_filter_single_and_multi(self):
        self.assertEqual(build_scope_filter(["aapl"]).tickers, ("AAPL",))
        self.assertEqual(
            build_scope_filter(["AAPL", "msft"]).tickers, ("AAPL", "MSFT")
        )

    def test_assert_scope_passes_when_in_scope(self):
        rows = _rows(["AAPL"], n=3)
        assert_scope(rows, RetrievalFilter(tickers=("AAPL",)), where="t")

    def test_assert_scope_noop_without_scope(self):
        rows = _rows(["AAPL", "NVDA"], n=3)
        assert_scope(rows, None, where="t")

    def test_assert_scope_raises_on_leak(self):
        rows = _rows(["AAPL", "NVDA"], n=3)
        with self.assertRaises(ScopeViolationError):
            assert_scope(rows, RetrievalFilter(tickers=("AAPL",)), where="t")


if __name__ == "__main__":
    unittest.main()


class HealthReadinessTests(unittest.TestCase):
    def test_health_reports_lexical_backend(self):
        from fastapi.testclient import TestClient
        from api.main import app

        body = TestClient(app).get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn(body["lexical_backend"], {"bm25s", "current"})
        self.assertIsInstance(body["bm25_documents"], int)

    def test_health_503_when_lexical_index_unavailable(self):
        from fastapi.testclient import TestClient
        from api.main import app
        from retrieval.lexical_backend import LexicalBackendError

        with patch("retrieval.bm25_search.get_index",
                   side_effect=LexicalBackendError("no index, no rebuild")):
            resp = TestClient(app, raise_server_exceptions=False).get("/health")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["status"], "unavailable")
