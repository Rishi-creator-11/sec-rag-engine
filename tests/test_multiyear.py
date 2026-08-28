"""Phase 5: fiscal-year scope model, filtering, API contract, discovery."""

import json
import unittest
from unittest.mock import patch

from retrieval.filters import RetrievalFilter
from retrieval.scope import Scope, expand_scopes, scope_of_chunk
from retrieval.scoped_search import scope_label
import api.rag as rag


# --------------------------------------------------------------------------- #
class ScopeModelTests(unittest.TestCase):
    def test_label(self):
        self.assertEqual(Scope("nvda").label, "NVDA")
        self.assertEqual(Scope("nvda", 2023).label, "NVDA:2023")

    def test_to_filter(self):
        self.assertEqual(
            Scope("NVDA", 2023).to_filter(),
            RetrievalFilter(tickers=("NVDA",), fiscal_years=(2023,)),
        )
        self.assertEqual(
            Scope("NVDA").to_filter(), RetrievalFilter(tickers=("NVDA",))
        )

    def test_matches_ticker_and_year(self):
        s = Scope("NVDA", 2023)
        self.assertTrue(s.matches({"ticker": "NVDA", "fiscal_year": 2023}))
        self.assertTrue(s.matches({"ticker": "NVDA", "fiscal_year": "2023"}))
        self.assertFalse(s.matches({"ticker": "NVDA", "fiscal_year": 2025}))
        self.assertFalse(s.matches({"ticker": "AAPL", "fiscal_year": 2023}))
        self.assertFalse(s.matches({"ticker": "NVDA"}))  # year required, absent

    def test_matches_ticker_only(self):
        s = Scope("NVDA")
        self.assertTrue(s.matches({"ticker": "NVDA", "fiscal_year": 2023}))
        self.assertTrue(s.matches({"ticker": "NVDA"}))
        self.assertFalse(s.matches({"ticker": "AAPL"}))

    def test_expand_scopes(self):
        self.assertEqual(expand_scopes(["NVDA"], []), [Scope("NVDA")])
        self.assertEqual(expand_scopes(["NVDA"], [2024]), [Scope("NVDA", 2024)])
        self.assertEqual(
            expand_scopes(["NVDA"], [2023, 2025]),
            [Scope("NVDA", 2023), Scope("NVDA", 2025)],
        )
        self.assertEqual(
            expand_scopes(["AAPL", "MSFT"], [2024]),
            [Scope("AAPL", 2024), Scope("MSFT", 2024)],
        )
        self.assertEqual(
            expand_scopes(["AAPL", "MSFT"], [2023, 2024]),
            [Scope("AAPL", 2023), Scope("AAPL", 2024),
             Scope("MSFT", 2023), Scope("MSFT", 2024)],
        )

    def test_comparison_mode_is_scope_count(self):
        self.assertEqual(len(expand_scopes(["NVDA"], [2024])), 1)          # single
        self.assertEqual(len(expand_scopes(["NVDA"], [2023, 2025])), 2)    # comparison
        self.assertEqual(len(expand_scopes(["AAPL", "MSFT"], [])), 2)      # comparison

    def test_scoped_search_label_year_form(self):
        self.assertEqual(scope_label(RetrievalFilter(tickers=("NVDA",))), "NVDA")
        self.assertEqual(
            scope_label(RetrievalFilter(tickers=("NVDA",), fiscal_years=(2023,))),
            "NVDA:2023",
        )


# --------------------------------------------------------------------------- #
class YearFilterTests(unittest.TestCase):
    """RetrievalFilter enforces ticker + fiscal_year; absent year never leaks."""

    CHUNKS = [
        {"ticker": "NVDA", "fiscal_year": 2023, "chunk_id": "a"},
        {"ticker": "NVDA", "fiscal_year": 2024, "chunk_id": "b"},
        {"ticker": "NVDA", "fiscal_year": 2025, "chunk_id": "c"},
        {"ticker": "AAPL", "fiscal_year": 2024, "chunk_id": "d"},
        {"ticker": "NVDA", "chunk_id": "e"},  # no year (un-backfilled)
    ]

    def test_ticker_and_single_year(self):
        f = RetrievalFilter(tickers=("NVDA",), fiscal_years=(2024,))
        self.assertEqual([c["chunk_id"] for c in self.CHUNKS if f.matches(c)], ["b"])

    def test_ticker_and_year_set(self):
        f = RetrievalFilter(tickers=("NVDA",), fiscal_years=(2023, 2025))
        self.assertEqual(
            [c["chunk_id"] for c in self.CHUNKS if f.matches(c)], ["a", "c"]
        )

    def test_no_cross_year_leak_and_missing_year_excluded(self):
        f = RetrievalFilter(tickers=("NVDA",), fiscal_years=(2023,))
        got = [c["chunk_id"] for c in self.CHUNKS if f.matches(c)]
        self.assertEqual(got, ["a"])          # not b/c (other years), not e (no year)
        self.assertNotIn("d", got)            # not AAPL

    def test_pinecone_filter_includes_year(self):
        self.assertEqual(
            RetrievalFilter(tickers=("NVDA",), fiscal_years=(2023, 2025))
            .to_pinecone_filter(),
            {"ticker": {"$in": ["NVDA"]}, "fiscal_year": {"$in": [2023, 2025]}},
        )


# --------------------------------------------------------------------------- #
class CoverageByScopeTests(unittest.TestCase):
    def _rows(self):
        return [
            {"chunk_id": "n23a", "ticker": "NVDA", "fiscal_year": 2023, "rerank_score": 0.9},
            {"chunk_id": "n23b", "ticker": "NVDA", "fiscal_year": 2023, "rerank_score": 0.8},
            {"chunk_id": "n23c", "ticker": "NVDA", "fiscal_year": 2023, "rerank_score": 0.7},
            {"chunk_id": "n25a", "ticker": "NVDA", "fiscal_year": 2025, "rerank_score": 0.6},
        ]

    def test_year_comparison_both_years_present(self):
        scopes = [Scope("NVDA", 2023), Scope("NVDA", 2025)]
        picked = rag.select_evidence_with_coverage(self._rows(), scopes, 5)
        got_years = {r["fiscal_year"] for r in picked}
        self.assertEqual(got_years, {2023, 2025})

    def test_evidence_by_scope_uses_year_labels(self):
        scopes = [Scope("NVDA", 2023), Scope("NVDA", 2025)]
        counts = rag.evidence_by_scope(self._rows(), scopes)
        self.assertEqual(set(counts), {"NVDA:2023", "NVDA:2025"})
        self.assertEqual(counts["NVDA:2023"], 3)
        self.assertEqual(counts["NVDA:2025"], 1)

    def test_company_only_comparison_contract_unchanged(self):
        rows = [
            {"chunk_id": "a1", "ticker": "AAPL", "rerank_score": 0.9},
            {"chunk_id": "m1", "ticker": "MSFT", "rerank_score": 0.8},
        ]
        counts = rag.evidence_by_scope(rows, [Scope("AAPL"), Scope("MSFT")])
        self.assertEqual(counts, {"AAPL": 1, "MSFT": 1})  # no ":year"

    def test_assert_scopes_rejects_other_year(self):
        scopes = [Scope("NVDA", 2023), Scope("NVDA", 2025)]
        rows = [{"chunk_id": "x", "ticker": "NVDA", "fiscal_year": 2024}]
        with self.assertRaises(rag.ScopeViolationError):
            rag.assert_scopes(rows, scopes, where="test")

    def test_assert_scopes_accepts_requested_years(self):
        scopes = [Scope("NVDA", 2023), Scope("NVDA", 2025)]
        rows = [{"chunk_id": "x", "ticker": "NVDA", "fiscal_year": 2023}]
        rag.assert_scopes(rows, scopes, where="test")  # no raise


# --------------------------------------------------------------------------- #
class ApiContractTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from api.main import app
        self.client = TestClient(app)

    def test_fiscal_years_without_tickers_rejected(self):
        r = self.client.post("/ask", json={"question": "hello world",
                                           "fiscal_years": [2024]})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["error"], "fiscal_years_without_tickers")

    def test_unavailable_year_rejected_not_widened(self):
        with patch("api.main.available_fiscal_years", return_value=[2026, 2025]):
            with patch("api.main.partition_tickers", return_value=(["NVDA"], [])):
                r = self.client.post("/ask", json={
                    "question": "hello world", "tickers": ["NVDA"],
                    "fiscal_years": [1999]})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["error"], "fiscal_year_not_available")

    def test_available_year_passes_through(self):
        captured = {}

        def fake_answer(question, top_k, tickers, fiscal_years):
            captured["tickers"] = tickers
            captured["fiscal_years"] = fiscal_years
            return {"answer": "x", "sources": [], "search_scope": {}, "timings": {},
                    "generation_model": "m", "reranker_fallback": False,
                    "reranker_fallback_reason": None, "question": question}

        with patch("api.main.available_fiscal_years", return_value=[2026, 2025, 2024]):
            with patch("api.main.partition_tickers", return_value=(["NVDA"], [])):
                with patch("api.main.answer_question", fake_answer):
                    r = self.client.post("/ask", json={
                        "question": "hello world", "tickers": ["NVDA"],
                        "fiscal_years": [2025, 2024]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(captured["fiscal_years"], [2025, 2024])

    def test_legacy_ticker_only_request_still_works(self):
        with patch("api.main.partition_tickers", return_value=(["NVDA"], [])):
            with patch("api.main.answer_question",
                       return_value={"answer": "x", "sources": [],
                                     "search_scope": {}, "timings": {},
                                     "generation_model": "m",
                                     "reranker_fallback": False,
                                     "reranker_fallback_reason": None,
                                     "question": "q"}) as fake:
                r = self.client.post("/ask", json={"question": "hello world",
                                                   "tickers": ["NVDA"]})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(fake.call_args.kwargs["fiscal_years"])


# --------------------------------------------------------------------------- #
class RegistryMultiYearTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from ingestion import registry
        self.registry = registry
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "companies.json"
        self.path.write_text(json.dumps({"companies": [
            {"ticker": "NVDA", "legal_name": "NVIDIA CORP", "cik": "0001045810"}]}))
        self.addCleanup(self._tmp.cleanup)
        p = patch.object(registry, "REGISTRY_PATH", self.path)
        p.start(); registry.reload()
        self.addCleanup(p.stop); self.addCleanup(registry.reload)

    def _add(self, fy, acc):
        self.registry.record_filing("NVDA", {
            "filing_type": "10-K", "fiscal_year": fy, "accession_number": acc,
            "filing_date": f"{fy}-03-01", "report_date": f"{fy}-01-28",
            "chunk_count": 100})

    def test_multiple_filings_newest_first_no_dupes(self):
        self._add(2024, "acc-24")
        self._add(2026, "acc-26")
        self._add(2025, "acc-25")
        self._add(2026, "acc-26")  # dupe accession
        raw = json.loads(self.path.read_text())["companies"][0]["filings"]
        self.assertEqual([f["fiscal_year"] for f in raw], [2026, 2025, 2024])
        self.assertEqual(len(raw), 3)

    def test_available_fiscal_years(self):
        self._add(2024, "acc-24")
        self._add(2026, "acc-26")
        self.assertEqual(self.registry.available_fiscal_years("NVDA"), [2026, 2024])
        self.assertEqual(self.registry.available_fiscal_years("ZZZZ"), [])


if __name__ == "__main__":
    unittest.main()
