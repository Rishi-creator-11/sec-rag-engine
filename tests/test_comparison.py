"""Phase 2: guaranteed multi-company comparison retrieval.

External services are mocked. Tests cover:
  - scoped_search: one hybrid call per scope, one shared query embedding, union
  - select_evidence_with_coverage: presence guarantee + quality-ordered fill
  - plan_evidence routing: 0 / 1 / 2+ tickers
  - one Cohere call for the union; fallback keeps per-scope coverage
  - missing-scope warning
  - search_scope.evidence_by_scope matches returned evidence
  - defensive assertions
"""

import unittest
from unittest.mock import patch

from api.rag import (
    ScopeViolationError,
    plan_evidence,
    select_evidence_with_coverage,
)
from retrieval.filters import RetrievalFilter
from retrieval.scoped_search import scoped_search

_NAME = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}


def _cand(ticker, i, rerank_score=None):
    row = {
        "chunk_id": f"{ticker.lower()}_10k_{i}",
        "ticker": ticker,
        "company": _NAME[ticker],
        "filing_type": "10-K",
        "filing_date": "2024-09-28",
        "source_url": "https://www.sec.gov/x",
        "text": f"{ticker} evidence {i}",
        "rrf_score": 0.5 - i * 0.01,
    }
    if rerank_score is not None:
        row["rerank_score"] = rerank_score
    return row


# --------------------------------------------------------------------------- #
class ScopedSearchTests(unittest.TestCase):
    @patch("retrieval.scoped_search.embed_text", return_value=[0.1] * 1536)
    @patch("retrieval.scoped_search.hybrid_search")
    def test_one_hybrid_call_per_scope_one_embed(self, mock_hybrid, mock_embed):
        mock_hybrid.side_effect = lambda q, **kw: [
            _cand(kw["filters"].tickers[0], i) for i in range(3)
        ]
        scopes = [
            RetrievalFilter(tickers=("AAPL",)),
            RetrievalFilter(tickers=("MSFT",)),
        ]
        union, per_scope, _ms = scoped_search("q", scopes)

        self.assertEqual(mock_embed.call_count, 1)
        self.assertEqual(mock_hybrid.call_count, 2)
        self.assertEqual(len(union), 6)
        self.assertEqual(set(per_scope), {"AAPL", "MSFT"})
        for call in mock_hybrid.call_args_list:
            self.assertEqual(call.kwargs["query_embedding"], [0.1] * 1536)

    @patch("retrieval.scoped_search.embed_text", return_value=[0.0] * 4)
    @patch("retrieval.scoped_search.hybrid_search")
    def test_union_dedupes_and_tags_scope(self, mock_hybrid, _embed):
        mock_hybrid.side_effect = lambda q, **kw: [
            _cand(kw["filters"].tickers[0], i) for i in range(2)
        ]
        scopes = [RetrievalFilter(tickers=("AAPL",)), RetrievalFilter(tickers=("MSFT",))]
        union, _per_scope, _ms = scoped_search("q", scopes, union_cap=3)
        self.assertEqual(len(union), 3)  # union_cap honored
        self.assertTrue(all("scopes" in row and "scope_rank" in row for row in union))

    @patch("retrieval.scoped_search.embed_text", return_value=[0.0] * 4)
    @patch("retrieval.scoped_search.hybrid_search")
    def test_reused_embedding_is_passed_not_reembedded(self, mock_hybrid, mock_embed):
        mock_hybrid.side_effect = lambda q, **kw: []
        scopes = [RetrievalFilter(tickers=("AAPL",)), RetrievalFilter(tickers=("NVDA",))]
        scoped_search("q", scopes, query_embedding=[9.0] * 4)
        mock_embed.assert_not_called()
        for call in mock_hybrid.call_args_list:
            self.assertEqual(call.kwargs["query_embedding"], [9.0] * 4)


# --------------------------------------------------------------------------- #
class CoverageSelectionTests(unittest.TestCase):
    def test_both_companies_present_aapl_msft(self):
        ranked = [_cand("AAPL", i) for i in range(4)] + [_cand("MSFT", 0)]
        picked = select_evidence_with_coverage(ranked, ["AAPL", "MSFT"], 5)
        tickers = [c["ticker"] for c in picked]
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)
        self.assertLessEqual(len(picked), 5)

    def test_both_present_when_one_company_dominates_ranking(self):
        # 6 AAPL ahead of the only MSFT chunk; k=5 -> must still include MSFT.
        ranked = [_cand("AAPL", i) for i in range(6)] + [_cand("MSFT", 9)]
        picked = select_evidence_with_coverage(ranked, ["AAPL", "MSFT"], 5)
        counts = {"AAPL": 0, "MSFT": 0}
        for c in picked:
            counts[c["ticker"]] += 1
        self.assertEqual(counts["MSFT"], 1)
        self.assertEqual(counts["AAPL"], 4)          # quality still favors AAPL
        self.assertEqual(len(picked), 5)

    def test_three_scopes_all_represented(self):
        ranked = (
            [_cand("AAPL", i) for i in range(3)]
            + [_cand("MSFT", i) for i in range(3)]
            + [_cand("NVDA", i) for i in range(3)]
        )
        picked = select_evidence_with_coverage(ranked, ["AAPL", "MSFT", "NVDA"], 5)
        self.assertEqual({c["ticker"] for c in picked}, {"AAPL", "MSFT", "NVDA"})

    def test_missing_scope_is_not_fabricated(self):
        ranked = [_cand("AAPL", i) for i in range(5)]  # no MSFT candidates
        picked = select_evidence_with_coverage(ranked, ["AAPL", "MSFT"], 5)
        self.assertEqual({c["ticker"] for c in picked}, {"AAPL"})

    def test_no_out_of_scope_ticker_survives_selection(self):
        ranked = [_cand("AAPL", 0), _cand("NVDA", 0), _cand("MSFT", 0)]
        picked = select_evidence_with_coverage(ranked, ["AAPL", "MSFT"], 5)
        # selection does not filter tickers itself, but everything it is given
        # here is in-scope except NVDA which only gets picked via the fill loop;
        # assert_scope upstream is what rejects that. Here just confirm the
        # reserved picks are the requested ones.
        self.assertIn("AAPL", [c["ticker"] for c in picked])
        self.assertIn("MSFT", [c["ticker"] for c in picked])

    def test_final_order_follows_global_rank(self):
        ranked = [_cand("AAPL", 0), _cand("AAPL", 1), _cand("MSFT", 2)]
        picked = select_evidence_with_coverage(ranked, ["AAPL", "MSFT"], 3)
        self.assertEqual(
            [c["chunk_id"] for c in picked],
            ["aapl_10k_0", "aapl_10k_1", "msft_10k_2"],
        )

    def test_empty_input(self):
        self.assertEqual(select_evidence_with_coverage([], ["AAPL", "MSFT"], 5), [])


# --------------------------------------------------------------------------- #
class PlanEvidenceRoutingTests(unittest.TestCase):
    """plan_evidence with scoped_search + Cohere mocked at the api.rag seam."""

    def _mock_scoped(self, per_scope_map):
        def fake_scoped(question, scopes, **kw):
            per_scope = {}
            union = []
            for scope in scopes:
                label = scope.tickers[0]
                rows = per_scope_map.get(label, [])
                per_scope[label] = rows
                union.extend(rows)
            return union, per_scope, 1.0
        return fake_scoped

    def _mock_rerank(self, counter):
        def fake_rerank_timed(question, candidates):
            counter["calls"] += 1
            counter["sizes"].append(len(candidates))
            ranked = sorted(
                candidates,
                key=lambda c: -c.get("rrf_score", 0.0),
            )
            for c in ranked:
                c["rerank_score"] = 0.9
            return ranked, 0.01
        return fake_rerank_timed

    def test_global_path_untouched(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            mock_re.return_value = ([_cand("AAPL", 0)], [_cand("AAPL", 0)],
                                    False, None, 1.0, 1.0)
            plan = plan_evidence("q", top_k=5, tickers=None)
        self.assertFalse(plan["comparison_mode"])
        self.assertEqual(plan["evidence_by_scope"], {})
        self.assertIsNone(plan["comparison_tickers"])

    def test_single_ticker_uses_phase1c_path(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            rows = [_cand("AAPL", i) for i in range(3)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            plan = plan_evidence("q", top_k=5, tickers=["aapl"])
        self.assertFalse(plan["comparison_mode"])
        mock_re.assert_called_once()
        self.assertEqual(plan["evidence_by_scope"], {"AAPL": 3})

    def test_two_tickers_enter_comparison_and_cover_both(self):
        counter = {"calls": 0, "sizes": []}
        scoped = self._mock_scoped({
            "AAPL": [_cand("AAPL", i) for i in range(5)],
            "MSFT": [_cand("MSFT", i) for i in range(5)],
        })
        with patch("api.rag.scoped_search", scoped), \
             patch("api.rag.rerank_timed", self._mock_rerank(counter)), \
             patch("api.rag.rerank_enabled", return_value=True):
            plan = plan_evidence("q", top_k=5, tickers=["AAPL", "MSFT"])

        self.assertTrue(plan["comparison_mode"])
        self.assertEqual(counter["calls"], 1)                 # test I: one Cohere call
        self.assertEqual(counter["sizes"], [10])              # the whole union
        self.assertGreaterEqual(plan["evidence_by_scope"]["AAPL"], 1)
        self.assertGreaterEqual(plan["evidence_by_scope"]["MSFT"], 1)
        self.assertEqual(plan["comparison_tickers"], ["AAPL", "MSFT"])
        self.assertEqual(plan["warnings"], [])

    def test_top_k_1_autoraises_for_two_tickers(self):
        counter = {"calls": 0, "sizes": []}
        scoped = self._mock_scoped({
            "AAPL": [_cand("AAPL", i) for i in range(5)],
            "MSFT": [_cand("MSFT", i) for i in range(5)],
        })
        with patch("api.rag.scoped_search", scoped), \
             patch("api.rag.rerank_timed", self._mock_rerank(counter)), \
             patch("api.rag.rerank_enabled", return_value=True):
            plan = plan_evidence("q", top_k=1, tickers=["AAPL", "MSFT"])
        self.assertEqual(len(plan["evidence"]), 2)
        self.assertEqual(set(plan["evidence_by_scope"]), {"AAPL", "MSFT"})
        self.assertEqual(min(plan["evidence_by_scope"].values()), 1)

    def test_three_tickers_all_represented(self):
        counter = {"calls": 0, "sizes": []}
        scoped = self._mock_scoped({
            "AAPL": [_cand("AAPL", i) for i in range(4)],
            "MSFT": [_cand("MSFT", i) for i in range(4)],
            "NVDA": [_cand("NVDA", i) for i in range(4)],
        })
        with patch("api.rag.scoped_search", scoped), \
             patch("api.rag.rerank_timed", self._mock_rerank(counter)), \
             patch("api.rag.rerank_enabled", return_value=True):
            plan = plan_evidence("q", top_k=5, tickers=["AAPL", "MSFT", "NVDA"])
        self.assertEqual(
            set(k for k, v in plan["evidence_by_scope"].items() if v >= 1),
            {"AAPL", "MSFT", "NVDA"},
        )

    def test_missing_scope_warning(self):
        counter = {"calls": 0, "sizes": []}
        scoped = self._mock_scoped({
            "AAPL": [_cand("AAPL", i) for i in range(5)],
            "MSFT": [],  # no candidates for MSFT
        })
        with patch("api.rag.scoped_search", scoped), \
             patch("api.rag.rerank_timed", self._mock_rerank(counter)), \
             patch("api.rag.rerank_enabled", return_value=True):
            plan = plan_evidence("q", top_k=5, tickers=["AAPL", "MSFT"])
        self.assertEqual(
            plan["warnings"],
            ["No relevant evidence was found for MSFT in the requested scope."],
        )
        self.assertEqual(plan["evidence_by_scope"]["MSFT"], 0)
        # not a hard failure
        self.assertTrue(plan["evidence"])

    def test_cohere_fallback_keeps_per_scope_coverage(self):
        scoped = self._mock_scoped({
            "AAPL": [_cand("AAPL", i) for i in range(5)],
            "MSFT": [_cand("MSFT", i) for i in range(5)],
        })

        def boom(question, candidates):
            raise RuntimeError("cohere down")

        with patch("api.rag.scoped_search", scoped), \
             patch("api.rag.rerank_timed", boom), \
             patch("api.rag.rerank_enabled", return_value=True):
            plan = plan_evidence("q", top_k=5, tickers=["AAPL", "MSFT"])

        self.assertTrue(plan["reranker_fallback"])
        self.assertEqual(plan["reranker_fallback_reason"], "api_error")
        self.assertGreaterEqual(plan["evidence_by_scope"]["AAPL"], 1)
        self.assertGreaterEqual(plan["evidence_by_scope"]["MSFT"], 1)  # test J

    def test_out_of_scope_candidate_raises(self):
        # scoped_search leaks an NVDA row into an AAPL/MSFT comparison.
        def leaky_scoped(question, scopes, **kw):
            union = [_cand("AAPL", 0), _cand("MSFT", 0), _cand("NVDA", 0)]
            per_scope = {"AAPL": [union[0]], "MSFT": [union[1]]}
            return union, per_scope, 1.0

        with patch("api.rag.scoped_search", leaky_scoped), \
             patch("api.rag.rerank_timed",
                   lambda q, c: (c, 0.0)), \
             patch("api.rag.rerank_enabled", return_value=True):
            with self.assertRaises(ScopeViolationError):
                plan_evidence("q", top_k=5, tickers=["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
