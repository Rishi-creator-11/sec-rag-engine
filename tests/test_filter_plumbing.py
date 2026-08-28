"""Phase 1B: unit tests for RetrievalFilter plumbing through the retrievers.

Network calls (Pinecone dense) are mocked. BM25 runs for real against the
local ``data/chunks`` corpus (no network).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from retrieval.filters import RetrievalFilter


def _fake_match(chunk_id, ticker):
    return SimpleNamespace(
        id=chunk_id,
        score=0.5,
        metadata={
            "text": f"body of {chunk_id}",
            "company": {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}.get(
                ticker, ticker
            ),
            "ticker": ticker,
            "filing_type": "10-K",
            "filing_date": "2024-09-28",
            "source_url": "https://www.sec.gov/x",
        },
    )


class DenseFilterPlumbingTests(unittest.TestCase):
    def _run(self, mock_index, **search_kwargs):
        mock_index.query.return_value = SimpleNamespace(
            matches=[_fake_match("apple_10k_1", "AAPL")]
        )
        from retrieval.pinecone_search import search

        return search("q", **search_kwargs)

    @patch("retrieval.pinecone_search.embed_text", return_value=[0.0] * 1536)
    @patch("retrieval.pinecone_search.index")
    def test_no_filter_sends_no_filter_kwarg(self, mock_index, _embed):
        self._run(mock_index)
        self.assertNotIn("filter", mock_index.query.call_args.kwargs)

    @patch("retrieval.pinecone_search.embed_text", return_value=[0.0] * 1536)
    @patch("retrieval.pinecone_search.index")
    def test_explicit_none_sends_no_filter_kwarg(self, mock_index, _embed):
        self._run(mock_index, filters=None)
        self.assertNotIn("filter", mock_index.query.call_args.kwargs)

    @patch("retrieval.pinecone_search.embed_text", return_value=[0.0] * 1536)
    @patch("retrieval.pinecone_search.index")
    def test_empty_filter_sends_no_filter_kwarg(self, mock_index, _embed):
        self._run(mock_index, filters=RetrievalFilter())
        self.assertNotIn("filter", mock_index.query.call_args.kwargs)

    @patch("retrieval.pinecone_search.embed_text", return_value=[0.0] * 1536)
    @patch("retrieval.pinecone_search.index")
    def test_ticker_filter_builds_pinecone_predicate(self, mock_index, _embed):
        self._run(mock_index, filters=RetrievalFilter(tickers=("aapl",)))
        self.assertEqual(
            mock_index.query.call_args.kwargs["filter"],
            {"ticker": {"$in": ["AAPL"]}},
        )

    @patch("retrieval.pinecone_search.embed_text", return_value=[0.0] * 1536)
    @patch("retrieval.pinecone_search.index")
    def test_multi_field_filter_predicate(self, mock_index, _embed):
        self._run(
            mock_index,
            filters=RetrievalFilter(tickers=("AAPL", "MSFT"), fiscal_years=(2024,)),
        )
        self.assertEqual(
            mock_index.query.call_args.kwargs["filter"],
            {"ticker": {"$in": ["AAPL", "MSFT"]}, "fiscal_year": {"$in": [2024]}},
        )

    @patch("retrieval.pinecone_search.embed_text", return_value=[0.0] * 1536)
    @patch("retrieval.pinecone_search.index")
    def test_result_shape_unchanged(self, mock_index, _embed):
        rows = self._run(mock_index)
        self.assertEqual(
            set(rows[0]),
            {
                "chunk_id",
                "score",
                "text",
                "company",
                "ticker",
                "filing_type",
                "filing_date",
                "source_url",
            },
        )


class Bm25FilterPlumbingTests(unittest.TestCase):
    QUERY = "What competitive risks does the company face?"

    @classmethod
    def setUpClass(cls):
        from retrieval.bm25_search import get_index, search

        cls.search = staticmethod(search)
        cls.index = get_index()

    def test_none_matches_default_behavior(self):
        a = self.search(self.QUERY, top_k=10)
        b = self.search(self.QUERY, top_k=10, filters=None)
        c = self.search(self.QUERY, top_k=10, filters=RetrievalFilter())
        self.assertEqual(
            [(r["chunk_id"], r["score"]) for r in a],
            [(r["chunk_id"], r["score"]) for r in b],
        )
        self.assertEqual(
            [(r["chunk_id"], r["score"]) for r in a],
            [(r["chunk_id"], r["score"]) for r in c],
        )

    def test_single_ticker_filter(self):
        rows = self.search(
            self.QUERY, top_k=10, filters=RetrievalFilter(tickers=("AAPL",))
        )
        self.assertTrue(rows)
        self.assertTrue(all(r["ticker"] == "AAPL" for r in rows))

    def test_other_ticker_filter(self):
        rows = self.search(
            self.QUERY, top_k=10, filters=RetrievalFilter(tickers=("MSFT",))
        )
        self.assertTrue(rows)
        self.assertTrue(all(r["ticker"] == "MSFT" for r in rows))

    def test_multi_ticker_filter(self):
        rows = self.search(
            self.QUERY, top_k=25, filters=RetrievalFilter(tickers=("AAPL", "MSFT"))
        )
        self.assertTrue(rows)
        self.assertTrue(all(r["ticker"] in {"AAPL", "MSFT"} for r in rows))
        self.assertNotIn("NVDA", {r["ticker"] for r in rows})

    def test_filtered_scores_match_global_scoring(self):
        """A filtered result keeps the same score it has in the unfiltered run."""
        unfiltered = {r["chunk_id"]: r["score"] for r in self.search(self.QUERY, top_k=300)}
        filtered = self.search(
            self.QUERY, top_k=300, filters=RetrievalFilter(tickers=("AAPL",))
        )
        for row in filtered:
            self.assertAlmostEqual(row["score"], unfiltered[row["chunk_id"]], places=12)

    def test_global_statistics_unchanged_after_filtered_query(self):
        avgdl_before = self.index.average_document_length
        idf_before = dict(self.index.inverse_document_frequencies)
        count_before = self.index.document_count

        self.search(
            self.QUERY, top_k=10, filters=RetrievalFilter(tickers=("NVDA",))
        )

        self.assertEqual(self.index.average_document_length, avgdl_before)
        self.assertEqual(self.index.document_count, count_before)
        self.assertEqual(
            dict(self.index.inverse_document_frequencies), idf_before
        )

    def test_no_matches_returns_empty_list(self):
        rows = self.search(
            self.QUERY, top_k=10, filters=RetrievalFilter(tickers=("NOSUCH",))
        )
        self.assertEqual(rows, [])


class HybridFilterPlumbingTests(unittest.TestCase):
    QUERY = "competitive risks"

    def _spies(self):
        seen = {}

        def fake_dense(query, top_k=10, filters=None, query_embedding=None):
            seen["dense"] = filters
            seen["dense_query_embedding"] = query_embedding
            pool = [
                {"chunk_id": "apple_10k_1", "score": 0.9, "ticker": "AAPL",
                 "company": "Apple Inc.", "filing_type": "10-K",
                 "filing_date": "2024-09-28", "source_url": "u", "text": "a"},
                {"chunk_id": "microsoft_10k_2", "score": 0.8, "ticker": "MSFT",
                 "company": "Microsoft Corporation", "filing_type": "10-K",
                 "filing_date": "2025-06-30", "source_url": "u", "text": "m"},
                {"chunk_id": "nvidia_10k_3", "score": 0.7, "ticker": "NVDA",
                 "company": "NVIDIA Corporation", "filing_type": "10-K",
                 "filing_date": "2026-01-25", "source_url": "u", "text": "n"},
            ]
            if filters is not None and not filters.is_empty():
                pool = [c for c in pool if filters.matches(c)]
            return pool[:top_k]

        def fake_bm25(query, top_k=10, filters=None):
            seen["bm25"] = filters
            pool = [
                {"chunk_id": "apple_10k_4", "score": 2.0, "ticker": "AAPL",
                 "company": "Apple Inc.", "filing_type": "10-K",
                 "filing_date": "2024-09-28", "source_url": "u", "text": "a4"},
                {"chunk_id": "nvidia_10k_5", "score": 1.5, "ticker": "NVDA",
                 "company": "NVIDIA Corporation", "filing_type": "10-K",
                 "filing_date": "2026-01-25", "source_url": "u", "text": "n5"},
            ]
            if filters is not None and not filters.is_empty():
                pool = [c for c in pool if filters.matches(c)]
            return pool[:top_k]

        def fake_sparse(query, top_k=10):
            seen["sparse_called"] = True
            return []

        return seen, fake_dense, fake_bm25, fake_sparse

    def test_filter_propagates_to_dense_and_bm25(self):
        seen, fd, fb, fs = self._spies()
        with patch("retrieval.hybrid_search.dense_search", fd), \
             patch("retrieval.hybrid_search.bm25_search", fb), \
             patch("retrieval.hybrid_search.sparse_search", fs):
            from retrieval.hybrid_search import search

            filt = RetrievalFilter(tickers=("AAPL",))
            rows = search(self.QUERY, top_k=5, candidate_k=10, filters=filt)

        self.assertEqual(seen["dense"], filt)
        self.assertEqual(seen["bm25"], filt)
        self.assertNotIn("sparse_called", seen)  # sparse stays disabled
        self.assertTrue(rows)
        self.assertTrue(all(r["ticker"] == "AAPL" for r in rows))

    def test_none_filter_propagates_as_none(self):
        seen, fd, fb, fs = self._spies()
        with patch("retrieval.hybrid_search.dense_search", fd), \
             patch("retrieval.hybrid_search.bm25_search", fb), \
             patch("retrieval.hybrid_search.sparse_search", fs):
            from retrieval.hybrid_search import search

            rows = search(self.QUERY, top_k=5, candidate_k=10, filters=None)

        self.assertIsNone(seen["dense"])
        self.assertIsNone(seen["bm25"])
        tickers = {r["ticker"] for r in rows}
        self.assertEqual(tickers, {"AAPL", "MSFT", "NVDA"})

    def test_empty_filter_propagates_as_none(self):
        seen, fd, fb, fs = self._spies()
        with patch("retrieval.hybrid_search.dense_search", fd), \
             patch("retrieval.hybrid_search.bm25_search", fb), \
             patch("retrieval.hybrid_search.sparse_search", fs):
            from retrieval.hybrid_search import search

            search(self.QUERY, top_k=5, candidate_k=10, filters=RetrievalFilter())

        self.assertIsNone(seen["dense"])
        self.assertIsNone(seen["bm25"])

    def test_fused_result_shape_unchanged(self):
        seen, fd, fb, fs = self._spies()
        with patch("retrieval.hybrid_search.dense_search", fd), \
             patch("retrieval.hybrid_search.bm25_search", fb), \
             patch("retrieval.hybrid_search.sparse_search", fs):
            from retrieval.hybrid_search import search

            rows = search(self.QUERY, top_k=5, candidate_k=10)
        for key in ("rrf_score", "retrieved_by", "ranks", "chunk_id"):
            self.assertIn(key, rows[0])


if __name__ == "__main__":
    unittest.main()
