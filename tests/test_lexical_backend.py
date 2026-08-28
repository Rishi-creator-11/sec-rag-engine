"""Phase 4.5: lexical backend abstraction — parity, filters, persistence."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval.filters import RetrievalFilter
from retrieval.lexical_backend import (
    BM25SBackend,
    CurrentBM25Backend,
    build_persisted_bm25s,
    corpus_version,
    get_lexical_backend,
    load_or_build_bm25s,
)

_TEXTS = {
    "AAPL": [
        "Apple faces intense competition in smartphones and services with pricing pressure.",
        "Apple relies on single source component suppliers and manufacturing partners in Asia.",
        "Apple describes privacy and data security regulation risk across jurisdictions.",
        "Apple research and development expense increased to support innovation.",
    ],
    "MSFT": [
        "Microsoft Azure competes with cloud service providers and open source offerings.",
        "Microsoft cybersecurity risk includes nation state actors targeting its software.",
        "Microsoft total revenue grew driven by cloud and productivity segments.",
        "Microsoft artificial intelligence investments carry model and data risk.",
    ],
    "NVDA": [
        "NVIDIA faces export controls that restrict sales of advanced chips to China.",
        "NVIDIA data center platform competition from custom accelerators is increasing.",
        "NVIDIA supply chain depends on a limited number of foundry partners.",
        "NVIDIA responsible AI risk and regulation such as the EU AI Act may raise costs.",
    ],
}


def _corpus():
    chunks = []
    for ticker, texts in _TEXTS.items():
        for i, text in enumerate(texts):
            chunks.append({
                "chunk_id": f"{ticker}_2025_10-K_x_{i}",
                "text": text,
                "company": f"{ticker} Inc.",
                "ticker": ticker,
                "filing_type": "10-K",
                "filing_date": "2025-12-31",
                "source_url": "https://www.sec.gov/x",
            })
    return chunks


_QUERIES = [
    "competition and pricing pressure",
    "cybersecurity and nation state actors",
    "export controls China chips",
    "supply chain single source suppliers",
    "artificial intelligence regulation risk",
    "total revenue growth cloud",
    "research and development expense",
    "data center accelerator competition",
]


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = _corpus()
        cls.cur = CurrentBM25Backend(cls.chunks)
        cls.b25s = BM25SBackend(cls.chunks)

    def test_top10_same_set_and_order(self):
        for q in _QUERIES:
            a = [r["chunk_id"] for r in self.cur.search(q, top_k=10)]
            b = [r["chunk_id"] for r in self.b25s.search(q, top_k=10)]
            self.assertEqual(a, b, f"ranking diverged for {q!r}")

    def test_result_dict_shape_matches_bm25_search(self):
        row = self.b25s.search(_QUERIES[0], top_k=1)[0]
        self.assertEqual(
            set(row),
            {"chunk_id", "score", "text", "company", "ticker",
             "filing_type", "filing_date", "source_url"},
        )

    def test_deterministic_ranking(self):
        for q in _QUERIES:
            self.assertEqual(self.b25s.search(q, 5), self.b25s.search(q, 5))

    def test_candidate_k_honored(self):
        self.assertEqual(len(self.b25s.search(_QUERIES[0], top_k=3)), 3)
        self.assertEqual(len(self.b25s.search(_QUERIES[0], top_k=7)), 7)

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.b25s.search("the of and", top_k=5), [])  # all stopwords


class FilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = BM25SBackend(_corpus())

    def _search(self, tickers):
        if isinstance(tickers, str):
            tickers = (tickers,)
        return self.b.search(
            "competition regulation risk revenue supply",
            top_k=99, filters=RetrievalFilter(tickers=tuple(tickers)),
        )

    def test_single_ticker_no_leakage(self):
        for t in ("AAPL", "MSFT", "NVDA"):
            rows = self._search(t)
            self.assertTrue(rows)
            self.assertTrue(all(r["ticker"] == t for r in rows))

    def test_multi_ticker_no_leakage(self):
        rows = self._search(("AAPL", "MSFT"))
        self.assertTrue(all(r["ticker"] in {"AAPL", "MSFT"} for r in rows))
        self.assertNotIn("NVDA", {r["ticker"] for r in rows})

    def test_filtered_scores_equal_unfiltered_scores(self):
        q = "competition regulation risk"
        unfiltered = {r["chunk_id"]: r["score"] for r in self.b.search(q, top_k=99)}
        for r in self.b.search(q, top_k=99, filters=RetrievalFilter(tickers=("NVDA",))):
            self.assertAlmostEqual(r["score"], unfiltered[r["chunk_id"]], places=10)

    def test_no_match_returns_empty(self):
        rows = self.b.search("competition", top_k=5,
                             filters=RetrievalFilter(tickers=("NOSUCH",)))
        self.assertEqual(rows, [])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "idx"
        self.chunks = _corpus()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_build_then_load_rank_identical(self):
        built = build_persisted_bm25s(self.chunks, self.dir)
        loaded = BM25SBackend.load(self.dir)
        for q in _QUERIES:
            self.assertEqual(
                [r["chunk_id"] for r in built.search(q, 5)],
                [r["chunk_id"] for r in loaded.search(q, 5)],
            )

    def test_load_or_build_second_call_is_a_load(self):
        load_or_build_bm25s(self.dir, self.chunks)
        with patch.object(BM25SBackend, "__init__",
                          side_effect=AssertionError("should not rebuild")):
            b = load_or_build_bm25s(self.dir, self.chunks)
        self.assertEqual(b.document_count, len(self.chunks))

    def test_missing_index_rebuilds(self):
        b = load_or_build_bm25s(self.dir, self.chunks)  # dir does not exist yet
        self.assertEqual(b.document_count, len(self.chunks))
        self.assertTrue((self.dir / "corpus_version.json").exists())

    def test_corrupt_version_file_rebuilds(self):
        load_or_build_bm25s(self.dir, self.chunks)
        (self.dir / "corpus_version.json").write_text("{ not json")
        b = load_or_build_bm25s(self.dir, self.chunks)
        self.assertEqual(b.document_count, len(self.chunks))

    def test_stale_corpus_version_rebuilds(self):
        load_or_build_bm25s(self.dir, self.chunks)
        (self.dir / "corpus_version.json").write_text(
            json.dumps({"corpus_version": "STALE", "document_count": 1})
        )
        extra = self.chunks + [{
            "chunk_id": "WMT_2026_10-K_y_0", "text": "Walmart retail logistics",
            "company": "Walmart Inc.", "ticker": "WMT", "filing_type": "10-K",
            "filing_date": "2026-01-31", "source_url": "https://www.sec.gov/x",
        }]
        b = load_or_build_bm25s(self.dir, extra)
        self.assertEqual(b.document_count, len(extra))
        meta = json.loads((self.dir / "corpus_version.json").read_text())
        self.assertEqual(meta["corpus_version"], corpus_version(extra))

    def test_corpus_version_is_deterministic_and_order_independent(self):
        import random
        shuffled = list(self.chunks)
        random.shuffle(shuffled)
        self.assertEqual(corpus_version(self.chunks), corpus_version(shuffled))


class SelectorTests(unittest.TestCase):
    def test_default_is_bm25s(self):
        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("SEC_RAG_LEXICAL_BACKEND", None)
            with patch("retrieval.lexical_backend.load_or_build_bm25s") as m:
                m.return_value = "sentinel"
                self.assertEqual(get_lexical_backend(_corpus()), "sentinel")

    def test_env_rollback_to_current(self):
        with patch.dict("os.environ", {"SEC_RAG_LEXICAL_BACKEND": "current"}):
            self.assertIsInstance(get_lexical_backend(_corpus()), CurrentBM25Backend)

    def test_force_rebuild_calls_builder(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("SEC_RAG_LEXICAL_BACKEND", None)
            with patch("retrieval.lexical_backend.build_persisted_bm25s") as m:
                m.return_value = "sentinel"
                self.assertEqual(
                    get_lexical_backend(_corpus(), force_rebuild=True), "sentinel"
                )


class ProductionWiringTests(unittest.TestCase):
    def test_bm25_search_delegates_to_lexical_backend(self):
        import retrieval.bm25_search as bs
        from retrieval.lexical_backend import LexicalBackend

        bs._index = None  # reset singleton
        idx = bs.get_index()
        self.assertIsInstance(idx, LexicalBackend)
        self.assertEqual(bs.document_count(), idx.document_count)

    def test_rollback_env_restores_current_impl(self):
        import retrieval.bm25_search as bs
        with patch.dict("os.environ", {"SEC_RAG_LEXICAL_BACKEND": "current"}):
            bs._index = None
            self.assertIsInstance(bs.get_index(), CurrentBM25Backend)
        bs._index = None  # leave clean for other tests


if __name__ == "__main__":
    unittest.main()


class LexicalBackendErrorTests(unittest.TestCase):
    def test_load_or_build_raises_when_rebuild_fails(self):
        from retrieval.lexical_backend import LexicalBackendError, load_or_build_bm25s

        with tempfile.TemporaryDirectory() as tmp:
            with patch("retrieval.lexical_backend.BM25SBackend",
                       side_effect=RuntimeError("scipy exploded")):
                with self.assertRaises(LexicalBackendError):
                    load_or_build_bm25s(Path(tmp) / "idx", _corpus())
