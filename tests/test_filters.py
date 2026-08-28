"""Unit tests for retrieval.filters.RetrievalFilter (stdlib unittest)."""

import unittest

from retrieval.filters import (
    RetrievalFilter,
    normalize_filing_types,
    normalize_int_values,
    normalize_tickers,
)


class NormalizationTests(unittest.TestCase):
    def test_tickers_uppercased_stripped_deduped_order_preserving(self):
        self.assertEqual(
            normalize_tickers([" aapl ", "MSFT", "aapl", "msft"]),
            ("AAPL", "MSFT"),
        )

    def test_ticker_string_is_not_iterated_as_characters(self):
        self.assertEqual(normalize_tickers("AAPL"), ("AAPL",))

    def test_empty_and_blank_collections_become_none(self):
        self.assertIsNone(normalize_tickers([]))
        self.assertIsNone(normalize_tickers(["", "   "]))
        self.assertIsNone(normalize_tickers(None))

    def test_filing_types_hyphen_inserted_and_deduped(self):
        self.assertEqual(normalize_filing_types(["10k", "10-K"]), ("10-K",))
        self.assertEqual(normalize_filing_types(["8k"]), ("8-K",))
        self.assertEqual(normalize_filing_types(["10-Q"]), ("10-Q",))

    def test_fiscal_years_coerced_and_deduped(self):
        self.assertEqual(normalize_int_values(["2024", 2024, 2025]), (2024, 2025))

    def test_fiscal_years_reject_bool(self):
        with self.assertRaises(TypeError):
            normalize_int_values([True])

    def test_fiscal_years_reject_non_numeric(self):
        with self.assertRaises(ValueError):
            normalize_int_values(["twenty-twenty-four"])

    def test_dataclass_normalizes_on_construction(self):
        f = RetrievalFilter(tickers=["aapl", "AAPL"], fiscal_years=["2024"])
        self.assertEqual(f.tickers, ("AAPL",))
        self.assertEqual(f.fiscal_years, (2024,))
        self.assertIsNone(f.filing_types)


class EmptyFilterTests(unittest.TestCase):
    def test_default_is_empty(self):
        self.assertTrue(RetrievalFilter().is_empty())

    def test_all_blank_values_is_empty(self):
        self.assertTrue(RetrievalFilter(tickers=[], fiscal_years=None).is_empty())

    def test_any_constraint_is_not_empty(self):
        self.assertFalse(RetrievalFilter(tickers=["AAPL"]).is_empty())

    def test_empty_filter_pinecone_is_none(self):
        self.assertIsNone(RetrievalFilter().to_pinecone_filter())

    def test_empty_filter_matches_everything(self):
        self.assertTrue(RetrievalFilter().matches({"ticker": "AAPL"}))
        self.assertTrue(RetrievalFilter().matches({}))


class PineconePredicateTests(unittest.TestCase):
    def test_single_ticker(self):
        self.assertEqual(
            RetrievalFilter(tickers=("AAPL",)).to_pinecone_filter(),
            {"ticker": {"$in": ["AAPL"]}},
        )

    def test_multiple_tickers_and_years(self):
        self.assertEqual(
            RetrievalFilter(
                tickers=("AAPL", "MSFT"), fiscal_years=(2024, 2025)
            ).to_pinecone_filter(),
            {
                "ticker": {"$in": ["AAPL", "MSFT"]},
                "fiscal_year": {"$in": [2024, 2025]},
            },
        )

    def test_only_supplied_fields_present(self):
        predicate = RetrievalFilter(filing_types=("10-K",)).to_pinecone_filter()
        self.assertEqual(predicate, {"filing_type": {"$in": ["10-K"]}})
        self.assertNotIn("ticker", predicate)

    def test_filing_ids(self):
        self.assertEqual(
            RetrievalFilter(filing_ids=("000032019324000123",)).to_pinecone_filter(),
            {"filing_id": {"$in": ["000032019324000123"]}},
        )

    def test_fiscal_year_values_are_ints(self):
        predicate = RetrievalFilter(fiscal_years=("2024",)).to_pinecone_filter()
        self.assertEqual(predicate["fiscal_year"]["$in"], [2024])
        self.assertIsInstance(predicate["fiscal_year"]["$in"][0], int)


class MatchesTests(unittest.TestCase):
    CHUNK = {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "fiscal_year": 2024,
        "filing_id": "000032019324000123",
    }

    def test_matching_ticker(self):
        self.assertTrue(RetrievalFilter(tickers=("AAPL",)).matches(self.CHUNK))

    def test_non_matching_ticker(self):
        self.assertFalse(RetrievalFilter(tickers=("MSFT",)).matches(self.CHUNK))

    def test_ticker_case_insensitive_on_chunk_side(self):
        self.assertTrue(
            RetrievalFilter(tickers=("AAPL",)).matches({"ticker": "aapl"})
        )

    def test_missing_constrained_field_is_non_match(self):
        self.assertFalse(RetrievalFilter(tickers=("AAPL",)).matches({}))
        self.assertFalse(
            RetrievalFilter(fiscal_years=(2024,)).matches({"ticker": "AAPL"})
        )

    def test_fiscal_year_string_on_chunk_is_coerced(self):
        self.assertTrue(
            RetrievalFilter(fiscal_years=(2024,)).matches(
                {"ticker": "AAPL", "fiscal_year": "2024"}
            )
        )

    def test_multi_value_membership(self):
        self.assertTrue(
            RetrievalFilter(tickers=("AAPL", "MSFT")).matches(self.CHUNK)
        )

    def test_all_constraints_must_hold(self):
        good = RetrievalFilter(tickers=("AAPL",), fiscal_years=(2024,))
        bad = RetrievalFilter(tickers=("AAPL",), fiscal_years=(2023,))
        self.assertTrue(good.matches(self.CHUNK))
        self.assertFalse(bad.matches(self.CHUNK))

    def test_filing_type_normalized_comparison(self):
        self.assertTrue(
            RetrievalFilter(filing_types=("10k",)).matches(self.CHUNK)
        )


class MiscTests(unittest.TestCase):
    def test_frozen(self):
        f = RetrievalFilter(tickers=("AAPL",))
        with self.assertRaises(Exception):
            f.tickers = ("MSFT",)  # type: ignore[misc]

    def test_describe(self):
        self.assertEqual(RetrievalFilter().describe(), "unfiltered")
        self.assertEqual(
            RetrievalFilter(tickers=("AAPL", "MSFT"), fiscal_years=(2024,)).describe(),
            "AAPL,MSFT | FY2024",
        )

    def test_hashable(self):
        {RetrievalFilter(tickers=("AAPL",)), RetrievalFilter(tickers=("AAPL",))}


if __name__ == "__main__":
    unittest.main()
