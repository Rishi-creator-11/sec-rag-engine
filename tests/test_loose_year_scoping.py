"""Regression coverage for the loose-phrasing year-scoping fix.

Previously untested request shape: a ticker scope with NO structured
``fiscal_years``, where the only fiscal-year signal is in the free-text
question (e.g. "What was ExxonMobil's revenue in FY2025?"). Before this
fix, that shape searched every ingested year for the ticker with no hard
filter -- verified live against production (see the investigation report):
5-chunk evidence sets mixing FY2023/FY2024/FY2025 in every trial.

These tests mock retrieval at the same ``api.rag`` seam the existing
comparison/filter tests use (no network calls), and assert on the
RetrievalFilter that actually reaches retrieval -- not on what an LLM
happens to answer. The deterministic claim under test is: an inferred
FY2025 scope must flow through the exact same Scope -> RetrievalFilter
-> assert_scopes path a structured ``fiscal_years=[2025]`` request uses.
"""

import unittest
from unittest.mock import patch

from api.rag import ScopeViolationError, plan_evidence


def _cand(ticker, fiscal_year, i):
    return {
        "chunk_id": f"{ticker.lower()}_{fiscal_year}_{i}",
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "company": "Exxon Mobil Corporation",
        "filing_type": "10-K",
        "filing_date": "2025-02-01",
        "source_url": "https://www.sec.gov/x",
        "text": f"{ticker} FY{fiscal_year} evidence {i}",
        "rrf_score": 0.5 - i * 0.01,
    }


class InferredScopeReachesRetrievalTests(unittest.TestCase):
    """The inferred year must reach retrieve_evidence as a real RetrievalFilter."""

    def test_single_year_question_infers_and_filters(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            rows = [_cand("XOM", 2025, i) for i in range(3)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            plan = plan_evidence(
                "What was ExxonMobil's revenue in FY2025?",
                top_k=5,
                tickers=["XOM"],
                fiscal_years=None,
            )

        mock_re.assert_called_once()
        _question, _evidence_k, passed_filter = mock_re.call_args[0]
        self.assertEqual(passed_filter.tickers, ("XOM",))
        self.assertEqual(passed_filter.fiscal_years, (2025,))
        self.assertEqual(plan["fiscal_years"], [2025])
        self.assertEqual(plan["scopes"], ["XOM:2025"])
        self.assertTrue(plan["fiscal_year_inferred"])

    def test_bare_year_question_also_infers(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            rows = [_cand("XOM", 2025, i) for i in range(3)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            plan_evidence(
                "What was ExxonMobil's revenue in 2025?",
                top_k=5,
                tickers=["XOM"],
                fiscal_years=None,
            )

        passed_filter = mock_re.call_args[0][2]
        self.assertEqual(passed_filter.fiscal_years, (2025,))


class StructuredScopeAlwaysWinsTests(unittest.TestCase):
    """Explicit fiscal_years must never be overridden or augmented by inference."""

    def test_structured_year_beats_conflicting_question_year(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            rows = [_cand("XOM", 2024, i) for i in range(3)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            plan = plan_evidence(
                "What was revenue in FY2025?",  # question says 2025
                top_k=5,
                tickers=["XOM"],
                fiscal_years=[2024],  # structured request says 2024
            )

        passed_filter = mock_re.call_args[0][2]
        self.assertEqual(passed_filter.fiscal_years, (2024,))
        self.assertEqual(plan["fiscal_years"], [2024])
        self.assertFalse(plan["fiscal_year_inferred"])

    def test_no_inference_attempted_when_years_supplied(self):
        # Even an unambiguous single-year question must not be consulted
        # once structured fiscal_years is present -- inference must not run.
        with patch("api.rag.retrieve_evidence") as mock_re, \
             patch("api.rag.infer_fiscal_year") as mock_infer:
            rows = [_cand("XOM", 2024, i) for i in range(3)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            plan_evidence(
                "revenue in FY2024",
                top_k=5,
                tickers=["XOM"],
                fiscal_years=[2024],
            )
        mock_infer.assert_not_called()
        self.assertIsNotNone(plan_evidence)  # sanity: import path intact


class AmbiguousQuestionsPreserveTodaysBehaviorTests(unittest.TestCase):
    """Multi-year / range / no-year questions must not collapse to one scope."""

    def test_two_year_comparison_stays_unfiltered_single_scope(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            rows = [_cand("XOM", 2023, 0), _cand("XOM", 2025, 1)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            plan = plan_evidence(
                "Compare FY2023 and FY2025 revenue",
                top_k=5,
                tickers=["XOM"],
                fiscal_years=None,
            )

        passed_filter = mock_re.call_args[0][2]
        self.assertIsNone(passed_filter.fiscal_years)
        self.assertFalse(plan["comparison_mode"])  # unchanged from today: 1 scope, no year
        self.assertFalse(plan["fiscal_year_inferred"])

    def test_no_year_mentioned_stays_unfiltered(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            rows = [_cand("XOM", 2025, 0)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            plan = plan_evidence(
                "What are ExxonMobil's main business segments?",
                top_k=5,
                tickers=["XOM"],
                fiscal_years=None,
            )

        passed_filter = mock_re.call_args[0][2]
        self.assertIsNone(passed_filter.fiscal_years)
        self.assertFalse(plan["fiscal_year_inferred"])


class HardFilterSafetyNetTests(unittest.TestCase):
    """If wrong-year evidence ever reached the plan despite the filter, the
    existing assert_scopes mechanism must still catch it -- proving the
    inferred scope is enforced exactly like an explicit one, not just
    requested."""

    def test_wrong_year_candidate_trips_existing_assertion(self):
        with patch("api.rag.retrieve_evidence") as mock_re:
            # Simulate a filter bug: retrieval returns a wrong-year row
            # alongside correct ones despite an inferred FY2025 scope.
            rows = [_cand("XOM", 2025, 0), _cand("XOM", 2024, 1)]
            mock_re.return_value = (rows, rows, False, None, 1.0, 1.0)
            with self.assertRaises(ScopeViolationError):
                plan_evidence(
                    "What was ExxonMobil's revenue in FY2025?",
                    top_k=5,
                    tickers=["XOM"],
                    fiscal_years=None,
                )


if __name__ == "__main__":
    unittest.main()
