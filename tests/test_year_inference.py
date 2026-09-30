"""Unit tests for retrieval.year_inference.infer_fiscal_year (stdlib unittest).

Deliberately conservative: only infers a year when the question names
exactly one and gives no signal of spanning, trending over, or comparing
across more than one.
"""

import unittest

from retrieval.year_inference import infer_fiscal_year


class ShouldInferTests(unittest.TestCase):
    def test_fy_prefixed_no_space(self):
        self.assertEqual(
            infer_fiscal_year("What was ExxonMobil's revenue in FY2025?"), 2025
        )

    def test_fy_prefixed_with_space(self):
        self.assertEqual(infer_fiscal_year("What was XOM revenue in FY 2025?"), 2025)

    def test_fiscal_year_phrase(self):
        self.assertEqual(
            infer_fiscal_year("What was revenue for fiscal year 2025?"), 2025
        )

    def test_fiscal_no_year_word(self):
        self.assertEqual(infer_fiscal_year("Total revenue for fiscal 2025."), 2025)

    def test_bare_year_supported_conservatively(self):
        self.assertEqual(
            infer_fiscal_year("What was ExxonMobil's revenue in 2025?"), 2025
        )

    def test_case_insensitive(self):
        self.assertEqual(infer_fiscal_year("revenue in fy2025"), 2025)

    def test_lowercase_fiscal_year(self):
        self.assertEqual(infer_fiscal_year("net income, fiscal year 2023"), 2023)


class ShouldNotInferSingleYearTests(unittest.TestCase):
    def test_two_explicit_years_with_and(self):
        self.assertIsNone(infer_fiscal_year("Compare FY2023 and FY2025"))

    def test_since_phrasing_with_one_year(self):
        self.assertIsNone(infer_fiscal_year("How has revenue changed since 2020?"))

    def test_from_to_range(self):
        self.assertIsNone(infer_fiscal_year("Compare revenue from 2023 to 2025"))

    def test_last_n_years_no_digit_year(self):
        self.assertIsNone(infer_fiscal_year("What changed over the last three years?"))

    def test_vs_phrasing(self):
        self.assertIsNone(infer_fiscal_year("2023 vs 2025 revenue"))

    def test_versus_phrasing(self):
        self.assertIsNone(infer_fiscal_year("FY2023 versus FY2025"))

    def test_no_year_present(self):
        self.assertIsNone(infer_fiscal_year("What is the capital of France?"))

    def test_empty_question(self):
        self.assertIsNone(infer_fiscal_year(""))

    def test_trend_word(self):
        self.assertIsNone(infer_fiscal_year("What is the revenue trend since 2021?"))

    def test_year_over_year(self):
        self.assertIsNone(
            infer_fiscal_year("How did revenue change year over year in the 2020s?")
        )

    def test_past_n_years(self):
        self.assertIsNone(infer_fiscal_year("Revenue over the past two years?"))

    def test_out_of_range_year_ignored(self):
        # Only 20xx years are considered fiscal-year candidates in this domain.
        self.assertIsNone(infer_fiscal_year("Founded in 1999, revenue today?"))


if __name__ == "__main__":
    unittest.main()
