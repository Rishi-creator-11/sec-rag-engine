"""Unit tests for retrieval.metadata (stdlib unittest)."""

import unittest

from retrieval.metadata import (
    ChunkMetadata,
    accession_to_filing_id,
    canonical_chunk_id,
    derive_fiscal_year,
    format_accession,
    normalize_cik,
    normalize_filing_type,
    normalize_ticker,
    parse_accession_from_sec_url,
    parse_cik_from_sec_url,
)

APPLE_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/"
    "000032019324000123/aapl-20240928.htm"
)
MSFT_URL = (
    "https://www.sec.gov/Archives/edgar/data/789019/"
    "000095017025100235/msft-20250630.htm"
)


class HelperTests(unittest.TestCase):
    def test_normalize_cik_from_string(self):
        self.assertEqual(normalize_cik("320193"), "0000320193")

    def test_normalize_cik_from_int_and_padded(self):
        self.assertEqual(normalize_cik(789019), "0000789019")
        self.assertEqual(normalize_cik("0000789019"), "0000789019")

    def test_normalize_cik_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            normalize_cik("abc")

    def test_normalize_cik_rejects_too_long(self):
        with self.assertRaises(ValueError):
            normalize_cik("123456789012")

    def test_format_accession_from_plain_and_idempotent(self):
        self.assertEqual(
            format_accession("000032019324000123"), "0000320193-24-000123"
        )
        self.assertEqual(
            format_accession("0000320193-24-000123"), "0000320193-24-000123"
        )

    def test_format_accession_rejects_garbage(self):
        with self.assertRaises(ValueError):
            format_accession("not-an-accession")

    def test_accession_to_filing_id_roundtrip(self):
        self.assertEqual(
            accession_to_filing_id("0000320193-24-000123"), "000032019324000123"
        )

    def test_derive_fiscal_year(self):
        self.assertEqual(derive_fiscal_year("2024-09-28"), 2024)
        self.assertEqual(derive_fiscal_year("2026-01-25"), 2026)

    def test_derive_fiscal_year_rejects_bad_date(self):
        with self.assertRaises(ValueError):
            derive_fiscal_year("2024/09/28")

    def test_normalize_ticker(self):
        self.assertEqual(normalize_ticker(" aapl "), "AAPL")
        with self.assertRaises(ValueError):
            normalize_ticker("")

    def test_normalize_filing_type(self):
        self.assertEqual(normalize_filing_type("10k"), "10-K")
        self.assertEqual(normalize_filing_type("10-Q"), "10-Q")
        with self.assertRaises(ValueError):
            normalize_filing_type("S-1")  # outside product scope, strict=True

    def test_parse_from_sec_url(self):
        self.assertEqual(parse_cik_from_sec_url(APPLE_URL), "0000320193")
        self.assertEqual(
            parse_accession_from_sec_url(APPLE_URL), "0000320193-24-000123"
        )
        # Microsoft's accession prefix is the filing agent, not Microsoft's CIK.
        self.assertEqual(parse_cik_from_sec_url(MSFT_URL), "0000789019")
        self.assertEqual(
            parse_accession_from_sec_url(MSFT_URL), "0000950170-25-100235"
        )

    def test_canonical_chunk_id(self):
        self.assertEqual(
            canonical_chunk_id(
                ticker="AAPL",
                fiscal_year=2024,
                filing_type="10-K",
                accession_number="0000320193-24-000123",
                chunk_index=28,
            ),
            "AAPL_2024_10-K_000032019324000123_28",
        )

    def test_canonical_chunk_id_rejects_negative_index(self):
        with self.assertRaises(ValueError):
            canonical_chunk_id(
                ticker="AAPL",
                fiscal_year=2024,
                filing_type="10-K",
                accession_number="0000320193-24-000123",
                chunk_index=-1,
            )


def _apple_kwargs(**overrides):
    kwargs = dict(
        cik="320193",
        accession_number="000032019324000123",
        chunk_id="apple_10k_0",
        chunk_index=0,
        ticker="aapl",
        filing_type="10-K",
        filing_date="2024-11-01",
        report_date="2024-09-28",
        company_name="Apple Inc.",
        source_url=APPLE_URL,
    )
    kwargs.update(overrides)
    return kwargs


class ChunkMetadataTests(unittest.TestCase):
    def test_valid_from_parts(self):
        meta = ChunkMetadata.from_parts(**_apple_kwargs())
        self.assertEqual(meta.cik, "0000320193")
        self.assertEqual(meta.accession_number, "0000320193-24-000123")
        self.assertEqual(meta.filing_id, "000032019324000123")
        self.assertEqual(meta.ticker, "AAPL")
        self.assertEqual(meta.fiscal_year, 2024)  # derived from report_date

    def test_explicit_fiscal_year_wins(self):
        meta = ChunkMetadata.from_parts(**_apple_kwargs(fiscal_year=2024))
        self.assertEqual(meta.fiscal_year, 2024)

    def test_to_pinecone_metadata_shape(self):
        meta = ChunkMetadata.from_parts(**_apple_kwargs())
        payload = meta.to_pinecone_metadata(text="hello")
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["fiscal_year"], 2024)
        self.assertIsInstance(payload["chunk_index"], int)
        self.assertNotIn(None, payload.values())

    def test_to_chunk_record_carries_text(self):
        meta = ChunkMetadata.from_parts(**_apple_kwargs())
        record = meta.to_chunk_record("body")
        self.assertEqual(record["text"], "body")
        self.assertEqual(record["report_date"], "2024-09-28")

    def test_rejects_filing_date_before_report_date(self):
        with self.assertRaises(ValueError):
            ChunkMetadata.from_parts(**_apple_kwargs(filing_date="2024-01-01"))

    def test_rejects_bad_date_format(self):
        with self.assertRaises(ValueError):
            ChunkMetadata.from_parts(**_apple_kwargs(report_date="Sept 28 2024"))

    def test_rejects_negative_chunk_index(self):
        with self.assertRaises(ValueError):
            ChunkMetadata.from_parts(**_apple_kwargs(chunk_index=-1))

    def test_rejects_non_sec_url(self):
        with self.assertRaises(ValueError):
            ChunkMetadata.from_parts(
                **_apple_kwargs(source_url="https://example.com/aapl.htm")
            )

    def test_rejects_empty_ticker(self):
        with self.assertRaises(ValueError):
            ChunkMetadata.from_parts(**_apple_kwargs(ticker="   "))

    def test_rejects_out_of_range_fiscal_year(self):
        with self.assertRaises(ValueError):
            ChunkMetadata.from_parts(**_apple_kwargs(fiscal_year=1200))

    def test_direct_construction_requires_normalized_values(self):
        with self.assertRaises(ValueError):
            ChunkMetadata(
                cik="320193",  # not padded
                accession_number="0000320193-24-000123",
                filing_id="000032019324000123",
                chunk_id="apple_10k_0",
                chunk_index=0,
                ticker="AAPL",
                filing_type="10-K",
                fiscal_year=2024,
                filing_date="2024-11-01",
                report_date="2024-09-28",
                company_name="Apple Inc.",
                source_url=APPLE_URL,
            )


if __name__ == "__main__":
    unittest.main()
