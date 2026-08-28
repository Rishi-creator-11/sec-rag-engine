"""Phase 4 Batch 3: explicit --cik override for ticker->registrant successions.

Covers:
  - --cik override discovers from the explicit CIK's SEC submissions
  - explicit CIK is validated (bad CIK rejected)
  - the override is never silent (logged + cik_override flag on the filing)
  - the filing's own registrant metadata stays SEC-authentic (not rewritten to
    the successor entity)
  - retrieval scope stays the product ticker (chunks + canonical ids use it)
  - ordinary tickers with no --cik still use normal ticker->CIK resolution
  - successor lineage is recorded in the registry, distinct from the registrant
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion import registry
from ingestion.sec_client import SecClient, SecNotFoundError
from ingestion.chunk_records import build_chunk_records
from retrieval.filters import RetrievalFilter
from retrieval.metadata import canonical_chunk_id

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
UA = "sec-rag-engine test@example.com"

# SEC company_tickers.json after the reorg: XOM points at the successor holdco.
COMPANY_TICKERS = {
    "0": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
    "1": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"},
}

# Legacy operating entity (CIK 34088) — still has the real FY2025 10-K.
XOM_LEGACY_SUBMISSIONS = {
    "name": "EXXON MOBIL CORP",
    "cik": "34088",
    "tickers": [],           # delisted — ticker moved to the successor
    "exchanges": [],
    "filings": {
        "recent": {
            "form": ["25-NSE", "10-Q", "10-K", "10-K"],
            "accessionNumber": [
                "0000034088-26-000060",
                "0000034088-26-000050",
                "0000034088-26-000045",
                "0000034088-25-000010",
            ],
            "filingDate": ["2026-07-02", "2026-05-05", "2026-02-18", "2025-02-19"],
            "reportDate": ["", "2026-03-31", "2025-12-31", "2024-12-31"],
            "primaryDocument": ["", "q.htm", "xom-20251231.htm", "xom-20241231.htm"],
            "primaryDocDescription": ["25-NSE", "10-Q", "10-K", "10-K"],
        }
    },
}

# Successor holdco (CIK 2115436) — no 10-K yet.
XOM_SUCCESSOR_SUBMISSIONS = {
    "name": "ExxonMobil Holdings Corp",
    "cik": "2115436",
    "tickers": ["XOM"],
    "exchanges": ["NYSE"],
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "S-8 POS"],
            "accessionNumber": [
                "0002115436-26-000010",
                "0002115436-26-000008",
                "0002115436-26-000004",
            ],
            "filingDate": ["2026-08-28", "2026-08-03", "2026-07-01"],
            "reportDate": ["2026-08-28", "2026-06-30", "2026-07-01"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            "primaryDocDescription": ["8-K", "10-Q", "S-8 POS"],
        }
    },
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.routes = {url: list(items) for url, items in routes.items()}
        self.calls = []
        self.headers = {}

    def get(self, url, timeout=None):
        self.calls.append(url)
        queue = self.routes.get(url)
        if not queue:
            return FakeResponse(404, text="not found")
        return queue.pop(0) if len(queue) > 1 else queue[0]


def make_client(routes):
    return SecClient(user_agent=UA, session=FakeSession(routes),
                     cache_dir="/tmp/sec-rag-nonexistent-cache")


LEGACY_CIK = "https://data.sec.gov/submissions/CIK0000034088.json"
SUCCESSOR_CIK = "https://data.sec.gov/submissions/CIK0002115436.json"
AMZN_CIK = "https://data.sec.gov/submissions/CIK0001018724.json"

AMZN_SUBMISSIONS = {
    "name": "AMAZON COM INC",
    "cik": "1018724",
    "tickers": ["AMZN"],
    "filings": {"recent": {
        "form": ["10-K"],
        "accessionNumber": ["0001018724-26-000004"],
        "filingDate": ["2026-02-06"],
        "reportDate": ["2025-12-31"],
        "primaryDocument": ["amzn-20251231.htm"],
    }},
}


class CikOverrideDiscoveryTests(unittest.TestCase):
    def _client(self):
        return make_client({
            TICKERS_URL: [FakeResponse(200, COMPANY_TICKERS)],
            LEGACY_CIK: [FakeResponse(200, XOM_LEGACY_SUBMISSIONS)],
            SUCCESSOR_CIK: [FakeResponse(200, XOM_SUCCESSOR_SUBMISSIONS)],
            AMZN_CIK: [FakeResponse(200, AMZN_SUBMISSIONS)],
        })

    def test_no_override_uses_ticker_resolution_and_finds_no_10k(self):
        # XOM without --cik resolves to the successor holdco, which has no 10-K.
        with self.assertRaises(SecNotFoundError):
            self._client().discover_latest_10k("XOM")

    def test_ordinary_ticker_still_uses_normal_resolution(self):
        filing = self._client().discover_latest_10k("AMZN")
        self.assertEqual(filing.cik, "0001018724")
        self.assertEqual(filing.accession_number, "0001018724-26-000004")
        self.assertFalse(filing.cik_override)

    def test_cik_override_discovers_from_explicit_cik(self):
        filing = self._client().discover_latest_10k("XOM", cik="0000034088")
        self.assertEqual(filing.accession_number, "0000034088-26-000045")
        self.assertEqual(filing.primary_document, "xom-20251231.htm")
        self.assertEqual(filing.fiscal_year, 2025)
        self.assertEqual(filing.report_date, "2025-12-31")
        self.assertIn("/data/34088/", filing.source_url)

    def test_override_carries_the_requested_ticker_as_scope(self):
        filing = self._client().discover_latest_10k("xom", cik="0000034088")
        self.assertEqual(filing.ticker, "XOM")  # normalized product ticker

    def test_registrant_metadata_stays_sec_authentic(self):
        filing = self._client().discover_latest_10k("XOM", cik="0000034088")
        # NOT rewritten to the successor:
        self.assertEqual(filing.company_name, "EXXON MOBIL CORP")
        self.assertEqual(filing.cik, "0000034088")

    def test_override_is_not_silent(self):
        with self.assertLogs("ingestion.sec_client", level=logging.WARNING) as cm:
            filing = self._client().discover_latest_10k("XOM", cik="0000034088")
        joined = "\n".join(cm.output)
        self.assertIn("event=cik_override", joined)
        self.assertIn("0000034088", joined)
        self.assertTrue(filing.cik_override)

    def test_override_warns_on_ticker_registrant_mismatch(self):
        # legacy CIK 34088 has tickers=[] so no mismatch line; force a populated
        # tickers list that does not contain XOM to exercise the warning.
        subs = json.loads(json.dumps(XOM_LEGACY_SUBMISSIONS))
        subs["tickers"] = ["XOMX"]
        client = make_client({
            TICKERS_URL: [FakeResponse(200, COMPANY_TICKERS)],
            LEGACY_CIK: [FakeResponse(200, subs)],
        })
        with self.assertLogs("ingestion.sec_client", level=logging.WARNING) as cm:
            client.discover_latest_10k("XOM", cik="0000034088")
        self.assertIn("cik_override_ticker_mismatch", "\n".join(cm.output))

    def test_bad_override_cik_rejected(self):
        with self.assertRaises(ValueError):
            self._client().discover_latest_10k("XOM", cik="not-a-cik")

    def test_override_records_successor_lineage_on_filing(self):
        filing = self._client().discover_latest_10k(
            "XOM", cik="0000034088",
            successor_cik="0002115436",
            successor_name="ExxonMobil Holdings Corporation",
            successor_effective_date="2026-07-01",
        )
        self.assertEqual(filing.successor_cik, "0002115436")
        self.assertEqual(filing.successor_name, "ExxonMobil Holdings Corporation")
        self.assertEqual(filing.successor_effective_date, "2026-07-01")

    def test_override_rejects_bad_successor_date(self):
        with self.assertRaises(ValueError):
            self._client().discover_latest_10k(
                "XOM", cik="0000034088", successor_effective_date="July 2026",
            )


class CikOverrideRetrievalScopeTests(unittest.TestCase):
    """The legacy FY2025 filing is reachable under the product ticker XOM."""

    def _filing(self):
        from ingestion.sec_client import DiscoveredFiling
        return DiscoveredFiling(
            company_name="EXXON MOBIL CORP",
            ticker="XOM",
            cik="0000034088",
            filing_type="10-K",
            fiscal_year=2025,
            accession_number="0000034088-26-000045",
            filing_id="000003408826000045",
            filing_date="2026-02-18",
            report_date="2025-12-31",
            primary_document="xom-20251231.htm",
            source_url="https://www.sec.gov/Archives/edgar/data/34088/"
            "000003408826000045/xom-20251231.htm",
            cik_override=True,
            successor_cik="0002115436",
            successor_name="ExxonMobil Holdings Corporation",
            successor_effective_date="2026-07-01",
        )

    def test_canonical_chunk_id_uses_product_ticker(self):
        cid = canonical_chunk_id(
            ticker="XOM", fiscal_year=2025, filing_type="10-K",
            accession_number="0000034088-26-000045", chunk_index=0,
        )
        self.assertTrue(cid.startswith("XOM_2025_10-K_000003408826000045_"))

    def test_chunks_are_tagged_with_xom_and_match_the_xom_filter(self):
        clean = (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 10-K "
            "ANNUAL REPORT. Exxon Mobil Corporation reported total revenues and "
            "described risks in energy markets, regulation, and climate policy. "
        ) * 200
        records = build_chunk_records(clean, self._filing())
        self.assertTrue(records)
        xom = RetrievalFilter(tickers=("XOM",))
        for r in records:
            self.assertEqual(r["ticker"], "XOM")
            self.assertTrue(xom.matches(r))
            self.assertNotIn("2115436", r["chunk_id"])  # not the successor CIK
        self.assertTrue(all(
            r["chunk_id"].startswith("XOM_2025_10-K_000003408826000045_")
            for r in records
        ))


class CikOverrideRegistryLineageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "companies.json"
        self.path.write_text(json.dumps({
            "schema_version": 2,
            "companies": [{"ticker": "AAPL", "legal_name": "Apple Inc.",
                           "display_name": "Apple Inc.", "cik": "0000320193"}],
        }), encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)
        p = patch.object(registry, "REGISTRY_PATH", self.path)
        p.start()
        registry.reload()
        self.addCleanup(p.stop)
        self.addCleanup(registry.reload)

    def _raw_xom(self):
        return next(c for c in json.loads(self.path.read_text())["companies"]
                    if c["ticker"] == "XOM")

    def test_lineage_written_and_distinct_from_registrant(self):
        registry.upsert_company(
            "XOM", legal_name="EXXON MOBIL CORP", cik="0000034088",
            lineage={
                "cik_override": True,
                "registrant_cik": "0000034088",
                "registrant_legal_name": "EXXON MOBIL CORP",
                "successor_cik": "0002115436",
                "successor_legal_name": "ExxonMobil Holdings Corporation",
                "successor_effective_date": "2026-07-01",
                "note": "ticker moved to successor",
            },
        )
        entry = self._raw_xom()
        self.assertEqual(entry["cik"], "0000034088")           # registrant
        self.assertEqual(entry["legal_name"], "EXXON MOBIL CORP")
        self.assertEqual(entry["lineage"]["successor_cik"], "0002115436")
        self.assertEqual(entry["lineage"]["successor_effective_date"], "2026-07-01")
        # surfaced through the read API
        self.assertEqual(
            registry.get_company("XOM")["lineage"]["successor_cik"], "0002115436"
        )

    def test_lineage_none_does_not_clear_existing(self):
        registry.upsert_company("XOM", legal_name="EXXON MOBIL CORP",
                                cik="0000034088",
                                lineage={"successor_cik": "0002115436"})
        registry.upsert_company("XOM", legal_name="EXXON MOBIL CORP",
                                cik="0000034088")  # plain re-ingest, no lineage
        self.assertEqual(self._raw_xom()["lineage"]["successor_cik"], "0002115436")

    def test_ordinary_company_has_no_lineage_key(self):
        registry.upsert_company("AMZN", legal_name="AMAZON COM INC",
                                cik="0001018724")
        entry = next(c for c in json.loads(self.path.read_text())["companies"]
                     if c["ticker"] == "AMZN")
        self.assertNotIn("lineage", entry)


class CikOverridePlumbingTests(unittest.TestCase):
    """--cik flows through ingest_company() and the CLI to the SEC client."""

    def test_ingest_company_passes_cik_and_lineage_to_discovery(self):
        from ingestion import ingest_company as ic

        seen = {}

        class RecordingClient:
            def discover_latest_10k(self, ticker, **kwargs):
                seen["ticker"] = ticker
                seen.update(kwargs)
                raise SecNotFoundError("stop after discovery")  # short-circuit

        with self.assertRaises(SecNotFoundError):
            ic.ingest_company(
                "XOM", dry_run=True, client=RecordingClient(),
                cik="0000034088", successor_cik="0002115436",
                successor_name="ExxonMobil Holdings Corporation",
                successor_effective_date="2026-07-01",
            )
        self.assertEqual(seen["cik"], "0000034088")
        self.assertEqual(seen["successor_cik"], "0002115436")
        self.assertEqual(seen["successor_effective_date"], "2026-07-01")

    def test_cli_successor_flags_require_cik(self):
        from ingestion import ingest_company as ic

        with self.assertRaises(SystemExit):
            ic.main(["--ticker", "XOM", "--successor-cik", "0002115436"])

    def test_cli_plain_ticker_needs_no_cik(self):
        # argparse should accept a bare --ticker (no --cik) without error;
        # we stop before any network by passing --dry-run and a patched client.
        from ingestion import ingest_company as ic

        class Boom:
            def discover_latest_10k(self, ticker, **kwargs):
                raise SecNotFoundError("network reached with cik=%r" % kwargs.get("cik"))

        with patch.object(ic, "SecClient", lambda *a, **k: Boom()):
            rc = ic.main(["--ticker", "AMZN", "--dry-run"])
        self.assertEqual(rc, 2)  # aborted at discovery, but argparse was happy


if __name__ == "__main__":
    unittest.main()
