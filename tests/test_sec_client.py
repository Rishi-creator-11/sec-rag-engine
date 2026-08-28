"""Phase 3: SEC client — resolution, discovery, retry/backoff, config."""

import json
import unittest
from unittest.mock import patch

import requests

from ingestion.sec_client import (
    SecClient,
    SecConfigError,
    SecNotFoundError,
    SecRateLimitError,
    SecTransientError,
)

UA = "sec-rag-engine test@example.com"

COMPANY_TICKERS = {
    "0": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

AMZN_SUBMISSIONS = {
    "name": "AMAZON COM INC",
    "cik": "1018724",
    "tickers": ["AMZN"],
    "filings": {
        "recent": {
            "form": ["8-K", "10-K/A", "10-K", "10-K"],
            "accessionNumber": [
                "0001018724-26-000010",
                "0001018724-26-000008",
                "0001018724-26-000004",
                "0001018724-25-000004",
            ],
            "filingDate": ["2026-03-01", "2026-02-20", "2026-02-06", "2025-02-07"],
            "reportDate": ["2026-02-01", "2025-12-31", "2025-12-31", "2024-12-31"],
            "primaryDocument": ["a.htm", "b.htm", "amzn-20251231.htm", "amzn-20241231.htm"],
            "primaryDocDescription": ["8-K", "10-K/A", "10-K", "10-K"],
        }
    },
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._payload


class FakeSession:
    """Serves a queue of responses per URL (last one repeats)."""

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


def make_client(routes, **kw):
    return SecClient(user_agent=UA, session=FakeSession(routes),
                     cache_dir="/tmp/sec-rag-nonexistent-cache", **kw)


class ConfigTests(unittest.TestCase):
    def test_missing_user_agent_raises(self):
        with patch.dict("os.environ", {"SEC_USER_AGENT": ""}, clear=False):
            with self.assertRaises(SecConfigError):
                SecClient(user_agent=None)

    def test_user_agent_without_contact_raises(self):
        with self.assertRaises(SecConfigError):
            SecClient(user_agent="sec-rag-engine")

    def test_max_rps_hard_capped(self):
        client = SecClient(user_agent=UA, max_rps=999, session=FakeSession({}))
        self.assertLessEqual(client.max_rps, 10.0)


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tickers_url = "https://www.sec.gov/files/company_tickers.json"

    def test_resolve_cik_pads_to_10(self):
        client = make_client({self.tickers_url: [FakeResponse(200, COMPANY_TICKERS)]})
        self.assertEqual(client.resolve_cik("amzn"), "0001018724")
        self.assertEqual(client.resolve_cik("AAPL"), "0000320193")

    def test_unknown_ticker_raises_not_found(self):
        client = make_client({self.tickers_url: [FakeResponse(200, COMPANY_TICKERS)]})
        with self.assertRaises(SecNotFoundError):
            client.resolve_cik("ZZZZ")


class DiscoveryTests(unittest.TestCase):
    def _client(self):
        return make_client({
            "https://www.sec.gov/files/company_tickers.json": [FakeResponse(200, COMPANY_TICKERS)],
            "https://data.sec.gov/submissions/CIK0001018724.json": [FakeResponse(200, AMZN_SUBMISSIONS)],
        })

    def test_latest_10k_skips_amendment_and_picks_newest(self):
        filing = self._client().discover_latest_10k("AMZN")
        self.assertEqual(filing.filing_type, "10-K")
        self.assertEqual(filing.accession_number, "0001018724-26-000004")  # not -26-000008 (10-K/A)
        self.assertEqual(filing.filing_id, "000101872426000004")
        self.assertEqual(filing.filing_date, "2026-02-06")
        self.assertEqual(filing.report_date, "2025-12-31")
        self.assertEqual(filing.fiscal_year, 2025)

    def test_filing_date_vs_report_date(self):
        filing = self._client().discover_latest_10k("AMZN")
        self.assertGreater(filing.filing_date, filing.report_date)

    def test_canonical_source_url(self):
        filing = self._client().discover_latest_10k("AMZN")
        self.assertEqual(
            filing.source_url,
            "https://www.sec.gov/Archives/edgar/data/1018724/"
            "000101872426000004/amzn-20251231.htm",
        )

    def test_accession_prefix_not_assumed_equal_to_cik(self):
        subs = json.loads(json.dumps(AMZN_SUBMISSIONS))
        subs["filings"]["recent"]["accessionNumber"][2] = "0000950170-26-000004"  # agent-filed
        client = make_client({
            "https://www.sec.gov/files/company_tickers.json": [FakeResponse(200, COMPANY_TICKERS)],
            "https://data.sec.gov/submissions/CIK0001018724.json": [FakeResponse(200, subs)],
        })
        filing = client.discover_latest_10k("AMZN")
        self.assertEqual(filing.cik, "0001018724")                      # from CIK lookup
        self.assertEqual(filing.accession_number, "0000950170-26-000004")  # from SEC
        self.assertIn("/data/1018724/", filing.source_url)              # CIK, not accession prefix

    def test_no_10k_raises_not_found(self):
        subs = json.loads(json.dumps(AMZN_SUBMISSIONS))
        subs["filings"]["recent"]["form"] = ["8-K", "10-Q", "4", "3"]
        client = make_client({
            "https://www.sec.gov/files/company_tickers.json": [FakeResponse(200, COMPANY_TICKERS)],
            "https://data.sec.gov/submissions/CIK0001018724.json": [FakeResponse(200, subs)],
        })
        with self.assertRaises(SecNotFoundError):
            client.discover_latest_10k("AMZN")


class RetryTests(unittest.TestCase):
    URL = "https://data.sec.gov/submissions/CIK0001018724.json"

    @patch("ingestion.sec_client.time.sleep")
    def test_429_then_success(self, mock_sleep):
        client = make_client({self.URL: [
            FakeResponse(429, headers={"Retry-After": "2"}),
            FakeResponse(200, AMZN_SUBMISSIONS),
        ]})
        self.assertEqual(client.submissions("0001018724")["name"], "AMAZON COM INC")
        mock_sleep.assert_called()  # honored a wait

    @patch("ingestion.sec_client.time.sleep")
    def test_retry_after_header_used(self, mock_sleep):
        client = make_client({self.URL: [
            FakeResponse(429, headers={"Retry-After": "7"}),
            FakeResponse(200, AMZN_SUBMISSIONS),
        ]})
        client.submissions("0001018724")
        self.assertIn(7.0, [call.args[0] for call in mock_sleep.call_args_list])

    @patch("ingestion.sec_client.time.sleep")
    def test_5xx_then_success(self, mock_sleep):
        client = make_client({self.URL: [
            FakeResponse(503),
            FakeResponse(500),
            FakeResponse(200, AMZN_SUBMISSIONS),
        ]})
        self.assertEqual(client.submissions("0001018724")["cik"], "1018724")

    @patch("ingestion.sec_client.time.sleep")
    def test_persistent_429_raises_rate_limit(self, mock_sleep):
        client = make_client({self.URL: [FakeResponse(429)]}, max_attempts=3)
        with self.assertRaises(SecRateLimitError):
            client.submissions("0001018724")

    @patch("ingestion.sec_client.time.sleep")
    def test_persistent_5xx_raises_transient(self, mock_sleep):
        client = make_client({self.URL: [FakeResponse(502)]}, max_attempts=3)
        with self.assertRaises(SecTransientError):
            client.submissions("0001018724")

    @patch("ingestion.sec_client.time.sleep")
    def test_404_not_retried(self, mock_sleep):
        session = FakeSession({self.URL: [FakeResponse(404)]})
        client = SecClient(user_agent=UA, session=session, max_attempts=5,
                           cache_dir="/tmp/none")
        with self.assertRaises(SecNotFoundError):
            client.submissions("0001018724")
        self.assertEqual(len(session.calls), 1)  # no retry loop
        mock_sleep.assert_not_called()

    @patch("ingestion.sec_client.time.sleep")
    def test_connection_error_retried(self, mock_sleep):
        class Flaky(FakeSession):
            def __init__(self):
                super().__init__({})
                self.n = 0

            def get(self, url, timeout=None):
                self.calls.append(url)
                self.n += 1
                if self.n == 1:
                    raise requests.ConnectionError("boom")
                return FakeResponse(200, AMZN_SUBMISSIONS)

        client = SecClient(user_agent=UA, session=Flaky(), cache_dir="/tmp/none")
        self.assertEqual(client.submissions("0001018724")["name"], "AMAZON COM INC")


if __name__ == "__main__":
    unittest.main()
