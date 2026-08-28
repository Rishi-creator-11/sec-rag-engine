"""Authoritative SEC EDGAR client.

Resolves ticker -> CIK, fetches company submissions, discovers the latest
10-K, and downloads the primary filing document. Everything comes from
SEC-published JSON / documents; there is no HTML-scraping fallback.

Configuration (environment)
---------------------------
SEC_USER_AGENT   required for any network call, e.g.
                 "sec-rag-engine you@example.com"  (must contain a contact).
SEC_MAX_RPS      optional float, default 5.0, hard-capped at 10.0.

Politeness / robustness
-----------------------
- global minimum interval between requests (<= SEC_MAX_RPS, hard cap 10/s)
- honors Retry-After on 429
- exponential backoff with jitter on 429 / 5xx / connection errors
- permanent 4xx (e.g. 404) raise immediately, no retry
- connect/read timeouts
- one requests.Session, gzip enabled
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import requests

from retrieval.metadata import (
    derive_fiscal_year,
    format_accession,
    normalize_cik,
    normalize_ticker,
    validate_iso_date,
)

logger = logging.getLogger("ingestion.sec_client")

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_document}"
)

HARD_MAX_RPS = 10.0
DEFAULT_MAX_RPS = 5.0
DEFAULT_TIMEOUT = (5, 30)          # (connect, read) seconds
DEFAULT_MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0
COMPANY_TICKERS_TTL_SECONDS = 7 * 24 * 3600

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "sec_cache"


class SecClientError(RuntimeError):
    """Base class for SEC client failures."""


class SecConfigError(SecClientError):
    """SEC_USER_AGENT missing or malformed."""


class SecNotFoundError(SecClientError):
    """A permanent 404 (unknown ticker / missing filing)."""


class SecRateLimitError(SecClientError):
    """Exhausted retries against repeated HTTP 429."""


class SecTransientError(SecClientError):
    """Exhausted retries against 5xx / connection errors."""


@dataclass(frozen=True)
class DiscoveredFiling:
    company_name: str       # SEC-authoritative registrant legal name (EDGAR)
    ticker: str             # product / retrieval-scope ticker
    cik: str                # 10-digit — the registrant CIK that FILED this filing
    filing_type: str        # "10-K"
    fiscal_year: int
    accession_number: str   # dashed
    filing_id: str          # 18-digit (accession without dashes)
    filing_date: str        # SEC submission date (YYYY-MM-DD)
    report_date: str        # fiscal period end (YYYY-MM-DD)
    primary_document: str
    source_url: str
    # --- ticker->CIK lineage (only set when an explicit CIK override is used) ---
    cik_override: bool = False          # True => ticker->CIK discovery was bypassed
    successor_cik: str | None = None    # CIK the ticker currently resolves to, if different
    successor_name: str | None = None   # successor registrant legal name
    successor_effective_date: str | None = None  # YYYY-MM-DD the succession took effect

    def to_dict(self) -> dict:
        return asdict(self)


def _looks_like_contact(user_agent: str) -> bool:
    return "@" in user_agent and len(user_agent.strip()) >= 5


class SecClient:
    def __init__(
        self,
        user_agent: str | None = None,
        *,
        max_rps: float | None = None,
        session: requests.Session | None = None,
        cache_dir: str | Path | None = None,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "").strip()
        if not self.user_agent or not _looks_like_contact(self.user_agent):
            raise SecConfigError(
                "SEC_USER_AGENT is required and must include a contact address, "
                'e.g. SEC_USER_AGENT="sec-rag-engine you@example.com". '
                "Do not use placeholder contact details."
            )

        if max_rps is None:
            try:
                max_rps = float(os.getenv("SEC_MAX_RPS", DEFAULT_MAX_RPS))
            except ValueError:
                max_rps = DEFAULT_MAX_RPS
        self.max_rps = max(0.5, min(max_rps, HARD_MAX_RPS))
        self._min_interval = 1.0 / self.max_rps
        self._last_request_ts = 0.0

        self.timeout = timeout
        self.max_attempts = max_attempts

        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), BACKOFF_CAP_SECONDS)
            except ValueError:
                pass
        raw = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        return min(raw, BACKOFF_CAP_SECONDS) + random.uniform(0, 0.5)

    def _get(self, url: str) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.timeout)
                self._last_request_ts = time.monotonic()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                wait = self._backoff(attempt)
                logger.warning(
                    "sec_request event=connection_error url=%s attempt=%d/%d "
                    "retry_in=%.1fs error_class=%s",
                    url, attempt, self.max_attempts, wait, type(exc).__name__,
                )
                time.sleep(wait)
                continue

            status = response.status_code
            if status == 200:
                return response
            if status == 404:
                raise SecNotFoundError(f"404 Not Found: {url}")
            if status == 429:
                wait = self._backoff(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "sec_request event=rate_limited url=%s attempt=%d/%d retry_in=%.1fs",
                    url, attempt, self.max_attempts, wait,
                )
                last_exc = SecRateLimitError(f"429 Too Many Requests: {url}")
                time.sleep(wait)
                continue
            if 500 <= status < 600:
                wait = self._backoff(attempt)
                logger.warning(
                    "sec_request event=server_error url=%s status=%d attempt=%d/%d "
                    "retry_in=%.1fs",
                    url, status, attempt, self.max_attempts, wait,
                )
                last_exc = SecTransientError(f"{status} server error: {url}")
                time.sleep(wait)
                continue
            # Other permanent 4xx.
            raise SecClientError(f"{status} error for {url}: {response.text[:200]}")

        if isinstance(last_exc, SecClientError):
            raise last_exc
        raise SecTransientError(f"exhausted {self.max_attempts} attempts for {url}: {last_exc}")

    def _get_json(self, url: str) -> dict:
        response = self._get(url)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise SecClientError(f"expected JSON from {url}: {exc}") from exc

    # ------------------------------------------------------------------ #
    def company_tickers(self, *, force_refresh: bool = False) -> dict:
        cache_path = self.cache_dir / "company_tickers.json"
        if not force_refresh and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < COMPANY_TICKERS_TTL_SECONDS:
                return json.loads(cache_path.read_text(encoding="utf-8"))

        data = self._get_json(COMPANY_TICKERS_URL)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, cache_path)
        logger.info("sec_cache event=refreshed file=company_tickers.json entries=%d", len(data))
        return data

    def resolve_cik(self, ticker: str) -> str:
        target = normalize_ticker(ticker)
        for entry in self.company_tickers().values():
            if str(entry.get("ticker", "")).strip().upper() == target:
                return normalize_cik(entry["cik_str"])
        raise SecNotFoundError(f"ticker not found in SEC company_tickers: {target}")

    def submissions(self, cik10: str) -> dict:
        return self._get_json(SUBMISSIONS_URL.format(cik10=normalize_cik(cik10)))

    def latest_filing(
        self,
        cik10: str,
        *,
        form: str = "10-K",
        ticker: str | None = None,
        submissions: dict | None = None,
    ) -> DiscoveredFiling:
        cik10 = normalize_cik(cik10)
        subs = submissions if submissions is not None else self.submissions(cik10)
        recent = subs.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        required = ("accessionNumber", "filingDate", "reportDate", "primaryDocument")
        if not forms or any(key not in recent for key in required):
            raise SecClientError(f"submissions for CIK {cik10} missing 'recent' filing arrays")

        chosen_index = None
        for index, current_form in enumerate(forms):
            if current_form == form:  # exact match: excludes "10-K/A"
                chosen_index = index
                break

        if chosen_index is None:
            raise SecNotFoundError(f"no {form} in recent submissions for CIK {cik10}")

        accession = format_accession(recent["accessionNumber"][chosen_index])
        report_date = recent["reportDate"][chosen_index]
        filing_date = recent["filingDate"][chosen_index]
        primary_document = recent["primaryDocument"][chosen_index]

        if not report_date:
            raise SecClientError(f"{accession}: empty reportDate; cannot derive fiscal year")

        cik_int = int(cik10)
        accession_nodash = accession.replace("-", "")
        source_url = ARCHIVES_DOC_URL.format(
            cik_int=cik_int,
            accession_nodash=accession_nodash,
            primary_document=primary_document,
        )

        resolved_ticker = (
            normalize_ticker(ticker)
            if ticker
            else normalize_ticker((subs.get("tickers") or ["?"])[0])
        )

        return DiscoveredFiling(
            company_name=str(subs.get("name", "")).strip(),
            ticker=resolved_ticker,
            cik=cik10,
            filing_type=form,
            fiscal_year=derive_fiscal_year(report_date),
            accession_number=accession,
            filing_id=accession_nodash,
            filing_date=filing_date,
            report_date=report_date,
            primary_document=primary_document,
            source_url=source_url,
        )

    def discover_latest_10k(
        self,
        ticker: str,
        *,
        cik: str | None = None,
        successor_cik: str | None = None,
        successor_name: str | None = None,
        successor_effective_date: str | None = None,
    ) -> DiscoveredFiling:
        """Discover the latest 10-K for ``ticker``.

        Normally the CIK is resolved from SEC's ``company_tickers.json``. Pass an
        explicit ``cik`` to bypass *only* that resolution step — filing discovery
        still runs against SEC's submissions feed for that exact CIK, and the
        requested ticker is validated and carried through as the retrieval scope.
        This exists for ticker->registrant successions (e.g. a holding-company
        reorganization) where the ticker now points at an entity that has not yet
        filed the annual report. An explicit override is always logged and is
        never applied silently.

        ``successor_*`` are optional lineage annotations recorded on the filing
        and (by the caller) in the registry; they never alter the filing's own
        SEC-authoritative registrant metadata.
        """
        normalized = normalize_ticker(ticker)  # raises ValueError on a bad ticker

        if cik is None:
            cik10 = self.resolve_cik(normalized)
            return self.latest_filing(cik10, form="10-K", ticker=normalized)

        cik10 = normalize_cik(cik)  # raises ValueError on a bad CIK
        logger.warning(
            "sec_client event=cik_override ticker=%s cik=%s "
            "reason=explicit_flag note=ticker->CIK_discovery_bypassed",
            normalized, cik10,
        )
        subs = self.submissions(cik10)
        registrant_tickers = [
            normalize_ticker(value)
            for value in (subs.get("tickers") or [])
            if str(value).strip()
        ]
        if registrant_tickers and normalized not in registrant_tickers:
            logger.warning(
                "sec_client event=cik_override_ticker_mismatch ticker=%s cik=%s "
                "registrant_tickers=%s note=expected_if_ticker_moved_to_successor",
                normalized, cik10, ",".join(registrant_tickers),
            )

        filing = self.latest_filing(
            cik10, form="10-K", ticker=normalized, submissions=subs
        )

        eff_date = None
        if successor_effective_date:
            eff_date = validate_iso_date(
                successor_effective_date, field="successor_effective_date"
            )
        return replace(
            filing,
            cik_override=True,
            successor_cik=normalize_cik(successor_cik) if successor_cik else None,
            successor_name=(successor_name or None),
            successor_effective_date=eff_date,
        )

    def download_document(self, url: str) -> requests.Response:
        return self._get(url)
