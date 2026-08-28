"""Canonical chunk metadata schema for SEC filings.

This module defines the metadata contract that *newly ingested* filings will
use from Phase 1B onward. It is pure and importable with no side effects and
makes no network calls.

Terminology (this is a deliberate fix of an existing naming bug)
--------------------------------------------------------------
- ``filing_date``  = the date the filing was actually submitted to the SEC
                     (EDGAR "Filing Date"). Example: Apple's FY2024 10-K was
                     submitted on 2024-11-01.
- ``report_date``  = the end of the fiscal period the filing covers
                     (EDGAR "Period of Report"). Example: 2024-09-28.

The current code base (ingestion/sec_loader.py, ingestion/chunker.py,
retrieval/*_store.py, retrieval/*_search.py, api/rag.py) stores a single field
called ``filing_date`` whose value is actually the *report date*. Phase 1A does
not rewrite that field in the running system; it only introduces the correct
vocabulary here and lets scripts/backfill_metadata.py add ``report_date`` (and
optionally correct ``filing_date``) as an explicit, reversible step.

Chunk IDs
---------
Existing seed chunks keep their legacy IDs (``apple_10k_28`` etc.) forever.
This module does NOT migrate them.

Filings ingested after Phase 1A use the canonical, collision-proof format::

    {TICKER}_{FISCAL_YEAR}_{FILING_TYPE}_{ACCESSION_NODASH}_{CHUNK_INDEX}

e.g. ``AAPL_2024_10-K_000032019324000123_28``.

The accession number alone already guarantees global uniqueness; the readable
prefix exists only for debuggability in logs and evaluation files. Treat the
whole string as opaque everywhere else.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

CANONICAL_CHUNK_ID_FORMAT = (
    "{ticker}_{fiscal_year}_{filing_type}_{accession_nodash}_{chunk_index}"
)

# Product scope for Phase 1. Amendments (e.g. "10-K/A") are accepted too.
KNOWN_FILING_TYPES = frozenset({"10-K", "10-Q", "8-K"})

# EDGAR's modern electronic-filing era. Used only as a sanity bound.
MIN_FISCAL_YEAR = 1994
MAX_FISCAL_YEAR = 2100

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ACCESSION_DASHED_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_ACCESSION_PLAIN_RE = re.compile(r"^\d{18}$")
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]*$")
_FILING_TYPE_RE = re.compile(r"^\d{1,2}-[A-Z]{1,3}(/A)?$")
_SEC_URL_CIK_RE = re.compile(r"/data/(\d{1,10})/")
_SEC_URL_ACCESSION_RE = re.compile(r"/data/\d{1,10}/(\d{18})/")

SEC_URL_PREFIX = "https://www.sec.gov/"


# --------------------------------------------------------------------------- #
# Pure normalization / derivation helpers                                     #
# --------------------------------------------------------------------------- #
def normalize_cik(value: str | int) -> str:
    """Return a 10-digit zero-padded CIK string.

    Accepts an int, a bare digit string, or a already-padded string.
    """
    if isinstance(value, bool):
        raise TypeError("CIK must not be a bool")

    text = str(value).strip()

    if text.upper().startswith("CIK"):
        text = text[3:]

    text = text.lstrip("0") or "0"

    if not text.isdigit():
        raise ValueError(f"CIK is not numeric: {value!r}")

    if len(text) > 10:
        raise ValueError(f"CIK has more than 10 digits: {value!r}")

    return text.zfill(10)


def format_accession(value: str) -> str:
    """Return the canonical dashed accession number ``NNNNNNNNNN-NN-NNNNNN``.

    Accepts either the dashed form or the 18-digit form.
    """
    text = str(value).strip()

    if _ACCESSION_DASHED_RE.match(text):
        return text

    if _ACCESSION_PLAIN_RE.match(text):
        return f"{text[:10]}-{text[10:12]}-{text[12:]}"

    raise ValueError(f"not a valid accession number: {value!r}")


def accession_to_filing_id(accession: str) -> str:
    """Return the 18-digit dashless accession ("filing_id")."""
    return format_accession(accession).replace("-", "")


def derive_fiscal_year(report_date: str) -> int:
    """Derive the fiscal year from the fiscal-period-end date.

    Uses the calendar year in which the fiscal period ends. This is correct for
    the three seed companies (AAPL, MSFT, NVDA) and for the large majority of
    filers. Companies whose fiscal year ends in January/February and who label
    that year with the *prior* calendar year (some retailers) are exceptions;
    ingestion must pass an explicit ``fiscal_year`` for those rather than rely
    on this helper.
    """
    validate_iso_date(report_date, field="report_date")
    return int(report_date[:4])


def validate_iso_date(value: str, *, field: str = "date") -> str:
    text = str(value).strip()
    if not _ISO_DATE_RE.match(text):
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}")
    try:
        _dt.date.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - message passthrough
        raise ValueError(f"{field} is not a real date: {value!r}") from exc
    return text


def normalize_ticker(value: str) -> str:
    text = str(value).strip().upper()
    if not _TICKER_RE.match(text):
        raise ValueError(f"not a valid ticker: {value!r}")
    return text


def normalize_filing_type(value: str, *, strict: bool = True) -> str:
    """Uppercase/trim a filing type and insert a missing hyphen (``10K`` -> ``10-K``)."""
    text = str(value).strip().upper().replace(" ", "")
    match = re.match(r"^(\d{1,2})-?([A-Z]{1,3})(/A)?$", text)
    if match:
        text = f"{match.group(1)}-{match.group(2)}{match.group(3) or ''}"
    if strict:
        base = text[:-2] if text.endswith("/A") else text
        if base not in KNOWN_FILING_TYPES:
            raise ValueError(
                f"unknown filing_type {value!r}; known: {sorted(KNOWN_FILING_TYPES)}"
            )
    elif not _FILING_TYPE_RE.match(text):
        raise ValueError(f"not a valid filing_type: {value!r}")
    return text


def parse_cik_from_sec_url(url: str) -> str:
    match = _SEC_URL_CIK_RE.search(str(url))
    if not match:
        raise ValueError(f"no CIK segment in SEC URL: {url!r}")
    return normalize_cik(match.group(1))


def parse_accession_from_sec_url(url: str) -> str:
    match = _SEC_URL_ACCESSION_RE.search(str(url))
    if not match:
        raise ValueError(f"no accession segment in SEC URL: {url!r}")
    return format_accession(match.group(1))


def canonical_chunk_id(
    *,
    ticker: str,
    fiscal_year: int,
    filing_type: str,
    accession_number: str,
    chunk_index: int,
) -> str:
    """Build the canonical chunk ID for a newly ingested filing."""
    if int(chunk_index) < 0:
        raise ValueError("chunk_index must be >= 0")
    return CANONICAL_CHUNK_ID_FORMAT.format(
        ticker=normalize_ticker(ticker),
        fiscal_year=int(fiscal_year),
        filing_type=normalize_filing_type(filing_type, strict=False),
        accession_nodash=accession_to_filing_id(accession_number),
        chunk_index=int(chunk_index),
    )


# --------------------------------------------------------------------------- #
# Canonical record                                                            #
# --------------------------------------------------------------------------- #
# Fields written into Pinecone metadata / chunk JSONL. ``text`` is carried
# alongside separately and is not part of the identity/filter contract.
_METADATA_FIELDS = (
    # stable identifiers
    "cik",
    "accession_number",
    "filing_id",
    "chunk_id",
    "chunk_index",
    # filters
    "ticker",
    "filing_type",
    "fiscal_year",
    "filing_date",
    "report_date",
    # display
    "company_name",
    "source_url",
)


@dataclass(frozen=True)
class ChunkMetadata:
    """Validated canonical metadata for a single chunk of a SEC filing.

    Construction validates every field and raises ``ValueError`` /
    ``TypeError`` on bad input. Use :meth:`from_parts` for light normalization
    of raw inputs.
    """

    cik: str
    accession_number: str
    filing_id: str
    chunk_id: str
    chunk_index: int
    ticker: str
    filing_type: str
    fiscal_year: int
    filing_date: str  # actual SEC submission date (YYYY-MM-DD)
    report_date: str  # fiscal period end (YYYY-MM-DD)
    company_name: str
    source_url: str

    def __post_init__(self) -> None:
        if self.cik != normalize_cik(self.cik):
            raise ValueError(f"cik is not normalized: {self.cik!r}")
        if self.accession_number != format_accession(self.accession_number):
            raise ValueError(
                f"accession_number is not normalized: {self.accession_number!r}"
            )
        if self.filing_id != accession_to_filing_id(self.accession_number):
            raise ValueError(
                "filing_id does not match accession_number "
                f"({self.filing_id!r} vs {self.accession_number!r})"
            )
        if not isinstance(self.chunk_index, int) or isinstance(self.chunk_index, bool):
            raise TypeError("chunk_index must be an int")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if not self.chunk_id or not str(self.chunk_id).strip():
            raise ValueError("chunk_id must be a non-empty string")
        if self.ticker != normalize_ticker(self.ticker):
            raise ValueError(f"ticker is not normalized: {self.ticker!r}")
        if self.filing_type != normalize_filing_type(self.filing_type, strict=False):
            raise ValueError(f"filing_type is not normalized: {self.filing_type!r}")
        if not isinstance(self.fiscal_year, int) or isinstance(self.fiscal_year, bool):
            raise TypeError("fiscal_year must be an int")
        if not (MIN_FISCAL_YEAR <= self.fiscal_year <= MAX_FISCAL_YEAR):
            raise ValueError(
                f"fiscal_year {self.fiscal_year} outside "
                f"[{MIN_FISCAL_YEAR}, {MAX_FISCAL_YEAR}]"
            )
        validate_iso_date(self.filing_date, field="filing_date")
        validate_iso_date(self.report_date, field="report_date")
        if self.filing_date < self.report_date:
            raise ValueError(
                "filing_date precedes report_date "
                f"({self.filing_date} < {self.report_date})"
            )
        if not self.company_name or not str(self.company_name).strip():
            raise ValueError("company_name must be a non-empty string")
        if not str(self.source_url).startswith(SEC_URL_PREFIX):
            raise ValueError(
                f"source_url must start with {SEC_URL_PREFIX!r}: {self.source_url!r}"
            )

    @classmethod
    def from_parts(
        cls,
        *,
        cik: str | int,
        accession_number: str,
        chunk_id: str,
        chunk_index: int,
        ticker: str,
        filing_type: str,
        filing_date: str,
        report_date: str,
        company_name: str,
        source_url: str,
        fiscal_year: int | None = None,
    ) -> "ChunkMetadata":
        """Normalize raw inputs then build a validated record."""
        accession = format_accession(accession_number)
        return cls(
            cik=normalize_cik(cik),
            accession_number=accession,
            filing_id=accession_to_filing_id(accession),
            chunk_id=str(chunk_id),
            chunk_index=int(chunk_index),
            ticker=normalize_ticker(ticker),
            filing_type=normalize_filing_type(filing_type, strict=False),
            fiscal_year=(
                int(fiscal_year)
                if fiscal_year is not None
                else derive_fiscal_year(report_date)
            ),
            filing_date=validate_iso_date(filing_date, field="filing_date"),
            report_date=validate_iso_date(report_date, field="report_date"),
            company_name=str(company_name).strip(),
            source_url=str(source_url).strip(),
        )

    def to_metadata(self) -> dict:
        """Return the canonical metadata mapping (no ``text``)."""
        return {field: getattr(self, field) for field in _METADATA_FIELDS}

    def to_pinecone_metadata(self, text: str | None = None) -> dict:
        """Return a Pinecone-safe metadata mapping.

        All values are ``str``/``int`` (Pinecone-compatible). ``text`` is
        included only when supplied.
        """
        payload = {
            key: value
            for key, value in self.to_metadata().items()
            if value is not None
        }
        if text is not None:
            payload["text"] = text
        return payload

    def to_chunk_record(self, text: str) -> dict:
        record = self.to_metadata()
        record["text"] = text
        return record


def metadata_fields() -> tuple[str, ...]:
    """Names of the canonical metadata fields, in declaration order."""
    return _METADATA_FIELDS
