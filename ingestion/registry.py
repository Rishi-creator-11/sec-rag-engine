"""Company registry — the small lookup that backs ticker validation.

``data/registry/companies.json`` is the single source of truth. There is no
database and no network call: the file is read once, cached in-process, and
served from memory. Later phases (automated ingestion) will grow the JSON file
and add a writer here; for Phase 1C this module is read-only.

Naming
------
- ``legal_name``   : SEC-authoritative company name (from EDGAR submissions).
                     Written automatically by ingestion. Never mutated by us.
- ``display_name`` : optional curated, human-friendly name. If absent, falls
                     back to ``legal_name``.
- ``name``         : the *effective* display name (display_name or legal_name).
                     This is what ``GET /companies`` returns. Kept for backward
                     compatibility with pre-3.5 callers.

Display name is NOT used for retrieval filtering — only ``ticker`` is.

Public API
----------
    list_companies()          -> list[dict]  (ticker, name, legal_name,
                                              display_name, cik, filings; sorted)
    get_company(ticker)       -> dict | None (case-insensitive)
    is_known_ticker(ticker)   -> bool        (case-insensitive)
    known_tickers()           -> set[str]    (normalized, uppercase)
    normalize_ticker(value)   -> str         (trim + uppercase)
    partition_tickers(values) -> (known: list[str], unknown: list[str])
    reload()                  -> None        (drop the cache; for tests / hot edits)
    upsert_company(ticker, legal_name, cik, display_name=None) -> add/update (atomic)
    record_filing(ticker, filing)            -> add a filing, dedup by accession

Writes are atomic (temp -> flush -> fsync -> os.replace) and idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

from ingestion.atomicio import atomic_write_json

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "registry" / "companies.json"
)
_FILING_FIELDS = (
    "filing_type",
    "fiscal_year",
    "accession_number",
    "filing_date",
    "report_date",
    "chunk_count",
)

_cache: dict[str, dict] | None = None


def normalize_ticker(value: str) -> str:
    """Trim and uppercase a ticker. Does not check existence."""
    return str(value).strip().upper()


def _load() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache

    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    companies = raw.get("companies", [])
    if not isinstance(companies, list) or not companies:
        raise ValueError(f"{REGISTRY_PATH}: 'companies' must be a non-empty list")

    table: dict[str, dict] = {}
    for entry in companies:
        if not entry.get("ticker") or not entry.get("cik"):
            raise ValueError(f"{REGISTRY_PATH}: company entry missing ticker/cik: {entry!r}")
        legal_name = (entry.get("legal_name") or entry.get("name") or "").strip()
        if not legal_name:
            raise ValueError(f"{REGISTRY_PATH}: company entry missing a name: {entry!r}")
        display_name = (entry.get("display_name") or "").strip() or None
        ticker = normalize_ticker(entry["ticker"])
        if ticker in table:
            raise ValueError(f"{REGISTRY_PATH}: duplicate ticker {ticker}")
        filings = entry.get("filings", [])
        record = {
            "ticker": ticker,
            "legal_name": legal_name,
            "display_name": display_name,
            "name": display_name or legal_name,  # effective display
            "cik": str(entry["cik"]).strip(),
            "filings": list(filings) if isinstance(filings, list) else [],
        }
        if isinstance(entry.get("lineage"), dict):
            record["lineage"] = dict(entry["lineage"])
        table[ticker] = record

    _cache = table
    return _cache


def reload() -> None:
    """Forget the cached registry so the next call re-reads the JSON file."""
    global _cache
    _cache = None


def _read_raw() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _write_raw(data: dict) -> None:
    atomic_write_json(REGISTRY_PATH, data, indent=2)
    reload()


def upsert_company(
    ticker: str,
    *,
    legal_name: str,
    cik: str,
    display_name: str | None = None,
    lineage: dict | None = None,
) -> dict:
    """Add a company or update its legal_name/cik (and display_name if given).

    ``legal_name`` is SEC-authoritative and always overwritten with the value
    passed. ``display_name`` is only set when a non-empty value is supplied, so
    ingestion never clobbers a curated name. Existing filings are preserved.
    Idempotent.

    ``lineage`` (optional): ticker->registrant succession metadata, written only
    when supplied. It records that the search ``ticker`` is distinct from the
    entity that filed (``cik`` / ``legal_name``) and, if known, from the entity
    the ticker currently resolves to. Passing ``None`` never clears an existing
    lineage block.
    """
    normalized = normalize_ticker(ticker)
    legal_name = str(legal_name).strip()
    cik = str(cik).strip()
    display_name = (display_name or "").strip() or None
    lineage = lineage if isinstance(lineage, dict) and lineage else None

    data = _read_raw()
    companies = data.setdefault("companies", [])
    for entry in companies:
        if normalize_ticker(entry.get("ticker", "")) == normalized:
            before = json.dumps(entry, sort_keys=True)
            entry.pop("name", None)  # migrate away from the old flat field
            entry["ticker"] = normalized
            entry["legal_name"] = legal_name
            entry["cik"] = cik
            if display_name is not None:
                entry["display_name"] = display_name
            if lineage is not None:
                entry["lineage"] = lineage
            entry.setdefault("filings", [])
            if json.dumps(entry, sort_keys=True) != before:
                _write_raw(data)
            else:
                reload()
            return dict(entry)

    new_entry = {
        "ticker": normalized,
        "legal_name": legal_name,
        "cik": cik,
        "filings": [],
    }
    if display_name is not None:
        new_entry["display_name"] = display_name
    if lineage is not None:
        new_entry["lineage"] = lineage
    companies.append(new_entry)
    companies.sort(key=lambda c: normalize_ticker(c.get("ticker", "")))
    _write_raw(data)
    return dict(new_entry)


def record_filing(ticker: str, filing: dict) -> None:
    """Append a filing to a company, deduped by accession_number. Idempotent."""
    normalized = normalize_ticker(ticker)
    record = {key: filing.get(key) for key in _FILING_FIELDS}
    missing = [key for key in _FILING_FIELDS if record.get(key) in (None, "")]
    if missing:
        raise ValueError(f"record_filing({normalized}): missing {missing}")

    data = _read_raw()
    for entry in data.get("companies", []):
        if normalize_ticker(entry.get("ticker", "")) == normalized:
            filings = entry.setdefault("filings", [])
            for index, existing in enumerate(filings):
                if existing.get("accession_number") == record["accession_number"]:
                    if existing == record:
                        reload()
                        return
                    filings[index] = record
                    _write_raw(data)
                    return
            filings.append(record)
            _sort_filings(filings)
            _write_raw(data)
            return
    raise ValueError(f"record_filing: company {normalized} is not registered")


def _sort_filings(filings: list[dict]) -> None:
    """Deterministic order: filing_type asc, then fiscal_year DESC (newest first)."""
    filings.sort(
        key=lambda f: (str(f.get("filing_type", "")), -int(f.get("fiscal_year") or 0))
    )


def available_fiscal_years(ticker: str) -> list[int]:
    """Fiscal years with an ingested 10-K for ``ticker`` (newest first).

    Backs API fiscal-year validation: a requested year not in this list is
    rejected rather than silently widened to an all-years search.
    """
    entry = _load().get(normalize_ticker(ticker))
    if not entry:
        return []
    years = {
        int(f["fiscal_year"])
        for f in entry.get("filings", [])
        if f.get("filing_type") == "10-K" and f.get("fiscal_year") is not None
    }
    return sorted(years, reverse=True)


def list_companies() -> list[dict]:
    """All companies as fresh dicts, sorted by ticker."""
    return [dict(entry) for entry in sorted(_load().values(), key=lambda c: c["ticker"])]


def get_company(ticker: str) -> dict | None:
    """Look up one company case-insensitively; ``None`` if not registered."""
    entry = _load().get(normalize_ticker(ticker))
    return dict(entry) if entry is not None else None


def known_tickers() -> set[str]:
    """The set of registered tickers, normalized to uppercase."""
    return set(_load())


def is_known_ticker(ticker: str) -> bool:
    return normalize_ticker(ticker) in _load()


def partition_tickers(values: list[str]) -> tuple[list[str], list[str]]:
    """Split an input list into (known, unknown), both normalized, order-preserving.

    Duplicates are collapsed while preserving first-seen order.
    """
    known_set = _load()
    known: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = normalize_ticker(value)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        (known if ticker in known_set else unknown).append(ticker)
    return known, unknown
