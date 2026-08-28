"""Generic retrieval scope: one ``(ticker, fiscal_year?)`` pair.

Phase 5 generalizes the Phase 2 per-ticker comparison. A *scope* is the unit of
retrieval coverage:

    Scope("NVDA")           -> all ingested NVDA filings
    Scope("NVDA", 2023)     -> only NVDA's FY2023 10-K

A request expands into a list of scopes (``expand_scopes``):

    tickers=[NVDA],       years=[]            -> [NVDA]                 (single)
    tickers=[NVDA],       years=[2024]        -> [NVDA:2024]           (single)
    tickers=[NVDA],       years=[2023, 2025]  -> [NVDA:2023, NVDA:2025](comparison)
    tickers=[AAPL, MSFT], years=[]            -> [AAPL, MSFT]          (comparison)
    tickers=[AAPL, MSFT], years=[2024]        -> [AAPL:2024, MSFT:2024](comparison)
    tickers=[AAPL, MSFT], years=[2023, 2024]  -> [AAPL:2023, AAPL:2024,
                                                  MSFT:2023, MSFT:2024](comparison)

``comparison_mode`` is simply ``len(scopes) >= 2`` — one code path for company
comparison, year comparison, and company+year comparison.

Backward compatibility: a ticker-only scope's label is just the ticker
(``"AAPL"``), so ``evidence_by_scope`` stays ``{"AAPL": 3, "MSFT": 2}`` for the
existing company-comparison contract; year scopes use ``"NVDA:2023"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from retrieval.filters import RetrievalFilter


@dataclass(frozen=True)
class Scope:
    ticker: str
    fiscal_year: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", str(self.ticker).strip().upper())
        if self.fiscal_year is not None:
            object.__setattr__(self, "fiscal_year", int(self.fiscal_year))

    @property
    def label(self) -> str:
        return f"{self.ticker}:{self.fiscal_year}" if self.fiscal_year is not None \
            else self.ticker

    @property
    def has_year(self) -> bool:
        return self.fiscal_year is not None

    def to_filter(self) -> RetrievalFilter:
        return RetrievalFilter(
            tickers=(self.ticker,),
            fiscal_years=(self.fiscal_year,) if self.fiscal_year is not None else None,
        )

    def matches(self, chunk: Mapping[str, Any]) -> bool:
        if str(chunk.get("ticker", "")).strip().upper() != self.ticker:
            return False
        if self.fiscal_year is None:
            return True
        value = chunk.get("fiscal_year")
        if value in (None, ""):
            return False
        try:
            return int(value) == self.fiscal_year
        except (TypeError, ValueError):
            return False


def expand_scopes(
    tickers: list[str],
    fiscal_years: list[int] | None = None,
) -> list[Scope]:
    """Cartesian product of tickers x fiscal_years, order-preserving.

    Empty ``fiscal_years`` => one all-years scope per ticker.
    """
    years = list(fiscal_years or [])
    scopes: list[Scope] = []
    seen: set[tuple[str, int | None]] = set()
    for ticker in tickers:
        for year in (years or [None]):
            key = (str(ticker).strip().upper(), int(year) if year is not None else None)
            if key in seen:
                continue
            seen.add(key)
            scopes.append(Scope(ticker, year))
    return scopes


def scope_of_chunk(chunk: Mapping[str, Any], scopes: list[Scope]) -> str | None:
    """Label of the first scope a chunk belongs to (for evidence_by_scope)."""
    for scope in scopes:
        if scope.matches(chunk):
            return scope.label
    return None
