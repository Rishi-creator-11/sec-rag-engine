"""Immutable retrieval-scope filter shared by every retriever.

This module is pure: no network calls, no I/O, no imports from the retrieval
back ends. It exists so that Phase 1B can thread one consistent filter object
through the Pinecone dense retriever, BM25, the (currently disabled) sparse
retriever, and hybrid RRF without each of them inventing its own filter shape.

Design rules
------------
- ``tickers`` and ``filing_types`` are normalized to uppercase, trimmed, and
  de-duplicated while preserving first-seen order.
- ``fiscal_years`` are coerced to ``int`` and de-duplicated.
- Any collection that ends up empty becomes ``None`` (i.e. "no constraint").
- A filter with every field ``None`` is "empty" and constrains nothing.
- :meth:`to_pinecone_filter` only emits keys for fields that were actually
  supplied.

Field -> stored metadata key mapping
------------------------------------
    tickers       -> "ticker"
    filing_types  -> "filing_type"
    fiscal_years  -> "fiscal_year"   (numeric in Pinecone metadata)
    filing_ids    -> "filing_id"
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_PINECONE_KEY_BY_FIELD = {
    "tickers": "ticker",
    "filing_types": "filing_type",
    "fiscal_years": "fiscal_year",
    "filing_ids": "filing_id",
}


def _as_iterable(value: Any) -> Iterable | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return value
    return [value]


def normalize_str_values(
    values: Any,
    *,
    upper: bool = False,
) -> tuple[str, ...] | None:
    """Trim, optionally uppercase, drop blanks, de-duplicate (order-preserving)."""
    iterable = _as_iterable(values)
    if iterable is None:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for item in iterable:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if upper:
            text = text.upper()
        if text not in seen:
            seen.add(text)
            out.append(text)

    return tuple(out) or None


def normalize_int_values(values: Any) -> tuple[int, ...] | None:
    """Coerce to ``int``, drop ``None``, de-duplicate (order-preserving)."""
    iterable = _as_iterable(values)
    if iterable is None:
        return None

    out: list[int] = []
    seen: set[int] = set()
    for item in iterable:
        if item is None:
            continue
        if isinstance(item, bool):
            raise TypeError("fiscal years must be integers, not booleans")
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fiscal year is not an integer: {item!r}") from exc
        if number not in seen:
            seen.add(number)
            out.append(number)

    return tuple(out) or None


def normalize_tickers(values: Any) -> tuple[str, ...] | None:
    return normalize_str_values(values, upper=True)


def normalize_filing_types(values: Any) -> tuple[str, ...] | None:
    normalized = normalize_str_values(values, upper=True)
    if normalized is None:
        return None
    fixed: list[str] = []
    seen: set[str] = set()
    for value in normalized:
        text = value.replace(" ", "")
        # "10K" -> "10-K", "8K" -> "8-K"; leave already-hyphenated forms alone.
        if len(text) >= 2 and text[0].isdigit() and "-" not in text:
            head = text[:-1] if text[-1].isalpha() else text
            tail = text[len(head):]
            if tail:
                text = f"{head}-{tail}"
        if text not in seen:
            seen.add(text)
            fixed.append(text)
    return tuple(fixed) or None


@dataclass(frozen=True)
class RetrievalFilter:
    """A normalized, immutable set of retrieval-scope constraints.

    Construct directly with raw values; normalization happens in
    ``__post_init__`` so ``RetrievalFilter(tickers=["aapl", "AAPL"])`` yields
    ``tickers=("AAPL",)``.
    """

    tickers: tuple[str, ...] | None = None
    filing_types: tuple[str, ...] | None = None
    fiscal_years: tuple[int, ...] | None = None
    filing_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tickers", normalize_tickers(self.tickers))
        object.__setattr__(
            self, "filing_types", normalize_filing_types(self.filing_types)
        )
        object.__setattr__(
            self, "fiscal_years", normalize_int_values(self.fiscal_years)
        )
        object.__setattr__(
            self, "filing_ids", normalize_str_values(self.filing_ids)
        )

    # ----------------------------------------------------------------- #
    def is_empty(self) -> bool:
        """True when the filter constrains nothing."""
        return (
            self.tickers is None
            and self.filing_types is None
            and self.fiscal_years is None
            and self.filing_ids is None
        )

    def matches(self, chunk_or_metadata: Mapping[str, Any]) -> bool:
        """True when a chunk/metadata mapping satisfies every supplied constraint.

        Used by BM25 (in-Python pre-filtering) and by tests. A constrained
        field that is absent from the mapping counts as a non-match.
        """
        if self.is_empty():
            return True

        meta = chunk_or_metadata

        if self.tickers is not None:
            value = meta.get("ticker")
            if value is None or str(value).strip().upper() not in self.tickers:
                return False

        if self.filing_types is not None:
            value = meta.get("filing_type")
            if value is None:
                return False
            normalized = normalize_filing_types([value])
            if normalized is None or normalized[0] not in self.filing_types:
                return False

        if self.fiscal_years is not None:
            value = meta.get("fiscal_year")
            if value is None:
                return False
            try:
                year = int(value)
            except (TypeError, ValueError):
                return False
            if year not in self.fiscal_years:
                return False

        if self.filing_ids is not None:
            value = meta.get("filing_id")
            if value is None or str(value).strip() not in self.filing_ids:
                return False

        return True

    def to_pinecone_filter(self) -> dict[str, Any] | None:
        """Return a Pinecone ``filter`` mapping, or ``None`` when empty.

        Only fields that were actually supplied appear in the result::

            RetrievalFilter(tickers=("AAPL",)).to_pinecone_filter()
            # {"ticker": {"$in": ["AAPL"]}}

            RetrievalFilter(
                tickers=("AAPL", "MSFT"), fiscal_years=(2024, 2025)
            ).to_pinecone_filter()
            # {"ticker": {"$in": ["AAPL", "MSFT"]},
            #  "fiscal_year": {"$in": [2024, 2025]}}
        """
        predicate: dict[str, Any] = {}
        for field, key in _PINECONE_KEY_BY_FIELD.items():
            values = getattr(self, field)
            if values:
                predicate[key] = {"$in": list(values)}
        return predicate or None

    def describe(self) -> str:
        """Short human label, e.g. ``AAPL,MSFT | 10-K | FY2024,FY2025``."""
        if self.is_empty():
            return "unfiltered"
        parts: list[str] = []
        if self.tickers:
            parts.append(",".join(self.tickers))
        if self.filing_types:
            parts.append(",".join(self.filing_types))
        if self.fiscal_years:
            parts.append(",".join(f"FY{year}" for year in self.fiscal_years))
        if self.filing_ids:
            parts.append(",".join(self.filing_ids))
        return " | ".join(parts)
