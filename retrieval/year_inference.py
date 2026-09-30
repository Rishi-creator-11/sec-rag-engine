"""Conservative single-fiscal-year inference from a free-text question.

Closes a specific scoping gap: a request can name a company (``tickers``)
without a structured ``fiscal_years`` value, in which case retrieval today
searches every ingested year for that ticker (see ``retrieval.scope``).
When the question itself names exactly one fiscal year in English --
"revenue in FY2025?" -- that's a real scoping signal the API otherwise
throws away, and its absence is what lets same-metric-different-year
chunks (e.g. a restated total in a reconciliation table) compete in the
same evidence set with no ranking signal capable of telling them apart.

This module only ever *narrows* scope. It never redirects, excludes, or
overrides anything: ambiguous or multi-year input returns ``None``, and
callers must fall back to exactly today's unconstrained-year behavior.

Precedence rule (enforced by the caller, not this module): a request that
already supplies structured ``fiscal_years`` must never reach this
function at all. Structured scope always wins; this is purely a fallback
for when structured scope is absent.
"""

from __future__ import annotations

import re

_YEAR_MIN = 2000
_YEAR_MAX = 2099

# "FY2025", "FY 2025", "fy2025"
_FY_PATTERN = re.compile(r"\bFY\s*(20\d{2})\b", re.IGNORECASE)

# "fiscal 2025", "fiscal year 2025"
_FISCAL_PATTERN = re.compile(r"\bfiscal(?:\s+year)?\s+(20\d{2})\b", re.IGNORECASE)

# A bare four-digit year, e.g. "revenue in 2025". Deliberately the widest
# net here -- everything else in this module exists to keep it safe.
_BARE_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# Any of these means the question is about a range, a trend, or an
# explicit comparison across years -- never collapse to one year, even if
# only one (or zero) year tokens are present ("since 2020" has one token
# but names a range, not a single year; "last three years" and "over the
# past decade" may have none at all but still must not infer).
_RANGE_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"since|vs\.?|versus|compare[ds]?|comparison|"
    r"over\s+the\s+(?:last|past)|"
    r"last\s+\w+\s+years?|past\s+\w+\s+years?|"
    r"from\s+\d{4}\s+to\s+\d{4}|"
    r"trend|historically|year[\s-]over[\s-]year"
    r")\b",
    re.IGNORECASE,
)


def infer_fiscal_year(question: str) -> int | None:
    """Return the single fiscal year ``question`` unambiguously names.

    Returns ``None`` -- meaning "don't narrow scope, behave as today" --
    when the question names zero years, more than one distinct year, or
    contains any range/comparison/trend signal regardless of year count.

    Never call this when the request already has a structured
    ``fiscal_years`` value; that value always takes precedence and this
    function must not be consulted in that case.
    """
    if not question or not question.strip():
        return None

    if _RANGE_SIGNAL_PATTERN.search(question):
        return None

    years: set[int] = set()
    for pattern in (_FY_PATTERN, _FISCAL_PATTERN, _BARE_YEAR_PATTERN):
        for match in pattern.finditer(question):
            year = int(match.group(1))
            if _YEAR_MIN <= year <= _YEAR_MAX:
                years.add(year)

    if len(years) == 1:
        return next(iter(years))
    return None
