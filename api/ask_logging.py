"""Best-effort logging of /ask interactions to Supabase.

Design notes (see the migration `create_alphabrief_ask_logs` for the table):

- Write-only: the key used here can INSERT and nothing else (no SELECT/UPDATE/
  DELETE grant or policy). Reading the log is done from the Supabase dashboard
  or a service-role key, never from this API. A leaked key from this module
  cannot expose or tamper with past logs.
- No-op by default: if SUPABASE_URL / SUPABASE_ANON_KEY are not set, every
  call here is a cheap no-op. Deploying this module cannot change existing
  behavior until those two env vars are added.
- Never raises: a logging failure (network error, Supabase outage, bad
  response) is caught and swallowed. Logging must never turn a successful or
  already-failed request into a different failure for the caller.
- Synchronous with a short timeout, not a "fire and forget" background task.
  Vercel's Python serverless functions are not guaranteed to keep running
  after the HTTP response is sent, so a true background write can silently
  never happen. A ~1.2s timeout on a single INSERT is the safer trade.
- Privacy-light: the caller's IP is never stored raw. It's salted and
  hashed so repeat calls from the same address can still be correlated
  (e.g. for abuse patterns) without keeping anyone's real IP.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_TIMEOUT_S = 1.2
_IP_SALT = "alphabrief-ask-log-v1"  # not a secret; only needs to be stable


def _supabase_target() -> tuple[str, str] | None:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None
    return url, key


def logging_enabled() -> bool:
    return _supabase_target() is not None


def hash_ip(ip: str | None) -> str | None:
    if not ip:
        return None
    digest = hashlib.sha256(f"{_IP_SALT}:{ip}".encode("utf-8")).hexdigest()
    return digest[:16]


def client_ip_from_headers(headers) -> str | None:
    """Best-effort client IP from proxy headers (Vercel sets x-forwarded-for)."""
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return headers.get("x-real-ip")


def log_ask(row: dict[str, Any]) -> None:
    """Insert one row into alphabrief_ask_logs. Never raises."""
    target = _supabase_target()
    if target is None:
        return
    url, key = target
    try:
        requests.post(
            f"{url}/rest/v1/alphabrief_ask_logs",
            json=row,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 - logging must never break the request
        logger.warning("ask_logging: insert failed", exc_info=True)


def build_ok_row(
    *,
    question: str,
    tickers: list[str] | None,
    fiscal_years: list[int] | None,
    result: dict[str, Any],
    ip_hash: str | None,
    user_agent: str | None,
    origin: str | None,
) -> dict[str, Any]:
    scope = result.get("search_scope") or {}
    sources = result.get("sources") or []
    timings = result.get("timings") or {}
    scope_mode = (
        "comparison"
        if scope.get("comparison_mode")
        else ("single" if tickers else "global")
    )
    rerank_scores = [
        s.get("rerank_score")
        for s in sources
        if isinstance(s.get("rerank_score"), (int, float))
    ]
    return {
        "question": question,
        "tickers": tickers,
        "fiscal_years": fiscal_years,
        "scope_mode": scope_mode,
        "status": "ok",
        "http_status": 200,
        "answer_length": len(result.get("answer") or ""),
        "sources_count": len(sources),
        "source_tickers": sorted({s.get("ticker") for s in sources if s.get("ticker")}),
        "reranker_fallback": bool(result.get("reranker_fallback")),
        "reranker_fallback_reason": result.get("reranker_fallback_reason"),
        "top_rerank_score": max(rerank_scores) if rerank_scores else None,
        "hybrid_ms": timings.get("hybrid_ms"),
        "rerank_ms": timings.get("rerank_ms"),
        "generation_ms": timings.get("generation_ms"),
        "total_ms": timings.get("total_ms"),
        "context_chunks_used": timings.get("context_chunks_used"),
        "ip_hash": ip_hash,
        "user_agent": user_agent,
        "origin": origin,
    }


def build_error_row(
    *,
    question: str | None,
    tickers: list[str] | None,
    fiscal_years: list[int] | None,
    status: str,
    http_status: int | None,
    error_code: str | None,
    error_detail: Any,
    ip_hash: str | None,
    user_agent: str | None,
    origin: str | None,
) -> dict[str, Any]:
    return {
        "question": question,
        "tickers": tickers,
        "fiscal_years": fiscal_years,
        "status": status,
        "http_status": http_status,
        "error_code": error_code,
        "error_detail": error_detail,
        "ip_hash": ip_hash,
        "user_agent": user_agent,
        "origin": origin,
    }


class Timer:
    """Tiny helper so call sites read `with Timer() as t: ...; t.ms`."""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = round((time.perf_counter() - self._t0) * 1000, 1)
