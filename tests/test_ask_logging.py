"""Unit tests for api.ask_logging: no-op by default, never raises, correct
shape when configured. No network calls -- requests.post is monkeypatched.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)


def _reload():
    import api.ask_logging as mod

    return importlib.reload(mod)


def test_disabled_when_env_vars_absent():
    mod = _reload()
    assert mod.logging_enabled() is False


def test_enabled_when_both_env_vars_present(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "sb_publishable_test")
    mod = _reload()
    assert mod.logging_enabled() is True


def test_log_ask_is_noop_without_env_vars(monkeypatch):
    mod = _reload()
    calls = []
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: calls.append((a, k)))
    mod.log_ask({"question": "hello"})
    assert calls == []


def test_log_ask_posts_with_insert_only_headers(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co/")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "sb_publishable_test")
    mod = _reload()
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers, timeout))

        class R:
            status_code = 201

        return R()

    monkeypatch.setattr(mod.requests, "post", fake_post)
    mod.log_ask({"question": "hello"})
    assert len(calls) == 1
    url, body, headers, timeout = calls[0]
    assert url == "https://example.supabase.co/rest/v1/alphabrief_ask_logs"
    assert body == {"question": "hello"}
    assert headers["apikey"] == "sb_publishable_test"
    assert headers["Authorization"] == "Bearer sb_publishable_test"
    assert headers["Prefer"] == "return=minimal"
    assert timeout <= 2.0


def test_log_ask_never_raises_on_network_error(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "sb_publishable_test")
    mod = _reload()

    def boom(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(mod.requests, "post", boom)
    mod.log_ask({"question": "hello"})  # must not raise


def test_hash_ip_is_stable_and_not_reversible():
    mod = _reload()
    h1 = mod.hash_ip("203.0.113.77")
    h2 = mod.hash_ip("203.0.113.77")
    h3 = mod.hash_ip("198.51.100.5")
    assert h1 == h2
    assert h1 != h3
    assert "203.0.113.77" not in h1
    assert mod.hash_ip(None) is None


def test_client_ip_prefers_x_forwarded_for():
    mod = _reload()
    headers = {"x-forwarded-for": "203.0.113.77, 10.0.0.1"}
    assert mod.client_ip_from_headers(headers) == "203.0.113.77"
    assert mod.client_ip_from_headers({"x-real-ip": "198.51.100.5"}) == "198.51.100.5"
    assert mod.client_ip_from_headers({}) is None


def test_build_ok_row_shape():
    mod = _reload()
    result = {
        "answer": "The answer. [Source 1]",
        "sources": [
            {"ticker": "AAPL", "rerank_score": 0.91},
            {"ticker": "AAPL", "rerank_score": 0.42},
        ],
        "search_scope": {"comparison_mode": False},
        "reranker_fallback": False,
        "reranker_fallback_reason": None,
        "timings": {"hybrid_ms": 10.0, "rerank_ms": 20.0, "generation_ms": 30.0,
                     "total_ms": 60.0, "context_chunks_used": 2},
    }
    row = mod.build_ok_row(
        question="q", tickers=["AAPL"], fiscal_years=[2024], result=result,
        ip_hash="abc", user_agent="ua", origin="https://secfrontend.vercel.app",
    )
    assert row["status"] == "ok"
    assert row["scope_mode"] == "single"
    assert row["sources_count"] == 2
    assert row["top_rerank_score"] == 0.91
    assert row["answer_length"] == len(result["answer"])
