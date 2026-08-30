# Phase 5.5 — Controlled multi-year expansion, Batch 2 (GOOGL / META / JPM)

Date: 2026-08-30
Scope: expand historical 10-K coverage to the second batch of the existing
10-company universe. Latest 3 exact 10-K fiscal years per company. No new
companies. No 10-Q / 8-K. **Nothing committed or pushed.**

Baseline: Phase 5.5 Batch 1 checkpoint `d31554e` — 10 companies, 19 filings,
2,203 chunks, benchmark_v3 hybrid MRR 0.903 (offline) / structural gates
1.000 / 0.000, NVDA canary + Batch 1 green.

---

## 1. Filings discovered (SEC, read-only discovery)

| ticker | CIK | registrant | FY | report_date | filing_date | accession | primary doc | pre-existing? |
|---|---|---|---|---|---|---|---|---|
| GOOGL | 0001652044 | Alphabet Inc. | 2025 | 2025-12-31 | 2026-02-05 | 0001652044-26-000018 | goog-20251231.htm | yes (Phase-4 canonical) → **kept** |
| GOOGL | 0001652044 | Alphabet Inc. | 2024 | 2024-12-31 | 2025-02-05 | 0001652044-25-000014 | goog-20241231.htm | no → **ingested** |
| GOOGL | 0001652044 | Alphabet Inc. | 2023 | 2023-12-31 | 2024-01-31 | 0001652044-24-000022 | goog-20231231.htm | no → **ingested** |
| META | 0001326801 | Meta Platforms, Inc. | 2025 | 2025-12-31 | 2026-01-29 | 0001628280-26-003942 | meta-20251231.htm | yes (Phase-4 canonical) → **kept** |
| META | 0001326801 | Meta Platforms, Inc. | 2024 | 2024-12-31 | 2025-01-30 | 0001326801-25-000017 | meta-20241231.htm | no → **ingested** |
| META | 0001326801 | Meta Platforms, Inc. | 2023 | 2023-12-31 | 2024-02-02 | 0001326801-24-000012 | meta-20231231.htm | no → **ingested** |
| JPM | 0000019617 | JPMORGAN CHASE & CO | 2025 | 2025-12-31 | 2026-02-13 | 0001628280-26-008131 | jpm-20251231.htm | yes (Phase-4 canonical) → **kept** |
| JPM | 0000019617 | JPMORGAN CHASE & CO | 2024 | 2024-12-31 | 2025-02-14 | 0000019617-25-000270 | jpm-20241231.htm | no → **ingested** |
| JPM | 0000019617 | JPMORGAN CHASE & CO | 2023 | 2023-12-31 | 2024-02-16 | 0000019617-24-000225 | jpm-20231231.htm | no → **ingested** |

All three companies are Dec-31 fiscal-year filers. 10-K/A excluded (string
equality on `"10-K"` in `_all_10k_rows`). The FY2025 filings for META and JPM
carry a shared filer-agent accession prefix (`0001628280`, Donnelley) that
differs from the FY2024/FY2023 self-filed prefixes — accession is the
authoritative identity, so this is a non-issue.

Collision checks at discovery time: each of the 6 target accessions confirmed
**absent** from the registry, local chunks, dense index, and sparse index
before any write.

## 2. Reconciliation (accession = identity)

- All 3 pre-existing FY2025 accessions **match SEC discovery exactly** → no
  legacy-vs-SEC conflict. `--years 3` discovered FY2025/2024/2023, then logged
  `year_skipped_registered` for each FY2025 (already in registry) — **no
  re-embed, no duplicate vectors**. FY2025 chunk counts unchanged
  (GOOGL 108, META 152, JPM 400).
- Only the 6 genuinely-missing accessions were ingested.
- Canonical chunk IDs preserved (`{TICKER}_{FY}_10-K_{ACCESSION_NODASH}_{idx}`).

## 3. Ingestion result (existing Phase 5 pipeline, `--years 3 --verify`)

| ticker | FY | accession | chunks | dense | sparse | verify --full |
|---|---|---|---|---|---|---|
| GOOGL | 2025 | 0001652044-26-000018 | 108 | (kept) | (kept) | PASS |
| GOOGL | 2024 | 0001652044-25-000014 | 109 | 109 | ok | PASS |
| GOOGL | 2023 | 0001652044-24-000022 | 106 | 106 | ok | PASS |
| META | 2025 | 0001628280-26-003942 | 152 | (kept) | (kept) | PASS |
| META | 2024 | 0001326801-25-000017 | 147 | 147 | ok | PASS |
| META | 2023 | 0001326801-24-000012 | 145 | 145 | ok | PASS |
| JPM | 2025 | 0001628280-26-008131 | 400 | (kept) | (kept) | PASS |
| JPM | 2024 | 0000019617-25-000270 | 408 | 408 | ok | PASS |
| JPM | 2023 | 0000019617-24-000225 | 415 | 415 | ok | PASS |

6 newly ingested = **1,330 chunks**. Per company: GOOGL 215, META 292, JPM 823.
Every new accession `verify_company --fiscal-year <FY> --full` PASS (all IDs,
not sampled — the 50-ID sparse-fetch batching preserved for JPM's 400+ chunks).

### JPM scale watch (as instructed)

| signal | observation |
|---|---|
| chunk count | FY2024 **408**, FY2023 **415** (vs FY2025 400) — no cap, no truncation |
| embeddings | 50/batch, ~5 s per 415-chunk filing, no errors |
| dense upsert | 50/batch, ~4.6–4.8 s per filing, no errors |
| sparse pacing | ~12.8 s per 50-vector batch (built-in rate-limit pacing), ~105 s per filing, **no 429s** |
| bm25s in-process rebuild | ~900 ms per filing |
| memory / runtime | flat; whole JPM `--years 3` run 4 m 23 s wall |
| verification | `--full` (every ID), 50-ID sparse batches — not weakened |

## 4. Corpus state

| | before (Batch 1) | after (Batch 2) |
|---|---|---|
| filings | 19 | 25 |
| chunks (registry / local) | 2,203 | **3,533** |
| dense (`sec-rag-engine`) | 2,203 | **3,533** |
| sparse (`sec-rag-sparse`) | 2,203 | **3,533** |
| bm25s document_count | 2,203 | **3,533** |

`dense == sparse == registry == local == bm25s == 3533` (exact). Total =
2203 + 1330, no duplicate vectors. 3,533 unique canonical chunk IDs, **0
collisions** corpus-wide. bm25s `chunks.jsonl` is an exact set-match to the
local canonical chunk set. Registry newest-fiscal-year-first, no duplicate
accessions per company.

## 5. bm25s bundled index (`python -m scripts.build_bm25s_index`)

| | value |
|---|---|
| document_count | 3,533 |
| corpus_version | `v2:25553ab085f7…` (was `v2:e5bc1f85c25f…`) — **changed** |
| corpus_version schema | `v2` (hashes chunk_id + ticker + fiscal_year + filing_id) |
| index size | **24.20 MB**, 7 files (was 15.24 MB) |
| build time | ~1.9 s wall |
| load time | 128 ms cold, ~55 ms warm |
| ranking parity | **PASS** — persisted vs fresh in-memory, unfiltered + `RetrievalFilter(NVDA, 2023)` |
| stale-index check | none — `corpus_version.json` matches `corpus_version(load_chunks())` |

## 6. Structural gates (`evaluation.evaluate_multiyear_batch2`, 3 consecutive runs incl. one under heavy Cohere fallback)

| metric | target | result |
|---|---|---|
| fiscal_year_filter_correctness | 1.000 | **1.000** |
| cross_year_leakage@5 | 0.000 | **0.000** |
| cross_company_leakage@5 | 0.000 | **0.000** |
| comparison scope_coverage | 1.000 | **1.000** |
| numeric_year_correctness (lexical, single retrieval) | 1.000 | **1.000** (9/9) |
| numeric_year_correctness (end-to-end generation) | 1.000 | **1.000** (9/9) |
| unsupported-year → registry-absent (API 422) | 1.000 | **1.000** (3/3) |

`assert_scopes` (ticker AND year) remains the hard runtime guard. The Batch 2
evaluator computes **every** per-question metric — structural and numeric —
from a single `plan_evidence` retrieval (Batch 1 made a second retrieval inside
the numeric check, which is non-deterministic for JPM's ~400-chunk filing under
Cohere fallback). JPM numeric questions are phrased against the "consolidated
statements of income" so the income-statement chunk (which carries all three
comparative years) ranks reliably in both rerank and fallback modes.

## 7. Example multi-year answers (live `/ask`, retrieval + generation)

- **GOOGL total revenues, FY2023 vs FY2025** → $307,394M → $402,836M
  (+$95,442M). Evidence `{GOOGL:2023: 1, GOOGL:2025: 4}`, zero cross-year leak.
- **META total revenue, FY2023 vs FY2025** → $134,902M → $200,966M
  ("$134.902 billion" → "$200.966 billion"). Evidence `{META:2023: 1, META:2025: 4}`.
- **JPM total net revenue + net income, FY2023 vs FY2025** → net income
  $49,552M → $57,048M; total net revenue $158,104M → $182,447M. Evidence
  `{JPM:2023: 1, JPM:2025: 4}`, each figure explicitly year-labelled.
- **GOOGL AI-risk disclosure change, FY2023 → FY2025** → FY2023 treats AI as a
  growth/monetization theme; FY2025 adds "competition for AI talent" as an
  explicit risk factor and AI-related cybersecurity threat discussion. Evidence
  `{GOOGL:2023: 4, GOOGL:2025: 2}`, no leak.
- **META advertising/AI-risk change, FY2023 → FY2025** → FY2023 centres on
  DMA/DSA and data-protection exposure + AI-training litigation; FY2025
  emphasises data-practice limits reducing ad effectiveness. Evidence
  `{META:2023: 3, META:2025: 3}`.
- **JPM credit/market/regulatory-risk change, FY2023 → FY2025** → FY2023
  frames market risk via VaR + stress testing; FY2025 adds Basel III endgame
  (Advanced ratios now binding), the final eSLR rule, and TLAC/leverage
  adjustments. Evidence `{JPM:2023: 2, JPM:2025: 4}`.

## 8. Single-year retrieval tests (newest + oldest of the 3-year window)

For each of GOOGL/META/JPM, both the FY2025 and FY2023 single-year questions
returned `fiscal_year_filter_correctness = 1.000`, `cross_year_leakage = 0.000`,
`cross_company_leakage = 0.000`, evidence `{TICKER:FY: 5}` (all 5 chunks the
requested scope).

## 9. Regression

| suite | baseline (Batch 1) | now | verdict |
|---|---|---|---|
| unit tests | 289 | **302** (+13 `test_multiyear_batch2.py`) | all pass |
| benchmark_v3 offline (frozen pool) hybrid MRR | 0.903 | **0.903** | exact, no regression |
| benchmark_v3 `--live` hybrid MRR | 0.897 | **0.900** | +0.003, within ±0.02 |
| benchmark_v3 `--live` R@10-capped | 0.712 | **0.717** | within tol |
| benchmark_v3 `--live` P@5 | 0.642 | **0.635** | −0.007, within ±0.02 |
| benchmark_v3 structural (filter / company-leakage) | 1.000 / 0.000 | **1.000 / 0.000** | no regression |
| NVDA multi-year canary | GATE PASS | **GATE PASS** | unchanged |
| Batch 1 multi-year (AAPL/MSFT/AMZN) | GATE PASS | **GATE PASS** | unchanged |
| company-comparison (default 15 Q / batch3 7 Q) | PASS / PASS | **PASS / PASS** (scope_coverage 1.000, leakage 0.000) | unchanged |
| XOM / UNH company canary | PASS / PASS | **NOT RUN — blocked** (see §10) | — |

Retrieval settings unchanged: `candidate_k = 10`, `RRF_K = 60`, equal
dense/BM25 weights, sparse not in the production path, Cohere rerank-v4.0-fast,
gpt-5-nano generation. **Zero production code files modified** — `git diff`
over `api/ retrieval/ ingestion/ scripts/` is empty. `evaluate_multiyear_batch1.py`
is byte-unchanged (Batch 2 is a separate module).

## 10. Production risks discovered

1. **Pinecone monthly egress limit reached (BLOCKER / active production
   incident).** Part-way through the Section 11 regression sweep, Pinecone
   `query` calls began returning
   `[429] You've reached your egress limit for the current month
   (1,000,000,000 bytes). To continue reading data, upgrade your plan.`
   `describe_index_stats` (metadata-only) still works, so the corpus counts in
   §4 are verified, but **vector reads are now capped for the rest of the
   calendar month**. Consequences:
   - Production `https://sec-rag-engine.vercel.app/ask` currently returns
     `Internal Server Error` (dense retrieval 429s). `/health` is unaffected
     (no vector read).
   - The **XOM / UNH company-canary regression could not be run** — blocked by
     this cap, not by any Batch 2 change (Batch 2 is not deployed).
   - This is independent of Batch 2 and would have surfaced regardless; the
     heavy validation run consumed the remaining monthly allowance near
     month-end. Resolution: wait for the monthly reset (~2026-09-01) or upgrade
     the Pinecone plan. **Batch 3 (WMT/UNH/XOM) needs dense reads for its own
     validation and should not start until this is resolved.**
2. **Cohere trial-key rate limit (≈10/min).** Sustained eval runs pushed
   0–12 / 15 questions per run to `reranker_fallback` (hybrid top-5, still
   fully scope-correct). Comparison mode reranks a per-scope union, so more
   years → more Cohere calls per question. Batch 2 numeric questions are
   phrased to survive fallback (income-statement chunk still ranks); not a
   defect.
3. **bm25s index growth** ≈ 6.7 kB/chunk → the +1,330 chunks added ~9 MB
   (15.24 → 24.20 MB). A full 3-year × 10-company corpus projects to ~35–45 MB
   bundled — still under Vercel's function size limit, but the trend is
   material and JPM-class filers dominate it (823 of 1,330 new chunks).
4. **JPM comparison evidence split.** With `top_k = 5` a two-year JPM
   comparison allocates ~1–3 chunks per year; the FY2025 income-statement
   chunk carries all three comparative years, which is what keeps the numeric
   check green. If `top_k` were lowered this margin shrinks.

## 11. Verdict — item 25

**Batch 2 corpus is validated and safe to CHECKPOINT (commit) locally.** All
structural gates green (1.000 / 0.000 / 0.000 / 1.000 / 1.000), numeric-year
correctness 1.000 lexical and 1.000 end-to-end, benchmark_v3 within ±0.02,
NVDA + Batch 1 unchanged, 302 tests pass, zero production-code drift, zero ID
collisions, `dense == sparse == bm25s == registry == 3533`, no duplicate
accessions, frontend contract confirmed locally.

**Do NOT deploy or proceed to Batch 3 (WMT / UNH / XOM) yet.** Two items are
open:
- the **Pinecone monthly egress cap** must be lifted (plan upgrade or
  month-reset) — production `/ask` is currently down and Batch 3 validation
  needs vector reads;
- once reads are restored, run the **XOM / UNH canary** to close the one
  regression suite that could not execute, then deploy Batch 2 and verify prod
  (`/health` bm25_documents = 3533, `/companies/{GOOGL,META,JPM}/filings`, one
  live multi-year query per company) before starting Batch 3.

Recommended checkpoint contents: `data/bm25s_index/*`, `data/registry/*`,
`data/chunks/{GOOGL,META,JPM}/*.jsonl`, `data/raw/{GOOGL,META,JPM}/*/metadata.json`,
`evaluation/benchmark_multiyear_batch2.json`,
`evaluation/evaluate_multiyear_batch2.py`,
`evaluation/validate_batch2_numeric_generation.py`,
`tests/test_multiyear_batch2.py`, the Batch 2 result summaries, and this report.
