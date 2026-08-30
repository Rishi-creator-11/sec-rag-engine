# Phase 5.5 — Controlled multi-year expansion, Batch 1 (AAPL / MSFT / AMZN)

Date: 2026-08-29
Scope: expand historical 10-K coverage from the NVDA canary to the first batch of
the existing 10-company universe. Latest 3 exact 10-K fiscal years per company.
No new companies. No 10-Q / 8-K. **Nothing committed or pushed.**

Baseline: Phase 5 checkpoint `c4549fa` + deploy fix `32bebf9` — 10 companies,
13 filings, 1,684 chunks, benchmark_v3 hybrid MRR 0.897 / R@10-capped 0.712 /
P@5 0.642, NVDA 4-fiscal-year canary green.

---

## 1. Filings discovered (SEC, read-only discovery)

| ticker | CIK | registrant | FY | report_date | filing_date | accession | primary doc | pre-existing? |
|---|---|---|---|---|---|---|---|---|
| AAPL | 0000320193 | Apple Inc. | 2025 | 2025-09-27 | 2025-10-31 | 0000320193-25-000079 | aapl-20250927.htm | no → **ingested** |
| AAPL | 0000320193 | Apple Inc. | 2024 | 2024-09-28 | 2024-11-01 | 0000320193-24-000123 | (seed) | yes (legacy seed `apple_10k_*`) → **kept** |
| AAPL | 0000320193 | Apple Inc. | 2023 | 2023-09-30 | 2023-11-03 | 0000320193-23-000106 | aapl-20230930.htm | no → **ingested** |
| MSFT | 0000789019 | MICROSOFT CORP | 2026 | 2026-06-30 | 2026-07-29 | 0001193125-26-323660 | msft-20260630.htm | no → **ingested** |
| MSFT | 0000789019 | MICROSOFT CORP | 2025 | 2025-06-30 | 2025-07-30 | 0000950170-25-100235 | (seed) | yes (legacy seed `microsoft_10k_*`) → **kept** |
| MSFT | 0000789019 | MICROSOFT CORP | 2024 | 2024-06-30 | 2024-07-30 | 0000950170-24-087843 | msft-20240630.htm | no → **ingested** |
| AMZN | 0001018724 | AMAZON COM INC | 2025 | 2025-12-31 | 2026-02-06 | 0001018724-26-000004 | (canonical) | yes (Phase-4 canonical ingest) → **kept** |
| AMZN | 0001018724 | AMAZON COM INC | 2024 | 2024-12-31 | 2025-02-07 | 0001018724-25-000004 | amzn-20241231.htm | no → **ingested** |
| AMZN | 0001018724 | AMAZON COM INC | 2023 | 2023-12-31 | 2024-02-02 | 0001018724-24-000008 | amzn-20231231.htm | no → **ingested** |

Non-calendar fiscal-year ends handled with no per-company special case: AAPL
end-of-September, MSFT end-of-June, AMZN end-of-December. 10-K/A excluded (string
equality on `"10-K"`). MSFT FY2026's accession carries a DFIN filer prefix
(`0001193125`) vs the FY2025 seed's `0000950170` — accession is the authoritative
identity, so this is a non-issue.

## 2. Reconciliation (accession = identity)

- All 3 pre-existing accessions **match SEC discovery exactly** for their fiscal
  year → no legacy-vs-SEC conflict, no re-embed, no duplicate vectors.
- Legacy seeds (`apple_10k_*`, `microsoft_10k_*`) verified: fully metadata-backed
  (`ticker`, `fiscal_year`, `accession_number`, `report_date` present and
  correct), present in dense + sparse. Kept as the canonical served filing.
- All 6 target accessions confirmed **absent** from dense + sparse before
  ingestion; dry-run chunk_id collision check PASS for each.

## 3. Ingestion result (existing Phase 5 pipeline, `--years 3 --verify`)

| ticker | FY | accession | chunks | dense | sparse | verify |
|---|---|---|---|---|---|---|
| AAPL | 2025 | 0000320193-25-000079 | 67 | 67 | ok | PASS |
| AAPL | 2024 | 0000320193-24-000123 | 66 | (kept) | (kept) | PASS |
| AAPL | 2023 | 0000320193-23-000106 | 65 | 65 | ok | PASS |
| MSFT | 2026 | 0001193125-26-323660 | 101 | 101 | ok | PASS |
| MSFT | 2025 | 0000950170-25-100235 | 98 | (kept) | (kept) | PASS |
| MSFT | 2024 | 0000950170-24-087843 | 109 | 109 | ok | PASS |
| AMZN | 2025 | 0001018724-26-000004 | 89 | (kept) | (kept) | PASS |
| AMZN | 2024 | 0001018724-25-000004 | 88 | 88 | ok | PASS |
| AMZN | 2023 | 0001018724-24-000008 | 89 | 89 | ok | PASS |

6 newly ingested = **519 chunks**. Per company: AAPL 198, MSFT 308, AMZN 266.

## 4. Corpus state

| | before | after |
|---|---|---|
| filings | 13 | 19 |
| chunks (registry) | 1,684 | **2,203** |
| dense (`sec-rag-engine`) | 1,684 | **2,203** |
| sparse (`sec-rag-sparse`) | 1,684 | **2,203** |
| bm25s document_count | 1,684 | **2,203** |

`dense == sparse == registry == bm25s == 2203` (exact). No duplicate vectors
(total = 1684 + 519). Registry newest-fiscal-year-first, no duplicate accessions.

## 5. bm25s bundled index (`python -m scripts.build_bm25s_index`)

| | value |
|---|---|
| document_count | 2,203 |
| corpus_version | `v2:e5bc1f85c25f…` (was `v2:ef5d6c2a1e…`) — **changed** |
| corpus_version schema | `v2` (hashes chunk_id + ticker + fiscal_year + filing_id) |
| index size | 15.24 MB, 7 files |
| build time | ~1.4 s wall (index build sub-second) |
| load time | 137 ms (cached 0 ms) |
| ranking parity | PASS — persisted vs fresh in-memory, unfiltered + ticker+fiscal_year filter |
| `--check` | PASS |

## 6. Structural gates

| metric | target | result |
|---|---|---|
| fiscal_year_filter_correctness | 1.000 | **1.000** |
| cross_year_leakage@5 | 0.000 | **0.000** |
| cross_company_leakage@5 | 0.000 | **0.000** |
| comparison scope_coverage | 1.000 | **1.000** |
| numeric_year_correctness (lexical, comparison-aware) | 1.000 | **1.000** |
| numeric_year_correctness (end-to-end generation) | 1.000 | **1.000** (9/9) |
| unsupported-year → 422 | 1.000 | **1.000** |

`assert_scopes` (ticker AND year) is the hard runtime guard — any out-of-scope
chunk raises `ScopeViolationError`. API 422 verified live: unavailable single
year, and a cartesian scope where one (ticker, year) pair is unavailable →
whole request rejected, never silently widened.

## 7. Example multi-year answers (live `/ask`)

- **AAPL total net sales, FY2023 vs FY2025** → $383,285M → $416,161M
- **MSFT total revenue, FY2024 vs FY2026** → $245,122M → $331,839M (+$86,717M, ~35%)
- **AMZN total net sales, FY2023 vs FY2025** → $574,785M → $716,924M (+$142,139M)
- **MSFT AI-risk disclosure change, FY2024 → FY2026** → FY2024 excerpts insufficient;
  FY2026 details AI export-control / regulatory / policy risk. Evidence scoped
  `{MSFT:2024: 1, MSFT:2026: 4}`, no cross-year leak.

## 8. Regression

| suite | baseline | now | verdict |
|---|---|---|---|
| unit tests | 277 | **289** (+12 `test_multiyear_batch1.py`) | all pass |
| benchmark_v3 `--live` hybrid MRR | 0.897 | **0.897** | no regression |
| benchmark_v3 `--live` R@10-capped | 0.712 | **0.712** | no regression |
| benchmark_v3 `--live` P@5 | 0.642 | **0.642** | no regression |
| benchmark_v3 structural (filter / leakage) | 1.000 / 0.000 | **1.000 / 0.000** | no regression |
| NVDA multi-year canary | GATE PASS | **GATE PASS** | unchanged |
| company-comparison (default + batch3) | PASS / PASS | **PASS / PASS** | unchanged |

Retrieval settings unchanged: `candidate_k=10`, `RRF_K=60`, equal dense/BM25
weights, sparse not in the production path, Cohere rerank-v4.0-fast, gpt-5-nano.
Zero production code files modified (`git diff` over `api/ retrieval/ ingestion/
scripts/` is empty).

## 9. Frontend

No frontend change. The unchanged production frontend discovers the new years
purely from `GET /companies/{ticker}/filings` → `available_fiscal_years`. Verified
locally: selecting Apple shows **FY2025 / FY2024 / FY2023** with zero code change.
Only hardcoded years in the frontend are in `lib/mockData.ts` (demo-mode
fallback, unused when `NEXT_PUBLIC_API_URL` is set).

## 10. Production risks

- **Cohere trial-key rate limit (10/min).** Rapid successive eval runs pushed
  5–7/15 questions to `reranker_fallback` (hybrid top-5, still fully
  scope-correct). Comparison mode reranks a per-scope union, so more years →
  more Cohere calls per question. Not a defect; watch if production QPS rises.
- **bm25s index growth** ≈ 0.6 MB/filing. Full 3-year × 10-company expansion
  lands ≈ 25–30 MB — under Vercel's function limit, but track it.
- **Comparison-mode numeric retrieval**: when one scope scores weaker on the
  joint query and Cohere is in fallback, the single `MIN_PER_SCOPE`-reserved
  chunk for that scope may not be the income statement. The generated answer is
  still correct (10-K comparative columns carry the prior-year figure), and both
  the comparison-aware lexical metric and the end-to-end generation validator
  pass; a strict same-year lexical check would flag it.
- **Filer-agent accession prefixes are not stable per company** (MSFT
  FY2026 vs FY2025) — handled, noted.

## 11. Verdict

**Batch 1 is safe to checkpoint and proceed to Batch 2 (GOOGL / META / JPM).**
All gates green, zero code drift, zero regression, index/registry/vectors
consistent at 2,203, frontend contract confirmed. Recommended checkpoint
contents: `data/bm25s_index/*`, `data/registry/*`, new
`data/chunks/{AAPL,MSFT,AMZN}/*.jsonl` + `data/raw/*/metadata.json`,
`evaluation/benchmark_multiyear_batch1.json`, `evaluation/evaluate_multiyear_batch1.py`,
`evaluation/validate_batch1_numeric_generation.py`, `tests/test_multiyear_batch1.py`,
this report. Then deploy, verify prod `/companies/{AAPL,MSFT,AMZN}/filings` and a
live multi-year query, then run Batch 2 with the identical procedure.
