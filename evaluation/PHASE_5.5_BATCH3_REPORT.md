# Phase 5.5 — Controlled multi-year expansion, Batch 3 (WMT / UNH / XOM)

Date: 2026-08-31
Scope: expand historical 10-K coverage to the final batch of the existing
10-company universe. Latest 3 exact 10-K fiscal years per company. No new
companies. No 10-Q / 8-K. **Nothing committed or pushed.**

Baseline: Phase 5.5 Batch 2 production commit `833d010` — 10 companies,
25 filings, 3,533 chunks, benchmark_v3 live hybrid MRR 0.900, all structural
leakage gates green, Batch 1 / Batch 2 / NVDA regressions green.

---

## 1. Filings discovered (SEC, read-only discovery)

| ticker | discovery CIK | registrant | FY | report_date | filing_date | accession | primary doc | pre-existing? |
|---|---|---|---|---|---|---|---|---|
| WMT | 0000104169 | Walmart Inc. | 2026 | 2026-01-31 | 2026-03-13 | 0000104169-26-000055 | wmt-20260131.htm | yes (Phase-4 canonical) → **kept** |
| WMT | 0000104169 | Walmart Inc. | 2025 | 2025-01-31 | 2025-03-14 | 0000104169-25-000021 | wmt-20250131.htm | no → **ingested** |
| WMT | 0000104169 | Walmart Inc. | 2024 | 2024-01-31 | 2024-03-15 | 0000104169-24-000056 | wmt-20240131.htm | no → **ingested** |
| UNH | 0000731766 | UNITEDHEALTH GROUP INC | 2025 | 2025-12-31 | 2026-03-02 | 0000731766-26-000062 | unh-20251231.htm | yes (Phase-4 Batch 3) → **kept** |
| UNH | 0000731766 | UNITEDHEALTH GROUP INC | 2024 | 2024-12-31 | 2025-02-27 | 0000731766-25-000063 | unh-20241231.htm | no → **ingested** |
| UNH | 0000731766 | UNITEDHEALTH GROUP INC | 2023 | 2023-12-31 | 2024-02-28 | 0000731766-24-000081 | unh-20231231.htm | no → **ingested** |
| XOM | **0000034088** | **EXXON MOBIL CORP** | 2025 | 2025-12-31 | 2026-02-18 | 0000034088-26-000045 | xom-20251231.htm | yes (Phase-4 Batch 3, CIK override) → **kept** |
| XOM | **0000034088** | **EXXON MOBIL CORP** | 2024 | 2024-12-31 | 2025-02-19 | 0000034088-25-000010 | xom-20241231.htm | no → **ingested (CIK override)** |
| XOM | **0000034088** | **EXXON MOBIL CORP** | 2023 | 2023-12-31 | 2024-02-28 | 0000034088-24-000018 | xom-20231231.htm | no → **ingested (CIK override)** |

10-K/A excluded (string equality on `"10-K"`). Collision checks at discovery
time: each of the 6 target accessions confirmed **absent** from registry,
local chunks, dense index and sparse index before any write.

**WMT non-calendar fiscal year** (§3): Walmart's fiscal year ends the last
Friday of January. `derive_fiscal_year(report_date)` = calendar year of the SEC
`reportDate`, so period-end 2026-01-31 → FY2026, 2025-01-31 → FY2025,
2024-01-31 → FY2024. Fiscal year is **never** inferred from the filing date
(all filed in March). Verified against SEC `reportDate` for all three.

## 2. XOM lineage / CIK behavior (§2, §11, §19)

XOM historical 10-Ks were discovered from the **filing registrant** CIK
`0000034088` (EXXON MOBIL CORP), **not** the 2026 successor `0002115436`
(ExxonMobil Holdings Corporation). The generic mechanism was used with no
XOM-specific code:

```
python -m ingestion.ingest_company --ticker XOM --years 3 \
  --cik 0000034088 \
  --successor-cik 0002115436 \
  --successor-name "ExxonMobil Holdings Corporation" \
  --successor-effective-date 2026-07-01 --verify
```

`--cik` bypasses **only** ticker→CIK discovery; filing discovery still runs
against SEC submissions for CIK 0000034088. `cik_override_active` logged for
each new filing. Provenance preserved and validated (all checks PASS):

| check | result |
|---|---|
| retrieval scope stays XOM (single-year + comparison) | ✓ every chunk `ticker == XOM`, every chunk in requested year |
| historical evidence registrant | ✓ `company: "EXXON MOBIL CORP"` on all FY2023/FY2024 chunks and `/ask` sources |
| CIK metadata on new chunk artifacts | ✓ `cik: "0000034088"` (historical) |
| source URLs / accessions authentic | ✓ `.../edgar/data/34088/000003408824000018/xom-20231231.htm`, accession `0000034088-24-000018` |
| successor entity never shown as filer | ✓ no `2115436` / "Holdings Corporation" in any `/ask` source |
| registry `legal_name` / `cik` | ✓ `EXXON MOBIL CORP` / `0000034088` (unchanged) |
| lineage block structured fields | ✓ `registrant_cik 0000034088`, `successor_cik 0002115436`, `successor_effective_date 2026-07-01`, `cik_override true` |
| lineage `note` | restored to the Phase-4 text (references the FY2025 canonical filing) — the pipeline rewrites this human-readable note per filing; the structured fields above are authoritative and unchanged |
| `/companies/XOM/filings` `registrant_lineage` | ✓ present, `filing_registrant_cik 0000034088` / `current_successor_cik 0002115436` |

No lineage regression from Phase 4. No provenance rewriting.

## 3. Reconciliation (accession = identity, §4)

- All 3 pre-existing FY-latest accessions match SEC discovery exactly →
  `year_skipped_registered` logged for each (WMT FY2026, UNH FY2025, XOM FY2025).
  **No re-embed, no duplicate vectors.** Latest-year chunk counts unchanged
  (WMT 110, UNH 103, XOM 149).
- Only the 6 genuinely-missing accessions were ingested.
- Canonical chunk IDs preserved (`{TICKER}_{FY}_10-K_{ACCESSION_NODASH}_{idx}`).

## 4. Ingestion result (existing Phase 5 pipeline, `--years 3 --verify`)

| ticker | FY | accession | chunks | dense | sparse | verify --full |
|---|---|---|---|---|---|---|
| WMT | 2026 | 0000104169-26-000055 | 110 | (kept) | (kept) | PASS |
| WMT | 2025 | 0000104169-25-000021 | 114 | 114 | ok | PASS |
| WMT | 2024 | 0000104169-24-000056 | 116 | 116 | ok | PASS |
| UNH | 2025 | 0000731766-26-000062 | 103 | (kept) | (kept) | PASS |
| UNH | 2024 | 0000731766-25-000063 | 99 | 99 | ok | PASS |
| UNH | 2023 | 0000731766-24-000081 | 97 | 97 | ok | PASS |
| XOM | 2025 | 0000034088-26-000045 | 149 | (kept) | (kept) | PASS |
| XOM | 2024 | 0000034088-25-000010 | 156 | 156 | ok | PASS |
| XOM | 2023 | 0000034088-24-000018 | 147 | 147 | ok | PASS |

6 newly ingested = **729 chunks** (WMT 230, UNH 196, XOM 303). Every new
accession `verify_company --fiscal-year <FY> --full` PASS (all IDs, 50-ID
sparse-fetch batching preserved).

## 5. Corpus state (§6, §8–11)

| | before (Batch 2) | after (Batch 3) |
|---|---|---|
| filings | 25 | 31 |
| chunks (registry / local) | 3,533 | **4,262** |
| dense (`sec-rag-engine`) | 3,533 | **4,262** |
| sparse (`sec-rag-sparse`) | 3,533 | **4,262** |
| bm25s document_count | 3,533 | **4,262** |

`dense == sparse == registry == local == bm25s == 4262` (exact). Total =
3533 + 729, no duplicate vectors. 4,262 unique canonical chunk IDs, **0
collisions** corpus-wide. bm25s `chunks.jsonl` is an exact set-match to the
local canonical chunk set. Registry newest-fiscal-year-first, no duplicate
accessions per company.

## 6. bm25s bundled index (`python -m scripts.build_bm25s_index`, §7, §12)

| | value |
|---|---|
| document_count | 4,262 |
| corpus_version | `v2:1a8d63947a72…` (was `v2:25553ab085f7…`) — **changed** |
| corpus_version schema | `v2` (hashes chunk_id + ticker + fiscal_year + filing_id) |
| index size | **29.07 MB**, 7 files (was 24.20 MB) |
| build time | ~2.2 s wall |
| cold load | 185 ms |
| warm load | 66 ms |
| ranking parity | **PASS** — persisted vs fresh in-memory, unfiltered + `RetrievalFilter(NVDA, 2023)` |
| stale-index check | none |

## 7. Structural gates — `evaluation.evaluate_multiyear_batch3` (3 consecutive runs, incl. one at 13/15 Cohere fallback)

| metric | target | result |
|---|---|---|
| fiscal_year_filter_correctness | 1.000 | **1.000** |
| cross_year_leakage@5 | 0.000 | **0.000** |
| cross_company_leakage@5 | 0.000 | **0.000** |
| comparison scope_coverage | 1.000 | **1.000** |
| numeric_year_correctness (lexical, single retrieval) | 1.000 | **1.000** (9/9) |
| numeric_year_correctness (end-to-end generation) | 1.000 | **1.000** (9/9) |
| unsupported-year → registry-absent (API 422) | 1.000 | **1.000** (3/3) |

### §8 single-year retrieval (newest + oldest of each 3-year window)

WMT FY2026 & FY2024, UNH FY2025 & FY2023, XOM FY2025 & FY2023 — every one
returned `fiscal_year_filter_correctness 1.000`, `cross_year_leakage 0.000`,
`cross_company_leakage 0.000`, evidence `{TICKER:FY: 5}`.

## 8. Example multi-year answers (live `/ask`, retrieval + generation)

### §10 numeric comparisons

- **WMT total revenues, FY2024 vs FY2026** → $648,125M → $713,163M.
  Evidence `{WMT:2024: 1, WMT:2026: 4}`, zero leak.
- **UNH total revenues, FY2023 vs FY2025** → $371,622M → $447,567M.
  Evidence `{UNH:2023: 1, UNH:2025: 4}`.
- **XOM total revenues and other income, FY2023 vs FY2025** → $344,582M →
  $332,238M (a decline). Evidence `{XOM:2023: 3, XOM:2025: 2}`.

### §9 disclosure-change comparisons

- **WMT e-commerce / supply-chain / competitive risk, FY2024 → FY2026** →
  FY2024 frames omni-channel execution and eCommerce-reliance / supply-chain
  disruption risk; FY2026 adds AI-powered associate and fulfillment tooling and
  a more integrated U.S. supply chain (Sam's Club U.S. merged into Walmart
  U.S.). Evidence `{WMT:2024: 2, WMT:2026: 4}`, zero leak.
- **UNH regulatory / reimbursement risk, FY2023 → FY2025** → both years centre
  on Medicare Advantage funding pressure and risk-adjustment model revisions;
  FY2025 adds explicit minimum-MLR and value-based-care shift detail.
  Evidence `{UNH:2023: 3, UNH:2025: 3}`.
- **XOM climate / regulatory / commodity-price risk, FY2023 → FY2025** →
  FY2023 references the "Climate Change and the Energy Transition" section and
  a per-$1/bbl Upstream sensitivity; FY2025 expands the climate/energy-
  transition and regulatory discussion within Item 1A. Evidence
  `{XOM:2023: 2, XOM:2025: 4}`.

All: both requested years represented, source tickers = the requested ticker
only, all source years in the requested set, zero warnings.

## 9. Regression (§12, §22–26)

| suite | baseline | now | verdict |
|---|---|---|---|
| unit tests | 302 | **320** (+18 `test_multiyear_batch3.py`, incl. 4 XOM-lineage tests) | all pass |
| benchmark_v3 offline (frozen pool) hybrid MRR | 0.903 | **0.903** | exact |
| benchmark_v3 `--live` hybrid MRR | 0.900 | **0.897** | −0.003, within ±0.02 (floor 0.877) |
| benchmark_v3 `--live` R@10-capped | 0.717 | **0.718** | within tol |
| benchmark_v3 `--live` P@5 | 0.635 | **0.638** | within tol |
| benchmark_v3 structural (filter / company-leakage) | 1.000 / 0.000 | **1.000 / 0.000** | no regression |
| `evaluate_scoped --mode filtered` | filter 1.000 / leak 0.000 / R@10 0.905 | **not re-run this batch** (v3 live structural gate covers it) | — |
| NVDA multi-year canary | GATE PASS | **GATE PASS** | unchanged |
| Batch 1 multi-year (AAPL/MSFT/AMZN) | GATE PASS | **GATE PASS** | unchanged |
| Batch 2 multi-year (GOOGL/META/JPM) | GATE PASS | **GATE PASS** | unchanged |
| default company-comparison (15 Q) | PASS | **PASS** (scope 1.000 / leak 0.000) | unchanged |
| XOM/UNH comparison (batch3, 7 Q) | PASS | **PASS** (scope 1.000 / leak 0.000) | unchanged |
| XOM company canary | PASS | **PASS** (filter 1.000 / leak 0.000) | unchanged |
| UNH company canary | PASS | **PASS** (filter 1.000 / leak 0.000) | unchanged |

Retrieval settings unchanged: `candidate_k = 10`, `RRF_K = 60`, equal
dense/BM25 weights, sparse not in the production path, Cohere rerank-v4.0-fast,
gpt-5-nano. **Zero production code files modified** — `git diff` over
`api/ retrieval/ ingestion/ scripts/` is empty. `evaluate_multiyear_batch1.py`
and `evaluate_multiyear_batch2.py` are byte-unchanged.

## 10. Pinecone / Cohere monitoring (§13, §20–21)

- **Pinecone reads: healthy.** Zero read failures this batch. The monthly
  egress cap was lifted by the Pinecone **Standard Trial**. `describe_index_stats`
  and every dense/sparse query succeeded. Dense + sparse each grew by 729
  vectors (ingress).
- **Cohere trial-key rate limit (≈10/min):** `reranker_fallback` ranged 5/15 →
  13/15 across the three batch3 eval runs; 12/15 on the default comparison
  suite, 0/7 on the batch3 comparison suite. Every fallback answer stayed
  fully scope-correct. Batch 3 numeric questions are phrased against the
  "consolidated statements of income / operations" so the income-statement
  chunk (which carries the comparative years) ranks in both rerank and
  fallback modes.
- **Index growth:** bm25s 24.20 → 29.07 MB (+4.87 MB / +729 chunks ≈ 6.7
  kB/chunk). A full 10-company × 3-year corpus is now essentially complete at
  ~4.3k chunks / ~29 MB — well under Vercel's function size limit.
- **Query latency:** batch3 structural eval mean 1.71 s, p50 0.70 s; company
  canaries ~0.7–1.0 s.

## 11. Frontend contract (§14 — local, no frontend change)

`GET /companies/{WMT,UNH,XOM}/filings` → 200, newest-first, no duplicate
accessions:

- WMT `available_fiscal_years [2026, 2025, 2024]`
- UNH `[2025, 2024, 2023]`
- XOM `[2025, 2024, 2023]` + `registrant_lineage` block (filing registrant
  0000034088 / successor 0002115436)

`/companies` unchanged (10 companies, `{ticker, name}` shape). Identical
endpoint contract the deployed frontend already consumes for the other seven
companies — no frontend code change required.

## 12. Production risks discovered (§27)

1. **XOM lineage `note` is filing-scoped.** The pipeline regenerates the
   human-readable `lineage.note` on every ingest, so after `--years 3` it
   referenced the last-processed accession (FY2023). Restored to the Phase-4
   text (FY2025 reference). The structured lineage fields are authoritative
   and were never at risk; consider making the note filing-agnostic if XOM-
   style successions recur.
2. **Cohere trial-key rate limit** remains the main source of eval-run
   non-determinism. Not a defect; the gated metrics are all fallback-robust.
   A paid Cohere key would remove the wobble in the ungated NE@k proxy.
3. **benchmark_v3 `--live` MRR drift**: 0.903 → 0.900 (Batch 2) → 0.897
   (Batch 3) as the corpus grew 1,684 → 4,262. Monotonic, small, and within
   ±0.02 of the frozen-pool baseline every batch — corpus-dilution on a
   depth-50 frozen qrel pool, not a ranking regression (offline frozen MRR is
   still exactly 0.903). Worth a fresh benchmark_v3 re-pool before the
   20-company pilot so the baseline tracks the live corpus.
4. **bm25s bundle at 29 MB.** Fine for Vercel today; the 20-company pilot at
   ~3× the corpus would push it toward ~90 MB — check the function size limit
   and consider a compressed or externally-hosted index before that scale-up.

## 13. Verdict — §29 and §30

**§29 — All 10 companies are now multi-year ready:**

| ticker | fiscal years | filings |
|---|---|---|
| AAPL | 2025, 2024, 2023 | 3 |
| AMZN | 2025, 2024, 2023 | 3 |
| GOOGL | 2025, 2024, 2023 | 3 |
| JPM | 2025, 2024, 2023 | 3 |
| META | 2025, 2024, 2023 | 3 |
| MSFT | 2026, 2025, 2024 | 3 |
| NVDA | 2026, 2025, 2024, 2023 | 4 |
| UNH | 2025, 2024, 2023 | 3 |
| WMT | 2026, 2025, 2024 | 3 |
| XOM | 2025, 2024, 2023 | 3 |

31 filings, 4,262 chunks. Every company has ≥ 3 fiscal years.

**§30 — Batch 3 is safe to checkpoint and deploy.** All structural gates green
(1.000 / 0.000 / 0.000 / 1.000 / 1.000), numeric-year correctness 1.000 lexical
and 1.000 end-to-end, benchmark_v3 within ±0.02, NVDA + Batch 1 + Batch 2 +
comparison + canary regressions all unchanged, 320 tests pass, zero
production-code drift, zero ID collisions, `dense == sparse == bm25s ==
registry == 4262`, no duplicate accessions, XOM provenance fully validated,
frontend contract confirmed locally. Recommended checkpoint contents:
`data/bm25s_index/*`, `data/registry/*`, `data/chunks/{WMT,UNH,XOM}/*.jsonl`,
`data/raw/{WMT,UNH,XOM}/*/metadata.json`,
`evaluation/benchmark_multiyear_batch3.json`,
`evaluation/evaluate_multiyear_batch3.py`,
`evaluation/validate_batch3_numeric_generation.py`,
`tests/test_multiyear_batch3.py`, the Batch 3 result summaries, and this report.

**The 20-company pilot should NOT begin from this checkpoint yet** — first
re-pool / re-judge benchmark_v3 against the now-complete 4.3k-chunk live corpus
(the `--live` MRR has drifted 0.006 below the frozen baseline over three
batches), and confirm the bm25s bundle size budget for 3× the corpus. Those
are follow-ups, not Batch 3 blockers.
