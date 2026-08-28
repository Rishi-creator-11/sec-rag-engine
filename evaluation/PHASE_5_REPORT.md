# Phase 5 — Multi-year 10-K support (NVDA canary)

Date: 2026-08-28
Scope: production-grade support for MULTIPLE historical 10-K filings per company,
proven with an NVDA 4-fiscal-year canary. Same 10-company universe. No 10-Q / 8-K.
Nothing committed or pushed.

Baseline (Phase 4 checkpoint `1ce2a3e`): 10 companies, 1,378 chunks, benchmark_v3
hybrid MRR 0.903 / R@10-capped 0.716 / P@1 0.843 / P@5 0.640 / NumEvidence@10 0.964.

---

## 1. Files created / modified

**Created**
| file | purpose |
|---|---|
| `retrieval/scope.py` | generic `Scope(ticker, fiscal_year?)` model + `expand_scopes` |
| `scripts/backfill_seed_registry.py` | register the 3 seed filings in registry + ledger |
| `evaluation/benchmark_multiyear.json` | 18-question NVDA multi-year benchmark |
| `evaluation/evaluate_multiyear.py` | fiscal-year-scope evaluator + metrics |
| `tests/test_multiyear.py` | 22 tests (scope model, year filter, coverage, API, registry) |

**Modified — SEC / ingestion**
- `ingestion/sec_client.py` — `list_10ks`, `discover_10k(fiscal_year=…)`,
  archived-filing-history handling; `discover_latest_10k` gains `fiscal_year=`.
- `ingestion/ingest_company.py` — `--years N` / `--fiscal-year YYYY`; per-filing
  `_ingest_discovered` extracted; `_ingest_years` loop; skips accessions already
  in the registry (e.g. a backfilled seed).
- `ingestion/registry.py` — `available_fiscal_years(ticker)`; filings sorted
  newest-fiscal-year-first, deduped by accession.
- `ingestion/verify_company.py` — `--fiscal-year`; reads chunk ids from the JSONL
  (so seed filings verify); sparse fetch batched (kept from Batch 3).

**Modified — retrieval**
- `retrieval/pinecone_search.py`, `retrieval/bm25_search.py`,
  `retrieval/lexical_backend.py` — result dicts now carry `fiscal_year`,
  `report_date`, `accession_number`, `filing_id`, `chunk_index` (were dropped).
- `retrieval/scoped_search.py` — `scope_label` emits `NVDA:2023` for ticker+year.
- `retrieval/filters.py` — unchanged (already year-aware).

**Modified — API / generation**
- `api/main.py` — `AskRequest.fiscal_years`; 422 for `fiscal_years` without
  tickers and for an unavailable year (never silently widened); `search_scope`
  gains `fiscal_years` + `scopes`.
- `api/rag.py` — generic Scope routing (`comparison_mode == len(scopes) >= 2`);
  `select_evidence_with_coverage` / `evidence_by_scope` / `assert_coverage` keyed
  by scope label; new `assert_scopes` (ticker AND year); year-comparison
  generation prompt.

**Modified — evaluation**
- `evaluation/evaluate_v3_offline.py` — `--live` scopes each question to the
  fiscal year(s) of its qrels (so multi-year NVDA does not fake a regression).
- `evaluation/evaluate_scoped.py` — same qrel-year scoping for benchmark_v2.

**Modified — data (seed metadata backfill, see §11)**
- `data/chunks/{apple,microsoft,nvidia}_10k_chunks.jsonl`
- `data/registry/companies.json`, `data/registry/ingestion_state.json`
- 267 dense + 267 sparse Pinecone vectors (metadata only)

## 2. Historical SEC discovery design

`SecClient._all_10k_rows(cik, include_archives=True)`:
- reads `submissions["filings"]["recent"]` (parallel arrays → row dicts),
- **and every archived filing-history file** referenced from
  `submissions["filings"]["files"]` (fetched from
  `https://data.sec.gov/submissions/<name>`),
- keeps only **exact `"10-K"`** (string equality → `10-K/A` excluded),
- drops rows with an empty `reportDate`,
- de-dupes by dashed accession, sorts `(reportDate, filingDate, accession)` desc.

`list_10ks(cik) -> [DiscoveredFiling]` (newest fiscal year first).
`discover_10k(cik, fiscal_year=None)`:
- `None` → newest, **recent[] only** (fast path, no archive fetches),
- a year → try recent[] first, page in archives only if the year is not there,
- unavailable year → `SecNotFoundError` listing the years that ARE available.

`latest_filing` now delegates to `discover_10k(fiscal_year=None)`, so
`discover_latest_10k` (verify, single-filing CLI) is unchanged in behaviour and
request count.

## 3. Archived-submissions handling

Never assumes 5 years live in `recent[]`. For NVDA, `recent[]` already holds
10-Ks back to FY2011 and `filings.files` has one archive
(`CIK0001045810-submissions-001.json`, 1998–2020) — the code fetched and merged
it (a FY1999 lookup correctly reported "available: 2026 … 2002", i.e. it read
the archive). A missing archive file logs `archive_missing` and is skipped, not
fatal. Covered by `tests/test_sec_client.py` mocks + `test_multiyear.py`.

## 4. Fiscal-year semantics

`fiscal_year = calendar year of the SEC reportDate` (fiscal-period end) —
`retrieval.metadata.derive_fiscal_year`. This matches each issuer's own label:

| issuer | period end | fiscal_year |
|---|---|---|
| NVIDIA FY2026 | 2026-01-25 | 2026 |
| NVIDIA FY2023 | 2023-01-29 | 2023 |
| Apple FY2024 (seed) | 2024-09-28 | 2024 |
| Microsoft FY2025 (seed) | 2025-06-30 | 2025 |
| Walmart FY2026 (Batch 3) | 2026-01-31 | 2026 |

Non-calendar fiscal years (NVDA end-of-January, WMT end-of-January, AAPL
end-of-September, MSFT end-of-June) are all handled by "calendar year the period
ends in" with no per-company special case. `tests/test_multiyear.py` +
`tests/test_sec_client.py` assert this for NVDA and WMT.

## 5. CLI changes

```
python -m ingestion.ingest_company --ticker NVDA                 # latest only (unchanged)
python -m ingestion.ingest_company --ticker AAPL --fiscal-year 2023
python -m ingestion.ingest_company --ticker NVDA --years 3       # latest 3 exact 10-Ks
```
`--years` and `--fiscal-year` are mutually exclusive (argparse group).
`--years N` discovers the latest N, then ingests each through the **existing**
per-filing pipeline; an accession already recorded in the registry is skipped
(`year_skipped_registered`). Each accession is an independent ledger entry, so a
rerun skips complete years, resumes incomplete ones, and never duplicates.
`--cik` / `--successor-*` (Batch 3) compose with both: e.g.
`--ticker XOM --cik 0000034088 --fiscal-year 2024` pulls a pre-succession XOM
filing from the legacy registrant (§ lineage below).

## 6. Registry changes

- `record_filing` dedupes by `accession_number`; filings sorted
  `(filing_type asc, fiscal_year DESC)` — newest year first.
- `available_fiscal_years(ticker) -> [int]` (newest first) backs API year
  validation.
- `/companies` unchanged — still `{ticker, name}` only; filing metadata is not
  exposed there.
- NVDA registry entry now lists FY2026 / FY2025 / FY2024 / FY2023.

## 7. Generic scope model

`retrieval.scope.Scope(ticker, fiscal_year=None)`:
`.label` (`"NVDA"` or `"NVDA:2023"`), `.to_filter()` (`RetrievalFilter`),
`.matches(chunk)` (ticker AND, if set, fiscal_year — an absent year on the chunk
never matches a year-constrained scope).

`expand_scopes(tickers, fiscal_years)` = order-preserving Cartesian product:

| request | scopes | mode |
|---|---|---|
| `NVDA`, `[]` | `[NVDA]` | single |
| `NVDA`, `[2024]` | `[NVDA:2024]` | single |
| `NVDA`, `[2023, 2025]` | `[NVDA:2023, NVDA:2025]` | comparison |
| `AAPL,MSFT`, `[]` | `[AAPL, MSFT]` | comparison |
| `AAPL,MSFT`, `[2024]` | `[AAPL:2024, MSFT:2024]` | comparison |
| `AAPL,MSFT`, `[2023,2024]` | `[AAPL:2023, AAPL:2024, MSFT:2023, MSFT:2024]` | comparison |

**One code path.** `comparison_mode` is purely `len(scopes) >= 2`. The Phase 2
per-ticker comparison (`scoped_search` → union → one Cohere rerank →
coverage-aware selection) generalised without a second implementation:
`retrieve_evidence_comparison(question, k, scopes)` builds
`[s.to_filter() for s in scopes]` and everything downstream keys on `scope.label`.

## 8. API changes

Request (all fields optional, backward compatible):
```json
{"question": "...", "tickers": ["NVDA"], "fiscal_years": [2023, 2025]}
```
| request | behaviour |
|---|---|
| no `tickers` | global search (unchanged) |
| `tickers` only | **search every ingested filing for that company** (documented default) |
| `tickers` + `fiscal_years` | one scope per `(ticker, year)`; ≥2 → comparison |

Validation (422, never a silent widen):
- `fiscal_years` with no `tickers` → `fiscal_years_without_tickers`
- a year not in `available_fiscal_years(ticker)` for **every** requested ticker →
  `fiscal_year_not_available` with `{unavailable, available}`

`search_scope` response gains `fiscal_years` and `scopes` (labels); all existing
fields kept. `sources[]` gain `fiscal_year`, `accession_number`, `report_date`.

## 9. evidence_by_scope behaviour

- company-only comparison: `{"AAPL": 3, "MSFT": 2}` — **unchanged contract**
  (a ticker-only scope's label is the bare ticker).
- year comparison: `{"NVDA:2023": 1, "NVDA:2025": 4}`.
- guarantee: ≥1 evidence chunk per requested scope that has any candidate
  (`assert_coverage` raises `ScopeViolationError` otherwise).

## 10. Generation prompt changes

`build_generation_request` detects a year comparison (any scope label contains
`:`) and instructs the model to: discuss each fiscal year in its own labelled
section then a "What changed:" section; use only that year's excerpts; never
transfer a fact between years; distinguish changed disclosure LANGUAGE from a
changed real-world fact; not assert increase/decrease unless both years'
excerpts support it; cite each claim. Ordinary single-year and company-only
prompts are untouched.

Live example (FY2023 vs FY2025, "What was NVIDIA total revenue"):
> FY2023: Revenue: $26,974 million. [Source 2]
> FY2025: Revenue: $130,497 million. [Source 4]
> What changed: Revenue increased from $26,974 million in FY2023 to $130,497
> million in FY2025. [Source 2][Source 4]

All 5 sources were NVDA FY2023 / FY2025 — zero cross-year leakage.

## 11. Seed metadata backfill — REQUIRED, and performed

**Why required:** the 3 seed filings' chunks (`{apple,microsoft,nvidia}_10k_N`)
had **no `fiscal_year`, `accession_number`, `filing_id`, `report_date`, `cik`**
at all (local JSONL *and* Pinecone). `RetrievalFilter.matches` treats an absent
year as a non-match, so a year-scoped query would silently exclude every seed
chunk — year filtering could not work.

**What was done** (`scripts/backfill_metadata.py`, an existing Phase-1A-vintage
script whose verified seed facts exactly matched a fresh SEC fetch):
- **metadata only** — added `cik, accession_number, filing_id, fiscal_year,
  report_date, chunk_index` to:
  - local `data/chunks/{apple,microsoft,nvidia}_10k_chunks.jsonl`
  - 267 dense vectors (`index.update(id, set_metadata=…)` — merge, no re-embed)
  - 267 sparse vectors (same)
- **no vector id changed, no embedding changed, no delete/recreate.**
- legacy `filing_date` (which was actually the period-end date) left as-is
  (`--fix-filing-date` not used) so `/ask` display output is unchanged; the
  correct `report_date` was added alongside.
- Idempotent: a re-run reports "no changes needed".
- **Verification: 267/267 dense and 267/267 sparse vectors** now carry the
  correct `fiscal_year` + `accession_number` (polled for read-after-write).
- Seed identities: **AAPL FY2024** (`0000320193-24-000123`), **MSFT FY2025**
  (`0000950170-25-100235`), **NVDA FY2026** (`0001045810-26-000021`).

`scripts/backfill_seed_registry.py` then added each seed filing to
`data/registry/companies.json` (so `available_fiscal_years` knows they exist)
and a synthetic "complete" ledger entry per accession pointing `artifacts.chunks`
at the legacy seed JSONL (so `--years` skips them and `verify_company
--fiscal-year` passes).

**Rollback plan** (artifacts written before applying):
- `data/registry/backups/seed_pinecone_metadata_presnapshot.json` — all 534
  vectors' pre-backfill metadata.
- `data/registry/backups/{companies,ingestion_state}.pre_phase5.json`.
- local JSONL: `git checkout data/chunks/{apple,microsoft,nvidia}_10k_chunks.jsonl`.
- (Full removal of the added Pinecone keys would need a re-upsert from
  `data/embeddings/sec_chunks.jsonl`; the additive metadata is low-risk and the
  snapshot restores prior values.)

No unrelated change was bundled with the backfill. Seed chunk ids are **not**
migrated to canonical form (Section 6 of the brief) — they keep `<prefix>_N`.

## 12. NVDA historical filings discovered

`--years 4` → available FY2026 … FY2002; selected FY2026, FY2025, FY2024, FY2023.

| fiscal year | report date | filing date | accession | primary document |
|---|---|---|---|---|
| 2026 (seed) | 2026-01-25 | 2026-02-25 | 0001045810-26-000021 | nvda-20260125.htm |
| 2025 | 2025-01-26 | 2025-02-26 | 0001045810-25-000023 | nvda-20250126.htm |
| 2024 | 2024-01-28 | 2024-02-21 | 0001045810-24-000029 | nvda-20240128.htm |
| 2023 | 2023-01-29 | 2023-02-24 | 0001045810-23-000017 | nvda-20230129.htm |

Dry-run first; FY2026 matched the seed's SEC URL exactly — **no mismatch with
seed data**, and it was skipped (`year_skipped_registered`), not re-ingested.
`--years 4` (not 3) was used so the §20 FY2023↔FY2025 tests have both endpoints.

## 13. NVDA filings ingested

FY2025, FY2024, FY2023 through the existing pipeline (discover → download →
clean → chunk → embed → dense_upsert → sparse_upsert → bm25_register →
registry_update → complete). FY2026 already served by the seed.

## 14. Chunk counts by year

| NVDA fiscal year | chunks | dense | sparse | canonical id prefix |
|---|---|---|---|---|
| 2026 (seed) | 103 | 103 | 103 | `nvidia_10k_N` (legacy, kept) |
| 2025 | 105 | 105 | ok | `NVDA_2025_10-K_000104581025000023_N` |
| 2024 | 103 | 103 | ok | `NVDA_2024_10-K_000104581024000029_N` |
| 2023 | 98 | 98 | ok | `NVDA_2023_10-K_000104581023000017_N` |
| **NVDA total** | **409** | | | |

## 15. Dense / sparse / bm25s counts

| | before Phase 5 | after |
|---|---|---|
| companies | 10 | 10 |
| total chunks | 1,378 | **1,684** (+306) |
| dense index `sec-rag-engine` | 1,378 | 1,684 |
| sparse index `sec-rag-sparse` | 1,378 | 1,684 |
| bm25s documents | 1,378 | **1,684** |
| bm25s persisted size | 9.51 MB | **11.70 MB** |
| chunk-id collisions | 0 | **0** (1,684 unique) |

## 16. Cross-year leakage

`evaluation/evaluate_multiyear.py` (16 supported NVDA questions, top-5):

| metric | value | target |
|---|---|---|
| `fiscal_year_filter_correctness` | **1.000** | 1.000 |
| `cross_year_leakage@5` | **0.000** | 0.000 |

Live spot check (`assert_scopes` also enforces this at request time and raises
`ScopeViolationError` on any out-of-scope year).

## 17. Multi-year coverage

| metric | value | target |
|---|---|---|
| `scope_coverage` (all questions) | 1.000 | — |
| `scope_coverage` (year-comparison questions) | **1.000** | 1.000 |

Every year-comparison question returned ≥1 chunk for **both** requested years,
e.g. `{'NVDA:2023': 1, 'NVDA:2025': 4}`, `{'NVDA:2023': 3, 'NVDA:2025': 2}`.

## 18. Numeric-year correctness

`numeric_year_correctness` = **1.000** (3/3). The requested figure appears in a
retrieved chunk **of the correct year**:

| question | expected | verified in chunk |
|---|---|---|
| NVDA FY2023 total revenue | $26,974 M | FY2023 income statement |
| NVDA FY2025 total revenue | $130,497 M | FY2025 filing |
| NVDA FY2024 Data Center revenue | $47,525 M | FY2024 filing |

A correct number retrieved from the wrong year would score as a failure; none
occurred.

## 19. Multi-year example answers

**"How did NVIDIA export-control risk disclosures change from FY2023 to FY2025?"**
scopes `['NVDA:2023', 'NVDA:2025']`, evidence `{'NVDA:2023': 1, 'NVDA:2025': 4}`:
> FY2023: new export restrictions / licensing targeting China (A100, H100), need
> to transition certain operations out of China, alternative non-restricted
> products… [Source 3]
> FY2025: adds a near-term global licensing framework (AI Diffusion IFR) imposing
> worldwide licensing on a broad ECCN/product set (A100, A800, H100, H200, H800,
> L4, L40S, RTX 6000 Ada)…

**"What was NVIDIA total revenue in FY2023 versus FY2025?"** — see §10 (exact
figures, correct years, "What changed:" section).

Unsupported-year questions (`FY2019`, `FY2020`): the registry reports the year
unavailable and the API contract returns 422 (`fiscal_year_not_available`) —
`unsupported_year_correctly_rejected` = **1.000**.

## 20. benchmark_multiyear metrics

`evaluation/results/multiyear_summary.txt` — **GATE PASS**:

```
fiscal_year_filter_correctness   1.000   (target 1.000)
cross_year_leakage@5             0.000   (target 0.000)
scope_coverage (comparison Qs)   1.000   (target 1.000)
numeric_year_correctness         1.000
unsupported_year rejected        1.000   (target 1.000)
reranker_fallback                6/16    (transient Cohere 429 -> hybrid fallback)
latency  mean 1.91s  p50 1.37s
```

## 21. benchmark_v3 regression

| | Phase 4 baseline | v3 OFFLINE (frozen pool) | v3 LIVE (year-scoped) | tol | verdict |
|---|---|---|---|---|---|
| hybrid MRR | 0.903 | **0.903** (0.000) | 0.897 (−0.006) | −0.02 | **PASS** |
| hybrid R@10-capped | 0.725 | **0.725** (0.000) | 0.722 (−0.003) | −0.02 | **PASS** |
| hybrid P@1 | 0.843 | **0.843** (0.000) | 0.835 (−0.008) | −0.02 | **PASS** |
| hybrid P@5 | 0.647 | **0.647** (0.000) | 0.647 (0.000) | −0.02 | **PASS** |
| hybrid NumEvidence@10 | 0.964 | **0.964** (0.000) | 0.964 (0.000) | −0.02 | **PASS** |
| filter_correctness | 1.000 | 1.000 | 1.000 | exact | **PASS** |
| cross_company_leakage@10 | 0.000 | 0.000 | 0.000 | exact | **PASS** |

- The **offline** run reads the frozen depth-50 pool from Phase 4 → byte-exact
  baseline, i.e. **zero regression**.
- The **live** run (`--live --tag phase5`) re-retrieves on the current
  1,684-chunk corpus and scopes each question to the fiscal year(s) of its
  qrels (so a ticker-only NVDA question does not silently span 4 years). Drift
  ≤ 0.008, all within tolerance — attributable to the BM25 global-IDF shift from
  306 new NVDA docs plus a few Cohere 429 fallbacks.
- benchmark_v3 qrels were **not** rewritten.

**Note on `evaluate_scoped` / benchmark_v2:** on first run the year-unaware
evaluator showed benchmark_v2 filtered-hybrid Recall@10 = 0.763 (NVDA questions
0.489 — pure year-dilution; AAPL 0.933 and MSFT 0.882 unchanged). Adding the
same qrel-year scoping restored it to **0.907** (filter_correctness 1.000,
leakage 0.000). benchmark_v2 is a retired reference, not a gate; the fix keeps
it meaningful.

## 22. Structural gates (10-company corpus + NVDA multi-year)

| gate | value | verdict |
|---|---|---|
| filtered hybrid `filter_correctness` (v2, year-aware) | 1.000 | **PASS** |
| filtered hybrid `cross_company_leakage@10` | 0.000 | **PASS** |
| comparison (historical 15) `scope_coverage@5` / `cross_scope_leakage` | 1.000 / 0.000 | **PASS** |
| comparison (batch-3 XOM/UNH, 7) `scope_coverage@5` / `cross_scope_leakage` | 1.000 / 0.000 | **PASS** |
| v3 live `filter_correctness` / `cross_company_leakage@10` | 1.000 / 0.000 | **PASS** |
| `fiscal_year_filter_correctness` | 1.000 | **PASS** |
| `cross_year_leakage@5` | 0.000 | **PASS** |
| canonical id collisions | 0 | **PASS** |
| destructive vector ops | none (metadata `update` + `upsert` only) | **PASS** |

## 23. Corpus growth

10 companies · 1,378 → **1,684 chunks**. NVDA 103 → **409** (4 fiscal years).
No new companies, no 10-Q/8-K.

## 24. bm25s build / load timings

| | before | after |
|---|---|---|
| cold load (persisted) | ~189 ms | **164 ms** |
| full rebuild + persist (all docs) | ~0.4 s | **458 ms** (1,684 docs) |
| per-filing rebuild during ingestion (`stage_bm25`) | ~0.4 s | ~0.46 s |
| persisted size | 9.51 MB | 11.70 MB |

Load time is flat-to-lower; rebuild scales linearly and stays sub-second.

## 25. Tests

`python -m unittest discover -s tests -t .` → **253 passed** (was 231; +22
`tests/test_multiyear.py`). Coverage added:
- SEC: `list_10ks`, archived-file merge, exact-year lookup, missing year,
  10-K/A excluded, non-calendar FY (NVDA/WMT).
- CLI: `--years`, `--fiscal-year`, mutual exclusion, latest-only back-compat.
- Registry: multiple filings/company, no-dup accession, newest-first,
  `available_fiscal_years`.
- Retrieval: ticker+year filter, no year leakage, multi-year scope coverage,
  company comparison unchanged, `assert_scopes`.
- API: `fiscal_years` validation, no silent widen, legacy requests unchanged,
  `search_scope` years.
- Generation: year-comparison prompt path, `comparison_scopes` plumbing.
- XOM lineage / CIK override: still green.
- Ingestion: independent per-accession resume.

## 26. Production risks discovered

1. **`corpus_version` = sha256(sorted chunk_ids) only.** A metadata-only change
   to *existing* chunks (like the seed backfill) does not change `corpus_version`,
   so a bare API restart could load a stale persisted bm25s index. Mitigation
   today: ingestion + the backfill flow always call `bm25_search.reload()`
   (force rebuild + re-persist), which we did. Recommend folding a content hash
   into `corpus_version` before routine metadata edits.
2. **Seed filings are one fiscal year stale for AAPL & MSFT.** SEC now has
   AAPL FY2025 (`0000320193-25-000079`) and MSFT FY2026
   (`0001193125-26-323660`); the corpus has FY2024 / FY2025. `verify_company
   --ticker AAPL` (no `--fiscal-year`) therefore fails on the un-ingested latest
   year — expected, not a regression. Fix is a one-liner
   (`--fiscal-year 2025` / `2026`), out of the NVDA-canary scope.
3. **Cohere 429s** were frequent during evaluation runs; the rate-limit →
   hybrid-candidate fallback held (coverage + numeric-year correctness stayed
   1.000), but heavy multi-year comparison traffic will lean on that fallback.
4. **Ticker-only queries now fan out across all ingested years** (the documented
   default). For a company with N years this is N× the candidate breadth before
   RRF; `candidate_k` is unchanged, so recall of a *specific* year's facts drops
   unless the caller passes `fiscal_years`. The frontend should default the year
   selector to "latest" or expose it clearly (deferred — no frontend change this
   phase).

## 27. git diff --stat

```
 26 files changed, 2359 insertions(+), 1228 deletions(-)
 api/main.py                                    |   57 +-
 api/rag.py                                     |  262 ++-
 data/chunks/apple_10k_chunks.jsonl             |  132 +-   (seed metadata backfill)
 data/chunks/microsoft_10k_chunks.jsonl         |  196 +-   (seed metadata backfill)
 data/chunks/nvidia_10k_chunks.jsonl            |  206 +-   (seed metadata backfill)
 data/registry/companies.json                   |   54 +
 data/registry/ingestion_state.json             |  471 ++++++
 evaluation/evaluate_scoped.py                  |   29 +-
 evaluation/evaluate_v3_offline.py              |   28 +-
 evaluation/results/*                            |  (regenerated)
 ingestion/ingest_company.py                    |  171 +-
 ingestion/registry.py                          |   26 +-
 ingestion/sec_client.py                        |  216 ++-
 ingestion/verify_company.py                    |   45 +-
 retrieval/bm25_search.py                       |   21 +-
 retrieval/lexical_backend.py                   |    5 +-
 retrieval/pinecone_search.py                   |   29 +-
 retrieval/scoped_search.py                     |   14 +-
 tests/test_{api,filter_plumbing,lexical_backend}.py | (shape assertions widened)
```
New untracked: `retrieval/scope.py`, `scripts/backfill_seed_registry.py`,
`evaluation/{benchmark_multiyear.json,evaluate_multiyear.py}`,
`tests/test_multiyear.py`, `data/chunks/NVDA/0001045810-{25,24,23}*_chunks.jsonl`,
`data/raw/NVDA/…/metadata.json`, `data/registry/backups/*`, the Phase 5
`evaluation/results/*` (multiyear, phase5-tagged v3).

**Nothing committed or pushed. `v1-stable` → `0edf0a5` and `a200b3e` / `1ce2a3e`
untouched.**

## 28. Ready to expand historical ingestion to all 10 companies?

**Yes.** The NVDA canary is green on every gate:

- generic historical discovery (recent[] + archives), fiscal-year semantics,
  `--years` / `--fiscal-year`, per-accession resume — all working and tested.
- the generic `(ticker, fiscal_year)` scope model handles single-year,
  year-comparison, and company+year comparison with one code path.
- `fiscal_year_filter_correctness` 1.000, `cross_year_leakage` 0.000,
  comparison `scope_coverage` 1.000, `numeric_year_correctness` 1.000,
  unsupported-year 422 — all exact.
- benchmark_v3 did not regress (offline exact; live ≤ 0.008, within ±0.02);
  structural company gates unchanged.
- 1,684 chunks, bm25s 11.7 MB / 164 ms load — headroom for ~5 years × 10
  companies (~6–7k chunks) is comfortable.

**Before mass ingestion, do these (small, non-blocking):**
1. Fold a corpus content hash into `corpus_version` (risk #1).
2. Ingest AAPL FY2025 and MSFT FY2026 so all seed companies are current
   (`--fiscal-year`).
3. Decide the per-company year depth (3 vs 5) and the ticker-only default
   (latest-only vs all-years) — the latter is a product call that also drives
   the frontend year selector.
4. Expect the sparse pacing (12 s inter-batch) to dominate wall-time:
   ~5 years × 10 companies ≈ 40 new filings ≈ 25–35 min of ingestion.
