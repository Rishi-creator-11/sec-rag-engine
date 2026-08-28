# Phase 4 Batch 3 — XOM + UNH (10-company pilot completion)

Date: 2026-08-28
Scope: ingest ExxonMobil (XOM) and UnitedHealth Group (UNH) 10-Ks, completing the
controlled 10-company pilot. Primary retrieval-quality gate: **benchmark_v3**
(the benchmark_v2 Recall@10 ≥ 0.900 gate is retired). Nothing committed or pushed.

---

## 1. New capability — explicit `--cik` override

XOM could not be ingested by the normal path: ExxonMobil did a holding-company
reorganization effective **2026-07-01**. SEC `company_tickers.json` now maps
`XOM` → **ExxonMobil Holdings Corp** (CIK `0002115436`), which has filed 8-Ks and
a 10-Q but **no 10-K**. The real FY2025 10-K was filed by the legacy operating
entity **EXXON MOBIL CORP** (CIK `0000034088`), since delisted.

A generic, explicit override was added (no XOM-specific logic anywhere):

```
python -m ingestion.ingest_company --ticker XOM --cik 0000034088 \
  --successor-cik 0002115436 \
  --successor-name "ExxonMobil Holdings Corporation" \
  --successor-effective-date 2026-07-01
```

**Behaviour (`ingestion/sec_client.py::discover_latest_10k`):**
- `--cik` bypasses **only** ticker→CIK resolution. Filing discovery still runs
  against `data.sec.gov/submissions/CIK{cik}.json` for that exact CIK.
- The requested ticker is validated (`normalize_ticker`) and carried through as
  the retrieval scope; the CIK is validated (`normalize_cik`).
- Every override logs `sec_client event=cik_override ticker=… cik=… reason=explicit_flag`
  and `ingest cik_override_active …`. **Never silent.**
- If the CIK's own `tickers` list is non-empty and does not contain the requested
  ticker, a `cik_override_ticker_mismatch` warning is logged (expected here).
- `--successor-*` are optional lineage annotations; they never alter the filing's
  SEC-authoritative registrant metadata.
- No `--cik` → unchanged behaviour (normal `resolve_cik`).

**Model / registry changes:**
- `DiscoveredFiling` gains `cik_override: bool` and `successor_cik/_name/_effective_date`.
- `registry.upsert_company(..., lineage=…)` writes a `lineage` block; passing
  `None` never clears an existing one.
- `ingestion/verify_company.py` reads `registry lineage.registrant_cik`
  automatically (also accepts `--cik`), so `verify_company --ticker XOM` works
  without repeating the flag.

**Registry entry for XOM (`data/registry/companies.json`):**
```json
{
  "ticker": "XOM",
  "cik": "0000034088",
  "legal_name": "EXXON MOBIL CORP",
  "lineage": {
    "cik_override": true,
    "registrant_cik": "0000034088",
    "registrant_legal_name": "EXXON MOBIL CORP",
    "successor_cik": "0002115436",
    "successor_legal_name": "ExxonMobil Holdings Corporation",
    "successor_effective_date": "2026-07-01",
    "note": "Ticker XOM is currently registered to CIK 0002115436 (ExxonMobil
             Holdings Corporation) effective 2026-07-01. This 10-K (accession
             0000034088-26-000045, FY2025) was filed by CIK 0000034088
             (EXXON MOBIL CORP); its metadata is kept SEC-authentic."
  },
  "filings": [{ "accession_number": "0000034088-26-000045", "fiscal_year": 2025,
               "chunk_count": 149, ... }]
}
```

The filing itself is **not** rewritten as if the holdco filed it: `company_name`
stays `EXXON MOBIL CORP`, `cik` stays `0000034088`, chunk metadata carries
`cik=0000034088`, and the source URL is `…/data/34088/…`.

**Tests added — `tests/test_cik_override.py` (18 tests):** override discovers
from the explicit CIK; bad CIK rejected; override is logged and sets
`cik_override=True`; registrant metadata stays SEC-authentic; canonical chunk
ids + `RetrievalFilter` use the product ticker `XOM` (not the successor CIK);
ordinary tickers still use normal resolution; successor lineage recorded and
distinct from the registrant; `--successor-*` requires `--cik`; `ingest_company`
and the CLI thread the flag through.

---

## 2. XOM filing metadata

| field | value |
|---|---|
| search ticker | XOM |
| registrant (SEC-authoritative) | EXXON MOBIL CORP |
| registrant CIK | 0000034088 |
| current successor | ExxonMobil Holdings Corporation — CIK 0002115436, effective 2026-07-01 |
| form / fiscal year | 10-K / 2025 |
| accession | 0000034088-26-000045 |
| primary document | xom-20251231.htm |
| filing date / report date | 2026-02-18 / 2025-12-31 |
| source | https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm |
| raw HTML | 5,591,068 bytes · clean text 419,592 chars |

## 3. UNH filing metadata

| field | value |
|---|---|
| search ticker | UNH |
| registrant | UNITEDHEALTH GROUP INC (normal ticker resolution, no override) |
| CIK | 0000731766 |
| form / fiscal year | 10-K / 2025 |
| accession | 0000731766-26-000062 |
| primary document | unh-20251231.htm |
| filing date / report date | 2026-03-02 / 2025-12-31 |

## 4. Chunk / dense / sparse counts

| ticker | chunks | dense upserted / verified | sparse | bm25s |
|---|---|---|---|---|
| XOM | **149** | 149 / 149 | ok (149) | registered |
| UNH | **103** | 103 / 103 | ok (103) | registered |

Content hashes recorded in the ledger for both (`primary_doc_sha256`,
`clean_text_sha256`, `chunks_sha256`, `embeddings_sha256`). Canonical ids
`XOM_2025_10-K_000003408826000045_{0..148}` and
`UNH_2025_10-K_000073176626000062_{0..102}` — collision check PASS, all ids
unique, no cross-filing contamination.

## 5. verify_company --full (all 10)

| ticker | LOCAL | DENSE | SPARSE | SERVING | result |
|---|---|---|---|---|---|
| XOM | PASS | PASS | PASS | PASS | **PASS** |
| UNH | PASS | PASS | PASS | PASS | **PASS** |
| AMZN GOOGL META WMT | PASS | PASS | PASS | PASS | **PASS** |
| JPM | PASS | PASS | PASS¹ | PASS | **PASS** |
| AAPL MSFT NVDA | fail² | — | — | present³ | fail² |

1. `verify_company --full` on JPM initially hit `ApiError [431] Request Header
   Fields Too Large` — the SPARSE check fetched all 400 ids in one request.
   Fixed: the sparse fetch now batches at 50 ids (mirrors the DENSE check). All
   400 JPM sparse records are present. Pre-existing latent bug, exposed by
   `--full` on the largest filing; unrelated to XOM/UNH.
2. AAPL/MSFT/NVDA are the pre-pipeline **seed** companies — they have no
   ingestion-ledger entry, so `verify_company` (which is ledger-driven) reports
   LOCAL fail. Not a regression.
3. Their chunks are present and correct in the lexical index (10/10 tickers
   return only their own chunks, zero leakage), the dense index, and the serving
   layer — verified directly.

Registry consistency: 10 companies, `legal_name`/`display_name` intact, every
filing recorded with its chunk count; XOM carries the lineage block.

## 6. Final 10-company corpus

| | |
|---|---|
| companies | AAPL AMZN GOOGL JPM META MSFT NVDA UNH WMT XOM |
| total chunks | **1,378** |
| seed (AAPL 66, MSFT 98, NVDA 103) | 267 |
| pipeline (AMZN 89, GOOGL 108, JPM 400, META 152, UNH 103, WMT 110, XOM 149) | 1,111 |
| dense index `sec-rag-engine` | 1,378 vectors (1536-d) |
| sparse index `sec-rag-sparse` | 1,378 records (not used in production retrieval) |

## 7. Final bm25s index stats

| | |
|---|---|
| documents | **1,378** |
| method / params | lucene, k1 = 1.5, b = 0.75, dtype float64 |
| persisted size | 9.51 MB (`data/bm25s_index/`, 7 files, gitignored) |
| cold load | ~189 ms |
| rebuild (149 + 103 new docs, twice) | ~0.4 s each (ingestion `stage_bm25`) |
| 10-ticker scoped probe | 100 hits, **0 cross-company leakage** |

## 8. benchmark_v3 retrieval quality — regression tracking

Live retrieval, scoped, top-10, `candidate_k=10` unchanged. Baseline is the
8-company v3 number; the gate is "no drop > 0.02".

| metric (hybrid) | v3 baseline (8 co) | after XOM (9 co) | **after UNH (10 co)** | floor | verdict |
|---|---|---|---|---|---|
| MRR | 0.903 | 0.901 | **0.901** | 0.881 | **PASS** (−0.002) |
| Recall@10-capped | 0.725 | 0.719 | **0.716** | 0.705 | **PASS** (−0.009) |
| R-precision | 0.553 | 0.550 | **0.552** | — | flat |
| P@1 | 0.843 | 0.843 | **0.843** | 0.823 | **PASS** (0.000) |
| P@5 | 0.647 | 0.642 | **0.640** | 0.622 | **PASS** (−0.007) |
| NumEvidence@10 | 0.964 | 0.964 | **0.964** | 0.944 | **PASS** (0.000) |

10-company retriever comparison (v3 live): hybrid MRR 0.901 ≥ bm25s 0.877 ≥
dense 0.871; hybrid P@1 0.843, NE@10 0.964. The dense + bm25s RRF mix stays the
best config.

The small cumulative drift (MRR −0.002, R@10-capped −0.009, P@5 −0.007 across
**both** ingestions) is the expected effect of 252 new documents shifting the
global BM25 IDF. No catastrophic per-question regression: no supported question
went from hit to complete miss because of XOM/UNH.

## 9. Structural gates (10-company corpus) — all PASS

| gate | value | verdict |
|---|---|---|
| filtered hybrid `filter_correctness == 1.000` | 1.000 | **PASS** |
| filtered hybrid `cross_company_leakage@10 == 0.000` | 0.000 | **PASS** |
| comparison (historical 15) `scope_coverage@5 == 1.000` | 1.000 | **PASS** |
| comparison (historical 15) `cross_scope_leakage == 0.000` | 0.000 | **PASS** |
| comparison (batch-3 XOM/UNH, 7 Qs) `scope_coverage@5 == 1.000` | 1.000 | **PASS** |
| comparison (batch-3 XOM/UNH, 7 Qs) `cross_scope_leakage == 0.000` | 0.000 | **PASS** |
| v3 live `filter_correctness` / `cross_company_leakage@10` | 1.000 / 0.000 | **PASS** |
| canonical id collisions | none | **PASS** |
| parser corruption | none (samples inspected) | **PASS** |
| destructive index ops | none (upsert-only, no create/delete) | **PASS** |

## 10. XOM / UNH canary suites

New: `evaluation/benchmark_xom.json`, `evaluation/benchmark_unh.json`
(10 questions each: 5 numeric with hints verified against the filed 10-K, plus
business/regulation/cyber/risk qualitative). Run with
`evaluation.evaluate_company_canary`.

| | XOM | UNH |
|---|---|---|
| result | **PASS** | **PASS** |
| filter_correctness | 1.000 | 1.000 |
| cross_company_leakage | 0.000 | 0.000 |
| scoped_retrieval_rate | 1.000 | 1.000 |
| numeric_evidence_present_rate (lexical proxy) | 0.80 (4/5) | 1.00 (5/5) |
| reranker_fallback | 0/10 | 10/10¹ |
| latency mean | 0.67 s | 3.61 s |

1. Cohere was returning `429` during the UNH canary run — all 10 questions fell
   back to hybrid candidates (graceful degradation working as designed; the
   XOM canary minutes earlier had 0 fallbacks). Numeric evidence still 5/5.
   Transient quota, not UNH-specific.

The one XOM numeric miss is `xom_rd_costs_2025_01` (hint `1,228` — R&D costs, a
minor line item): the exact chunk was outside the top-5. Not judged; it is a
strict digit-string proxy.

End-to-end sanity (retrieval scope = ticker XOM despite the CIK override):
> Q: "What total revenues did ExxonMobil report for 2025?"
> A: **"$332,238 million."** (matches xom-20251231.htm)

## 11. Cross-company comparison examples (`benchmark_comparisons_batch3.json`)

All 7 hit `scope_coverage 1.00 / leak 0.00`:

| id | scopes | evidence by scope |
|---|---|---|
| xom_cvx_style_climate_msft_01 | XOM+MSFT | {XOM: 3, MSFT: 2} |
| xom_wmt_supply_chain_01 | XOM+WMT | {XOM: 2, WMT: 3} |
| xom_nvda_cybersecurity_01 | XOM+NVDA | {XOM: 3, NVDA: 2} |
| unh_jpm_regulation_01 | UNH+JPM | {UNH: 2, JPM: 3} |
| unh_aapl_cybersecurity_01 | UNH+AAPL | {UNH: 3, AAPL: 2} |
| xom_unh_competition_01 | XOM+UNH | {XOM: 3, UNH: 2} |
| xom_unh_jpm_regulation_01 | XOM+UNH+JPM | {XOM: 2, UNH: 1, JPM: 2} |

The 3-company XOM+UNH+JPM comparison guarantees ≥1 chunk per requested scope.

## 12. Parser / filing surprises

1. **XOM ticker→registrant succession** (the whole reason for §1). Handled with
   the explicit, logged, tested `--cik` override + registry lineage.
2. XOM 10-K text confirms authenticity — references the **Pioneer Natural
   Resources acquisition (2024)**, ROCE methodology, sensitivity tables. Clean
   parse: cover page → MD&A/notes → exhibits → signature/POA. 149 chunks,
   1,428–4,614 chars.
3. UNH 10-K — clean. References the **Change Healthcare 2024 cyberattack** in
   Item 1C. Auditor's report (PCAOB) mid-document, signature page
   ("Dated: March 2, 2026 … Stephen Hemsley, Chief Executive Officer") at the
   end. 103 chunks.
4. No encoding damage, no boilerplate bleed, no HTML artifacts in either.

## 13. Latency / build / load observations

| operation | XOM | UNH |
|---|---|---|
| download | 0.39 s | 0.36 s |
| clean | 0.78 s | 0.42 s |
| chunk | 0.08 s | 0.07 s |
| embed | 3.20 s | 1.86 s |
| dense upsert + verify | 3.54 s | 2.67 s |
| sparse upsert (12 s inter-batch pacing) | 29.0 s | 28.2 s |
| bm25s rebuild + persist | 0.41 s | 0.42 s |
| **total** | **37.5 s** | **34.1 s** |

- Sparse dominates ingestion wall-time purely from the deliberate 12 s
  inter-batch sleep (rate-limit safety from the JPM incident). Sparse is not on
  the production retrieval path.
- bm25s: 189 ms cold load for 1,378 docs; ~0.4 s full rebuild.
- Retrieval latency unchanged: comparison `plan_evidence` p50 0.54 s / p95
  3.56 s; XOM canary mean 0.67 s.

## 14. Tests

`python -m unittest discover -s tests -t .` → **231 passed** (was 213 before
this batch; +18 `test_cik_override.py`). Includes the updated
`test_ingestion_hardening` (verify fake-client signature + no change to
behaviour) and `test_ingest_company` (fake client `**kwargs`).

## 15. git diff --stat

```
 .gitignore                                     |    7 +
 api/main.py                                    |   35 +-
 data/registry/companies.json                   |  140 ++-
 data/registry/ingestion_state.json             |  534 ++++++++
 evaluation/corpus_growth.json                  |   53 +-
 evaluation/evaluate_comparison.py              |   49 +-
 evaluation/results/…                           |  (regenerated)
 ingestion/ingest_company.py                    |  110 +-
 ingestion/registry.py                          |   17 +-
 ingestion/sec_client.py                        |   86 +-
 ingestion/stages.py                            |   18 +-
 ingestion/verify_company.py                    |   41 +-
 requirements.txt                               |    1 +
 retrieval/bm25_search.py                       |   41 +-
 retrieval/pinecone_store.py                    |   28 +-
 retrieval/sparse_store.py                      |   39 +-
 tests/test_api.py                              |   22 +
 tests/test_filter_plumbing.py                  |   32 +-
 tests/test_ingest_company.py                   |    2 +-
 tests/test_ingestion_hardening.py              |   57 +-
 25 files changed, 1970 insertions(+), 872 deletions(-)
```
New untracked files for this batch: `tests/test_cik_override.py`,
`evaluation/benchmark_xom.json`, `evaluation/benchmark_unh.json`,
`evaluation/benchmark_comparisons_batch3.json`, plus the regenerated
`evaluation/results/v3_retrieval_*_10co.*`, `comparison_*_batch3.*`,
`{xom,unh}_canary.json`, `{xom,unh}_ingest.log`. (`ingestion/stages.py`,
`retrieval/{bm25_search,pinecone_store,sparse_store}.py`, `api/main.py`,
`requirements.txt`, `tests/test_{api,filter_plumbing,lexical_backend}.py`,
`retrieval/lexical_backend.py` are Phase 4.5 work carried in the same
uncommitted tree.)

**Nothing committed or pushed. `v1-stable` untouched. No destructive index
operations.**

## 16. Is the 10-company pilot complete and ready for multi-year 10-K support?

**Yes.**

- 10 companies ingested, all `verify_company --full` PASS (the 3 seed-company
  LOCAL failures are the known pre-pipeline ledger gap, not a regression).
- All structural gates hold on the 10-company corpus; all benchmark_v3 quality
  gates hold (worst drift −0.009).
- The ingestion pipeline now handles ticker→registrant successions generically
  (`--cik` + lineage), which is exactly the kind of real-world messiness
  multi-year backfill will hit more often.
- bm25s scales cleanly: 1,378 docs, 9.5 MB, 189 ms load, sub-second rebuild.

**Readiness notes / follow-ups for multi-year (none blocking):**
1. **Canonical ids are already multi-year-safe** — `{TICKER}_{FY}_{TYPE}_{ACCESSION}_{IDX}`
   namespaces by fiscal year and accession. `record_filing` dedups by accession
   and sorts by `(filing_type, fiscal_year)`, so multiple years per company are
   already representable in the registry and ledger.
2. **`discover_latest_10k` only finds the latest** — multi-year needs a
   `discover_10k(cik, fiscal_year=…)` / "all 10-Ks" path (the `latest_filing`
   internals already parse the full `recent` list; it just picks the first
   match).
3. **`RetrievalFilter` already has `fiscal_years`** — retrieval can scope by year
   today; the benchmark/evals would need year-aware questions.
4. **XOM lineage** — when ExxonMobil Holdings Corp files its first 10-K (~Feb
   2027), decide whether that filing lands under the same `XOM` ticker (likely
   yes) with the lineage note updated, or as a distinct registrant series.
5. **`microsoft_cash_2025_01`** (from Phase 4.5) — the MSFT balance-sheet
   single-number retrieval weakness is still open; unrelated to Batch 3.
6. Consider splitting the broad NVDA questions (`nvidia_china_01` etc., 10–19
   relevant chunks each) in a future benchmark revision.
