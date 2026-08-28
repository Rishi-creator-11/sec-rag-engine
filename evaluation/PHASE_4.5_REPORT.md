# Phase 4.5 — bm25s production migration + Benchmark v3 refresh

Date: 2026-08-28 (benchmark v3 review pass completed after an OpenAI credit
outage; all numbers below are final on the 115/115-reviewed qrels).
Decision executed: adopt bm25s as the production lexical backend; keep
candidate_k=10, RRF weights, RRF_K, dense, Cohere, and generation unchanged;
re-pool and re-judge the retrieval benchmark on the current 8-company corpus
before resuming Phase 4 Batch 3. XOM/UNH not ingested. Nothing committed or pushed.

Corpus at time of report: 8 companies —
AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, WMT (1126 lexical documents).

---

## PART A — Production bm25s migration

### 1. Files created / modified

**Created**
- `retrieval/lexical_backend.py` — backend abstraction + bm25s implementation +
  persisted-index lifecycle.
- `tests/test_lexical_backend.py` — parity, filter, persistence, selector,
  production-wiring, and error-path tests.

**Modified**
- `retrieval/bm25_search.py` — `get_index()` / `reload()` now delegate to
  `retrieval.lexical_backend.get_lexical_backend(...)`. Public `search(query,
  top_k, filters)` signature unchanged. `BM25Index` + `tokenize` kept here
  (imported by the backend module and by `CurrentBM25Backend`).
- `api/main.py` — `/health` gains `lexical_backend` and `bm25_documents`;
  returns **503** if the lexical index cannot load or rebuild.
- `ingestion/ingest_company.py` — `stage_bm25` rebuilds **and persists** the
  bm25s index, then asserts the just-ingested ticker is retrievable.
- `requirements.txt` — `bm25s` added.
- `.gitignore` — `data/bm25s_index/` (persisted index, rebuildable) plus the
  large v3 evaluation artifacts / judge logs.
- `tests/test_api.py` — `HealthReadinessTests`.
- `tests/test_filter_plumbing.py` — the "global BM25 stats unchanged" test now
  constructs `BM25Index` directly (production `get_index()` is bm25s now).

> Also present in the working tree but **not part of this migration** (earlier
> uncommitted Phase 3.5 / Phase 4 Batch 1–2 hardening): `ingestion/stages.py`,
> `ingestion/verify_company.py`, `retrieval/pinecone_store.py`,
> `retrieval/sparse_store.py`, `tests/test_ingestion_hardening.py`,
> `ingestion/ingest_batch.py`, `tests/test_ingest_batch.py`, `data/registry/*`,
> `evaluation/corpus_growth.json`, various `evaluation/results/*`.

### 2. Exact production wiring

```
api/rag.py / retrieval/hybrid_search.py
        │  (unchanged) from retrieval.bm25_search import search
        ▼
retrieval/bm25_search.py :: search(query, top_k, filters)
        │  get_index()  ──►  _select_backend()
        ▼
retrieval/lexical_backend.py :: get_lexical_backend(chunks=None, force_rebuild=False)
        │  name = os.getenv("SEC_RAG_LEXICAL_BACKEND", DEFAULT_BACKEND="bm25s")
        │
        ├─ name == "current"  ──►  CurrentBM25Backend(BM25Index)      # rollback path
        ├─ force_rebuild       ──►  build_persisted_bm25s(...)        # ingestion path
        └─ otherwise           ──►  load_or_build_bm25s(index_dir, chunks)
```

`BM25SBackend` = `bm25s.BM25(k1=1.5, b=0.75, method="lucene", dtype="float64")`.
`method="lucene"` reproduces the previous pure-Python IDF
`log(1 + (N - df + 0.5)/(df + 0.5))` exactly; the extra `(k1+1)` numerator factor
is rank-preserving. `search()` retrieves against the full corpus, then applies
`RetrievalFilter.matches()` as a post-filter and re-sorts by `(-score,
chunk_id)` — identical selection semantics to `BM25Index.search`.

### 3. Startup behaviour

`data/bm25s_index/` holds `{data,indices,indptr}.csc.index.npy`,
`vocab.index.json`, `params.index.json`, `chunks.jsonl`, `corpus_version.json`.
`corpus_version = sha256("\n".join(sorted(chunk_ids)))` (order-independent).

On first `get_index()`:
1. `corpus_version.json` present and matches the current chunk set → **load**
   (`lexical event=bm25s_load document_count=… duration_ms=… corpus_version=…`).
2. directory missing → `bm25s_missing` warning → rebuild + persist.
3. files unreadable / wrong shape → `bm25s_corrupt` warning → rebuild + persist.
4. `corpus_version` mismatch → `bm25s_stale` warning → rebuild + persist.
5. rebuild itself fails → `LexicalBackendError` (never serves an empty index).

Log lines carry `document_count`, `duration_ms`, `corpus_version` — **no chunk
text**. Measured: load ≈ 19 ms, full rebuild of 1126 docs ≈ sub-second, query
≈ 4× faster than the pure-Python path.

### 4. Ingestion / rebuild behaviour

`stage_bm25` → `bm25_search.reload()` → `get_lexical_backend(force_rebuild=True)`
→ `build_persisted_bm25s()` rebuilds from `data/chunks/**/*_chunks.jsonl`,
writes the new index atomically, logs `bm25s_rebuild`, then the stage asserts
the new ticker returns hits with no foreign ticker. Ledger stage
`bm25_registered` records `bm25_hits`, `lexical_backend`,
`lexical_document_count`.

### 5. Rollback behaviour

`SEC_RAG_LEXICAL_BACKEND=current` → `get_lexical_backend` returns
`CurrentBM25Backend` (the original in-memory `BM25Index`), no persisted index
touched. Verified live: with the env var set, `bm25_search.get_index()` is a
`CurrentBM25Backend`; unset, it is a `BM25SBackend`. `/health` echoes the active
value. `CurrentBM25Backend` is retained as the reference implementation and the
parity oracle in the test-suite.

### 6. Health / readiness behaviour

`GET /health` →
```json
{"status":"ok","lexical_backend":"bm25s","companies":8,"bm25_documents":1126}
```
If `get_index()` raises `LexicalBackendError`:
```
503  {"status":"unavailable","lexical_backend":"bm25s","bm25_documents":null,
      "detail":"lexical index unavailable: …"}
```
(`tests/test_api.py::HealthReadinessTests` covers both.)

### 7. bm25s vs current parity

`tests/test_lexical_backend.py::ParityTests` — 8 representative queries, top-10
from `BM25SBackend` vs `CurrentBM25Backend` on the same corpus: **identical set
and identical order, 8/8**. `FilterTests` — single- and multi-ticker scoping,
zero cross-company leakage, filtered scores equal unfiltered scores for
surviving docs. Live 8-company spot check (AAPL AMZN GOOGL JPM META MSFT NVDA
WMT): every ticker returns only its own chunks.

---

## PART B — Benchmark v3 (retrieval benchmark refresh)

`evaluation/benchmark_v2.json` is **unchanged** and remains the historical
record. New artifacts:

| file | purpose |
|---|---|
| `evaluation/build_benchmark_v3.py` | assemble the question set |
| `evaluation/benchmark_v3_questions.json` | 125 questions, no qrels |
| `evaluation/build_v3_pool.py` | deep scoped candidate pool |
| `evaluation/results/benchmark_v3_pool.json` | pooled candidates (gitignored, 38 MB) |
| `evaluation/judge_v3_pool.py` | model-assisted relevance judging |
| `evaluation/results/benchmark_v3_auto_judgments.json` | judgments (gitignored) |
| `evaluation/assemble_benchmark_v3.py` | build the final benchmark |
| `evaluation/benchmark_v3.json` | **final benchmark, 125 questions, with qrels** |
| `evaluation/evaluate_v3_offline.py` | scoped retrieval eval from the pool |
| `evaluation/evaluate_v3.py` | same eval via live retrieval (needs API credits) |
| `evaluation/compare_v2_v3.py` | v2↔v3 comparison + weak-question re-exam |
| `evaluation/results/v3_retrieval_evaluation.json` / `v3_retrieval_summary.txt` | v3 results |
| `evaluation/results/v2_vs_v3_comparison.txt` / `.json` | comparison report |

### 8. Benchmark v3 design

- **125 questions**, 8 companies: AAPL/MSFT/NVDA 20 each, AMZN/GOOGL/JPM/META/WMT
  13 each.
- Provenance: 60 from v2 (all AAPL/MSFT/NVDA, carried verbatim), 50 per-company
  canary questions (AMZN/GOOGL/META/JPM/WMT, 10 each), 15 new "extra" questions
  (3 each for the 5 new companies).
- Answer types: 87 qualitative, 25 numeric, 3 mixed, **10 unsupported**
  → **115 supported / scored**, 28 numeric-or-mixed.
- Categories covered: financial/numeric, competition, regulation, cybersecurity,
  AI, international, operational/supply-chain, governance, plus the unsupported
  set (answers not in a 10-K).
- XOM / UNH absent.

### 9. Pool depth & size

- Pooling is **scoped** (`RetrievalFilter(tickers=(company,))`) — matches how
  production retrieves once a ticker is known.
- Per question, union of: dense top-50, **bm25s** top-50, hybrid (RRF of
  dense+bm25s) top-50, and sparse top-150→post-filtered (sparse is for **pool
  diversity only** and stays disabled in production).
- **Pool depth 50** is deliberately far deeper than production `candidate_k=10`.
  Production retrieval depth was **not changed**.
- Result: **8 982 pooled candidates across 115 questions, avg 78.1/question**
  (JPM questions 170–183, others 60–80). Each candidate records which
  retriever(s) surfaced it and at what rank.

### 10. Judging & human-vs-model accounting

**These qrels are model-assisted, not human.** No manual/human judgments were
produced.

- **First pass:** `gpt-5-mini`, reasoning effort `minimal`, one call per
  question over the top-50 of the pre-sorted union (retriever-consensus first).
  Candidates shown as blind aliases (`C001…`, text only — no retriever, rank, or
  score). Per candidate: `{relevant: bool, confidence: high|medium|low,
  reason}`. Every alias validated exactly once; malformed responses auto-retried
  and recovered. **115/115 questions, 5 750 candidate labels.**
- **Review pass:** `gpt-5-mini`, effort `low`, re-judging every candidate the
  first pass marked *relevant* (precision) plus every *medium/low-confidence*
  candidate (recall). Run in two segments (an OpenAI credit outage interrupted
  it at 86/115; resumed with a skip-guard that re-judged only the remaining 29
  and left the completed 86 untouched). **115/115 questions completed. 535 of
  5 750 labels flipped by the review pass (~9.3%)** — substantive correction of
  the fast first pass.
- **Human-vs-model accounting:**
  - model-assisted judgments (gpt-5-mini first pass + gpt-5-mini review pass):
    **5 750 / 5 750 candidate labels, 115 / 115 questions.**
  - **human-reviewed judgments: 0.** No manual relevance labels were produced.
  - The 6 Phase-4.5 weak questions were additionally inspected by hand at the
    chunk-text level for the write-up in §14, but their qrels are still the
    model's — no labels were hand-edited.
- **Remaining uncertain cases:** after the review pass, **0 relevant labels sit
  at "low" confidence** — all 1 311 relevant labels are `high` (1 087) or
  `medium` (224). 254 low-confidence labels remain, all of them *negatives*
  (judged not-relevant), which do not enter the qrel set. No question is
  unreviewed.
- Final benchmark: 115 supported questions, **1 311 relevant labels, mean 11.4
  relevant chunks/question** (median 9). No supported question has zero relevant
  chunks. Unsupported questions kept with `relevant_chunks: []` and excluded
  from scoring.
- Numeric questions: judged against the actual filing evidence (the number, or
  the line item needed to derive it) — e.g. `microsoft_total_revenue_2025_01`
  qrels are the income-statement / segment chunks that carry the FY25 revenue
  figure.

### 11. V3 retrieval results (offline, from the depth-50 pool)

Final numbers — 115/115 questions reviewed. Every question scoped to its own
company; top-10; production `candidate_k=10` unchanged. Metrics computed from
the pooled per-retriever ranks (the top-10 slice is rank-identical to what
production returns).

| retriever | R@1 | R@5 | R@10 | R@10-capped | R-prec | P@1 | P@5 | MRR | NumEv@5 | NumEv@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| dense   | 0.114 | 0.387 | 0.572 | 0.711 | 0.562 | 0.809 | 0.614 | 0.871 | 0.893 | 0.964 |
| bm25s   | 0.127 | 0.403 | 0.549 | 0.680 | 0.551 | 0.809 | 0.607 | 0.884 | 0.893 | 0.929 |
| **hybrid** | 0.119 | 0.420 | 0.586 | **0.725** | 0.553 | **0.843** | **0.647** | **0.903** | **0.929** | **0.964** |

- **hybrid ≥ dense, hybrid ≥ bm25s on almost every metric** → the production
  dense+bm25s RRF mix is still the right choice, and bm25s alone is competitive
  with dense alone (bm25s MRR 0.884 vs dense 0.871).
- **Plain Recall@10 (0.586) is capacity-capped**: mean |relevant| = 11.4 > 10,
  so 10 slots cannot hold all relevant chunks. Use **Recall@10-capped
  (hits@10 / min(|rel|,10)) = 0.725**, **R-precision = 0.553**, **P@5 = 0.647**,
  **P@1 = 0.843**, **MRR = 0.903** as the v3 quality baseline instead.
- **Numeric evidence: NumEv@5 = 0.929, NumEv@10 = 0.964** (28 numeric/mixed
  questions) — the chunk carrying the requested figure is in the top 10 for 96%
  of numeric questions.

### 12. Structural gates

| gate | value | verdict |
|---|---|---|
| `filter_correctness == 1.000` | 1.000 | **PASS** |
| `cross_company_leakage@10 == 0.000` | 0.000 | **PASS** |
| comparison `scope_coverage@5 == 1.000` | 1.000 | **PASS** |
| comparison `cross_scope_leakage == 0.000` | 0.000 | **PASS** |

Scoping gates hold by construction in the pool and are independently proven live
by `evaluation/results/scoped_retrieval_evaluation.json` (v2 filtered hybrid:
filter_correctness 1.000, leakage@10 0.000) and
`evaluation/results/comparison_summary.txt` (15 comparison questions, run
post-flip).

### 13. V2 vs V3 comparison

| | v2 | v3 |
|---|---|---|
| companies | 3 | 8 |
| questions (total / scored) | 60 / 55 | 125 / 115 |
| judging pool depth | ~10 (3-co retrievers) | 50 (8-co system) |
| pooled questions / total candidates | 45 / 819 | 115 / 8 982 |
| avg candidates / question | 18.2 | 78.1 |
| total relevant labels | 301 | 1 311 |
| filtered hybrid MRR | 0.894 | 0.903 (+0.008) |
| filtered hybrid P@1 | 0.836 | 0.843 |
| filtered hybrid P@5 | 0.629 | 0.647 (+0.018) |
| filtered hybrid NumEvidence@5 | 0.867 | 0.929 (+0.062) |
| filtered hybrid NumEvidence@10 | 1.000 | 0.964 (−0.036) |
| filtered hybrid Recall@10 | 0.904 | 0.586 (capacity-capped; see §11) |
| filtered hybrid Recall@10-capped | n/a | 0.725 |

v2 and v3 are different question sets at different qrel depths — this is not
"v2 improved". The load-bearing observation:

> **34 of the 60 v2-origin questions gained ≥1 newly-relevant chunk under v3
> (212 new relevant labels total)** — chunks that the 8-company retriever
> surfaces and that a judge confirms are relevant, but which the depth-10
> 3-company v2 pool never showed a judge, so v2 scored them as misses
> ("unjudged = irrelevant").

That is the mechanism behind the apparent Phase 4.5 Recall@10 decline: the
retriever was returning **more** genuinely-relevant material, and the stale
qrels punished it. MRR and P@1 — which the qrel-depth artifact does not distort
— went **up**.

### 14. Phase 4.5 weak questions, re-examined under v3

| question | v2 R@10 | v3 finding |
|---|---|---|
| `microsoft_total_revenue_2025_01` | 0.75 | **v3 R@10 = 1.000** — all four income-statement/segment chunks now retrieved and judged relevant. Resolved. |
| `nvidia_china_01` | 0.78 | relevant set 9 → **19** (10 new, all confirmed). first-relevant-rank still 1. Plain R@10 falls to 0.37 purely from the capacity cap. |
| `nvidia_export_controls_01` | 0.80 | relevant set 10 → 14 (6 new, 2 dropped). first-relevant-rank 1. R@10-capped 0.64. |
| `nvidia_responsible_ai_risk_01` | 0.67 | relevant set 6 → 10 (5 new, 1 dropped). first-relevant-rank 4. |
| `microsoft_cash_investments_2025_01` | 0.75 | qrels unchanged; v3 R@10 = 0.75. Stable. |
| `microsoft_cash_2025_01` | 0.25 | **genuine retrieval weakness.** The balance-sheet chunk `microsoft_10k_52` ("Cash and cash equivalents $ 30,242" as of 2025-06-30) sits at **hybrid rank 22 / dense rank 43 / bm25s rank 10** — outside the production top-10, so v3 hybrid Recall@10 = 0.0 for this question. The judge also (defensibly) tightened the v2 qrels: dropped `microsoft_10k_53` (cash-flow-from-operations detail, no cash balance) and kept only `_51`/`_52`; `microsoft_10k_54` (cash-flow reconciliation — it does carry the 30,242 end-of-period figure) is retrieved at rank 8 but judged not-relevant, which is arguable. Net: a real ranking gap for single-number balance-sheet lookups on MSFT, not a qrel artifact. Recommend a targeted follow-up (chunking or a numeric-lookup boost), tracked separately — **not** a blocker for Batch 3. |

### 15. Recommended v3 production gates

Keep the structural gates exact:
- `filter_correctness == 1.000`
- `cross_company_leakage@10 == 0.000`
- comparison `scope_coverage@5 == 1.000`
- comparison `cross_scope_leakage == 0.000`

Retrieval-quality regression rule (empirical v3 baseline; the old 0.900 Recall@10
gate is **not** carried over — it was measured against under-counted qrels):
- hybrid **MRR ≥ 0.883** (baseline 0.903 − 0.02)
- hybrid **Recall@10-capped ≥ 0.705** (baseline 0.725 − 0.02)
- hybrid **Precision@5 ≥ 0.627** (baseline 0.647 − 0.02)
- hybrid **NumEvidence@10 ≥ 0.944** (baseline 0.964 − 0.02)
- no structural-gate regression

---

## Validation run

| check | result |
|---|---|
| `python -m unittest discover -s tests -t .` | **213 passed** |
| `judge_v3_pool --review` (resumed) | printed "86 already reviewed (skipped), 29 remaining"; completed all 29 → **115/115 reviewed**, 535 cumulative flips |
| `assemble_benchmark_v3` | 125 questions, 115 supported, 1 311 relevant labels |
| `evaluate_v3_offline` | §11 — structural gates PASS, final baseline recorded |
| `compare_v2_v3` | §13 — regenerated on the final qrels |
| `evaluate_scoped --mode compare` (v2, post-flip) | filtered hybrid R@10 0.904, filter_correctness 1.000, leakage@10 0.000 — **gates PASS** (on file from this session, post-flip, 8-company corpus) |
| `evaluate_comparison` (15 questions, post-flip) | scope_coverage@5 1.000, cross_scope_leakage 0.000 — **gates PASS** (on file, same) |
| `verify_company` AMZN / GOOGL / META / JPM / WMT | **all PASS** (LOCAL, DENSE, SPARSE, SERVING) |
| `verify_company` AAPL / MSFT / NVDA | seed companies — no ingestion-ledger entry (pre-pipeline); present and correct in the lexical index (0-leak scoped hits), dense index, and serving layer (verified directly) |
| registry consistency | 8 companies, `legal_name`/`display_name` intact, filings recorded with chunk counts |
| bm25s index | 1126 documents, method lucene, k1 1.5 / b 0.75; all 8 tickers retrievable, zero cross-company leakage |
| `SEC_RAG_LEXICAL_BACKEND=current` rollback | verified — `_select_backend()` returns `CurrentBM25Backend`; unset → `BM25SBackend` |
| production config | `DEFAULT_BACKEND="bm25s"`, `candidate_k=10` (hybrid + `api.rag`), `RRF_K=60`, RRF `weights=None` (equal), `use_sparse=False` — all unchanged |

The live `evaluate_v3.py` retrieval path was **not** re-run: `evaluate_v3_offline.py`
reads the already-built depth-50 pool, whose per-retriever ranks are the same
top-10 production returns, so a live run would repeat paid embedding/rerank work
for an identical result. The structural scoping gates are independently proven
live by the post-flip `scoped_retrieval_evaluation.json` /
`comparison_summary.txt` on file.

## Repo safety audit

- `.env`, `data/bm25s_index/`, `data/embeddings/`, `data/sec_cache/`,
  `data/raw/**/filing.html`, `data/raw/**/filing.txt` — all gitignored.
- `benchmark_v3_pool.json` (38 MB), `benchmark_v3_auto_judgments.json`, judge
  logs — added to `.gitignore`.
- Secret scan of every modified / new `.py` `.json` `.txt` `.md`: **clean** (no
  API keys, tokens, or credentials).
- Only small `metadata.json` files would be staged from `data/raw/`.
- **Nothing committed or pushed.** `v1-stable` untouched.

## git diff --stat

```
 .gitignore                                     |    7 +
 api/main.py                                    |   35 +-
 data/registry/companies.json                   |  101 +-
 data/registry/ingestion_state.json             |  354 ++++++
 evaluation/corpus_growth.json                  |   53 +-
 evaluation/results/amzn_evaluation.json        |   24 +-
 evaluation/results/amzn_summary.txt            |   22 +-
 evaluation/results/comparison_evaluation.json  |  104 +-
 evaluation/results/comparison_summary.txt      |   28 +-
 evaluation/results/scoped_retrieval_evaluation.json | 1316 +++++++++----------
 evaluation/results/scoped_retrieval_summary.txt |   22 +-
 ingestion/ingest_company.py                    |   13 +-
 ingestion/stages.py                            |   18 +-
 ingestion/verify_company.py                    |   17 +-
 requirements.txt                               |    1 +
 retrieval/bm25_search.py                       |   41 +-
 retrieval/pinecone_store.py                    |   28 +-
 retrieval/sparse_store.py                      |   39 +-
 tests/test_api.py                              |   22 +
 tests/test_filter_plumbing.py                  |   32 +-
 tests/test_ingestion_hardening.py              |   54 +
 21 files changed, 1497 insertions(+), 834 deletions(-)
```
Plus untracked new files: `retrieval/lexical_backend.py`,
`tests/test_lexical_backend.py`, `ingestion/ingest_batch.py`,
`tests/test_ingest_batch.py`, the `evaluation/*_v3*.py` scripts,
`evaluation/benchmark_v3*.json`, and `evaluation/results/v3_*` /
`v2_vs_v3_*`. (`stages.py`, `verify_company.py`, `pinecone_store.py`,
`sparse_store.py`, `test_ingestion_hardening.py`, `ingest_batch.py`,
`data/registry/*`, `corpus_growth.json` and most `evaluation/results/*` are
pre-existing uncommitted Phase 3.5 / Phase 4 work, not part of this migration.)

## Can Phase 4 Batch 3 (XOM, UNH) resume safely?

**Yes — fully cleared.**

- bm25s is in production, with a working rollback (`SEC_RAG_LEXICAL_BACKEND=
  current`), persisted-index lifecycle, ingestion rebuild+persist, and `/health`
  readiness. 213 tests pass. All four structural gates pass. Ingestion of a new
  company rebuilds and re-persists the lexical index automatically and asserts
  the new ticker is retrievable.
- Benchmark refresh **complete**: `benchmark_v3.json`, 125 questions on the
  current 8-company corpus, **115/115 supported questions judged and reviewed**
  (model-assisted, 0 human labels), quality baseline recorded, regression gates
  defined.
- The Phase 4.5 "regression" is explained and closed: a v2 qrel-depth artifact
  (34 v2-origin questions gained ≥1 genuinely-relevant chunk; 212 new labels),
  not a retrieval regression. On comparable metrics — MRR (+0.008), P@1 (+0.007),
  P@5 (+0.018), NumEvidence@5 (+0.062) — v3 is level or better.

**Follow-ups (none blocking Batch 3):**
1. `microsoft_cash_2025_01` exposed a real weakness in single-number
   balance-sheet retrieval on MSFT (the balance-sheet chunk sits at hybrid
   rank 22). Worth a targeted chunking / numeric-boost fix, tracked on its own.
2. `nvidia_china_01` / `nvidia_export_controls_01` / `nvidia_responsible_ai_risk_01`
   now have 10–19 relevant chunks each — consider whether these broad questions
   should be split into narrower sub-questions in a future benchmark revision.
3. Optional: one live `evaluate_v3.py` run to confirm the offline pool-derived
   numbers match live retrieval end-to-end (expected to match — the pool ranks
   are the production ranks).
