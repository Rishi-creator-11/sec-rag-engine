# Production deployment fix — Vercel-safe bm25s + available-filings API

Date: 2026-08-28
Scope: two backend contract/deployment issues found by the frontend inspection.
No ranking / candidate_k / RRF / Cohere / OpenAI / frontend change. Not committed,
not pushed.

---

## 1. Root cause of the Vercel failure

`GET /health` → 503 on Vercel because the lexical backend tried to **build** the
bm25s index at request time and **write** it under `/var/task/data/bm25s_index`,
which is read-only on Vercel (AWS Lambda) serverless functions.

Chain: first `/ask` (or `/health`) → `get_lexical_backend()` →
`load_or_build_bm25s()` → the persisted index is **absent from the function
bundle** (nothing bundled `data/bm25s_index/**` — Vercel's `@vercel/python`
only ships imported modules unless `includeFiles` says otherwise) → falls to
`build_persisted_bm25s()` → `BM25SBackend.save()` → `path.mkdir()` +
`np.save(...)` → `OSError: [Errno 30] Read-only file system` →
`LexicalBackendError` → `/health` returns 503 and `/ask` cannot retrieve.

Two design gaps behind it:
- no deployment-aware branch — the runtime always assumed a writable FS;
- `corpus_version` hashed **chunk ids only**, so a metadata-only change (the
  Phase-5 seed `fiscal_year` backfill) would not have invalidated a stale
  bundled index even once one existed.

## 2. Index packaging strategy — **A: commit the prebuilt index**

`data/bm25s_index/` is now **git-tracked** and bundled read-only into the Vercel
function via `vercel.json` `includeFiles`. Chosen over "build during Vercel
build" because:
- there is currently **no committed Vercel build pipeline** to hang a reliable
  `buildCommand` on (no `vercel.json`, no `.vercelignore` in the repo);
- deterministic — the exact index validated locally is what runs in production;
- 11.7 MB / 7 files is trivial for the bundle (Vercel unzipped limit is 250 MB);
- it fits the existing operator workflow — ingestion already rebuilds the index,
  and phase checkpoints already commit `data/` serving artifacts.

Rebuild + re-commit `data/bm25s_index/` after any ingestion or metadata change:
`python -m scripts.build_bm25s_index`.

## 3. Files changed

**Modified**
| file | change |
|---|---|
| `retrieval/lexical_backend.py` | `is_read_only_runtime()`; `load_readonly_bm25s()` (load-only, fails clearly); `get_lexical_backend()` read-only branch (never rebuilds); content-aware `corpus_version()` (`v2:` schema); `save()` records `corpus_version_schema` |
| `api/main.py` | `GET /companies/{ticker}/filings`; `/health` adds `read_only_runtime` and documents the read-only readiness contract; `/companies` doc clarified (unchanged shape) |
| `.gitignore` | `data/bm25s_index/` now tracked (only `*.tmp` ignored) |

**New**
| file | purpose |
|---|---|
| `scripts/build_bm25s_index.py` | deterministic prebuild + verify (load round-trip + ranking parity) |
| `api/index.py` | Vercel Python entrypoint (`from api.main import app`) |
| `vercel.json` | function config + `includeFiles: "data/**"` + rewrite all routes to `api/index` |
| `.vercelignore` | drop `evaluation/`, `tests/`, `scripts/`, dev trees from the bundle |
| `tests/test_deploy_readonly.py` | 24 tests (see §10) |
| `data/bm25s_index/` (7 files) | the committed prebuilt index (`v2:` corpus_version, 1,684 docs, 11.7 MB) |

## 4. Build / index generation command

```
python -m scripts.build_bm25s_index          # build data/bm25s_index/ + verify
python -m scripts.build_bm25s_index --check   # verify only, no write
```
Build settings are read from the production code path (`method="lucene"`,
`k1=1.5`, `b=0.75`, `dtype="float64"`, `retrieval.bm25_search.tokenize`). Verify:
- fresh `BM25SBackend.load()` round-trips;
- `corpus_version` + `document_count` recorded and match `load_chunks()`;
- **ranking parity** — for 6 fixed queries, unfiltered and with a
  `NVDA` + `fiscal_year=2023` filter, the persisted index returns byte-identical
  top-10 (ids + scores) to a fresh in-memory build.

Current output: `document_count 1684`, `corpus_version
v2:ef5d6c2a1e426f2481fdde176658ac8deb01359ef88c1b67405f9985b4aea163`,
parity **OK**.

## 5. Runtime Vercel behaviour

`is_read_only_runtime()` is True when `VERCEL`, `VERCEL_ENV`,
`AWS_LAMBDA_FUNCTION_NAME`, or `AWS_EXECUTION_ENV` is set (or
`SEC_RAG_READ_ONLY_FS=1`; `SEC_RAG_FORCE_WRITABLE_FS=1` forces it off).

In that mode `get_lexical_backend()`:
- loads the bundled index **read-only** via `BM25SBackend.load(BM25S_INDEX_DIR)`
  — bm25s 0.3.11 `load()` reads only, no write, verified against a
  `chmod`-read-only directory (both `mmap=False` and `mmap=True`);
- **never** calls `build_persisted_bm25s()`; `force_rebuild=True` raises
  `LexicalBackendError` ("build the index before deploy");
- fails **clearly** (→ `LexicalBackendError` → `/health` 503, `status:
  "unavailable"`, `detail: "...read-only runtime: ..."`) when the bundle is
  **missing**, its `corpus_version.json` is **unreadable**, the
  `corpus_version` does **not match** `corpus_version(load_chunks())` (stale), the
  bm25s load **throws**, or the loaded index has **0 documents**. No silent empty
  retriever; no write under `/var/task`.

A `/tmp` copy is **not** needed — direct read-only load works.

Local verification (`VERCEL=1`, `TestClient`):
- current bundle → `/health` **200 ok** (`read_only_runtime: true`,
  `bm25_documents: 1684`); `/ask NVDA FY2023` **200**, sources all FY2023.
- stale bundle → `/health` **503** "read-only runtime: bundled bm25s index is
  stale …".

## 6. Local behaviour (unchanged)

Writable FS (default): `load_or_build_bm25s()` — loads the persisted index if
`corpus_version` matches, otherwise deterministically **rebuilds + re-persists**
(missing / corrupt / stale), and only fails readiness if a rebuild also fails.
`bm25_search.reload()` (called by ingestion `stage_bm25`) still force-rebuilds
locally. `SEC_RAG_LEXICAL_BACKEND=current` rollback unchanged.

## 7. corpus_version change

Before: `sha256("\n".join(sorted(chunk_id)))`.
After: `"v2:" + sha256` over the sorted per-chunk fingerprint

```
chunk_id | ticker | fiscal_year | (filing_id or accession_number, dashes stripped)
```

- Order-independent; **excludes** chunk text and transient fields
  (`filing_date` display, scores, timestamps).
- A metadata-only change — e.g. backfilling `fiscal_year` onto the seed chunks,
  re-attributing a filing, adding a fiscal year for a company — now changes
  `corpus_version`, so a stale bundled index is detected (503 on Vercel; rebuild
  locally).
- The `v2:` schema prefix means any index persisted by the old code is treated
  as stale. The committed `data/bm25s_index/` was rebuilt with the new function.
- Tests: `CorpusVersionTests` (7) — schema prefix, determinism/order-independence,
  changes on `fiscal_year` / `filing_id` / `ticker`, `accession_number` ⇄
  `filing_id` equivalence, ignores transient fields, ≠ the old chunk-id-only hash.

## 8. GET /companies/{ticker}/filings contract

```
GET /companies/NVDA/filings  -> 200
{
  "ticker": "NVDA",
  "name": "NVIDIA Corporation",
  "cik": "0001045810",
  "filings": [
    {"filing_type":"10-K","fiscal_year":2026,"report_date":"2026-01-25",
     "filing_date":"2026-02-25","accession_number":"0001045810-26-000021",
     "chunk_count":103},
    ... FY2025, FY2024, FY2023
  ],
  "available_fiscal_years": [2026, 2025, 2024, 2023]
}
```
- backed by `ingestion.registry.get_company` — **no chunk text, no secrets, no
  internal paths**;
- deterministic order: `filing_type` asc, then `fiscal_year` **desc**
  (newest-year-first);
- unknown ticker → **404** `{"detail": {"error": "unknown_ticker", "ticker": …}}`;
- `GET /companies` **unchanged** — still `{"companies": [{"ticker","name"}]}`.

## 9. XOM provenance response

When a company carries registry lineage (ticker→registrant succession), the
response adds a factual `registrant_lineage` block — the historical filing's
registrant is **not** the entity the ticker currently resolves to:

```
GET /companies/XOM/filings -> "registrant_lineage": {
  "note": "Ticker XOM is currently registered to CIK 0002115436 (ExxonMobil
           Holdings Corporation) effective 2026-07-01. This 10-K (accession
           0000034088-26-000045, FY2025) was filed by CIK 0000034088
           (EXXON MOBIL CORP); its metadata is kept SEC-authentic.",
  "filing_registrant_cik": "0000034088",
  "filing_registrant_legal_name": "EXXON MOBIL CORP",
  "current_successor_cik": "0002115436",
  "current_successor_legal_name": "ExxonMobil Holdings Corporation",
  "successor_effective_date": "2026-07-01"
}
```
The XOM FY2025 filing itself stays attributed to the historical registrant.
Companies without lineage have no such block. Generic — driven by the registry,
no ticker hardcoding.

## 10. Tests

`python -m unittest discover -s tests -t .` → **277 passed** (was 253; +24
`tests/test_deploy_readonly.py`). Existing suites all green (ranking, comparison,
multi-year, XOM lineage, ingestion resume, API contract).

New coverage:
- **corpus_version** (7): schema, determinism, sensitivity to
  fiscal_year/filing_id/ticker, transient-field immunity, ≠ old hash.
- **read-only detection** (4): VERCEL / AWS_LAMBDA env, explicit on/off flags,
  local default writable.
- **read-only backend** (5, against the real prebuilt index): loads without
  writing a byte to the dir, Vercel mode never calls the builder,
  `force_rebuild` raises, missing bundle fails clearly, stale corpus_version
  fails clearly, **ranking identical** read-only-load vs fresh build.
- **local mode** (1): missing index still rebuilds when not on Vercel.
- **/companies/{ticker}/filings** (7): NVDA FY2026/2025/2024/2023 newest-first,
  lowercase normalised, AAPL current registered filing, XOM lineage factual
  (registrant ≠ successor), no-lineage company has no block, unknown ticker 404,
  `/companies` unchanged.

## 11. Local API checks

| check | result |
|---|---|
| `GET /health` | **200** `{"status":"ok","lexical_backend":"bm25s","read_only_runtime":false,"bm25_documents":1684,"companies":10}` |
| `GET /companies` | **200**, 10 companies, `{ticker,name}` only |
| `GET /companies/NVDA/filings` | **200**, years `[2026,2025,2024,2023]`, `available_fiscal_years` matches |
| `GET /companies/XOM/filings` | **200**, `registrant_lineage` `0000034088 → 0002115436` |
| `GET /companies/ZZZZ/filings` | **404** `unknown_ticker` |
| `POST /ask` NVDA FY2023 | **200**, `search_scope.scopes == ["NVDA:2023"]`, every source `fiscal_year == 2023` (zero cross-year leakage) |
| `POST /ask` NVDA FY2023+FY2025 | **200**, `evidence_by_scope {"NVDA:2023":1,"NVDA:2025":4}`, sources only FY2023/FY2025 |
| `POST /ask` NVDA FY1999 | **422** `fiscal_year_not_available` (not silently widened) |
| `VERCEL=1` `GET /health` | **200 ok**, `read_only_runtime: true` |
| `VERCEL=1` + stale index `GET /health` | **503** "bundled bm25s index is stale" |
| benchmark_v3 offline (frozen pool) | hybrid MRR **0.903**, filter_correctness 1.000, leakage 0.000 — **ranking unchanged** |
| `build_bm25s_index --check` | **OK** — read-only load ranks identically to a fresh build |

## 12. Exact Vercel deployment steps

**Files to deploy (all in the repo, this task's diff):**
- `vercel.json`, `.vercelignore`, `api/index.py`
- `retrieval/lexical_backend.py`, `api/main.py`, `.gitignore`
- `scripts/build_bm25s_index.py`, `tests/test_deploy_readonly.py`
- **`data/bm25s_index/` (7 files, 11.7 MB)** — the prebuilt index, now git-tracked

**Steps:**
1. `python -m scripts.build_bm25s_index` — regenerate `data/bm25s_index/` from
   the current corpus, confirm parity `OK`.
2. Commit the diff (this task did **not** commit) including `data/bm25s_index/`.
3. Push `main` → Vercel deploys.
4. Vercel picks up `vercel.json`:
   - `functions."api/index.py".includeFiles: "data/**"` bundles
     `data/bm25s_index/**`, `data/chunks/**`, `data/registry/*.json`,
     `data/raw/**/metadata.json` (gitignored `data/raw/**/filing.*`,
     `data/embeddings/`, `data/sec_cache/`, `data/registry/backups/` are absent
     from the git checkout and not bundled);
   - `rewrites: [{ "source": "/(.*)", "destination": "/api/index" }]` routes
     `/health`, `/companies`, `/companies/*/filings`, `/ask` to the FastAPI app;
   - `maxDuration: 30`.
5. **Environment variables** on the Vercel project (Production):
   - `OPENAI_API_KEY`, `PINECONE_API_KEY`, `COHERE_API_KEY` — required
   - `SEC_USER_AGENT` — required for any SEC call (`/ask` does not call SEC, but
     `verify` tooling does; safe to set)
   - `FRONTEND_ORIGINS=https://secfrontend.vercel.app` (or rely on the built-in
     default from the CORS commit)
   - `VERCEL` is set automatically by Vercel → read-only bm25s path activates.
   - do **not** set `SEC_RAG_LEXICAL_BACKEND` (defaults to `bm25s`).
6. If the Vercel project currently has a dashboard "Root Directory" / build
   override or an out-of-repo `vercel.json`, reconcile it with the committed
   `vercel.json` (this repo had none committed).

**Post-deploy production checks (must pass):**
| request | expect |
|---|---|
| `GET /health` | 200, `{"status":"ok","read_only_runtime":true,"bm25_documents":1684}` |
| `GET /companies` | 200 |
| `GET /companies/NVDA/filings` | 200, 4 fiscal years newest-first |
| `POST /ask` `{"question":"…","tickers":["NVDA"],"fiscal_years":[2023]}` | 200, sources FY2023 only |
| `POST /ask` `{"question":"…","tickers":["NVDA"],"fiscal_years":[2023,2025]}` | 200, evidence_by_scope both years |
| `POST /ask` `{"question":"…","tickers":["NVDA"],"fiscal_years":[1999]}` | 422 `fiscal_year_not_available` |

If `/health` returns 503 with "bundled bm25s index is stale" or "no bundled
bm25s index", step 1 was skipped or `includeFiles` did not ship the directory —
rerun the build, confirm `data/bm25s_index/` is committed, redeploy.

## 13. git diff --stat (staged preview — NOT committed)

```
 .gitignore                             |    9 +-
 .vercelignore                          |   24 +
 api/index.py                           |   12 +
 api/main.py                            |   79 +-
 data/bm25s_index/chunks.jsonl          | 1684 ++++++++++++++++++++++++++++++
 data/bm25s_index/corpus_version.json   |    9 +
 data/bm25s_index/data.csc.index.npy    |  Bin 0 -> 3179048 bytes
 data/bm25s_index/indices.csc.index.npy |  Bin 0 -> 1589588 bytes
 data/bm25s_index/indptr.csc.index.npy  |  Bin 0 -> 92472 bytes
 data/bm25s_index/params.index.json     |   12 +
 data/bm25s_index/vocab.index.json      |    1 +
 retrieval/lexical_backend.py           |  115 ++-
 scripts/build_bm25s_index.py           |  130 +++
 tests/test_deploy_readonly.py          |  233 +++++
 vercel.json                            |   12 +
 15 files changed, 2310 insertions(+), 10 deletions(-)
```

Secret scan of the staged diff: **clean**. No raw filing HTML, embeddings, SEC
cache, judge pools, backups, or test scratch staged. `data/bm25s_index/chunks.jsonl`
carries public SEC 10-K text identical to the already-tracked `data/chunks/**`.

**Nothing committed, nothing pushed.** `HEAD` = `c4549fa`; `v1-stable` → `0edf0a5`,
`a200b3e`, `1ce2a3e` untouched. Frontend not modified. No new companies ingested.
