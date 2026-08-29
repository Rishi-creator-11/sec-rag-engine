# Production deployment fix — Vercel-safe bm25s + available-filings API

Date: 2026-08-29
Deployed: `https://sec-rag-engine.vercel.app`
Commits: `5aa624c` (fix) + `9ba93c0` (routing correction) on `origin/main`.
No retrieval-ranking / candidate_k / RRF / Cohere / OpenAI / frontend change.

---

## 1. Root cause of the Vercel failure

`GET /health` → 503:

```
lexical index unavailable: bm25s rebuild failed:
OSError: [Errno 30] Read-only file system: '/var/task/data/bm25s_index'
```

The persisted bm25s index (`data/bm25s_index/`) was **git-ignored**, so it was
absent from the Vercel git checkout and therefore absent from the Python
function bundle (Vercel's Python runtime bundles *tracked* project files by
default; a gitignored directory ships nothing). At the first request
`get_lexical_backend()` → `load_or_build_bm25s()` found no bundled index →
fell through to `build_persisted_bm25s()` → `BM25SBackend.save()` →
`Path.mkdir()` + `np.save(...)` under `/var/task`, which is read-only on Vercel
(AWS Lambda) → `OSError` → `LexicalBackendError` → `/health` 503, `/ask`
un-retrievable.

Two design gaps behind it:
- the runtime always assumed a writable filesystem — no read-only branch;
- `corpus_version` hashed **chunk ids only**, so a metadata-only change (the
  Phase-5 seed `fiscal_year` backfill) would not invalidate a stale bundled
  index once one existed.

## 2. Index packaging strategy — **A: commit the prebuilt index**

`data/bm25s_index/` is now **git-tracked** (7 files, 11.7 MB). Vercel's Python
runtime bundles every reachable tracked file into the function **by default —
no `includeFiles` needed**. Chosen over a Vercel build step because there is no
committed build pipeline to hang a `buildCommand` on, the committed index is
deterministic (exactly what was validated locally runs in production), and it
fits the operator workflow (ingestion already rebuilds the index and phase
checkpoints already commit `data/` serving artifacts).

Rebuild + re-commit after any ingestion or metadata change:
`python -m scripts.build_bm25s_index`.

## 3. Files changed

**`5aa624c` — Make bm25s deployment Vercel-safe and expose filing years**

| file | change |
|---|---|
| `retrieval/lexical_backend.py` | `is_read_only_runtime()`; `load_readonly_bm25s()` (load-only, fails clearly); `get_lexical_backend()` read-only branch (never rebuilds); content-aware `corpus_version()` (`v2:` schema); `save()` records `corpus_version_schema` |
| `api/main.py` | `GET /companies/{ticker}/filings`; `/health` adds `read_only_runtime`; readiness contract doc |
| `.gitignore` | `data/bm25s_index/` now tracked (only `*.tmp` ignored) |
| `scripts/build_bm25s_index.py` (new) | deterministic prebuild + verify (load round-trip + ranking parity, unfiltered and year-filtered) |
| `tests/test_deploy_readonly.py` (new) | 24 tests (§10) |
| `data/bm25s_index/` (new, 7 files) | committed prebuilt index (`v2:` corpus_version, 1,684 docs) |
| `evaluation/DEPLOY_FIX_REPORT.md` (new) | this report |

**`9ba93c0` — Fix Vercel routing: drop vercel.json rewrite + api/index.py**

`5aa624c` also added `vercel.json` (a `/(.*) → /api/index` rewrite), an
`api/index.py` re-export entrypoint, and `.vercelignore`. On the live project
this **broke routing** — every path returned Starlette's `{"detail":"Not
Found"}` (`/openapi.json` still showed the routes; `/docs` still rendered):
`api/index.py` took precedence over `api/main.py` as the FastAPI entrypoint and
its `from api.main import app` fails on Vercel's import path, and the rewrite
forced every request onto `/api/index`. The pre-existing zero-config deployment
routed all paths to the `api/main.py` FastAPI `app`; `9ba93c0` restores that by
**removing all three files**. Net config change for routing: none.

## 4. Build / index generation command

```
python -m scripts.build_bm25s_index          # build data/bm25s_index/ + verify
python -m scripts.build_bm25s_index --check   # verify only, no write
```
Settings come from the production code path (`method="lucene"`, `k1=1.5`,
`b=0.75`, `dtype="float64"`, `retrieval.bm25_search.tokenize`). Verifies:
fresh `BM25SBackend.load()` round-trips; `corpus_version` + `document_count`
recorded and match `load_chunks()`; **ranking parity** — 6 fixed queries,
unfiltered and with a `NVDA` + `fiscal_year=2023` filter, byte-identical top-10
(ids + scores) between the persisted index and a fresh in-memory build.

Current: `document_count 1684`, `corpus_version
v2:ef5d6c2a1e426f2481fdde176658ac8deb01359ef88c1b67405f9985b4aea163`,
parity **OK**.

## 5. Runtime Vercel behaviour

`is_read_only_runtime()` → True when `VERCEL` / `VERCEL_ENV` /
`AWS_LAMBDA_FUNCTION_NAME` / `AWS_EXECUTION_ENV` is set (or
`SEC_RAG_READ_ONLY_FS=1`; `SEC_RAG_FORCE_WRITABLE_FS=1` forces off).

In that mode `get_lexical_backend()`:
- loads the bundled index **read-only** via `BM25SBackend.load(BM25S_INDEX_DIR)`
  — bm25s 0.3.11 `load()` reads only, verified against a `chmod`-read-only
  directory (`mmap=False` and `mmap=True`);
- **never** calls `build_persisted_bm25s()`; `force_rebuild=True` raises
  `LexicalBackendError`;
- fails **clearly** (→ `LexicalBackendError` → `/health` 503,
  `status: "unavailable"`) when the bundle is **missing**, `corpus_version.json`
  is **unreadable**, the `corpus_version` does **not match**
  `corpus_version(load_chunks())` (stale), the bm25s load **throws**, or the
  loaded index has **0 documents**. No silent empty retriever; no write under
  `/var/task`. A `/tmp` copy is not needed.

**Production, live:** `GET /health` → `{"status":"ok","lexical_backend":"bm25s",
"read_only_runtime":true,"companies":10,"bm25_documents":1684}`.

## 6. Local behaviour (unchanged)

Writable FS: `load_or_build_bm25s()` loads if `corpus_version` matches,
otherwise deterministically **rebuilds + re-persists** (missing/corrupt/stale),
fails readiness only if a rebuild also fails. `bm25_search.reload()` (ingestion
`stage_bm25`) still force-rebuilds. `SEC_RAG_LEXICAL_BACKEND=current` rollback
unchanged.

## 7. corpus_version change

`"v2:" + sha256` over the sorted per-chunk fingerprint
`chunk_id | ticker | fiscal_year | (filing_id or accession_number, dashes
stripped)` — order-independent, **excludes** chunk text and transient fields
(`filing_date` display, scores, timestamps). A metadata-only change (seed
`fiscal_year` backfill, re-attribution, adding a company year) now changes
`corpus_version`, so a stale bundled index is detected (503 on Vercel; rebuild
locally). The `v2:` prefix treats any index persisted by the old code as stale;
the committed index was rebuilt with the new function.

Tests: `CorpusVersionTests` (7) — schema prefix, determinism/order-independence,
sensitivity to `fiscal_year` / `filing_id` / `ticker`, `accession_number` ⇄
`filing_id` equivalence, transient-field immunity, ≠ old chunk-id-only hash.

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
    ...FY2025, FY2024, FY2023
  ],
  "available_fiscal_years": [2026, 2025, 2024, 2023]
}
```
- backed by `ingestion.registry.get_company` — **no chunk text, no secrets, no
  internal paths**;
- deterministic order: `filing_type` asc, then `fiscal_year` **desc**;
- unknown ticker → **404** `{"detail":{"error":"unknown_ticker","ticker":…}}`;
- `GET /companies` **unchanged** — `{"companies":[{"ticker","name"}]}`.

## 9. XOM provenance response

Companies with registry lineage get a factual `registrant_lineage` block —
the historical filing's registrant is not the entity the ticker currently
resolves to:

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
Production check confirmed: `filing_registrant 0000034088` → `successor
0002115436`, XOM filing years `[2025]`. Companies without lineage have no block.
Generic — registry-driven, no ticker hardcoding.

## 10. Tests

`python -m unittest discover -s tests -t .` → **277 passed** (+24
`tests/test_deploy_readonly.py`; existing suites all green).

New coverage: `corpus_version` (7); read-only detection (4: VERCEL/AWS env,
explicit flags, local default); read-only backend against the real prebuilt
index (5: loads without writing a byte, Vercel mode never calls the builder,
`force_rebuild` raises, missing bundle fails clearly, stale `corpus_version`
fails clearly, **ranking identical** read-only-load vs fresh); local mode still
rebuilds when not on Vercel (1); `/companies/{ticker}/filings` (7:
NVDA 4 years newest-first, lowercase normalised, AAPL current filing, XOM
lineage factual, no-lineage company has no block, unknown ticker 404,
`/companies` unchanged).

## 11. Local API checks (pre-push)

| check | result |
|---|---|
| `GET /health` | 200, `read_only_runtime:false`, `bm25_documents:1684` |
| `GET /companies` | 200, 10 companies |
| `GET /companies/NVDA/filings` | 200, `[2026,2025,2024,2023]` |
| `POST /ask` NVDA FY2023 | 200, `scopes ["NVDA:2023"]`, every source `fiscal_year==2023` |
| `POST /ask` NVDA FY2023+FY2025 | 200, `evidence_by_scope {"NVDA:2023":1,"NVDA:2025":4}`, sources FY2023/2025 only |
| `POST /ask` NVDA FY1999 | 422 `fiscal_year_not_available` |
| `VERCEL=1 GET /health` | 200 ok; stale index → 503 "bundled bm25s index is stale" |
| benchmark_v3 offline (frozen pool) | hybrid MRR **0.903**, filter_correctness 1.000, leakage 0.000 — ranking unchanged |
| `build_bm25s_index --check` | OK — read-only load ranks identically to a fresh build |

## 12. Vercel deployment — what shipped

- **Routing**: pre-existing zero-config — Vercel's FastAPI preset (`fastapi` in
  `requirements.txt`) serves the `api/main.py` `app` and routes every path to
  it. No `vercel.json`, no rewrites.
- **Bundling**: every tracked project file is bundled by default, so the
  committed `data/bm25s_index/` (11.7 MB) + `data/chunks/**` + `data/registry/
  *.json` ship into the function. Gitignored paths
  (`data/raw/**/filing.*`, `data/embeddings/`, `data/sec_cache/`,
  `data/registry/backups/`) are absent from the git checkout and not bundled.
- **Environment variables** (Vercel Production project): `OPENAI_API_KEY`,
  `PINECONE_API_KEY`, `COHERE_API_KEY` (required for `/ask`); `SEC_USER_AGENT`;
  `FRONTEND_ORIGINS` (or the built-in default). `VERCEL` is set automatically →
  read-only bm25s path activates. Do **not** set `SEC_RAG_LEXICAL_BACKEND`.
- Redeploy after any ingestion / metadata change: `python -m
  scripts.build_bm25s_index`, commit `data/bm25s_index/`, push.

## 13. git diff --stat

`5aa624c`:
```
 .gitignore                             |    9 +-
 api/main.py                            |   79 +-
 data/bm25s_index/chunks.jsonl          | 1684 ++++++++++++++++++++++++++++++
 data/bm25s_index/corpus_version.json   |    9 +
 data/bm25s_index/data.csc.index.npy    |  Bin 0 -> 3179048
 data/bm25s_index/indices.csc.index.npy |  Bin 0 -> 1589588
 data/bm25s_index/indptr.csc.index.npy  |  Bin 0 -> 92472
 data/bm25s_index/params.index.json     |   12 +
 data/bm25s_index/vocab.index.json      |    1 +
 evaluation/DEPLOY_FIX_REPORT.md        |  297 +++++
 retrieval/lexical_backend.py           |  115 ++-
 scripts/build_bm25s_index.py           |  130 +++
 tests/test_deploy_readonly.py          |  233 +++++
 vercel.json / .vercelignore / api/index.py  (added, then reverted in 9ba93c0)
```
`9ba93c0`: `-vercel.json -.vercelignore -api/index.py`, docstring tweak.

Secret scan of both commits' staged diffs: **clean**. `data/bm25s_index/
chunks.jsonl` carries public SEC 10-K text identical to the already-tracked
`data/chunks/**`. No raw filing HTML, embeddings, SEC cache, judge pools,
backups, or test scratch committed.

`HEAD` = `9ba93c0` = `origin/main`. `v1-stable` → `0edf0a5`, `a200b3e`,
`1ce2a3e`, `c4549fa` untouched. Frontend not modified. No new companies ingested.
