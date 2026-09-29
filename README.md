# AlphaBrief

AI-powered research for SEC 10-K filings with grounded answers and source-level evidence.

Ask a plain-English question about a public company's annual report and get a response built
only from retrieved filing text, with every answer backed by retrieved filing evidence. When
the filings do not support an answer, the system refuses instead of guessing.

## Live Demo

- Frontend: https://secfrontend.vercel.app
- Backend API docs (Swagger): https://sec-rag-engine.vercel.app/docs

## Current Scale

**92 companies · 274 annual 10-K filings · 45,252 indexed chunks**, as of the live
[`/health`](https://sec-rag-engine.vercel.app/health) and [`/companies`](https://sec-rag-engine.vercel.app/companies)
endpoints. This is a V1 launch corpus (large-cap US filers), not full SEC coverage.

## What It Does

- Pick one or more companies and fiscal years, or ask across the whole corpus.
- Ask a natural-language question ("How did operating margin change?", "What does the company
  say about AI-related risk?").
- Get a grounded answer with inline `[Source N]` citations.
- Inspect the evidence: each citation opens the underlying 10-K passage with a link to the
  filing on SEC.gov.
- Compare across years or companies in a single question; each scope stays tied to its own
  filing.

## Architecture

```mermaid
flowchart LR
    EDGAR[SEC EDGAR] --> ING[Ingestion + cleaning]
    ING --> CHUNK[Token-window chunking]
    CHUNK --> EMB[OpenAI embeddings]
    EMB --> DENSE[(Pinecone dense)]
    CHUNK --> BM25[(bm25s lexical index)]

    Q[Question + ticker/year scope] --> FILTER[Structured scope filter]
    FILTER --> DENSE
    FILTER --> BM25
    DENSE --> RRF[Weighted RRF fusion]
    BM25 --> RRF
    RRF --> RERANK[Cohere rerank]
    RERANK --> SELECT[Scope-aware evidence selection]
    SELECT --> GEN[OpenAI generation]
    GEN --> API[FastAPI /ask]
    API --> UI[Next.js UI]
```

Pipeline: SEC EDGAR → ingestion and cleaning → chunking → embeddings → Pinecone dense index →
bm25s lexical index → RRF fusion → Cohere reranking → scope-aware evidence selection → OpenAI
generation → FastAPI → Next.js UI.

## Retrieval Design

Retrieval is a dense + lexical hybrid:

- **Dense**: OpenAI `text-embedding-3-small` (1536-dim), stored in the Pinecone `sec-rag-engine`
  index.
- **Lexical**: `bm25s` (Lucene-style BM25, `k1=1.5`, `b=0.75`) over a persisted, version-pinned
  index that ships with the deployment.
- **Fusion**: each retriever returns `candidate_k = 10` results; they are combined with
  Reciprocal Rank Fusion (`RRF_K = 60`, equal weight for dense and BM25).
- **Reranking**: the fused candidates go to Cohere `rerank-v4.0-fast`, which produces the final
  top-5 evidence set.

Structured scoping happens **before** generation. A request carries an optional list of tickers
and fiscal years; these become a `RetrievalFilter` that constrains every retriever, so the
generator never sees a chunk outside the requested scope. If a requested year has no ingested
10-K for a requested company, the API returns a structured `422` rather than widening scope.

For comparison questions (two or more scopes), retrieval runs independently per scope, using
one shared query embedding across scopes for comparability. The per-scope candidate sets are
deduplicated into a union (capped at 60), and a **single** Cohere rerank is run over that union.
Evidence selection then guarantees at least one chunk per requested scope. Because each scope is
retrieved and filtered on its own before the union is formed, one company's or year's text
cannot leak into another scope's evidence.

## Multi-Year / Comparison Support

Scope is derived from what the request specifies:

- **Ticker only** → all ingested filings for that company.
- **Ticker + year** → that exact fiscal year.
- **Multiple scopes** (several tickers, several years, or both) → comparison mode, one shared
  code path for company-vs-company, year-vs-year, and company+year comparisons.
- **Unsupported year** → structured `422` (`fiscal_year_not_available`) with the available
  years, or `fiscal_years_without_tickers` when years are given with no ticker.

There is no silent scope widening. A scope-validation failure is surfaced to the caller; it is
never retried against a different year or company.

## SEC Ingestion

Ingestion uses the official SEC submissions API:

- Exact `10-K` discovery only; `10-K/A` amendments are excluded by string equality.
- Fiscal year is derived from the filing's `reportDate`, so non-calendar filers (Apple,
  Microsoft, Walmart) are dated correctly.
- Historical filings are discovered from the same submissions history, not just the latest.
- A resumable, accession-keyed ledger re-validates each completed stage's on-disk artifacts, so
  a rerun picks up at the earliest invalid stage. Accession numbers are deduplicated.
- Chunk IDs are deterministic (`{TICKER}_{FY}_{FILING_TYPE}_{ACCESSION}_{INDEX}`), so
  re-ingestion overwrites rather than duplicates.
- Verification checks dense / sparse / BM25 parity for a filing.
- The bm25s index is rebuilt and ranking-parity-checked by a dedicated script and committed;
  the read-only deployment loads the bundled index and never rebuilds it.

CIK-lineage edge cases are handled generically: ExxonMobil's recent 10-Ks were filed under a
legacy registrant CIK, which is recorded as a `lineage` block in the registry without any
company-specific code in the SEC client.

## Evaluation & Testing

**Correctness** (scope filtering, cross-year/cross-company isolation, numeric-year attribution)
is enforced in code, not just measured after the fact: every retrieval path applies the same
structured `RetrievalFilter` before generation, and that guarantee is checked by dedicated
structural tests, not sampled from a benchmark.

| Check | Result |
|---|---|
| Backend tests (`python -m pytest`) | **336 / 336** |
| Fiscal-year filter correctness, cross-year leakage, cross-company leakage, comparison scope coverage, numeric-year attribution | **1.000 / 0.000 / 0.000 / 1.000 / 1.000** — asserted by `tests/test_multiyear*.py`, `test_comparison.py`, `test_filters.py` |

**Retrieval quality** (MRR, Recall@K, Precision@K) is evaluated against `benchmark_v3_repool` —
125 questions (115 answerable, 10 deliberately unsupported fiscal years) covering 8 companies
(AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, WMT), with model-assisted relevance judgments (a
first-pass judgment plus a lower-temperature review pass over gpt-5-mini; not human-labeled).
Those judgments are a property of the filing text, not of corpus size, so they stay valid as
companies are added — but the retrieval *ranking* isn't: BM25 scores the whole corpus before
scope-filtering, so its ranking shifts as the corpus grows. The table below is `python -m
evaluation.evaluate_v3_current_corpus` — live retrieval against the current 92-company /
45,252-chunk corpus, scored against those same reviewed judgments (2026-09-29):

| Metric | Result |
|---|---|
| MRR (hybrid) | 0.86 |
| Recall@10, capacity-capped (hybrid) | 0.66 |
| Precision@5 (hybrid) | 0.60 |

Down from the last full run at 10 companies / 4,262 chunks (MRR 0.88 / 0.70 / 0.62) — consistent
with the drift this repo's own scale report predicted from BM25's whole-corpus IDF as the corpus
grows roughly 10x, not a regression introduced by anything else. **Caveat:** the question set
still covers 8 of the 92 companies; extending it to the full corpus is future work, not done
here. See `evaluation/evaluate_v3_current_corpus.py` and
`evaluation/results/v3_retrieval_evaluation_current_corpus.json` for the exact run, and
`evaluation/SCALE_READINESS_REPORT.md` for the original methodology writeup.

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn.
- **Frontend**: Next.js, TypeScript (deployed on Vercel).
- **Retrieval**: OpenAI `text-embedding-3-small` + Pinecone (dense), `bm25s` (lexical), RRF
  fusion, Cohere `rerank-v4.0-fast`.
- **Generation**: OpenAI `gpt-5-nano`, constrained to the retrieved evidence.
- **Infrastructure**: Pinecone managed indexes, Vercel (read-only Python runtime for the API,
  static + SSR for the UI).
- **Evaluation**: offline pooled benchmark, model-assisted judging, `unittest` structural gates.

## API

```text
GET  /health
GET  /companies
GET  /companies/{ticker}/filings
POST /ask
```

`POST /ask` request:

```json
{
  "question": "How did Apple's total net sales change from fiscal 2023 to fiscal 2024?",
  "tickers": ["AAPL"],
  "fiscal_years": [2023, 2024],
  "top_k": 5
}
```

`tickers` and `fiscal_years` are optional. Omitting both searches the whole corpus; giving
`fiscal_years` without `tickers` is rejected with `422`.

Response fields: `question`, `answer`, `sources`, `generation_model`, `reranker_fallback`,
`reranker_fallback_reason`, `search_scope` (including `comparison_mode` and `evidence_by_scope`),
and `timings`.

## Reliability / Safety

- Scope is explicit and structured; it is applied to retrieval before generation.
- Cross-year and cross-company leakage are covered by dedicated tests and stay at 0.000 on the
  benchmark.
- Numeric answers are validated for correct fiscal-year attribution on the multi-year suites.
- The generator answers only from the supplied evidence and refuses when the evidence is
  insufficient.
- There is no silent fallback to a different year or company; an unavailable scope returns a
  structured error.
- If Cohere reranking fails or is rate-limited, the request continues with the hybrid top-5 and
  sets `reranker_fallback = true`. The fallback path is still scope-filtered, so it cannot
  introduce out-of-scope evidence.

## Known Limitation

Loose, directly phrased revenue questions for ExxonMobil can safely refuse. ExxonMobil's 10-K
restates consolidated revenue across several segment and reconciliation tables, and in that
filing's layout a bare per-year total can land in a chunk with no adjacent fiscal-year column
header, which makes the consolidated income-statement figure hard to retrieve by a loose query.
The system prefers to refuse rather than return a number it cannot attribute confidently.
Anchored queries that name the statement ("What does the consolidated statement of income show
for total revenues?") return the correct value. This is specific to that filing's table
structure and does not affect the other companies.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```text
OPENAI_API_KEY=
PINECONE_API_KEY=
COHERE_API_KEY=
FRONTEND_ORIGINS=http://localhost:3000
SEC_USER_AGENT=your-project you@example.com
```

Run the API:

```bash
fastapi dev api/main.py
```

Run the tests:

```bash
python -m unittest discover -s tests -t .
```

Offline retrieval evaluation:

```bash
python -m evaluation.evaluate_v3_offline
```

Ingest a company's recent 10-Ks (writes to Pinecone and the local registry):

```bash
python -m ingestion.ingest_company --ticker AMZN --years 3 --verify
python -m scripts.build_bm25s_index
```

## Repository Structure

```text
api/          FastAPI app, request validation, RAG orchestration, generation
ingestion/    SEC discovery, download, cleaning, chunking, ingestion ledger
retrieval/    embeddings, Pinecone clients, bm25s backend, RRF hybrid, Cohere reranker, scope
evaluation/   benchmarks, pooled judging, offline retrieval and multi-year validation
scripts/      bm25s index build, metadata backfills, filter regression checks
tests/        unit and structural tests (scope, leakage, numeric attribution, fallback)
data/         chunks, registry, persisted bm25s index, SEC cache (embeddings/raw gitignored)
```

See `INGESTION.md` for ingestion operations and `FAILURES.md` for recorded failure analysis.
