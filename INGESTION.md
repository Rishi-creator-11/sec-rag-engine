# Ingestion Operations

Automated single-filing SEC ingestion (latest 10-K per company).

## Commands

```bash
# discover + plan only, no writes / no network mutation
python -m ingestion.ingest_company --ticker AMZN --dry-run

# ingest the latest 10-K
python -m ingestion.ingest_company --ticker AMZN

# ingest then run consistency verification
python -m ingestion.ingest_company --ticker AMZN --verify

# rebuild a completed filing (same deterministic IDs, upserts overwrite)
python -m ingestion.ingest_company --ticker AMZN --force

# skip the sparse-parity stage (sparse is not in the production retrieval path)
python -m ingestion.ingest_company --ticker AMZN --skip-sparse

# verify an already-ingested filing (local + Pinecone + BM25; no OpenAI/Cohere)
python -m ingestion.verify_company --ticker AMZN [--full] [--json]
```

Requires `SEC_USER_AGENT="project-name you@example.com"` in the environment
(`.env`), max SEC request rate `SEC_MAX_RPS` (default 5, hard cap 10).

## Pipeline stages and their dependencies

```
discovered
  └─ downloaded        needs: SEC        produces: raw_html            [rebuild]
       └─ cleaned      needs: raw_html   produces: clean_text, metadata [rebuild]
            └─ chunked needs: clean_text produces: chunks JSONL         [SERVING]
                 ├─ embedded         needs: chunks   produces: embeddings JSONL [rebuild]
                 │    └─ dense_upserted   needs: embeddings   -> Pinecone sec-rag-engine
                 ├─ sparse_upserted  needs: chunks (NOT embeddings) -> Pinecone sec-rag-sparse (soft)
                 └─ bm25_registered  needs: chunks
            registry_updated  needs: chunks + dense_upserted
  └─ complete   needs: every hard stage ok (sparse may be skipped/failed)
```

## Artifact classes

| class | artifacts | on deletion |
|---|---|---|
| **SERVING** | `data/chunks/{ticker}/{filing_id}_chunks.jsonl` | filing is no longer queryable via BM25 → `complete` invalidated → re-chunk (+ downstream) |
| **REBUILD** | `filing.html`, `filing.txt`, `metadata.json`, `*_embeddings.jsonl` | large / gitignored; absence does **not** break a complete filing. Regenerated only on `--force` or if a later re-run needs them. |

Dense and sparse vectors live in Pinecone, not on local disk.

## Resume behavior

A rerun does **not** trust `stage == "ok"` blindly. `ingestion.stages.assess_filing`
re-validates each completed stage's artifacts on disk (existence, size, JSONL
integrity, count, sha256 recorded in the ledger). The resume point is the
earliest invalid or incomplete stage, backed up further if the artifact feeding
it is also gone:

| deleted | resume from | not repeated |
|---|---|---|
| `*_embeddings.jsonl` | `embedded` (re-embed) → `dense_upserted` | download, clean, chunk |
| `*_chunks.jsonl` | `chunked` (re-chunk from clean text) | download, clean |
| `filing.txt` | `cleaned` (re-clean from raw HTML) | download |
| `filing.txt` **and** `filing.html` | `downloaded` | — |
| `filing.html` only, everything else intact & complete | nothing — stays `already_ingested` | everything |

## Content hashes (ledger `hashes`)

| hash | set by | detects |
|---|---|---|
| `primary_doc_sha256` | downloaded | changed source document (on `--force` re-download) |
| `clean_text_sha256` | cleaned | tampered/corrupt clean text; confirms re-chunk input is unchanged |
| `chunks_sha256` | chunked | corrupt serving artifact |
| `embeddings_sha256` | embedded | stale/corrupt embedding artifact |

Deliberately one hash per persisted artifact file — not per chunk / per vector.

## Atomic writes

The ledger, the registry, and every ingestion artifact
(`filing.txt`, `metadata.json`, chunk JSONL, embedding JSONL) are written via
`ingestion.atomicio` (temp file in the same dir → flush → `fsync` →
`os.replace`). An interrupted process never replaces a good file with a
half-written one.

## BM25 refresh — running API picks up new filings

BM25 is an in-process index built lazily from `data/chunks/**/*_chunks.jsonl`.

- The ingestion process calls `retrieval.bm25_search.reload()` itself (its
  `bm25_registered` stage) and verifies the new filing is visible.
- **A separately running API process does not share that memory.** After
  ingesting a company, **restart the API** so it rebuilds BM25.
- `GET /health` reports `bm25_documents` (chunk count, or `null` before first
  build) and `companies` so an operator can confirm the restart worked.

Dense (Pinecone) retrieval is live immediately — no restart needed for that.

## Registry naming

- `legal_name` — SEC-authoritative, written by ingestion, never mutated by us.
- `display_name` — optional curated name; `GET /companies` returns it (falls
  back to `legal_name`). Ingestion never overwrites a curated `display_name`.
- Retrieval filtering uses `ticker` only — never a name.
