# Scale Readiness — 10 → 20 → 50 → 100 companies

Date: 2026-08-31
Production checkpoint: `2e33141` (10 companies, 31 filings, 4,262 chunks).
Nothing in production was changed. No companies ingested. No commit.

---

## 0. Measured constants (used for every projection)

| quantity | measured value | how |
|---|---|---|
| bm25s bundle bytes / chunk | **7,153 B** | 29.07 MB ÷ 4,262 (4,273 B `chunks.jsonl` + 2,881 B index arrays) |
| `data/chunks/**` per-filing jsonl | **~4,100 B / chunk** | 17.0 MB ÷ 4,262 (near-duplicate of `chunks.jsonl`; not loaded at runtime) |
| Pinecone dense metadata / vector | **~3,838 B** | fetched-match size; `text` field dominates (~3,800 chars) |
| production `/ask` dense query egress | **~37.5 KB** | 10 matches × metadata, `include_metadata=True` |
| depth-80 pool query egress | **~292 KB** | re-pool build query |
| dense query egress *without* `text` in metadata | **~0.7 KB** | `include_metadata=False` — **54× smaller** |
| bm25s cold load | **185 ms @ 4,262** → ~43 µs/chunk | `BM25SBackend.load()` cold |
| bm25s warm load | 66 ms | cached |
| linux deps (`requirements.txt`, no scipy — bm25s needs only numpy) | **~100 MB** est. | numpy ~35 + openai ~15 + pinecone ~9 + cohere ~5 + pydantic ~8 + fastapi/uvicorn ~5 + tiktoken ~5 + misc |
| Cohere calls per `/ask` | **1 rerank, always** | per-scope hybrid → union (cap 60) → **one** `rerank_once` — verified in `api/rag.retrieve_evidence_comparison` |
| dense (Pinecone) queries per `/ask` | 1 single-scope; **N for an N-scope comparison** | one query per scope filter |

### Company chunk profiles (3 fiscal years, measured)

| company | chunks/3yr | chunks/filing | profile |
|---|---|---|---|
| AAPL | 198 | 66 | smallest — terse 10-K |
| MSFT | 308 | ~103 | |
| NVDA | 409 | ~102 | (4 years) |
| GOOGL | 323 | 108 | |
| AMZN | 266 | 89 | |
| UNH | 299 | ~100 | |
| WMT | 340 | 113 | |
| META | 444 | 148 | verbose risk factors |
| XOM | 452 | 151 | segment-heavy |
| **JPM** | **1,223** | **~408** | **large bank — 3–4× the median** |

median company ≈ **300 chunks / 3yr**; mean (current, tech-heavy) ≈ 328; large filer ≈ **1,100–1,300**.

---

## 1. Benchmark_v3 re-pool

### Methodology (model-assisted, **not** human labels)

* **Same 125-question set** (`benchmark_v3_questions.json`), verbatim. `benchmark_v3.json` is **untouched**; output is the new `benchmark_v3_repool.json`.
* Each question **year-scoped** to the fiscal year(s) its existing v3 qrels live in (`evaluate_v3_offline._qrel_years`) — the benchmark is single-year-per-question by construction, and scoping keeps the pool comparable instead of diluting it with other-year chunks now that every company has 3–4 filings.
* **Pool depth 80** (was 50) — justified: the corpus tripled (1,684 → 4,262), so pool deeper than production `candidate_k=10` by a wider margin. Retrievers pooled: **dense + bm25s + hybrid + sparse**.
* **Judged head 60** per question (was 50). Unjudged pool chunks are **not** treated as non-relevant — they are simply unlabeled and excluded from precision denominators exactly as `benchmark_v3` did.
* Judge: **gpt-5-mini** minimal-effort first pass (all head candidates in one call) → **gpt-5-mini** low-effort review pass re-judging **every positive + every medium/low-confidence** candidate.

### Pool + judgment counts

| | value |
|---|---|
| pool depth | 80 |
| candidates pooled | **10,179** (avg **88.5**/q; min 66 AAPL, max 132 JPM) |
| candidate-judgments | **6,900** (head 60 × 115) |
| relevant labels — first pass | 1,604 |
| relevant labels — **after review** | **1,359** (595 flips, 9% of judged) |
| positive rate | 20% of judged |
| confidence of relevant labels | **1,131 high / 227 medium / 1 low** (83% high) |

### Old → new label expansion

| | frozen v3 | repool | Δ |
|---|---|---|---|
| relevant labels (supported) | 1,311 | **1,359** | **+48 (+3.7%)** |
| mean relevant / question | 11.4 | 11.8 | |
| label churn vs frozen | — | **kept 992 · added 367 · dropped 319** | ~25% of labels changed |

Per company: META **+33**, NVDA **+25**, AMZN +12, AAPL +8 gained (multi-year depth + semantically rich risk disclosure); JPM **−14**, MSFT **−12** lost (stricter re-judge demoting keyword-heavy financial chunks the old gpt-5-nano first pass over-labeled). **GOOGL/WMT ≈ flat.**

> **Benchmark gap found:** `benchmark_v3_questions.json` covers **8 companies only** — it has **no XOM and no UNH questions**, and none for future pilot companies. This must be extended before the 20-company pilot is meaningfully gated.

### Old-vs-new metrics — *same* current-corpus retrieval, two qrel sets

(One year-scoped live retrieval per question, scored against frozen vs re-pooled qrels. Deterministic — retrieval is called once and reused.)

| retriever | metric | frozen v3 | **repool** | Δ |
|---|---|---|---|---|
| **hybrid** (production) | **MRR** | 0.897 | **0.884** | **−0.013** |
| | R@10-capped | 0.718 | 0.701 | −0.017 |
| | P@1 | 0.835 | 0.817 | −0.017 |
| | P@5 | 0.638 | 0.623 | −0.016 |
| | R-precision | 0.551 | 0.543 | −0.008 |
| dense | MRR | 0.871 | **0.885** | **+0.014** |
| bm25s | MRR | 0.879 | **0.822** | **−0.057** |
| bm25s | P@1 | 0.800 | 0.722 | −0.078 |

### Was the MRR drift (0.903 → 0.900 → 0.897) a benchmark artifact?

**Partly — not entirely.**

* It was **never a gate breach** — 0.884 re-anchored is well within the ±0.02 band around the historical 0.903.
* The drift is **not pure stale-pool inflation**: re-judging against the current corpus, **dense retrieval got *better* (+0.014) while bm25s degraded sharply (−0.057)**.
* **Mechanism:** bm25s scores the *entire* 4,262-doc corpus (global IDF) and *then* post-filters to the query's scope (`retrieval/lexical_backend.py:197` — "retrieve the full ranked corpus, then filter"). As the corpus grew 1,300 → 4,262, global IDF shifted, so within-scope lexical rankings drifted. Dense (Pinecone, server-side filtered) is immune. The frozen benchmark also over-credited bm25s because its first-pass judge was gpt-5-nano (known to over-label keyword matches; the memory notes it "marked everything relevant" and needed heavy review).
* **Net:** hybrid retrieval quality on the current corpus is genuinely **~0.884 MRR** — adopt this as the new baseline. **bm25s weakening as the corpus grows is itself a scale signal** (see §3/§8-D).

### Recommended new regression gates (repool hybrid baseline − 0.02)

| metric | new baseline | floor |
|---|---|---|
| hybrid MRR | 0.884 | **0.864** |
| hybrid R@10-capped | 0.701 | **0.681** |
| hybrid P@5 | 0.623 | **0.603** |
| structural filter_correctness / cross-company leakage@10 | 1.000 / 0.000 | exact (unchanged) |

Artifacts: `evaluation/benchmark_v3_repool.json`, `evaluation/results/benchmark_v3_repool{,_judgments}.json`, `evaluation/results/v3_repool_comparison.json`, and the new scripts `build_v3_repool.py` / `judge_v3_repool.py` / `assemble_v3_repool.py` / `compare_v3_repool.py`.

---

## 2. XOM numeric sensitivity — is it a generic repeated-table failure?

**Partially. It is a filing-layout pattern, not a universal retrieval bug.**

Scan: loose vs anchored phrasing, newest FY, headline revenue metric, 3 runs each (`evaluation/results/numeric_ambiguity_scan.json`).

| company | metric | loose: value correct | loose: **wrong-year value** | anchored: value correct |
|---|---|---|---|---|
| **XOM** | total revenues & other income | 2/3 | **1/3 — 349,585 (FY2024) labelled FY2025** | 3/3 |
| JPM | net income | 3/3¹ | 0/3 | 3/3 |
| WMT | total revenues | 3/3 | 0/3 | 3/3 |
| AAPL | total net sales | 3/3 | 0/3 | 3/3 |
| MSFT | total revenue | 3/3 | 0/3 | 3/3 |

¹ two JPM runs answered "$57.0 billion" (correct, rounded) — an exact-digit check under-counts; no wrong-year error.

**Only XOM reproduced an actual wrong-year value.** All source *years* were correct (2025); the wrong *number* came from a prior-year comparative sub-total physically inside a FY2025-filing chunk.

**Root cause (measured):** the XOM 10-K carries the same metric across **4+ tables**, and its **segment reconciliation restates one bare total per year in separate chunks with no adjacent fiscal-year column** — chunk 94 = FY2025 `"Total consolidated revenues and other income 332,238"`, chunks 95/96 = FY2024 `"... 349,585"`. A loose query retrieves the FY2024 reconciliation chunk and the generator, seeing a bare number in requested-year context, mislabels it.

JPM/WMT/AAPL/MSFT present headline figures **column-labelled** (`"713,163 680,985 648,125"` under `2026 2025 2024` headers), so a loose query still lands on a correctly-labelled row → no mislabel.

**Generic risk:** latent for **segment-heavy filers with per-year reconciliation sub-totals** — energy (CVX), industrials / conglomerates (BRK.B, GE, HON, MMM), insurers with statutory tables. **Not active in the current 10 beyond XOM.** Anchored phrasing: **15/15 correct** across all 5.

**Not patched** (assessment only; failure does not reproduce generically). If/when addressed, three non-XOM-specific options, in order of preference:
1. **Generation prompt:** instruct the model to prefer a figure that appears alongside an explicit fiscal-year column header matching the requested year, and to distrust a bare total in a reconciliation/segment table.
2. **Retrieval:** a chunk-type signal — down-weight `reconciliation` / `selected financial data` / `segment` chunks for single-year numeric questions.
3. **Benchmark:** every current numeric question is anchored ("consolidated statement of income"), so the gates stay 1.000 and **cannot catch this**. Add a handful of loose-phrasing numeric probes to `benchmark_v3_repool` / the multiyear suites.

---

## 3. bm25s / Vercel scale projection

**Vercel limits (docs, last-updated 2026-08-24):** Python function uncompressed bundle **500 MB** standard; **Large Functions** (beta) up to **5 GB** (needs fluid compute + Active CPU, opt-in `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`, supported on Python). Memory: Hobby 2 GB / Pro 4 GB. Duration: Hobby 300 s / Pro 800 s. Response body cap 4.5 MB. Bundle = code + linux deps + **every tracked project file** (FastAPI zero-config preset).

**Company mix model:** ~18% large filers (banks/insurers/Berkshire) at ~1,150 chunks/3yr, ~82% median at ~300 → **blended ≈ 453 chunks/company/3yr**. Optimistic ≈ 330, conservative ≈ 650.

| companies | chunks (opt / **blended** / consv) | bm25s bundle @ 7,153 B (opt / **blend** / consv) | cold load (blended) |
|---|---|---|---|
| **10 (now)** | 4,262 (measured) | **29 MB** (measured) | 185 ms |
| **20** | 7,600 / **8,800** / 10,800 | 54 / **63** / 77 MB | ~380 ms |
| **50** | 17,500 / **22,400** / 30,300 | 125 / **160** / 217 MB | ~960 ms |
| **100** | 34,000 / **45,000** / 62,800 | 243 / **322** / 449 MB | ~1.9 s (blended), ~2.7 s (consv) |

**Full Vercel function bundle** (bm25s + `data/chunks/**` duplicate + ~100 MB deps), blended:

| companies | `data/chunks/**` **bundled** | `data/chunks/**` **`.vercelignore`'d** (not loaded at runtime) |
|---|---|---|
| 20 | 63 + 36 + 100 = **199 MB** ✅ | 163 MB ✅ |
| 50 | 160 + 92 + 100 = **352 MB** ✅ | **260 MB** ✅ |
| 100 (blended) | 322 + 184 + 100 = **606 MB** ❌ (> 500) | **422 MB** ✅ (no margin) |
| 100 (conservative) | 806 MB ❌ | **549 MB** ❌ |

**Deploy artifact (git-tracked `data/`):** 48 MB now → ~510 MB at 100co blended. **git LFS or a build-time index fetch is advisable beyond ~50 companies.**

**Vercel constraints, in order they bite:**
1. **git repo weight** (~250 MB at 50co) — annoying before it's fatal; LFS by 50co.
2. **500 MB function bundle** — fine to ~50co; at 100co needs `.vercelignore data/chunks/**` (trivial, non-runtime) *and* still no margin under the conservative mix → Large Functions beta or externalized lexical.
3. cold-load latency ~1 s at 50co, ~2 s at 100co — acceptable with fluid compute keeping instances warm, but a cold `/ask` at 100co is a 2 s+ tax.
4. memory: a 322 MB index + numpy + FastAPI ≈ 500–700 MB RSS at 100co — under Hobby 2 GB, fine.

---

## 4. Pinecone scale projection

| companies | vectors (dense = sparse, each) | dense storage (vec 6,144 B + meta ~3,838 B) | sparse storage (est.) | total | storage $/mo @ $0.33/GB |
|---|---|---|---|---|---|
| 10 (now) | 4,262 | ~43 MB | ~15 MB | ~58 MB | ~$0.02 |
| 20 | ~8,800 | ~88 MB | ~30 MB | ~120 MB | ~$0.04 |
| 50 | ~22,400 | ~224 MB | ~78 MB | ~300 MB | ~$0.10 |
| 100 | ~45,000 | ~450 MB | ~155 MB | ~605 MB | ~$0.20 |

* **Read units:** `1 RU per 1 GB of namespace, min 0.25 RU/query`; **`top_k` and `include_metadata` do *not* affect query cost.** Our namespace stays < 1 GB even at 100co → **0.25 RU/query flat.** At $16/M RU: 1M production queries/mo ≈ 250k RU ≈ **$4/mo.** Negligible at every scale.
* **Egress** — the one that bit us:
  * Starter tier: **hard 1 GB/mo cap → `RESOURCE_EXHAUSTED` (429)**. Production `/ask` = **37.5 KB/query** → 1 GB ≈ **27,000 queries/mo**. A single re-pool run ≈ 67 MB; a full eval sweep ≈ 100–200 MB; a dev month of iteration ≈ 1 GB. **This is exactly what exhausted the quota.**
  * **Standard ($50/mo minimum):** Vercel `iad1` and the Pinecone index (`aped-4627-b74a`) are **both AWS us-east-1 → no cross-region egress charge**, and the 1 GB cap is Starter-only. On Standard, **egress is effectively unmetered for our pattern.**
* **Eval cost:** unchanged by company count *per query*, but a bigger benchmark = proportionally more queries. A 100-company benchmark re-pool at depth 80 ≈ 300 queries × 292 KB ≈ **90 MB egress** — a rounding error on Standard, a third of the Starter cap.
* **Production monthly query usage:** unknown (traffic-dependent). Even 500k `/ask`/mo = ~19 GB egress (Standard: free, same-region) + 125k RU (~$2). **Not a constraint on Standard.**

**Big lever:** removing `text` from Pinecone metadata (look it up by `chunk_id` from the bm25s bundle after retrieval) drops per-query egress **54×** (37.5 KB → 0.7 KB) → even Starter would sustain ~1.4M queries/mo. Worth doing regardless of tier.

---

## 5. Cohere scale projection

* **1 rerank call per `/ask`, always** — single-scope and comparison alike (per-scope hybrid → deduped union capped at 60 → one `rerank_once`). **Comparison queries do *not* amplify Cohere** (they amplify *dense Pinecone* calls — N per N-scope comparison).
* **Trial key: 10 requests/minute** → **600 `/ask`/hour ceiling**, and it is hit constantly when eval scripts run back-to-back.
* **Eval load:** retrieval-only pooling (`build_v3_repool`, `evaluate_v3_offline`) makes **zero** Cohere calls. The multiyear batch evals + numeric-generation validators call `plan_evidence` → **~15–50 rerank calls per run**. Running 3–4 eval scripts in sequence → sustained > 10/min → `reranker_fallback` (hybrid top-k, still scope-correct, non-deterministic NE@k).
* **Fallback frequency at scale:** driven by **benchmark size, not corpus size**. 8-company benchmark ≈ 15 rerank/batch-eval; a 20-company benchmark ≈ 40/run; a 100-company benchmark ≈ 200/run — any single eval sweep then exceeds 10/min for its whole duration on the trial key.
* **Production:** at any real QPS the trial key throttles immediately. A **paid key** (`rerank-v3.5` / v4 ≈ **$1–2 per 1,000 searches**): 100 companies at 100k `/ask`/mo ≈ **$100–200/mo**; eval usage negligible.

**Cohere is the binding operational constraint *today*, at 10 companies**, independent of corpus size. The fix (paid key) is small and well understood.

---

## 6. Frontend scale readiness (no changes made)

| element | 20 | 50 | 100 | bottleneck |
|---|---|---|---|---|
| `GET /companies` payload (`{ticker,name}` ≈ 30 B each) | 0.6 KB | 1.5 KB | 3 KB | none; consider a search endpoint > 500 |
| **CompanySelector** — renders every company as a `cmdk` `CommandItem`, client-side filter | fine | fine | **100 DOM nodes in a popover; cmdk handles it, but sluggish typing risk** | **virtualize > ~300 companies** |
| **company chips** — one `<span>` per selected ticker, flex-wrap | fine | messy at 15–20 selected | unusable at 30+ selected | UX limit ~15 selected — product decision, not a code bug |
| **`useAvailableYears`** — fetches `/companies/{ticker}/filings` **per selected ticker**, session-cached | 5 selected = 5 parallel reqs | 15 selected = queues behind browser's ~6/host cap | 30 selected = noticeable stall | **add `GET /companies/filings?tickers=a,b,c` batch endpoint by ~50co** |
| **year intersection** (`intersectYears`, pure client) | O(t·y), trivial | trivial | trivial | none |
| **evidence-by-scope / ScopeBar** — one row per (ticker,year) scope | 2–3 scope comparison = clean | a 6-scope comparison already shows ~1 chunk/scope | a 10-company × 3-year "comparison" = 30 rows, 1 chunk each — unreadable | **already a soft limit today**; `MIN_PER_SCOPE` + `evidence_k` cap means big comparisons degrade to noise. Product-design limit, not frontend code. |

**None of this blocks a 20-company pilot.** Priorities if scaling past 50: (1) batch filings endpoint, (2) selector virtualization, (3) a hard cap or different UI for comparisons beyond ~4 scopes.

---

## 7. Proposed next 10 companies (20-company pilot) — **not ingested**

Chosen to stress every filing profile the current 10 don't: large-bank, conglomerate, pharma, medtech, energy-reconciliation, embedded-finance industrial, transport, and **two more non-calendar fiscal years**.

| # | ticker | company | sector | why included | filing-size risk | lineage / fiscal-year edge case |
|---|---|---|---|---|---|---|
| 1 | **BAC** | Bank of America Corp | Financials (bank) | 2nd large bank → confirms JPM isn't a one-off; RRF/bundle stress | **High** — JPM-class, ~380–430 chunks/filing (~1,200/3yr) | none; standard ticker→CIK (0000070858), calendar FY |
| 2 | **BRK.B** | Berkshire Hathaway Inc | Financials / conglomerate | Largest 10-K by breadth (insurance + BNSF + energy + manufacturing + retail segments) — the true bundle stress test | **Very high** — expect 900–1,600 chunks/filing | **class-B ticker**; SEC registrant `BERKSHIRE HATHAWAY INC`, CIK **0001067983**; calendar FY |
| 3 | **LLY** | Eli Lilly and Co | Healthcare (pharma) | Pharma R&D / patent-cliff / pipeline / pricing-litigation disclosure — a profile absent from the set | Medium (~110–140/filing) | none; calendar FY |
| 4 | **JNJ** | Johnson & Johnson | Healthcare (pharma + medtech) | Dual pharma/device; **massive talc-litigation footnotes**; **Kenvue spin-off (Aug 2023)** → 2023 10-K has consumer-health as discontinued operations | Medium-high (~140–170/filing) | **discontinued-ops restatement** across the FY2023↔FY2025 window — stresses year-over-year numeric comparability; calendar FY |
| 5 | **CVX** | Chevron Corp | Energy | XOM peer → directly tests whether the segment-reconciliation numeric-ambiguity risk (§2) is generic to energy filers | Medium-high (~150/filing) | **Hess Corp acquisition closed ~2025** (post-arbitration) → combined-entity / pro-forma disclosure; segment-reconciliation tables; calendar FY |
| 6 | **CAT** | Caterpillar Inc | Industrials | Heavy machinery **with an embedded financial-services segment (Cat Financial)** — a bank-inside-an-industrial filing | Medium (~120–150/filing) | none; calendar FY |
| 7 | **UNP** | Union Pacific Corp | Industrials / transport | Railroad — capital-intensive, single-segment, clean; a baseline industrial | Low-medium (~90–120/filing) | none; calendar FY |
| 8 | **PG** | Procter & Gamble Co | Consumer staples | Global brand portfolio, FX exposure, clean mid-size filing | Low-medium (~100–130/filing) | **non-calendar FY — ends June 30** (like MSFT); reportDate ~2026-06-30 |
| 9 | **HD** | Home Depot Inc | Consumer discretionary (retail) | WMT peer; big retail supply-chain / e-commerce disclosure | Medium (~110–140/filing) | **non-calendar FY — ends the Sunday nearest Jan 31** (like WMT); reportDate ~2026-02-01 |
| 10 | **KO** | Coca-Cola Co | Consumer staples | Global operations, heavy FX + **bottler consolidation/deconsolidation** disclosure; equity-method complexity | Low-medium (~100–130/filing) | none; calendar FY |

Sector coverage after pilot: **tech** 6, **financials** 3 (JPM, BAC, BRK.B), **healthcare** 3 (UNH, LLY, JNJ), **industrials** 3 (CAT, UNP + BRK segments), **consumer** 4 (WMT, HD, PG, KO), **energy** 2 (XOM, CVX). Non-calendar FY filers: 4 (MSFT, WMT, PG, HD).

Projected pilot corpus: current 4,262 + ~**3,400** (blended, with BAC + BRK.B as large filers) → **~7,700 chunks**, bm25s bundle **~55 MB**, full function bundle **~185 MB**.

---

## 8. Decision gates

**A. Can we safely start a 20-company pilot with the current architecture?**
**YES.** Projected bundle ~185–200 MB (well under the 500 MB Python limit), cold load ~380 ms, Pinecone trivial on Standard, git repo ~80 MB. **Three prerequisites, none blocking:**
1. **Move Cohere off the trial key** onto a paid key (the actual current constraint — §5/§F).
2. **Stay on Pinecone Standard** (never fall back to Starter — the 1 GB egress cap is a hard wall).
3. **Extend the benchmark** — `benchmark_v3_questions.json` has no XOM/UNH questions; add XOM/UNH + the 10 pilot companies (and a few loose-phrasing numeric probes per §2) before the pilot's regression gate means anything.

**B. Can bundled bm25s reasonably survive to 50 companies?**
**YES, with one trivial change:** `.vercelignore data/chunks/**` (it is not read at runtime — only `data/bm25s_index/chunks.jsonl` is). That gives ~260 MB bundle + ~1 s cold load at 50co. This is the **"working but plan the exit"** zone — have the migration (§D) designed and mostly built by ~40 companies.

**C. Can it reasonably survive to 100?**
**NO, not comfortably.** Even with `data/chunks/**` excluded: blended ~422 MB (no margin under 500 MB), conservative ~549 MB (**over**). git repo ~510 MB. Cold load ~2 s. Options at that point: Vercel **Large Functions** beta (5 GB, needs fluid compute + Active CPU) as a stopgap, or — better — the lexical backend is already out of the bundle (§D).

**D. When should lexical retrieval move out of the Vercel bundle?**
**Between ~40 and ~60 companies** (~18k–25k chunks, ~150–200 MB bm25s bundle, ~1 s cold load, git repo passing ~250 MB). Cheapest path, **already 90% built:** **enable the existing Pinecone `sparse` index in production and retire the bm25s bundle** — the sparse index already exists, mirrors bm25s 1:1 at 4,262 vectors, is populated on every ingest, and Pinecone filters server-side (no global-IDF-over-the-whole-corpus drift — which is exactly the bm25s weakness §1 surfaced). Cost: re-tune RRF weights (sparse ≠ bm25s ranking; the benchmark shows sparse weaker standalone) and accept a little more Pinecone RU/egress (both cheap on Standard). Alternative: **managed OpenSearch/Elasticsearch** (~$50–100/mo small cluster) — more faithful to the current bm25s ranking, more ops surface. **Recommendation: sparse-in-Pinecone**, decided and RRF re-tuned by 40 companies, cut over by 50.

**E. Does Pinecone pricing/egress become the bigger constraint first?**
**NO** — *only* on the free Starter tier, where the 1 GB/mo egress cap already bit us and is already mitigated by being on Standard. On Standard, at 100 companies: storage ~$0.20/mo, RU ~$4/mo at 1M queries, same-region egress free. Pinecone is **not** a top-3 constraint.

**F. Does Cohere quota become the bigger constraint first?**
**YES.** It is the binding constraint **right now at 10 companies** — the trial key's 10 req/min throttles every multi-script eval session and would throttle any real production traffic. It scales with **benchmark size**, so it gets worse with the pilot. The fix (paid rerank key, ~$1–2/1k searches) is small, well-understood, and should land **before** the 20-company pilot.

---

## 9. Final report — answers

| # | question | answer |
|---|---|---|
| 1 | refreshed benchmark methodology | Same 125 questions; year-scoped depth-80 pool (dense+bm25s+hybrid+sparse); judged head 60; gpt-5-mini minimal first pass + gpt-5-mini low review re-judging every positive + medium/low-confidence candidate. Model-assisted, **not human**. New file `benchmark_v3_repool.json`; `benchmark_v3.json` untouched. |
| 2 | refreshed benchmark metrics | **hybrid MRR 0.884**, R@10-capped 0.701, P@1 0.817, P@5 0.623, R-prec 0.543. Dense MRR 0.885, bm25s MRR 0.822. Structural gates unchanged (1.000 / 0.000). New floors: MRR 0.864 / R@10c 0.681 / P@5 0.603. |
| 3 | was the old MRR drift a benchmark artifact? | **Partly.** Never a gate breach (0.884 is within ±0.02 of 0.903). But re-anchoring shows **dense +0.014, bm25s −0.057** — the drift is real and **dominated by bm25s** (global-IDF-then-filter drifts as the corpus grows), plus the old gpt-5-nano first pass over-credited keyword chunks. |
| 4 | XOM numeric ambiguity findings | Reproduced **1/3** on loose phrasing (FY2024 $349,585M labelled FY2025); **0/3** on anchored; **3/3** anchored correct. Cause: XOM's segment reconciliation restates a **bare per-year total in separate chunks with no fiscal-year column**; a loose query grabs the FY2024 one. |
| 5 | generic numeric-retrieval risk | **Not generic today.** JPM/WMT/AAPL/MSFT are **0/3 wrong-year on loose phrasing** because they column-label every figure. Latent risk for **segment-heavy filers with per-year reconciliation sub-totals** (CVX, BRK.B, industrials, insurers). Anchored phrasing 15/15. |
| 6 | current bytes/chunk | **bm25s bundle 7,153 B/chunk** (4,273 text + 2,881 index); Pinecone dense metadata ~3,838 B/vector; `data/chunks/**` ~4,100 B/chunk (runtime-unused duplicate). |
| 7 | projected bm25s size at 20/50/100 | blended **63 MB / 160 MB / 322 MB**; conservative 77 / 217 / 449 MB. Full function bundle (blended, `data/chunks/**` excluded): 163 / 260 / 422 MB. Cold load ~0.38 / ~0.96 / ~1.9 s. |
| 8 | projected vector counts | dense = sparse, each: **~8,800 / ~22,400 / ~45,000**. |
| 9 | Pinecone scale risk | **Low on Standard** (storage < $0.25/mo, RU 0.25/query flat = ~$4/mo at 1M queries, same-region egress free). **High on Starter** (1 GB/mo egress hard cap — already hit). Lever: drop `text` from metadata → 54× less egress. |
| 10 | Cohere scale risk | **Highest near-term risk.** Trial key 10/min already throttles evals at 10 companies; scales with benchmark size. 1 rerank/`/ask` (no comparison amplification). Needs paid key (~$1–2/1k) before the pilot. |
| 11 | Vercel scale risk | Fine to ~50co (`.vercelignore data/chunks/**`, ~260 MB, ~1 s cold). At 100co blended ~422 MB (no margin), conservative over 500 MB → Large Functions beta or externalized lexical. git repo weight (~250 MB @ 50co) wants LFS. |
| 12 | frontend scale risk | None blocks the pilot. By ~50co: batch `/companies/filings` endpoint, selector virtualization. Comparisons beyond ~4–6 scopes are **already** unreadable (1 chunk/scope) — product-design limit. |
| 13 | recommended next 10 companies | **BAC, BRK.B, LLY, JNJ, CVX, CAT, UNP, PG, HD, KO** — see §7 for per-company rationale, size risk, and edge cases (BRK.B class-B ticker; JNJ Kenvue discontinued-ops; CVX Hess acquisition; PG June FY; HD end-Jan FY). |
| 14 | **go / no-go for the 20-company pilot** | **GO** — with 3 prerequisites first: (a) Cohere **paid** rerank key, (b) **stay on Pinecone Standard**, (c) **add XOM/UNH + pilot-company questions** (and loose-phrasing numeric probes) to the benchmark so its regression gate is real. Architecture headroom at 20 companies is comfortable (~185 MB bundle, ~0.4 s cold load). |
| 15 | likely architecture change point before 100 | **~40–60 companies.** Move lexical retrieval out of the Vercel bundle — preferred path is **enabling the already-populated Pinecone sparse index in production** (retire the bm25s bundle, re-tune RRF), decided by ~40co and cut over by ~50co. This also fixes the bm25s global-IDF drift that §1 surfaced. Secondary: git LFS for `data/` at ~50co. |
