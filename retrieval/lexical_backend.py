"""Pluggable lexical (BM25) retrieval backends for the Phase 4.5 experiment.

Both backends share the SAME tokenizer, stopwords, k1=1.5, b=0.75, and global
corpus statistics (IDF / avgdl computed over every chunk). Filtering is applied
AFTER scoring against the full corpus, exactly like the current
``retrieval.bm25_search.BM25Index.search`` — so ticker filters never change a
score, only which already-scored docs are eligible.

Result dicts match ``retrieval.bm25_search.search``:
    {chunk_id, score, text, company, ticker, filing_type, filing_date, source_url}

Nothing here is wired into production. ``retrieval.bm25_search`` remains the
control until Phase 4.5 picks a winner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from retrieval.bm25_search import BM25Index, load_chunks, tokenize
from retrieval.filters import RetrievalFilter

logger = logging.getLogger("retrieval.lexical")

REPO_ROOT = Path(__file__).resolve().parent.parent
BM25S_INDEX_DIR = REPO_ROOT / "data" / "bm25s_index"

DEFAULT_BACKEND = "bm25s"  # production default since Phase 4.5; "current" for rollback


class LexicalBackendError(RuntimeError):
    """The selected lexical backend could not be loaded or rebuilt."""

_RESULT_FIELDS = (
    "text", "company", "ticker", "filing_type", "filing_date", "source_url",
)


def _to_result(chunk: dict, score: float) -> dict:
    out = {"chunk_id": chunk["chunk_id"], "score": float(score)}
    for field in _RESULT_FIELDS:
        out[field] = chunk.get(field)
    return out


class LexicalBackend(ABC):
    name: str

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> list[dict]:
        ...

    # diagnostics
    build_ms: float = 0.0
    load_ms: float = 0.0
    document_count: int = 0
    average_document_length: float = 0.0
    vocabulary_size: int = 0


# --------------------------------------------------------------------------- #
class CurrentBM25Backend(LexicalBackend):
    """Wraps the production ``BM25Index`` (pure-Python)."""

    name = "current"

    def __init__(self, chunks: list[dict] | None = None):
        t0 = time.perf_counter()
        chunks = chunks if chunks is not None else load_chunks()
        self.load_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        self._index = BM25Index(chunks)
        self.build_ms = (time.perf_counter() - t0) * 1000
        self.document_count = self._index.document_count
        self.average_document_length = self._index.average_document_length
        self.vocabulary_size = len(self._index.inverse_document_frequencies)

    def search(self, query, top_k=5, filters=None):
        return self._index.search(query, top_k=top_k, filters=filters)


# --------------------------------------------------------------------------- #
class BM25SBackend(LexicalBackend):
    """bm25s with method='lucene' — the same IDF formula the current impl uses:
    ``log(1 + (N - df + 0.5) / (df + 0.5))``. The ``(k1+1)`` numerator factor in
    the current impl is a rank-preserving constant, so with matched tokenization
    the document ordering is expected to be identical.
    """

    name = "bm25s"

    def __init__(
        self,
        chunks: list[dict] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
        dtype: str = "float64",
        save_dir: str | None = None,
    ):
        import bm25s

        t0 = time.perf_counter()
        chunks = chunks if chunks is not None else load_chunks()
        self._chunks = chunks
        self.load_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self._retriever = bm25s.BM25(k1=k1, b=b, method=method, dtype=dtype)
        self._retriever.index(corpus_tokens, show_progress=False)
        self.build_ms = (time.perf_counter() - t0) * 1000

        self.document_count = len(chunks)
        # bm25s stores per-doc lengths; avgdl = mean
        try:
            import numpy as np

            self.average_document_length = float(
                np.mean([len(t) for t in corpus_tokens])
            )
        except Exception:  # noqa: BLE001
            self.average_document_length = (
                sum(len(t) for t in corpus_tokens) / max(1, len(corpus_tokens))
            )
        self.vocabulary_size = len(getattr(self._retriever, "vocab_dict", {}) or {})
        self._save_dir = save_dir
        if save_dir:
            self.save(save_dir)

    # -- persistence -------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self._retriever.save(str(path))
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in self._chunks), encoding="utf-8"
        )
        (path / "corpus_version.json").write_text(
            json.dumps({
                "corpus_version": corpus_version(self._chunks),
                "document_count": len(self._chunks),
                "k1": 1.5, "b": 0.75, "method": "lucene", "built": time.time(),
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path):
        import bm25s

        path = Path(path)
        obj = cls.__new__(cls)
        obj.name = "bm25s"
        t0 = time.perf_counter()
        obj._retriever = bm25s.BM25.load(str(path), mmap=False)
        obj._chunks = [
            json.loads(line)
            for line in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        obj.load_ms = (time.perf_counter() - t0) * 1000
        obj.build_ms = 0.0
        obj.document_count = len(obj._chunks)
        obj.average_document_length = 0.0
        obj.vocabulary_size = len(getattr(obj._retriever, "vocab_dict", {}) or {})
        obj._save_dir = str(path)
        return obj

    def corpus_version(self) -> str:
        return corpus_version(self._chunks)

    # -- search ------------------------------------------------------- #
    def search(self, query, top_k=5, filters=None):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        n = len(self._chunks)
        # retrieve the full ranked corpus, then filter + take top_k, mirroring
        # BM25Index.search (score-all -> filter -> sort -> top_k).
        docs, scores = self._retriever.retrieve(
            [query_tokens], k=n, show_progress=False, n_threads=0
        )
        ranked = []
        for doc_idx, score in zip(docs[0].tolist(), scores[0].tolist()):
            chunk = self._chunks[doc_idx]
            if filters is not None and not filters.is_empty() and not filters.matches(chunk):
                continue
            ranked.append((float(score), chunk))
        # tie-break by chunk_id, matching the current impl
        ranked.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        return [_to_result(chunk, score) for score, chunk in ranked[:top_k]]


# --------------------------------------------------------------------------- #
# Corpus versioning + backend selection                                       #
# --------------------------------------------------------------------------- #
def corpus_version(chunks: list[dict]) -> str:
    """Deterministic id of the lexical corpus: sha256 of sorted chunk_ids."""
    joined = "\n".join(sorted(str(c["chunk_id"]) for c in chunks))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_persisted_bm25s(
    chunks: list[dict] | None = None,
    index_dir: str | Path = BM25S_INDEX_DIR,
) -> "BM25SBackend":
    """Deterministic (re)build of the bm25s index from canonical chunks + persist."""
    t0 = time.perf_counter()
    chunks = chunks if chunks is not None else load_chunks()
    version = corpus_version(chunks)
    try:
        backend = BM25SBackend(chunks)
        backend.save(index_dir)
    except Exception as exc:  # noqa: BLE001
        raise LexicalBackendError(
            f"bm25s rebuild failed: {type(exc).__name__}: {exc}"
        ) from exc
    logger.info(
        "lexical event=bm25s_rebuild document_count=%d duration_ms=%.0f "
        "corpus_version=%s",
        len(chunks), (time.perf_counter() - t0) * 1000, version[:12],
    )
    return backend


def load_or_build_bm25s(
    index_dir: str | Path = BM25S_INDEX_DIR,
    chunks: list[dict] | None = None,
) -> "BM25SBackend":
    """Load the persisted bm25s index; rebuild from canonical chunks if it is
    missing, unreadable, or its corpus_version does not match the chunks on disk.

    Never returns an empty index. Rebuild is chosen over "fail readiness"
    because the canonical chunk JSONL is always present (committed serving
    artifact) and a deterministic rebuild beats a hard outage on a stale/corrupt
    cache file. If BOTH load and rebuild fail, raises LexicalBackendError so the
    caller can fail readiness rather than serve partial lexical search.
    """
    index_dir = Path(index_dir)
    chunks = chunks if chunks is not None else load_chunks()
    expected = corpus_version(chunks)

    version_file = index_dir / "corpus_version.json"
    if index_dir.is_dir() and version_file.exists():
        try:
            meta = json.loads(version_file.read_text(encoding="utf-8"))
            if meta.get("corpus_version") == expected:
                t0 = time.perf_counter()
                backend = BM25SBackend.load(index_dir)
                logger.info(
                    "lexical event=bm25s_load document_count=%d duration_ms=%.0f "
                    "corpus_version=%s",
                    backend.document_count, (time.perf_counter() - t0) * 1000,
                    expected[:12],
                )
                return backend
            logger.warning(
                "lexical event=bm25s_stale expected=%s found=%s (rebuilding)",
                expected[:12], str(meta.get("corpus_version"))[:12],
            )
        except LexicalBackendError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "lexical event=bm25s_corrupt error_class=%s (rebuilding)",
                type(exc).__name__,
            )
    else:
        logger.info("lexical event=bm25s_missing dir=%s (building)", index_dir)

    return build_persisted_bm25s(chunks, index_dir)


def get_lexical_backend(
    chunks: list[dict] | None = None,
    *,
    force_rebuild: bool = False,
) -> LexicalBackend:
    """Return the selected lexical backend.

    Default backend: ``bm25s`` (persisted index; loads in ~20ms, rebuilds
    deterministically from canonical chunks if missing/corrupt/stale).
    Set ``SEC_RAG_LEXICAL_BACKEND=current`` to roll back to the pure-Python
    reference implementation (kept for parity/debugging).

    ``force_rebuild`` (used by reload() after ingestion) rebuilds + re-persists.
    """
    name = os.getenv("SEC_RAG_LEXICAL_BACKEND", DEFAULT_BACKEND).strip().lower()
    if name == "current":
        return CurrentBM25Backend(chunks)
    if force_rebuild:
        return build_persisted_bm25s(chunks)
    return load_or_build_bm25s(chunks=chunks)
