"""Explicit ingestion stage dependency graph + artifact validation.

Not a workflow engine. Just enough structure to:
  1. know which prior stages and which local artifacts a stage needs
  2. tell whether a stage recorded ``ok`` is *still* valid on disk
  3. compute the earliest stage a resume must restart from

Artifact classes
----------------
SERVING  artifacts are read at query time and must exist for a filing to be
         considered queryable. Currently: the chunk JSONL (BM25 reads it).
         Dense/sparse vectors live in Pinecone, not on local disk.
REBUILD  artifacts (raw HTML, cleaned text, embedding JSONL) only speed up
         re-running the pipeline. They are large / gitignored and may be
         deleted deliberately. Their absence does NOT make a COMPLETE filing
         broken — the pipeline just regenerates them from the earliest point
         that is still intact if a later stage actually needs them.

Resume policy
-------------
- A stage is "valid" iff status is ok/skipped AND every artifact it declares
  exists and passes validation AND every stage it depends on is valid.
- ``resume_stage`` = the earliest stage that is not valid (walking in order).
- A COMPLETE filing with its SERVING artifact intact but REBUILD artifacts
  missing is reported ``servable=True`` and ``resume_stage=None`` — a plain
  rerun is a no-op ("already ingested"); only ``--force`` regenerates.
- A COMPLETE filing whose SERVING artifact (chunks) is missing/corrupt is
  ``servable=False`` — ``complete`` is invalidated and the resume restarts at
  the earliest stage that can rebuild the chunks (re-chunk from cleaned text,
  or re-clean, or re-download, depending on what survives).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from retrieval.build_embeddings import EMBEDDING_DIMENSION

# Ordered pipeline (mirrors ingestion.state.STAGES minus the terminal "complete").
STAGE_ORDER = (
    "downloaded",
    "cleaned",
    "chunked",
    "embedded",
    "dense_upserted",
    "sparse_upserted",
    "bm25_registered",
    "registry_updated",
)

MIN_CLEAN_CHARS = 40_000
MIN_HTML_BYTES = 50_000


@dataclass(frozen=True)
class StageSpec:
    name: str
    depends_on: tuple[str, ...]
    # ledger "artifacts" keys this stage produces on local disk
    serving_artifacts: tuple[str, ...] = ()
    rebuild_artifacts: tuple[str, ...] = ()
    remote: bool = False  # writes to Pinecone rather than local disk

    @property
    def artifacts(self) -> tuple[str, ...]:
        return self.serving_artifacts + self.rebuild_artifacts


STAGE_SPECS: dict[str, StageSpec] = {
    "downloaded": StageSpec("downloaded", ("discovered",), rebuild_artifacts=("raw_html",)),
    "cleaned": StageSpec("cleaned", ("downloaded",),
                         rebuild_artifacts=("clean_text", "metadata")),
    "chunked": StageSpec("chunked", ("cleaned",), serving_artifacts=("chunks",)),
    "embedded": StageSpec("embedded", ("chunked",), rebuild_artifacts=("embeddings",)),
    "dense_upserted": StageSpec("dense_upserted", ("embedded",), remote=True),
    # sparse + bm25 consume the CHUNK text, not the OpenAI embeddings
    "sparse_upserted": StageSpec("sparse_upserted", ("chunked",), remote=True),
    "bm25_registered": StageSpec("bm25_registered", ("chunked",)),
    # registry is written only once the filing is actually servable
    "registry_updated": StageSpec("registry_updated", ("chunked", "dense_upserted")),
}

# When a stage's artifact is invalidated, which later stages must also re-run
# because they consumed (a derivative of) that artifact.
_DOWNSTREAM_OF = {
    "downloaded": ("cleaned", "chunked", "embedded", "dense_upserted",
                   "sparse_upserted", "bm25_registered", "registry_updated"),
    "cleaned": ("chunked", "embedded", "dense_upserted", "sparse_upserted",
                "bm25_registered", "registry_updated"),
    "chunked": ("embedded", "dense_upserted", "sparse_upserted",
                "bm25_registered", "registry_updated"),
    "embedded": ("dense_upserted", "registry_updated"),
    "dense_upserted": ("registry_updated",),
    "sparse_upserted": (),
    "bm25_registered": (),
    "registry_updated": (),
}


# --------------------------------------------------------------------------- #
# Artifact validators                                                         #
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_raw_html(path: Path) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "raw HTML missing"
    if p.stat().st_size < MIN_HTML_BYTES:
        return False, f"raw HTML too small ({p.stat().st_size} bytes)"
    return True, "ok"


def validate_clean_text(path: Path, *, expected_sha256: str | None = None) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "cleaned text missing"
    if p.stat().st_size < MIN_CLEAN_CHARS:
        return False, f"cleaned text too small ({p.stat().st_size} bytes)"
    if expected_sha256 and _sha256_file(p) != expected_sha256:
        return False, "cleaned text sha256 mismatch"
    return True, "ok"


def validate_metadata(path: Path) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "metadata.json missing"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"metadata.json unparseable: {exc}"
    if not data.get("accession_number"):
        return False, "metadata.json missing accession_number"
    return True, "ok"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_chunks_artifact(
    path: Path,
    *,
    expected_count: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "chunk JSONL missing"
    try:
        rows = _read_jsonl(p)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"chunk JSONL corrupt: {exc}"
    if not rows:
        return False, "chunk JSONL empty"
    ids = [r.get("chunk_id") for r in rows]
    if any(not cid for cid in ids):
        return False, "chunk JSONL row missing chunk_id"
    if len(ids) != len(set(ids)):
        return False, "chunk JSONL has duplicate chunk_id"
    indices = [r.get("chunk_index") for r in rows]
    if indices != list(range(len(rows))):
        return False, "chunk_index not contiguous from 0"
    if any(not str(r.get("text", "")).strip() for r in rows):
        return False, "chunk JSONL has empty text"
    if expected_count is not None and len(rows) != expected_count:
        return False, f"chunk count {len(rows)} != expected {expected_count}"
    if expected_sha256 and _sha256_file(p) != expected_sha256:
        return False, "chunk JSONL sha256 mismatch"
    return True, "ok"


def validate_embeddings_artifact(
    path: Path,
    *,
    expected_count: int | None = None,
    expected_chunk_ids: list[str] | None = None,
    expected_sha256: str | None = None,
) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "embedding JSONL missing"
    try:
        rows = _read_jsonl(p)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"embedding JSONL corrupt: {exc}"
    if not rows:
        return False, "embedding JSONL empty"
    for row in rows:
        vector = row.get("embedding")
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
            return False, f"{row.get('chunk_id')}: embedding dim != {EMBEDDING_DIMENSION}"
    if expected_count is not None and len(rows) != expected_count:
        return False, f"embedding count {len(rows)} != expected {expected_count}"
    if expected_chunk_ids is not None:
        if [r.get("chunk_id") for r in rows] != list(expected_chunk_ids):
            return False, "embedding chunk_id set/order does not match chunks"
    if expected_sha256 and _sha256_file(p) != expected_sha256:
        return False, "embedding JSONL sha256 mismatch"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Filing assessment                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class FilingAssessment:
    complete: bool
    servable: bool
    resume_stage: str | None
    invalid_stages: list[str] = field(default_factory=list)
    missing_rebuild_artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _artifact_path(paths: dict, key: str) -> Path | None:
    value = paths.get(key)
    return Path(value) if value else None


def assess_filing(
    entry: dict,
    artifact_paths: dict[str, Path],
    *,
    skip_sparse: bool = False,
) -> FilingAssessment:
    """Evaluate a ledger entry against what is actually on disk.

    ``artifact_paths`` maps artifact keys (raw_html, clean_text, metadata,
    chunks, embeddings) to absolute Paths for this filing.
    """
    stages = entry.get("stages", {})
    hashes = entry.get("hashes", {})
    chunk_count = entry.get("chunk_count")

    def stage_ok(name: str) -> bool:
        return stages.get(name, {}).get("status") in ("ok", "skipped")

    invalid: set[str] = set()
    notes: list[str] = []
    missing_rebuild: list[str] = []

    # 1. artifact-level validation for each locally-produced stage
    validators = {
        "raw_html": lambda p: validate_raw_html(p),
        "clean_text": lambda p: validate_clean_text(
            p, expected_sha256=hashes.get("clean_text_sha256")
        ),
        "metadata": lambda p: validate_metadata(p),
        "chunks": lambda p: validate_chunks_artifact(
            p, expected_count=chunk_count,
            expected_sha256=hashes.get("chunks_sha256"),
        ),
        "embeddings": lambda p: validate_embeddings_artifact(
            p, expected_count=chunk_count,
            expected_sha256=hashes.get("embeddings_sha256"),
        ),
    }

    # rebuild artifact -> (producing stage, the stage that consumes it)
    _rebuild_consumer = {
        "raw_html": ("downloaded", "cleaned"),
        "clean_text": ("cleaned", "chunked"),
        "embeddings": ("embedded", "dense_upserted"),
    }

    for stage_name in STAGE_ORDER:
        spec = STAGE_SPECS[stage_name]
        if spec.remote or not spec.artifacts:
            continue
        if not stage_ok(stage_name):
            continue
        for artifact_key in spec.artifacts:
            path = _artifact_path(artifact_paths, artifact_key)
            if path is None:
                continue
            valid, reason = validators[artifact_key](path)
            if valid:
                continue
            if artifact_key in spec.serving_artifacts:
                invalid.add(stage_name)
                notes.append(f"{stage_name}: {reason} (serving artifact)")
                continue
            # rebuild artifact: only a problem if a downstream consumer still
            # needs to run (e.g. embeddings gone while dense_upserted != ok).
            missing_rebuild.append(artifact_key)
            producer, consumer = _rebuild_consumer.get(artifact_key, (stage_name, None))
            if consumer is not None and not stage_ok(consumer):
                invalid.add(producer)
                notes.append(
                    f"{producer}: {reason} — needed because {consumer} is not complete"
                )
            else:
                notes.append(f"{stage_name}: {reason} (rebuild artifact, regenerable)")

    # 2. cascade: an invalid stage invalidates its downstream stages
    for stage_name in list(invalid):
        for downstream in _DOWNSTREAM_OF.get(stage_name, ()):
            if stage_ok(downstream):
                invalid.add(downstream)

    # 3. if a rebuild artifact is missing AND a still-"ok" later stage will need
    #    to be re-run for another reason, we must also rebuild that artifact.
    #    Handled implicitly: resume_stage walks from the earliest invalid stage,
    #    and any earlier stage whose rebuild artifact is gone is added below.
    complete = stages.get("complete", {}).get("status") == "ok"
    servable = "chunked" not in invalid and stage_ok("chunked")

    # 4. compute resume_stage
    resume_stage: str | None = None
    for stage_name in STAGE_ORDER:
        if skip_sparse and stage_name == "sparse_upserted":
            continue
        if stage_name in invalid:
            resume_stage = stage_name
            break
        if not stage_ok(stage_name):
            resume_stage = stage_name
            break

    # 5. if we must resume at stage X but an EARLIER rebuild artifact needed to
    #    produce X's inputs is missing, back the resume point up.
    if resume_stage is not None:
        resume_stage = _back_up_for_missing_inputs(
            resume_stage, stages, artifact_paths, hashes, notes
        )

    # A COMPLETE + servable filing whose soft sparse stage was ATTEMPTED and
    # FAILED (distinct from deliberately "skipped") is still worth repairing on
    # a plain rerun — sparse is idempotent and re-upserting is cheap relative to
    # --force. "skipped" is left alone.
    repair_soft_stage: str | None = None
    if complete and servable and not skip_sparse:
        if stages.get("sparse_upserted", {}).get("status") == "failed":
            repair_soft_stage = "sparse_upserted"
            notes.append("sparse_upserted was attempted and failed; a plain "
                         "rerun will retry it (idempotent)")

    if complete and not servable:
        notes.append("COMPLETE filing is not servable: chunk artifact missing/corrupt")
    elif complete and missing_rebuild:
        notes.append(
            "COMPLETE filing is servable; missing rebuild artifacts "
            f"{sorted(set(missing_rebuild))} will be regenerated only on --force "
            "or if a later re-run needs them"
        )

    if complete and servable:
        final_resume = repair_soft_stage
    else:
        final_resume = resume_stage

    return FilingAssessment(
        complete=complete,
        servable=servable,
        resume_stage=final_resume,
        invalid_stages=sorted(invalid),
        missing_rebuild_artifacts=sorted(set(missing_rebuild)),
        notes=notes,
    )


def _back_up_for_missing_inputs(
    resume_stage: str,
    stages: dict,
    artifact_paths: dict[str, Path],
    hashes: dict,
    notes: list[str],
) -> str:
    """If we plan to resume at ``resume_stage`` but the artifact feeding it is
    gone, move the resume point earlier so that artifact is regenerated."""
    order = list(STAGE_ORDER)
    idx = order.index(resume_stage)

    # chunked needs cleaned text; if it's gone, re-clean (which needs raw HTML;
    # if that's gone too, re-download).
    def ok(name):
        return stages.get(name, {}).get("status") in ("ok", "skipped")

    if resume_stage in ("chunked",):
        valid, _ = validate_clean_text(
            artifact_paths.get("clean_text", Path("/nonexistent")),
            expected_sha256=hashes.get("clean_text_sha256"),
        )
        if not valid:
            valid_html, _ = validate_raw_html(
                artifact_paths.get("raw_html", Path("/nonexistent"))
            )
            notes.append("re-chunk needs cleaned text; backing up to "
                         + ("cleaned" if valid_html else "downloaded"))
            return "cleaned" if valid_html else "downloaded"

    if resume_stage in ("cleaned",):
        valid_html, _ = validate_raw_html(
            artifact_paths.get("raw_html", Path("/nonexistent"))
        )
        if not valid_html:
            notes.append("re-clean needs raw HTML; backing up to downloaded")
            return "downloaded"

    if resume_stage in ("embedded", "dense_upserted"):
        valid, _ = validate_chunks_artifact(
            artifact_paths.get("chunks", Path("/nonexistent")),
            expected_sha256=hashes.get("chunks_sha256"),
        )
        if not valid:
            notes.append("re-embed needs chunk JSONL; backing up to chunked")
            return _back_up_for_missing_inputs("chunked", stages, artifact_paths, hashes, notes)

    return resume_stage
