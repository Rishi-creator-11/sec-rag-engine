import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from api.rag import answer_question
from ingestion.registry import list_companies, partition_tickers


load_dotenv()

MAX_TICKERS = 10


def parse_frontend_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000")
    origins = [origin.strip() for origin in raw.split(",")]
    return [origin for origin in origins if origin]


app = FastAPI(
    title="SEC RAG Engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_frontend_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )
    # Optional retrieval scope. Absent / null / [] => global search (legacy).
    # One ticker  => strict company-specific retrieval.
    # Several     => joint "ticker IN (...)" filter. Phase 1C does NOT balance
    #                evidence across companies; Phase 2 adds per-company quota
    #                retrieval (scoped_search) for guaranteed comparison coverage.
    tickers: list[str] | None = Field(
        default=None,
        max_length=MAX_TICKERS,
    )

    @field_validator("tickers", mode="after")
    @classmethod
    def _normalize_tickers(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in value:
            ticker = str(raw).strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            normalized.append(ticker)
        return normalized or None


@app.get("/health")
def health():
    """Liveness + readiness.

    Additive diagnostics: `lexical_backend` ("bm25s" | "current"), `bm25_documents`
    (null until the index is built on the first /ask), `companies`.

    Readiness: the lexical retriever must be able to load OR deterministically
    rebuild its index. If neither works we return 503 rather than silently
    serving partial lexical search.
    """
    import os

    from fastapi.responses import JSONResponse

    from retrieval.bm25_search import document_count, get_index
    from retrieval.lexical_backend import DEFAULT_BACKEND, LexicalBackendError

    backend = os.getenv("SEC_RAG_LEXICAL_BACKEND", DEFAULT_BACKEND).strip().lower()
    body = {
        "status": "ok",
        "lexical_backend": backend,
        "companies": len(list_companies()),
        "bm25_documents": document_count(),
    }
    try:
        get_index()  # builds/loads once; cheap thereafter
        body["bm25_documents"] = document_count()
    except LexicalBackendError as exc:
        return JSONResponse(
            status_code=503,
            content={**body, "status": "unavailable",
                     "detail": f"lexical index unavailable: {exc}"},
        )
    return body


@app.get("/companies")
def companies() -> dict:
    """Registered companies, sorted by ticker. Powers the frontend selector."""
    return {
        "companies": [
            {"ticker": company["ticker"], "name": company["name"]}
            for company in list_companies()
        ]
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    tickers = request.tickers

    if tickers:
        known, unknown = partition_tickers(tickers)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_tickers",
                    "unknown_tickers": unknown,
                },
            )
        tickers = known

    return answer_question(
        question=request.question,
        top_k=request.top_k,
        tickers=tickers,
    )