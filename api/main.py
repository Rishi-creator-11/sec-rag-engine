import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.rag import answer_question


load_dotenv()


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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok"
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return answer_question(
        question=request.question,
        top_k=request.top_k,
    )