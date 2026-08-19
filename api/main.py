from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.rag import answer_question


app = FastAPI(
    title="SEC RAG Engine",
    version="0.1.0",
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