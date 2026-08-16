import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":
    text_a = "Apple faces competition in global smartphone markets."

    text_b = (
        "Apple competes with other companies that sell smartphones."
    )

    text_c = "Microsoft paid a quarterly dividend."

    embedding_a = embed_text(text_a)
    embedding_b = embed_text(text_b)
    embedding_c = embed_text(text_c)

    similarity_ab = cosine_similarity(
        embedding_a,
        embedding_b,
    )

    similarity_ac = cosine_similarity(
        embedding_a,
        embedding_c,
    )

    print(f"A vs B: {similarity_ab:.4f}")
    print(f"A vs C: {similarity_ac:.4f}")