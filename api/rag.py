from openai import OpenAI

from retrieval.pinecone_search import search


client = OpenAI()

ANSWER_MODEL = "gpt-5-mini"


def build_context(results: list[dict]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"""
SOURCE {index}
Company: {result['company']}
Filing: {result['filing_type']}
Filing Date: {result['filing_date']}
Chunk ID: {result['chunk_id']}
Source URL: {result['source_url']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


def answer_question(
    question: str,
    top_k: int = 5,
) -> dict:
    results = search(
        question,
        top_k=top_k,
    )

    context = build_context(results)

    prompt = f"""
You are answering questions using SEC filing excerpts.

Rules:
1. Answer only using the supplied SEC filing context.
2. Do not use outside knowledge.
3. If the context is insufficient, say:
   "The provided SEC filing excerpts do not contain enough information to answer this question."
4. Cite claims using [Source 1], [Source 2], etc.
5. Be concise and factual.

QUESTION:
{question}

SEC FILING CONTEXT:
{context}
"""

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=prompt,
    )

    answer = response.output_text

    sources = [
        {
            "chunk_id": result["chunk_id"],
            "company": result["company"],
            "filing_type": result["filing_type"],
            "filing_date": result["filing_date"],
            "source_url": result["source_url"],
            "retrieval_score": result["score"],
        }
        for result in results
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    question = input("Ask an SEC question: ")

    result = answer_question(question)

    print("\nANSWER\n")
    print(result["answer"])

    print("\nSOURCES\n")

    for source in result["sources"]:
        print(
            f"{source['chunk_id']} | "
            f"{source['company']} | "
            f"{source['retrieval_score']:.4f}"
        )