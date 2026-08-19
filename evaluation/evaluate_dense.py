import json
import statistics
import time
from pathlib import Path

from retrieval.pinecone_search import search


BENCHMARK_PATH = "evaluation/benchmark.json"


def load_benchmark(
    path: str = BENCHMARK_PATH,
) -> list[dict]:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def first_relevant_rank(
    results: list[dict],
    relevant_chunks: set[str],
) -> int | None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["chunk_id"] in relevant_chunks:
            return rank

    return None


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunks: set[str],
    k: int,
) -> float:
    top_k = set(
        retrieved_chunk_ids[:k]
    )

    relevant_retrieved = (
        top_k & relevant_chunks
    )

    return (
        len(relevant_retrieved)
        / len(relevant_chunks)
    )


def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunks: set[str],
    k: int,
) -> float:
    top_k = set(
        retrieved_chunk_ids[:k]
    )

    relevant_retrieved = (
        top_k & relevant_chunks
    )

    return (
        len(relevant_retrieved)
        / k
    )


def evaluate() -> None:
    benchmark = load_benchmark()

    recall_1_scores = []
    recall_3_scores = []
    recall_5_scores = []

    precision_1_scores = []
    precision_3_scores = []
    precision_5_scores = []

    reciprocal_rank_sum = 0.0
    latencies = []

    print(
        f"Evaluating {len(benchmark)} questions...\n"
    )

    for item in benchmark:
        question = item["question"]

        relevant_chunks = set(
            item["relevant_chunks"]
        )

        start_time = time.perf_counter()

        results = search(
            question,
            top_k=5,
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        latencies.append(latency)

        retrieved_chunk_ids = [
            result["chunk_id"]
            for result in results
        ]

        recall_1 = recall_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            1,
        )

        recall_3 = recall_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            3,
        )

        recall_5 = recall_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            5,
        )

        precision_1 = precision_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            1,
        )

        precision_3 = precision_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            3,
        )

        precision_5 = precision_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            5,
        )

        recall_1_scores.append(recall_1)
        recall_3_scores.append(recall_3)
        recall_5_scores.append(recall_5)

        precision_1_scores.append(
            precision_1
        )

        precision_3_scores.append(
            precision_3
        )

        precision_5_scores.append(
            precision_5
        )

        rank = first_relevant_rank(
            results,
            relevant_chunks,
        )

        if rank is not None:
            reciprocal_rank_sum += (
                1 / rank
            )

        print(
            f"{item['id']}: "
            f"first_relevant_rank={rank} "
            f"R@5={recall_5:.2f} "
            f"P@5={precision_5:.2f} "
            f"latency={latency:.3f}s"
        )

    total = len(benchmark)

    mean_recall_1 = (
        sum(recall_1_scores) / total
    )

    mean_recall_3 = (
        sum(recall_3_scores) / total
    )

    mean_recall_5 = (
        sum(recall_5_scores) / total
    )

    mean_precision_1 = (
        sum(precision_1_scores) / total
    )

    mean_precision_3 = (
        sum(precision_3_scores) / total
    )

    mean_precision_5 = (
        sum(precision_5_scores) / total
    )

    mrr = (
        reciprocal_rank_sum / total
    )

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    median_latency = statistics.median(
        latencies
    )

    sorted_latencies = sorted(latencies)

    p95_index = int(
        0.95 * (len(sorted_latencies) - 1)
    )

    p95_latency = sorted_latencies[
        p95_index
    ]

    print("\n" + "=" * 50)
    print("DENSE RETRIEVAL BASELINE")
    print("=" * 50)

    print(
        f"Questions: {total}"
    )

    print(
        f"Mean Recall@1:    {mean_recall_1:.3f}"
    )

    print(
        f"Mean Recall@3:    {mean_recall_3:.3f}"
    )

    print(
        f"Mean Recall@5:    {mean_recall_5:.3f}"
    )

    print(
        f"Mean Precision@1: {mean_precision_1:.3f}"
    )

    print(
        f"Mean Precision@3: {mean_precision_3:.3f}"
    )

    print(
        f"Mean Precision@5: {mean_precision_5:.3f}"
    )

    print(
        f"MRR:              {mrr:.3f}"
    )

    print(
        f"Average latency:  "
        f"{average_latency * 1000:.1f} ms"
    )

    print(
        f"Median latency:   "
        f"{median_latency * 1000:.1f} ms"
    )

    print(
        f"P95 latency:      "
        f"{p95_latency * 1000:.1f} ms"
    )


if __name__ == "__main__":
    evaluate()