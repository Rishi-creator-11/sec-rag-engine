import json
import statistics
import time
from pathlib import Path

from retrieval.bm25_search import get_index


BENCHMARK_PATH = Path(
    "evaluation/benchmark.json"
)
OUTPUT_PATH = Path(
    "evaluation/results/bm25_evaluation.json"
)


def load_benchmark(
    path: Path = BENCHMARK_PATH,
) -> list[dict]:
    return json.loads(
        path.read_text(
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
    relevant_retrieved = (
        set(retrieved_chunk_ids[:k])
        & relevant_chunks
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
    relevant_retrieved = (
        set(retrieved_chunk_ids[:k])
        & relevant_chunks
    )

    return len(relevant_retrieved) / k


def evaluate() -> None:
    benchmark = load_benchmark()
    index = get_index()

    recall_scores = {
        1: [],
        3: [],
        5: [],
    }
    precision_scores = {
        1: [],
        3: [],
        5: [],
    }

    reciprocal_rank_sum = 0.0
    latencies = []
    question_results = []

    print(
        f"Loaded {index.document_count} chunks"
    )
    print(
        f"Evaluating {len(benchmark)} "
        f"questions...\n"
    )

    for item in benchmark:
        relevant_chunks = set(
            item["relevant_chunks"]
        )

        start_time = time.perf_counter()

        results = index.search(
            item["question"],
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

        per_question_recall = {}
        per_question_precision = {}

        for k in (1, 3, 5):
            recall = recall_at_k(
                retrieved_chunk_ids,
                relevant_chunks,
                k,
            )
            precision = precision_at_k(
                retrieved_chunk_ids,
                relevant_chunks,
                k,
            )

            recall_scores[k].append(recall)
            precision_scores[k].append(
                precision
            )
            per_question_recall[str(k)] = (
                recall
            )
            per_question_precision[str(k)] = (
                precision
            )

        rank = first_relevant_rank(
            results,
            relevant_chunks,
        )

        if rank is not None:
            reciprocal_rank_sum += 1 / rank

        question_results.append({
            "id": item["id"],
            "question": item["question"],
            "relevant_chunks": sorted(
                relevant_chunks
            ),
            "retrieved_chunk_ids": (
                retrieved_chunk_ids
            ),
            "scores": [
                result["score"]
                for result in results
            ],
            "first_relevant_rank": rank,
            "recall_at_k": (
                per_question_recall
            ),
            "precision_at_k": (
                per_question_precision
            ),
            "latency_ms": latency * 1000,
        })

        print(
            f"{item['id']}: "
            f"first_relevant_rank={rank} "
            f"R@5={per_question_recall['5']:.2f} "
            f"P@5="
            f"{per_question_precision['5']:.2f} "
            f"latency={latency:.6f}s"
        )

    total = len(benchmark)
    sorted_latencies = sorted(latencies)
    p95_index = int(
        0.95 * (len(sorted_latencies) - 1)
    )

    summary = {
        "questions": total,
        "mean_recall_at_1": (
            sum(recall_scores[1]) / total
        ),
        "mean_recall_at_3": (
            sum(recall_scores[3]) / total
        ),
        "mean_recall_at_5": (
            sum(recall_scores[5]) / total
        ),
        "mean_precision_at_1": (
            sum(precision_scores[1]) / total
        ),
        "mean_precision_at_3": (
            sum(precision_scores[3]) / total
        ),
        "mean_precision_at_5": (
            sum(precision_scores[5]) / total
        ),
        "mrr": reciprocal_rank_sum / total,
        "average_latency_ms": (
            sum(latencies)
            / len(latencies)
            * 1000
        ),
        "median_latency_ms": (
            statistics.median(latencies)
            * 1000
        ),
        "p95_latency_ms": (
            sorted_latencies[p95_index]
            * 1000
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "retriever": "bm25",
                "parameters": {
                    "k1": index.k1,
                    "b": index.b,
                    "documents": (
                        index.document_count
                    ),
                },
                "summary": summary,
                "questions": question_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("BM25 RETRIEVAL BASELINE")
    print("=" * 50)
    print(f"Questions: {total}")
    print(
        "Mean Recall@1:    "
        f"{summary['mean_recall_at_1']:.3f}"
    )
    print(
        "Mean Recall@3:    "
        f"{summary['mean_recall_at_3']:.3f}"
    )
    print(
        "Mean Recall@5:    "
        f"{summary['mean_recall_at_5']:.3f}"
    )
    print(
        "Mean Precision@1: "
        f"{summary['mean_precision_at_1']:.3f}"
    )
    print(
        "Mean Precision@3: "
        f"{summary['mean_precision_at_3']:.3f}"
    )
    print(
        "Mean Precision@5: "
        f"{summary['mean_precision_at_5']:.3f}"
    )
    print(f"MRR:              {summary['mrr']:.3f}")
    print(
        "Average latency:  "
        f"{summary['average_latency_ms']:.3f} ms"
    )
    print(
        "Median latency:   "
        f"{summary['median_latency_ms']:.3f} ms"
    )
    print(
        "P95 latency:      "
        f"{summary['p95_latency_ms']:.3f} ms"
    )
    print(
        f"Saved results to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate()
