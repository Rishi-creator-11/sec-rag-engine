import json
import statistics
import time
from pathlib import Path

from retrieval.sparse_search import search


BENCHMARK_PATH = Path(
    "evaluation/benchmark.json"
)

RESULTS_PATH = Path(
    "evaluation/results/sparse_evaluation.json"
)

TOP_K = 5


def load_benchmark() -> list[dict]:
    with BENCHMARK_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    retrieved_at_k = set(
        retrieved_ids[:k]
    )

    relevant_retrieved = (
        retrieved_at_k
        & relevant_ids
    )

    return (
        len(relevant_retrieved)
        / len(relevant_ids)
    )


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    retrieved_at_k = (
        retrieved_ids[:k]
    )

    relevant_count = sum(
        1
        for chunk_id
        in retrieved_at_k
        if chunk_id in relevant_ids
    )

    return relevant_count / k


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def percentile(
    values: list[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    index = int(
        round(
            (len(values) - 1)
            * p
        )
    )

    return values[index]


def evaluate() -> None:
    benchmark = load_benchmark()

    recall_1 = []
    recall_3 = []
    recall_5 = []

    precision_1 = []
    precision_3 = []
    precision_5 = []

    reciprocal_ranks = []
    latencies_ms = []

    per_question = []

    print(
        f"Evaluating "
        f"{len(benchmark)} questions..."
    )

    print()

    for item in benchmark:
        question_id = item["id"]
        question = item["question"]

        relevant_ids = set(
            item["relevant_chunks"]
        )

        start = time.perf_counter()

        results = search(
            question,
            top_k=TOP_K,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        r1 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            1,
        )

        r3 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            3,
        )

        r5 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            5,
        )

        p1 = precision_at_k(
            retrieved_ids,
            relevant_ids,
            1,
        )

        p3 = precision_at_k(
            retrieved_ids,
            relevant_ids,
            3,
        )

        p5 = precision_at_k(
            retrieved_ids,
            relevant_ids,
            5,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)

        precision_1.append(p1)
        precision_3.append(p3)
        precision_5.append(p5)

        reciprocal_ranks.append(rr)

        latencies_ms.append(
            elapsed_ms
        )

        first_relevant_rank = None

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if chunk_id in relevant_ids:
                first_relevant_rank = rank
                break

        per_question.append({
            "id": question_id,
            "question": question,
            "company": item.get(
                "company"
            ),
            "category": item.get(
                "category"
            ),
            "relevant_chunks": sorted(
                relevant_ids
            ),
            "retrieved_chunks": (
                retrieved_ids
            ),
            "recall_at_1": r1,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "precision_at_1": p1,
            "precision_at_3": p3,
            "precision_at_5": p5,
            "reciprocal_rank": rr,
            "first_relevant_rank": (
                first_relevant_rank
            ),
            "latency_ms": elapsed_ms,
        })

        print(
            f"{question_id}: "
            f"first_relevant_rank="
            f"{first_relevant_rank} "
            f"R@5={r5:.2f} "
            f"P@5={p5:.2f} "
            f"latency={elapsed_ms:.1f} ms"
        )

    metrics = {
        "questions": len(
            benchmark
        ),

        "mean_recall_at_1": (
            statistics.mean(
                recall_1
            )
        ),

        "mean_recall_at_3": (
            statistics.mean(
                recall_3
            )
        ),

        "mean_recall_at_5": (
            statistics.mean(
                recall_5
            )
        ),

        "mean_precision_at_1": (
            statistics.mean(
                precision_1
            )
        ),

        "mean_precision_at_3": (
            statistics.mean(
                precision_3
            )
        ),

        "mean_precision_at_5": (
            statistics.mean(
                precision_5
            )
        ),

        "mrr": statistics.mean(
            reciprocal_ranks
        ),

        "average_latency_ms": (
            statistics.mean(
                latencies_ms
            )
        ),

        "median_latency_ms": (
            statistics.median(
                latencies_ms
            )
        ),

        "p95_latency_ms": (
            percentile(
                latencies_ms,
                0.95,
            )
        ),
    }

    print()

    print("=" * 50)

    print(
        "SPARSE RETRIEVAL BASELINE"
    )

    print("=" * 50)

    print(
        f"Questions: "
        f"{metrics['questions']}"
    )

    print(
        f"Mean Recall@1:    "
        f"{metrics['mean_recall_at_1']:.3f}"
    )

    print(
        f"Mean Recall@3:    "
        f"{metrics['mean_recall_at_3']:.3f}"
    )

    print(
        f"Mean Recall@5:    "
        f"{metrics['mean_recall_at_5']:.3f}"
    )

    print(
        f"Mean Precision@1: "
        f"{metrics['mean_precision_at_1']:.3f}"
    )

    print(
        f"Mean Precision@3: "
        f"{metrics['mean_precision_at_3']:.3f}"
    )

    print(
        f"Mean Precision@5: "
        f"{metrics['mean_precision_at_5']:.3f}"
    )

    print(
        f"MRR:              "
        f"{metrics['mrr']:.3f}"
    )

    print(
        f"Average latency:  "
        f"{metrics['average_latency_ms']:.1f} ms"
    )

    print(
        f"Median latency:   "
        f"{metrics['median_latency_ms']:.1f} ms"
    )

    print(
        f"P95 latency:      "
        f"{metrics['p95_latency_ms']:.1f} ms"
    )

    output = {
        "metrics": metrics,
        "per_question": (
            per_question
        ),
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()

    print(
        f"Saved results to "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    evaluate()