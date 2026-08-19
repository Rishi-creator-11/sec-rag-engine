import json
from pathlib import Path

from api.rag import answer_question


QUESTIONS = [
    "What privacy risks does Apple face?",
    "What are Apple's major competitive risks?",
    "What supply chain risks does Apple describe?",
    "What does Apple say about Greater China?",
    "What cybersecurity risks does Apple face?",

    "What does Microsoft say about artificial intelligence?",
    "What AI risks does Microsoft identify?",
    "What cybersecurity risks does Microsoft describe?",
    "What does Microsoft say about cloud services?",
    "What competitive risks does Microsoft face?",

    "What risks does NVIDIA face from export controls?",
    "What risks does NVIDIA face in China?",
    "What supply chain risks does NVIDIA face?",
    "What competitive risks does NVIDIA describe?",
    "What regulatory risks does NVIDIA face?",

    "Which company discusses AI-related risks most directly?",
    "Compare Apple and Microsoft cybersecurity risks.",
    "Which company appears most exposed to China-related restrictions?",

    "What will NVIDIA's exact stock price be on December 31, 2027?",
    "What will Apple's revenue be in 2030?",
]


def run_tests() -> list[dict]:
    results = []

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\nRunning {index}/{len(QUESTIONS)}")
        print(question)

        try:
            response = answer_question(
                question=question,
                top_k=5,
            )

            results.append(response)

            print("Done.")

        except Exception as error:
            results.append({
                "question": question,
                "error": str(error),
            })

            print(f"ERROR: {error}")

    return results


def save_results(
    results: list[dict],
    output_path: str,
) -> None:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    results = run_tests()

    save_results(
        results,
        "evaluation/results/rag_test_results.json",
    )

    print(
        f"\nFinished {len(results)} tests."
    )

    print(
        "Saved results to "
        "evaluation/results/rag_test_results.json"
    )