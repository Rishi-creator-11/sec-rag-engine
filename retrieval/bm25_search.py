import json
import math
import re
from collections import Counter
from pathlib import Path


CHUNKS_DIR = Path("data/chunks")
K1 = 1.5
B = 0.75

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "or",
    "say",
    "says",
    "the",
    "to",
    "what",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(
            text.lower()
        )
        if token not in STOP_WORDS
    ]


def load_chunks(
    input_dir: str | Path = CHUNKS_DIR,
) -> list[dict]:
    directory = Path(input_dir)
    paths = sorted(
        directory.glob("*_chunks.jsonl")
    )

    if not paths:
        raise FileNotFoundError(
            f"No *_chunks.jsonl files found in "
            f"{directory}"
        )

    chunks = []

    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                if not line.strip():
                    continue

                chunk = json.loads(line)

                if "chunk_id" not in chunk:
                    raise ValueError(
                        f"Missing chunk_id in "
                        f"{path}:{line_number}"
                    )

                if "text" not in chunk:
                    raise ValueError(
                        f"Missing text in "
                        f"{path}:{line_number}"
                    )

                chunks.append(chunk)

    return chunks


class BM25Index:
    def __init__(
        self,
        chunks: list[dict],
        k1: float = K1,
        b: float = B,
    ) -> None:
        if not chunks:
            raise ValueError(
                "BM25 requires at least one chunk"
            )

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.term_frequencies = []
        self.document_lengths = []
        document_frequencies: Counter[str] = (
            Counter()
        )

        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            frequencies = Counter(tokens)

            self.term_frequencies.append(
                frequencies
            )
            self.document_lengths.append(
                len(tokens)
            )
            document_frequencies.update(
                frequencies.keys()
            )

        self.document_count = len(chunks)
        self.average_document_length = (
            sum(self.document_lengths)
            / self.document_count
        )

        self.inverse_document_frequencies = {
            term: math.log(
                1
                + (
                    self.document_count
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )
            for term, frequency
            in document_frequencies.items()
        }

    def score(
        self,
        query: str,
    ) -> list[float]:
        query_frequencies = Counter(
            tokenize(query)
        )

        scores = []

        for frequencies, document_length in zip(
            self.term_frequencies,
            self.document_lengths,
        ):
            score = 0.0

            length_normalization = (
                1
                - self.b
                + self.b
                * document_length
                / self.average_document_length
            )

            for term, query_frequency in (
                query_frequencies.items()
            ):
                term_frequency = frequencies.get(
                    term,
                    0,
                )

                if term_frequency == 0:
                    continue

                inverse_document_frequency = (
                    self.inverse_document_frequencies[
                        term
                    ]
                )

                numerator = (
                    term_frequency
                    * (self.k1 + 1)
                )

                denominator = (
                    term_frequency
                    + self.k1
                    * length_normalization
                )

                score += (
                    query_frequency
                    * inverse_document_frequency
                    * numerator
                    / denominator
                )

            scores.append(score)

        return scores

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: str | None = None,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        scores = self.score(query)

        candidates = [
            (score, chunk)
            for score, chunk in zip(
                scores,
                self.chunks,
            )
            if (
                ticker is None
                or chunk["ticker"].upper()
                == ticker.upper()
            )
        ]

        if not candidates:
            raise ValueError(
                f"No chunks found for ticker "
                f"{ticker}"
            )

        ranked = sorted(
            candidates,
            key=lambda item: (
                -item[0],
                item[1]["chunk_id"],
            ),
        )

        results = []

        for score, chunk in ranked[:top_k]:
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "text": chunk["text"],
                "company": chunk["company"],
                "ticker": chunk["ticker"],
                "filing_type": chunk[
                    "filing_type"
                ],
                "filing_date": chunk[
                    "filing_date"
                ],
                "source_url": chunk[
                    "source_url"
                ],
            })

        return results


_index: BM25Index | None = None


def get_index() -> BM25Index:
    global _index

    if _index is None:
        _index = BM25Index(
            load_chunks()
        )

    return _index


def search(
    query: str,
    top_k: int = 5,
    ticker: str | None = None,
) -> list[dict]:
    return get_index().search(
        query,
        top_k=top_k,
        ticker=ticker,
    )


if __name__ == "__main__":
    index = get_index()

    print(
        f"Loaded {index.document_count} chunks"
    )

    query = input(
        "Enter your SEC question: "
    )

    results = index.search(
        query,
        top_k=5,
    )

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Company: {result['company']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Source: {result['source_url']}")
        print()
        print(result["text"][:1200])
        print()
