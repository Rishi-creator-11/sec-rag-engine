import json
from pathlib import Path

import tiktoken


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):
        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunk = encoding.decode(chunk_tokens)

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunk_records(
    text: str,
    metadata: dict,
) -> list[dict]:

    chunks = chunk_text(text)

    records = []

    for index, chunk in enumerate(chunks):

        record = {
            "chunk_id": f"{metadata['filename']}_{index}",
            "chunk_index": index,
            "text": chunk,
            "company": metadata["company"],
            "ticker": metadata["ticker"],
            "filing_type": metadata["filing_type"],
            "filing_date": metadata["filing_date"],
            "source_url": metadata["source_url"],
        }

        records.append(record)

    return records


def save_chunks(
    records: list[dict],
    output_path: str,
) -> None:

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(record) + "\n"
            )


if __name__ == "__main__":

    raw_dir = Path("data/raw")
    output_dir = Path("data/chunks")

    metadata_files = raw_dir.glob("*.json")

    for metadata_path in metadata_files:

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        text_path = (
            raw_dir
            / f"{metadata['filename']}.txt"
        )

        text = text_path.read_text(
            encoding="utf-8"
        )

        records = create_chunk_records(
            text,
            metadata,
        )

        output_path = (
            output_dir
            / f"{metadata['filename']}_chunks.jsonl"
        )

        save_chunks(
            records,
            output_path,
        )

        print(
            f"{metadata['company']}: "
            f"{len(records)} chunks"
        )