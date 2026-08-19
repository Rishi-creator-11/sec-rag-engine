import argparse
import os

from dotenv import load_dotenv
from pinecone import Pinecone

from retrieval.bm25_search import load_chunks


INDEX_NAME = "sec-rag-sparse"
NAMESPACE = "__default__"


def get_client() -> Pinecone:
    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing. Add it to your .env file."
        )

    return Pinecone(
        api_key=api_key
    )


def create_sparse_index() -> None:
    pc = get_client()

    if pc.has_index(INDEX_NAME):
        print(
            f"Index '{INDEX_NAME}' already exists."
        )
        return

    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "pinecone-sparse-english-v0",
            "field_map": {
                "text": "chunk_text",
            },
            "read_parameters": {
                "max_tokens_per_sequence": 2048,
            },
            "write_parameters": {
                "max_tokens_per_sequence": 2048,
            },
        },
    )

    print(
        f"Created sparse index '{INDEX_NAME}'."
    )


def chunk_to_record(
    chunk: dict,
) -> dict:
    return {
        "_id": chunk["chunk_id"],
        "chunk_text": chunk["text"],
        "chunk_id": chunk["chunk_id"],
        "company": chunk["company"],
        "ticker": chunk["ticker"],
        "filing_type": chunk["filing_type"],
        "filing_date": chunk["filing_date"],
        "source_url": chunk["source_url"],
    }


def inspect_one_record() -> None:
    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks were loaded."
        )

    first_chunk = chunks[0]

    record = chunk_to_record(
        first_chunk
    )

    print("=" * 80)
    print("ORIGINAL CHUNK")
    print("=" * 80)

    print(
        f"chunk_id: "
        f"{first_chunk['chunk_id']}"
    )

    print(
        f"company: "
        f"{first_chunk['company']}"
    )

    print(
        f"ticker: "
        f"{first_chunk['ticker']}"
    )

    print(
        f"filing_type: "
        f"{first_chunk['filing_type']}"
    )

    print(
        f"filing_date: "
        f"{first_chunk['filing_date']}"
    )

    print(
        f"source_url: "
        f"{first_chunk['source_url']}"
    )

    print()

    print(
        first_chunk["text"][:1000]
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PINECONE SPARSE RECORD"
    )

    print("=" * 80)

    print(
        f"_id: {record['_id']}"
    )

    print(
        f"chunk_id: "
        f"{record['chunk_id']}"
    )

    print(
        f"company: "
        f"{record['company']}"
    )

    print(
        f"ticker: "
        f"{record['ticker']}"
    )

    print(
        f"filing_type: "
        f"{record['filing_type']}"
    )

    print(
        f"filing_date: "
        f"{record['filing_date']}"
    )

    print(
        f"source_url: "
        f"{record['source_url']}"
    )

    print()

    print(
        "chunk_text preview:"
    )

    print(
        record["chunk_text"][:1000]
    )


def upsert_one_record() -> None:
    pc = get_client()

    if not pc.has_index(INDEX_NAME):
        raise RuntimeError(
            f"Index '{INDEX_NAME}' does not exist. "
            "Run the create command first."
        )

    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks were loaded."
        )

    record = chunk_to_record(
        chunks[0]
    )

    index = pc.Index(
        INDEX_NAME
    )

    index.upsert_records(
        namespace=NAMESPACE,
        records=[
            record
        ],
    )

    print(
        f"Upserted one record into "
        f"'{INDEX_NAME}': "
        f"{record['_id']}"
    )


def upsert_all_records(
    batch_size: int = 50,
) -> None:
    pc = get_client()

    if not pc.has_index(INDEX_NAME):
        raise RuntimeError(
            f"Index '{INDEX_NAME}' does not exist. "
            "Run the create command first."
        )

    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks were loaded."
        )

    records = [
        chunk_to_record(chunk)
        for chunk in chunks
    ]

    index = pc.Index(
        INDEX_NAME
    )

    total = len(records)

    print(
        f"Uploading {total} records "
        f"in batches of {batch_size}..."
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        batch = records[
            start:end
        ]

        index.upsert_records(
            namespace=NAMESPACE,
            records=batch,
        )

        print(
            f"Uploaded {end}/{total}"
        )

    print(
        f"Finished uploading "
        f"{total} records."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create and manage "
            "the Pinecone sparse index."
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "create",
            "inspect",
            "upsert-one",
            "upsert-all",
        ],
        help="Action to run.",
    )

    args = parser.parse_args()

    if args.command == "create":
        create_sparse_index()

    elif args.command == "inspect":
        inspect_one_record()

    elif args.command == "upsert-one":
        upsert_one_record()

    elif args.command == "upsert-all":
        upsert_all_records()


if __name__ == "__main__":
    main()