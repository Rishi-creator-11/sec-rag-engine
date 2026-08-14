import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "sec-rag-engine hrishitkumar628@gmail.com"
}


def download_filing(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return response.text


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def save_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        text,
        encoding="utf-8",
    )


def save_metadata(metadata: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    filings = [
        {
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-09-28",
            "filename": "apple_10k",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/"
                "320193/000032019324000123/aapl-20240928.htm"
            ),
        },
        {
            "company": "Microsoft Corporation",
            "ticker": "MSFT",
            "filing_type": "10-K",
            "filing_date": "2025-06-30",
            "filename": "microsoft_10k",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/"
                "789019/000095017025100235/msft-20250630.htm"
            ),
        },
        {
            "company": "NVIDIA Corporation",
            "ticker": "NVDA",
            "filing_type": "10-K",
            "filing_date": "2026-01-25",
            "filename": "nvidia_10k",
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/"
                "1045810/000104581026000021/nvda-20260125.htm"
            ),
        },
    ]

    for filing in filings:
        print(f"Downloading {filing['company']}...")

        html = download_filing(filing["source_url"])
        clean_text = clean_html(html)

        text_path = f"data/raw/{filing['filename']}.txt"
        metadata_path = f"data/raw/{filing['filename']}.json"

        save_text(clean_text, text_path)
        save_metadata(filing, metadata_path)

        print(
            f"Saved {filing['company']} "
            f"({len(clean_text):,} characters)"
        )