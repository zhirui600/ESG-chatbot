"""
Step 1: Extract text from ESG PDF reports page by page.
Output: data/raw_pages.csv with columns [company, year, page_num, raw_text, word_count, is_low_text]
"""

import pdfplumber
import pandas as pd
import re
import os
from pathlib import Path

REPORT_DIR = Path(__file__).parent.parent / "report"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw_pages.csv"
LOW_TEXT_THRESHOLD = 50  # pages with fewer words are flagged as low-text (charts/covers)


def parse_filename(filename: str) -> tuple[str, str]:
    """Extract company name and year from filename."""
    # e.g. "Apple Inc - 2024 - Environmental Progress Report ..."
    parts = filename.replace(".pdf", "").split(" - ")
    company = parts[0].strip() if len(parts) > 0 else "Unknown"
    year = parts[1].strip() if len(parts) > 1 else "Unknown"
    return company, year


def clean_raw_text(text: str) -> str:
    """Basic cleanup of extracted PDF text."""
    if not text:
        return ""
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf(filepath: str) -> list[dict]:
    """Extract text from each page of a PDF file."""
    company, year = parse_filename(os.path.basename(filepath))
    pages = []

    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            cleaned = clean_raw_text(text)
            word_count = len(cleaned.split()) if cleaned else 0

            pages.append({
                "company": company,
                "year": year,
                "page_num": i,
                "total_pages": len(pdf.pages),
                "raw_text": cleaned,
                "word_count": word_count,
                "is_low_text": word_count < LOW_TEXT_THRESHOLD,
            })

    return pages


def main():
    pdf_files = sorted(REPORT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {REPORT_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF reports")

    all_pages = []
    for pdf_path in pdf_files:
        print(f"  Extracting: {pdf_path.name}")
        pages = extract_pdf(str(pdf_path))
        all_pages.extend(pages)

    df = pd.DataFrame(all_pages)

    # Summary stats
    total = len(df)
    low_text = df["is_low_text"].sum()
    print(f"\nTotal pages extracted: {total}")
    print(f"Low-text pages (charts/covers, <{LOW_TEXT_THRESHOLD} words): {low_text}")
    print(f"Content pages for analysis: {total - low_text}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_PATH}")

    # Print per-report summary
    print("\n--- Per Report Summary ---")
    for _, row in df.groupby(["company", "year"]).agg(
        total_pages=("page_num", "count"),
        content_pages=("is_low_text", lambda x: (~x).sum()),
        low_text_pages=("is_low_text", "sum"),
    ).reset_index().iterrows():
        print(f"  {row['company']} ({row['year']}): "
              f"{row['total_pages']} total, "
              f"{row['content_pages']} content, "
              f"{row['low_text_pages']} low-text")


if __name__ == "__main__":
    main()
