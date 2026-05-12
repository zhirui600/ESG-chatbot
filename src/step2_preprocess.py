"""
Step 2: Text preprocessing - clean text, tokenize, remove stopwords, lemmatize.
Input:  data/raw_pages.csv
Output: data/cleaned_pages.csv
"""

import pandas as pd
import re
import spacy
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent / "data" / "raw_pages.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "cleaned_pages.csv"

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """Remove noise from extracted PDF text."""
    if not text:
        return ""
    # Remove cid patterns from embedded fonts (e.g. cid:38, cid 83)
    text = re.sub(r"cid[:\s]*\d+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove repeated special characters (keep meaningful ones like %)
    text = re.sub(r"[^a-zA-Z0-9\s\-.,;:!?%$'\"()]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def spacy_process(text: str) -> tuple[str, list[str]]:
    """Tokenize, remove stopwords, lemmatize using spaCy."""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.lemma_) > 1
    ]
    return " ".join(tokens), tokens


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} pages from {INPUT_PATH}")

    content_mask = ~df["is_low_text"]
    print(f"Processing {content_mask.sum()} content pages "
          f"(skipping {(~content_mask).sum()} low-text pages)")

    cleaned_texts = []
    token_lists = []

    for idx, row in df.iterrows():
        if row["is_low_text"] or pd.isna(row["raw_text"]) or row["raw_text"].strip() == "":
            cleaned_texts.append("")
            token_lists.append([])
            continue

        cleaned = clean_text(row["raw_text"])
        processed_text, tokens = spacy_process(cleaned)
        cleaned_texts.append(processed_text)
        token_lists.append(tokens)

    df["cleaned_text"] = cleaned_texts
    df["tokens"] = [" ".join(t) for t in token_lists]
    df["token_count"] = [len(t) for t in token_lists]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Token counts: mean={df.loc[content_mask, 'token_count'].mean():.0f}, "
          f"median={df.loc[content_mask, 'token_count'].median():.0f}")


if __name__ == "__main__":
    main()
