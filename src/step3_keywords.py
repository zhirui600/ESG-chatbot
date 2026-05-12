"""
Step 3: Keyword extraction using TF-IDF and KeyBERT.
Input:  data/cleaned_pages.csv
Output: data/keywords_tfidf.csv, data/keywords_keybert.csv
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent / "data" / "cleaned_pages.csv"
KEYWORDS_CONFIG = Path(__file__).parent.parent / "config" / "esg_keywords.json"
OUTPUT_TFIDF = Path(__file__).parent.parent / "data" / "keywords_tfidf.csv"
OUTPUT_KEYBERT = Path(__file__).parent.parent / "data" / "keywords_keybert.csv"

TOP_N_TFIDF = 15
TOP_N_KEYBERT = 10

# Custom stopwords: company names and brand names that appear in reports
CUSTOM_STOPWORDS = [
    # Company names and abbreviations from our 10 reports
    "ge", "general electric", "caterpillar", "honeywell", "broadcom",
    "amazon", "amazon com", "apple", "adobe", "nvidia", "deere",
    "john deere", "ups", "united parcel service",
    # Common non-ESG noise words
    "company", "report", "year", "page", "use", "using", "used",
    "include", "including", "also", "new", "make", "made",
    " sustainabilityreports com",
]


def get_all_stopwords() -> list[str]:
    """Merge spaCy English stopwords with custom stopwords."""
    from spacy.lang.en.stop_words import STOP_WORDS
    all_stops = set(STOP_WORDS)
    for w in CUSTOM_STOPWORDS:
        all_stops.add(w.lower())
    return list(all_stops)


def extract_tfidf_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Extract top-N TF-IDF keywords for each page."""
    content = df[df["token_count"] > 10].copy()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85,
        stop_words=get_all_stopwords(),
    )
    tfidf_matrix = vectorizer.fit_transform(content["cleaned_text"])
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for idx, (_, row) in enumerate(content.iterrows()):
        row_vector = tfidf_matrix[idx].toarray().flatten()
        top_indices = row_vector.argsort()[-TOP_N_TFIDF:][::-1]
        keywords = [(feature_names[i], round(float(row_vector[i]), 4)) for i in top_indices if row_vector[i] > 0]

        results.append({
            "company": row["company"],
            "year": row["year"],
            "page_num": row["page_num"],
            "tfidf_keywords": "; ".join([kw for kw, _ in keywords]),
            "tfidf_scores": "; ".join([str(sc) for _, sc in keywords]),
        })

    return pd.DataFrame(results)


def extract_keybert_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Extract top-N KeyBERT keywords for each page."""
    kw_model = KeyBERT()
    content = df[df["token_count"] > 10].copy()

    results = []
    for _, row in content.iterrows():
        text = row["cleaned_text"]
        if len(text.split()) < 15:
            continue

        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=get_all_stopwords(),
            top_n=TOP_N_KEYBERT,
        )

        results.append({
            "company": row["company"],
            "year": row["year"],
            "page_num": row["page_num"],
            "keybert_keywords": "; ".join([kw for kw, _ in keywords]),
            "keybert_scores": "; ".join([str(round(sc, 4)) for _, sc in keywords]),
        })

    return pd.DataFrame(results)


def extract_esg_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Identify ESG performance measurement metrics from keyword matches."""
    with open(KEYWORDS_CONFIG, "r", encoding="utf-8") as f:
        esg_config = json.load(f)

    # Flatten all keywords into a single set for metric detection
    all_esg_terms = set()
    for category in esg_config.values():
        for subcategory, terms in category.items():
            for term in terms:
                all_esg_terms.add(term.lower())

    content = df[df["token_count"] > 10].copy()

    results = []
    for _, row in content.iterrows():
        text_lower = row["cleaned_text"].lower()
        matched = sorted({term for term in all_esg_terms if term in text_lower})

        results.append({
            "company": row["company"],
            "year": row["year"],
            "page_num": row["page_num"],
            "esg_terms_matched": "; ".join(matched),
            "esg_terms_count": len(matched),
        })

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} pages from {INPUT_PATH}")

    # TF-IDF keywords
    print("\nExtracting TF-IDF keywords...")
    tfidf_df = extract_tfidf_keywords(df)
    tfidf_df.to_csv(OUTPUT_TFIDF, index=False, encoding="utf-8-sig")
    print(f"TF-IDF keywords saved to {OUTPUT_TFIDF} ({len(tfidf_df)} pages)")

    # KeyBERT keywords
    print("\nExtracting KeyBERT keywords (this may take a while)...")
    keybert_df = extract_keybert_keywords(df)
    keybert_df.to_csv(OUTPUT_KEYBERT, index=False, encoding="utf-8-sig")
    print(f"KeyBERT keywords saved to {OUTPUT_KEYBERT} ({len(keybert_df)} pages)")

    # ESG performance metrics summary
    print("\nIdentifying ESG performance metrics...")
    metrics_df = extract_esg_performance_metrics(df)
    metrics_path = Path(__file__).parent.parent / "data" / "esg_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"ESG metrics saved to {metrics_path}")

    # Print top keywords across all reports
    print("\n--- Top 20 TF-IDF Keywords (All Reports) ---")
    from collections import Counter
    all_kw = Counter()
    for kws in tfidf_df["tfidf_keywords"]:
        for kw in str(kws).split("; "):
            all_kw[kw.strip()] += 1
    for kw, count in all_kw.most_common(20):
        print(f"  {kw}: {count}")


if __name__ == "__main__":
    main()
