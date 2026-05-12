"""
Step 4: ESG topic classification using keyword rule matching.
Input:  data/cleaned_pages.csv, config/esg_keywords.json
Output: data/page_classification.csv, data/report_summary.csv
"""

import pandas as pd
import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent / "data" / "cleaned_pages.csv"
KEYWORDS_CONFIG = Path(__file__).parent.parent / "config" / "esg_keywords.json"
OUTPUT_PAGES = Path(__file__).parent.parent / "data" / "page_classification.csv"
OUTPUT_SUMMARY = Path(__file__).parent.parent / "data" / "report_summary.csv"


def load_esg_keywords(config_path: Path) -> dict[str, dict[str, list[str]]]:
    """Load ESG keyword taxonomy from JSON config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_page(text: str, taxonomy: dict) -> tuple[str, str, dict[str, int]]:
    """Classify a page's text against the ESG taxonomy.
    Returns (major_category, sub_category, score_dict).
    """
    if not text or len(text.strip()) == 0:
        return "Unclassified", "Unclassified", {}

    text_lower = text.lower()
    scores = {}

    for major, subcategories in taxonomy.items():
        for sub, keywords in subcategories.items():
            count = sum(1 for kw in keywords if kw.lower() in text_lower)
            if count > 0:
                scores[f"{major} > {sub}"] = count

    if not scores:
        return "Unclassified", "Unclassified", {}

    # Find the best matching category
    best = max(scores, key=scores.get)
    major, sub = best.split(" > ")

    # Also find all categories with matches (for multi-topic pages)
    all_matches = {k: v for k, v in sorted(scores.items(), key=lambda x: -x[1])}

    return major, sub, all_matches


def main():
    df = pd.read_csv(INPUT_PATH)
    taxonomy = load_esg_keywords(KEYWORDS_CONFIG)

    print(f"Loaded {len(df)} pages, {df['company'].nunique()} companies")
    print(f"Taxonomy: {len(taxonomy)} major categories")

    # Print taxonomy summary
    for major, subs in taxonomy.items():
        total_kw = sum(len(kws) for kws in subs.values())
        print(f"  {major}: {len(subs)} subcategories, {total_kw} keywords")

    # Classify each page
    print("\nClassifying pages...")
    major_cats = []
    sub_cats = []
    all_matches_list = []
    match_counts = []

    for _, row in df.iterrows():
        if row["is_low_text"]:
            major_cats.append("LowText")
            sub_cats.append("LowText")
            all_matches_list.append("")
            match_counts.append(0)
            continue

        text = row["cleaned_text"] if pd.notna(row["cleaned_text"]) else ""
        major, sub, matches = classify_page(text, taxonomy)

        major_cats.append(major)
        sub_cats.append(sub)
        all_matches_list.append("; ".join([f"{k}({v})" for k, v in matches.items()]))
        match_counts.append(sum(matches.values()))

    df["major_category"] = major_cats
    df["sub_category"] = sub_cats
    df["all_matches"] = all_matches_list
    df["match_count"] = match_counts

    # Save page-level classification
    output_cols = [
        "company", "year", "page_num", "total_pages",
        "word_count", "token_count",
        "major_category", "sub_category", "all_matches", "match_count",
    ]
    # Ensure categories are clean strings
    out_df = df[output_cols].copy()
    out_df["major_category"] = out_df["major_category"].fillna("Unclassified").astype(str)
    out_df["sub_category"] = out_df["sub_category"].fillna("Unclassified").astype(str)
    out_df.to_csv(OUTPUT_PAGES, index=False, encoding="utf-8-sig")
    print(f"Page classification saved to {OUTPUT_PAGES}")

    # Generate report-level summary
    content_df = df[~df["is_low_text"]].copy()
    summary_rows = []

    for (company, year), group in content_df.groupby(["company", "year"]):
        total_content = len(group)
        row = {
            "company": company,
            "year": year,
            "total_content_pages": total_content,
        }

        # Major category distribution
        for cat in list(taxonomy.keys()) + ["Unclassified", "LowText"]:
            count = (group["major_category"] == cat).sum()
            row[f"{cat}_pages"] = count
            row[f"{cat}_pct"] = round(count / total_content * 100, 1) if total_content > 0 else 0

        # Top subcategories
        sub_dist = group[~group["major_category"].isin(["Unclassified", "LowText"])]["sub_category"].value_counts()
        row["top_subcategories"] = "; ".join(
            [f"{cat}({cnt})" for cat, cnt in sub_dist.head(5).items()]
        )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"Report summary saved to {OUTPUT_SUMMARY}")

    # Print classification stats
    print("\n--- Classification Statistics ---")
    classifiable = (~df["major_category"].isin(["Unclassified", "LowText"])).sum()
    unclass_count = (df["major_category"] == "Unclassified").sum()
    lowtext_count = (df["major_category"] == "LowText").sum()
    print(f"Classified pages: {classifiable}")
    print(f"Unclassified pages: {unclass_count}")
    print(f"Low-text pages: {lowtext_count}")

    print("\n--- Major Category Distribution (All Reports) ---")
    cat_counts = df[df["is_low_text"] == False]["major_category"].value_counts()
    for cat, cnt in cat_counts.items():
        if cat not in ("Unclassified", "LowText"):
            print(f"  {cat}: {cnt} pages ({cnt / classifiable * 100:.1f}%)")
    if "Unclassified" in cat_counts:
        print(f"  Unclassified: {cat_counts['Unclassified']} pages")

    print("\n--- Top Subcategories (All Reports) ---")
    sub_counts = df[~df["major_category"].isin(["Unclassified", "LowText"])]["sub_category"].value_counts()
    for cat, cnt in sub_counts.head(10).items():
        print(f"  {cat}: {cnt} pages")


if __name__ == "__main__":
    main()
