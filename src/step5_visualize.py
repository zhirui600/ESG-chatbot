"""
Step 5: Key visualizations and summary for downstream stages.
Input:  data/page_classification.csv, data/report_summary.csv, data/keywords_tfidf.csv
Output: output/ directory with charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

PAGE_CLS = DATA_DIR / "page_classification.csv"
REPORT_SUM = DATA_DIR / "report_summary.csv"
TFIDF_KW = DATA_DIR / "keywords_tfidf.csv"

# Color scheme for ESG categories
ESG_COLORS = {
    "Environmental": "#2ecc71",
    "Social": "#3498db",
    "Governance": "#9b59b6",
    "None": "#bdc3c7",
    "Unclassified": "#bdc3c7",
    "LowText": "#95a5a6",
}


def plot_esg_distribution_by_company(report_df: pd.DataFrame):
    """Stacked bar chart: ESG page distribution per company."""
    categories = ["Environmental", "Social", "Governance", "Unclassified", "LowText"]
    companies = report_df["company"]

    fig, ax = plt.subplots(figsize=(12, 6))

    bottoms = np.zeros(len(companies))
    for cat in categories:
        col = f"{cat}_pages"
        if col not in report_df.columns:
            continue
        values = report_df[col].values
        ax.barh(companies, values, left=bottoms, label=cat,
                color=ESG_COLORS.get(cat, "#95a5a6"))
        bottoms += values

    ax.set_xlabel("Number of Pages")
    ax.set_title("ESG Topic Distribution by Company")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = OUTPUT_DIR / "esg_distribution_by_company.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_esg_percentage_stacked(report_df: pd.DataFrame):
    """100% stacked bar chart: ESG percentage per company."""
    categories = ["Environmental", "Social", "Governance"]
    companies = report_df["company"]

    fig, ax = plt.subplots(figsize=(12, 6))

    bottoms = np.zeros(len(companies))
    for cat in categories:
        col = f"{cat}_pct"
        if col not in report_df.columns:
            continue
        values = report_df[col].values
        ax.barh(companies, values, left=bottoms, label=cat,
                color=ESG_COLORS.get(cat, "#95a5a6"))
        bottoms += values

    ax.set_xlabel("Percentage (%)")
    ax.set_title("ESG Topic Proportion by Company (Excluding None)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = OUTPUT_DIR / "esg_proportion_by_company.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_top_keywords(tfidf_df: pd.DataFrame):
    """Bar chart: top 20 keywords across all reports."""
    from collections import Counter
    kw_counter = Counter()
    for kws in tfidf_df["tfidf_keywords"].dropna():
        for kw in str(kws).split("; "):
            kw = kw.strip()
            if kw:
                kw_counter[kw] += 1

    top20 = kw_counter.most_common(20)
    if not top20:
        print("  No keywords found, skipping top keywords chart.")
        return

    keywords, counts = zip(*top20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(keywords)), counts, color="#2ecc71")
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency (number of pages)")
    ax.set_title("Top 20 TF-IDF Keywords Across All Reports")
    plt.tight_layout()

    path = OUTPUT_DIR / "top20_keywords.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_category_heatmap(page_df: pd.DataFrame):
    """Heatmap: subcategory distribution per company."""
    content = page_df[~page_df["major_category"].isin(["Unclassified", "LowText", "nan"])].copy()
    pivot = content.groupby(["company", "sub_category"]).size().unstack(fill_value=0)
    # Normalize by company total
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot_pct.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(pivot_pct.columns)))
    ax.set_xticklabels(pivot_pct.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_pct.index)))
    ax.set_yticklabels(pivot_pct.index, fontsize=9)
    ax.set_title("ESG Subcategory Distribution by Company (%)")

    fig.colorbar(im, ax=ax, label="Percentage (%)")
    plt.tight_layout()

    path = OUTPUT_DIR / "subcategory_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_summary_stats(page_df: pd.DataFrame, report_df: pd.DataFrame):
    """Export summary statistics CSV for downstream stages."""
    output = {}

    # Per-company ESG scores (page ratio as proxy)
    for _, row in report_df.iterrows():
        total = row["total_content_pages"]
        output[row["company"]] = {
            "company": row["company"],
            "year": row["year"],
            "total_content_pages": total,
            "env_score": row.get("Environmental_pct", 0),
            "soc_score": row.get("Social_pct", 0),
            "gov_score": row.get("Governance_pct", 0),
        }

    stats_df = pd.DataFrame(output).T
    path = OUTPUT_DIR / "esg_scores_summary.csv"
    stats_df.to_csv(path, encoding="utf-8-sig")
    print(f"  Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    page_df = pd.read_csv(PAGE_CLS)
    report_df = pd.read_csv(REPORT_SUM)

    print("Generating visualizations...\n")

    plot_esg_distribution_by_company(report_df)
    plot_esg_percentage_stacked(report_df)

    if TFIDF_KW.exists():
        tfidf_df = pd.read_csv(TFIDF_KW)
        plot_top_keywords(tfidf_df)

    plot_category_heatmap(page_df)
    generate_summary_stats(page_df, report_df)

    print("\nDone. All outputs saved to output/")


if __name__ == "__main__":
    main()
