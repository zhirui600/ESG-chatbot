"""
Step 14: ESG Commitment & Target Extraction
- Extract quantitative ESG targets/goals from raw text (e.g. "reduce emissions by 50% by 2030")
- Classify language as forward-looking (commitments) vs backward-looking (achievements)
- Build structured commitment database for chatbot knowledge base

Input:  data/raw_pages.csv, data/page_classification.csv
Output: data/esg_commitments.csv, data/forward_backward_scores.csv,
        output/commitment_timeline.png, output/forward_vs_backward.png
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ─── Regex patterns for extracting quantitative ESG targets ───
TARGET_PATTERNS = [
    # "reduce X by Y% by YEAR" / "achieve X by YEAR"
    re.compile(
        r'(?P<action>reduce|decrease|cut|lower|achieve|reach|attain|increase|improve)'
        r'[^.]{0,80}?'
        r'(?P<value>\d+[\.,]?\d*)\s*(?P<unit>%|percent|metric tons?|tonnes?|mwh|gwh|kwh|million|billion|m3)'
        r'[^.]{0,60}?'
        r'(?:by|before|in|through)\s+(?P<year>20[2-9]\d)',
        re.IGNORECASE
    ),
    # "X% reduction by YEAR"
    re.compile(
        r'(?P<value>\d+[\.,]?\d*)\s*(?P<unit>%|percent)'
        r'\s+(?P<action>reduction|decrease|increase|improvement)'
        r'[^.]{0,60}?'
        r'(?:by|before|in|through)\s+(?P<year>20[2-9]\d)',
        re.IGNORECASE
    ),
    # "net zero by YEAR" / "carbon neutral by YEAR"
    re.compile(
        r'(?P<action>net[\s-]?zero|carbon[\s-]?neutral|100\s*%\s*renewable)'
        r'[^.]{0,80}?'
        r'(?:by|before|in|through)\s+(?P<year>20[2-9]\d)',
        re.IGNORECASE
    ),
    # "target/goal of X% by YEAR"
    re.compile(
        r'(?:target|goal|objective|commitment|pledge|aim)\s+(?:of|to|is)'
        r'[^.]{0,80}?'
        r'(?P<value>\d+[\.,]?\d*)\s*(?P<unit>%|percent|metric tons?|mwh|gwh|million|billion)'
        r'[^.]{0,60}?'
        r'(?:by|before|in|through)\s+(?P<year>20[2-9]\d)',
        re.IGNORECASE
    ),
]

# ─── Forward-looking vs Backward-looking language ───
FORWARD_WORDS = [
    'will', 'plan', 'target', 'goal', 'aim', 'commit', 'committed', 'pledge',
    'intend', 'aspire', 'strive', 'expect', 'anticipate', 'forecast',
    'by 2025', 'by 2030', 'by 2035', 'by 2040', 'by 2050',
    'roadmap', 'strategy', 'future', 'vision', 'ambition', 'objective'
]

BACKWARD_WORDS = [
    'achieved', 'accomplished', 'reduced', 'decreased', 'improved', 'completed',
    'delivered', 'invested', 'generated', 'saved', 'recycled', 'diverted',
    'installed', 'implemented', 'exceeded', 'surpassed', 'reached',
    'in 2022', 'in 2023', 'in 2024', 'since 2020', 'year-over-year',
    'compared to', 'from baseline', 'last year', 'this year'
]


def extract_commitments(df_raw):
    """Extract structured ESG commitments/targets from raw text."""
    print("Extracting ESG commitments and quantitative targets...")

    commitments = []
    for _, row in df_raw.iterrows():
        text = str(row['raw_text'])
        if len(text) < 50:
            continue

        for pattern in TARGET_PATTERNS:
            for match in pattern.finditer(text):
                d = match.groupdict()
                # Get some context around the match
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].replace('\n', ' ').strip()

                commitments.append({
                    'company': row['company'],
                    'year': row['year'],
                    'page_num': row['page_num'],
                    'action': d.get('action', ''),
                    'value': d.get('value', ''),
                    'unit': d.get('unit', ''),
                    'target_year': d.get('year', ''),
                    'context': context,
                })

    df_commit = pd.DataFrame(commitments)
    if len(df_commit) > 0:
        df_commit = df_commit.drop_duplicates(subset=['company', 'context'])

    out_path = DATA_DIR / "esg_commitments.csv"
    df_commit.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Extracted {len(df_commit)} unique commitments → {out_path}")
    return df_commit


def compute_forward_backward_scores(df_raw):
    """Score each page and company for forward-looking vs backward-looking language."""
    print("Scoring forward-looking vs backward-looking language...")

    results = []
    for _, row in df_raw.iterrows():
        text = str(row['raw_text']).lower()
        if len(text.split()) < 20:
            continue

        fwd_count = sum(text.count(w) for w in FORWARD_WORDS)
        bwd_count = sum(text.count(w) for w in BACKWARD_WORDS)
        total = fwd_count + bwd_count + 1  # avoid div/0

        results.append({
            'company': row['company'],
            'year': row['year'],
            'page_num': row['page_num'],
            'forward_count': fwd_count,
            'backward_count': bwd_count,
            'forward_ratio': fwd_count / total,
            'backward_ratio': bwd_count / total,
        })

    df_fb = pd.DataFrame(results)

    # Company-level aggregation
    company_fb = df_fb.groupby('company').agg(
        total_forward=('forward_count', 'sum'),
        total_backward=('backward_count', 'sum'),
        avg_forward_ratio=('forward_ratio', 'mean'),
        avg_backward_ratio=('backward_ratio', 'mean'),
    ).reset_index()

    company_fb['fwd_bwd_balance'] = (
        company_fb['total_forward'] / (company_fb['total_forward'] + company_fb['total_backward'] + 1)
    )

    out_path = DATA_DIR / "forward_backward_scores.csv"
    company_fb.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved forward/backward scores → {out_path}")
    return df_fb, company_fb


def plot_commitment_timeline(df_commit):
    """Plot extracted commitments by target year."""
    if len(df_commit) == 0:
        print("  No commitments to plot.")
        return

    df_commit['target_year'] = pd.to_numeric(df_commit['target_year'], errors='coerce')
    df_valid = df_commit.dropna(subset=['target_year'])
    df_valid = df_valid[(df_valid['target_year'] >= 2024) & (df_valid['target_year'] <= 2060)]

    if len(df_valid) == 0:
        print("  No valid target years to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_valid.groupby(['company', 'target_year']).size().unstack(fill_value=0)

    pivot.T.plot(kind='bar', stacked=True, ax=ax, cmap='Set3', edgecolor='black', linewidth=0.5)
    ax.set_title('ESG Commitment Target Years by Company', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Year')
    ax.set_ylabel('Number of Commitments')
    plt.legend(title='Company', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "commitment_timeline.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_forward_backward(company_fb):
    """Diverging bar chart: forward-looking vs backward-looking language ratio."""
    fig, ax = plt.subplots(figsize=(12, 6))

    companies = company_fb.sort_values('fwd_bwd_balance')['company']
    fwd = company_fb.set_index('company').loc[companies, 'avg_forward_ratio']
    bwd = company_fb.set_index('company').loc[companies, 'avg_backward_ratio']

    y = range(len(companies))
    ax.barh(y, fwd, color='#3498db', label='Forward-Looking (Commitments)', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.barh(y, -bwd, color='#e74c3c', label='Backward-Looking (Achievements)', alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(companies, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Average Language Ratio per Page')
    ax.set_title('Forward-Looking vs Backward-Looking Language by Company', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()

    out_path = OUTPUT_DIR / "forward_vs_backward.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(DATA_DIR / "raw_pages.csv")
    df_commit = extract_commitments(df_raw)
    df_fb_pages, company_fb = compute_forward_backward_scores(df_raw)

    plot_commitment_timeline(df_commit)
    plot_forward_backward(company_fb)

    # Print summary insights
    print("\n=== ESG Commitment Extraction Insights ===")
    if len(df_commit) > 0:
        print(f"\nCommitments by company:")
        print(df_commit['company'].value_counts().to_string())
        print(f"\nMost common target years:")
        print(df_commit['target_year'].value_counts().head(5).to_string())

    print(f"\nForward vs Backward balance (higher = more forward-looking):")
    print(company_fb[['company', 'fwd_bwd_balance']].sort_values('fwd_bwd_balance', ascending=False).to_string(index=False))

    print("\nStep 14 Complete!")


if __name__ == "__main__":
    main()
