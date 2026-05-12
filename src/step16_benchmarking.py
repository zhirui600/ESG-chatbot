"""
Step 16: ESG Peer Benchmarking & Industry Positioning
- Rank companies across multiple ESG dimensions
- Z-score based percentile positioning
- Identify leaders, laggards, and emerging strengths/weaknesses
- Generate a "company profile card" data structure for chatbot

Input:  data/company_features_matrix.csv, data/disclosure_gaps.csv,
        data/forward_backward_scores.csv, data/esg_commitments.csv
Output: data/esg_benchmarks.csv, data/company_profiles.json,
        output/benchmark_ranking.png, output/strength_weakness.png
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def compute_benchmarks(df_feat, df_gaps, df_fb):
    """Compute z-score based percentile rankings across multiple dimensions."""
    print("Computing peer benchmarks...")

    # Merge all available data
    bench = df_feat[['company', 'env_score', 'soc_score', 'gov_score',
                      'avg_fog_index', 'avg_vagueness', 'avg_quant_score',
                      'composite_esg_score']].copy()

    # Add disclosure breadth
    if df_gaps is not None:
        bench = bench.merge(df_gaps[['company', 'breadth_score', 'avg_depth']], on='company', how='left')

    # Add forward/backward balance
    if df_fb is not None:
        bench = bench.merge(df_fb[['company', 'fwd_bwd_balance']], on='company', how='left')

    # Compute z-scores for each metric
    score_cols = [c for c in bench.columns if c != 'company']
    for col in score_cols:
        bench[f'{col}_zscore'] = (bench[col] - bench[col].mean()) / (bench[col].std() + 1e-9)

    # Compute percentile ranks (0-100)
    for col in score_cols:
        bench[f'{col}_pctile'] = bench[col].rank(pct=True) * 100

    # For negative indicators, flip the ranking (lower is better)
    for neg_col in ['avg_fog_index', 'avg_vagueness']:
        if f'{neg_col}_pctile' in bench.columns:
            bench[f'{neg_col}_pctile'] = 100 - bench[f'{neg_col}_pctile']

    out_path = DATA_DIR / "esg_benchmarks.csv"
    bench.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved benchmarks → {out_path}")
    return bench


def generate_company_profiles(bench, df_commit):
    """Generate structured company profile cards for chatbot use."""
    print("Generating company profile cards for chatbot...")

    profiles = {}

    # Define dimension descriptions for the chatbot
    dimension_labels = {
        'env_score': 'Environmental Focus',
        'soc_score': 'Social Responsibility',
        'gov_score': 'Governance Strength',
        'breadth_score': 'Disclosure Breadth',
        'avg_quant_score': 'Quantitative Rigor',
        'composite_esg_score': 'Overall ESG Score',
    }

    for _, row in bench.iterrows():
        company = row['company']

        # Find strengths (top percentile dimensions)
        pctile_cols = {c.replace('_pctile', ''): row[c]
                       for c in bench.columns if c.endswith('_pctile')}
        strengths = sorted(pctile_cols.items(), key=lambda x: -x[1])[:3]
        weaknesses = sorted(pctile_cols.items(), key=lambda x: x[1])[:3]

        # Commitments for this company
        commitments_list = []
        if df_commit is not None and len(df_commit) > 0:
            company_commits = df_commit[df_commit['company'] == company]
            for _, c_row in company_commits.head(5).iterrows():
                commitments_list.append({
                    'target_year': str(c_row.get('target_year', '')),
                    'action': str(c_row.get('action', '')),
                    'context': str(c_row.get('context', ''))[:200],
                })

        profiles[company] = {
            'company': company,
            'overall_esg_score': round(row.get('composite_esg_score', 0), 1),
            'environmental_score': round(row['env_score'], 1),
            'social_score': round(row['soc_score'], 1),
            'governance_score': round(row['gov_score'], 1),
            'strengths': [
                {'dimension': dimension_labels.get(s[0], s[0]),
                 'percentile': round(s[1], 1)}
                for s in strengths
            ],
            'weaknesses': [
                {'dimension': dimension_labels.get(w[0], w[0]),
                 'percentile': round(w[1], 1)}
                for w in weaknesses
            ],
            'key_commitments': commitments_list,
            'disclosure_breadth': round(row.get('breadth_score', 0), 1),
            'transparency_score': round(100 - row.get('avg_vagueness', 0) * 10, 1)
                                  if 'avg_vagueness' in row.index else None,
        }

    out_path = DATA_DIR / "company_profiles.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(profiles)} company profiles → {out_path}")
    return profiles


def plot_benchmark_ranking(bench):
    """Horizontal multi-bar ranking chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    companies = bench.sort_values('composite_esg_score', ascending=True)['company']
    y_pos = range(len(companies))

    metrics = [
        ('env_score', 'Environmental Score', '#2ecc71'),
        ('soc_score', 'Social Score', '#3498db'),
        ('gov_score', 'Governance Score', '#9b59b6'),
    ]

    for ax, (col, title, color) in zip(axes, metrics):
        values = bench.set_index('company').loc[companies, col]
        bars = ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Score (%)')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=9)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(companies, fontsize=10)

    plt.suptitle('ESG Pillar Scores: Peer Benchmarking', fontsize=15, fontweight='bold')
    plt.tight_layout()

    out_path = OUTPUT_DIR / "benchmark_ranking.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_strength_weakness(bench):
    """Matrix showing strengths and weaknesses per company across dimensions."""
    dims = ['env_score', 'soc_score', 'gov_score', 'avg_quant_score']
    dim_labels = ['Environmental', 'Social', 'Governance', 'Quant. Rigor']

    # Available dims only
    available_dims = [d for d in dims if d in bench.columns]
    available_labels = [dim_labels[dims.index(d)] for d in available_dims]

    if 'breadth_score' in bench.columns:
        available_dims.append('breadth_score')
        available_labels.append('Disclosure Breadth')

    if 'fwd_bwd_balance' in bench.columns:
        available_dims.append('fwd_bwd_balance')
        available_labels.append('Forward-Looking')

    companies = bench['company'].values

    # Build z-score matrix for coloring
    z_matrix = np.zeros((len(companies), len(available_dims)))
    for j, dim in enumerate(available_dims):
        vals = bench[dim].values
        z_matrix[:, j] = (vals - vals.mean()) / (vals.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(z_matrix, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)

    ax.set_xticks(range(len(available_labels)))
    ax.set_xticklabels(available_labels, rotation=30, ha='right', fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(companies)))
    ax.set_yticklabels(companies, fontsize=10)

    # Annotations
    for i in range(len(companies)):
        for j in range(len(available_dims)):
            val = bench.iloc[i][available_dims[j]]
            z = z_matrix[i, j]
            marker = '★' if z > 1.0 else ('▼' if z < -1.0 else '')
            ax.text(j, i, f'{val:.1f}\n{marker}', ha='center', va='center', fontsize=8,
                    color='black' if abs(z) < 1.5 else 'white', fontweight='bold')

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('Z-Score (Green=Strong, Red=Weak)', fontsize=11)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#27ae60', label='★ Strength (z > 1.0)'),
        mpatches.Patch(facecolor='#c0392b', label='▼ Weakness (z < -1.0)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.12), fontsize=10)

    ax.set_title('ESG Strength & Weakness Matrix (Peer Comparison)\n'
                 'Green = Above Average, Red = Below Average', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = OUTPUT_DIR / "strength_weakness.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all available data
    df_feat = pd.read_csv(DATA_DIR / "company_features_matrix.csv")

    df_gaps = None
    if (DATA_DIR / "disclosure_gaps.csv").exists():
        df_gaps = pd.read_csv(DATA_DIR / "disclosure_gaps.csv")

    df_fb = None
    if (DATA_DIR / "forward_backward_scores.csv").exists():
        df_fb = pd.read_csv(DATA_DIR / "forward_backward_scores.csv")

    df_commit = None
    if (DATA_DIR / "esg_commitments.csv").exists():
        df_commit = pd.read_csv(DATA_DIR / "esg_commitments.csv")

    bench = compute_benchmarks(df_feat, df_gaps, df_fb)
    profiles = generate_company_profiles(bench, df_commit)

    plot_benchmark_ranking(bench)
    plot_strength_weakness(bench)

    # Print top insights
    print("\n=== ESG Peer Benchmarking Insights ===")
    print("\nOverall ESG Score Ranking:")
    for _, row in bench.sort_values('composite_esg_score', ascending=False).iterrows():
        print(f"  {row['company']}: {row['composite_esg_score']:.1f}")

    print("\nCompany profiles saved as JSON for chatbot integration.")
    print("Step 16 Complete!")


if __name__ == "__main__":
    main()
