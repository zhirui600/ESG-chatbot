"""
Step 15: ESG Disclosure Gap Analysis
- Map each company's disclosure coverage against a comprehensive ESG framework
- Identify which ESG sub-topics each company covers vs. misses
- Score disclosure breadth and depth
- Provide actionable recommendations per company

Input:  data/page_classification.csv, config/esg_keywords.json
Output: data/disclosure_gaps.csv, data/disclosure_coverage_matrix.csv,
        output/disclosure_gap_heatmap.png, output/disclosure_radar.png
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from math import pi

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
CONFIG_DIR = Path(__file__).parent.parent / "config"


def build_coverage_matrix(df_cls, taxonomy):
    """Build binary + intensity matrix: which companies cover which sub-topics."""
    print("Building disclosure coverage matrix...")

    # Get all sub-categories from taxonomy
    all_subcats = []
    subcat_to_major = {}
    for major, subs in taxonomy.items():
        for sub in subs:
            all_subcats.append(sub)
            subcat_to_major[sub] = major

    companies = sorted(df_cls['company'].unique())

    # Filter to content pages only
    content = df_cls[~df_cls['major_category'].isin(['LowText', 'Unclassified'])].copy()

    # Build coverage matrix: pages per sub-category per company
    coverage = pd.DataFrame(0, index=companies, columns=all_subcats)
    for _, row in content.iterrows():
        matches_str = str(row['all_matches'])
        if matches_str == 'nan' or matches_str == '':
            continue
        for match in matches_str.split('; '):
            parts = match.split('(')
            if len(parts) == 2:
                cat_path = parts[0].strip()
                count = int(parts[1].replace(')', ''))
                # cat_path is like "Environmental > Climate Change"
                sub = cat_path.split(' > ')[-1].strip() if ' > ' in cat_path else cat_path
                if sub in coverage.columns:
                    coverage.at[row['company'], sub] += count

    # Save raw coverage
    coverage.to_csv(DATA_DIR / "disclosure_coverage_matrix.csv", encoding='utf-8-sig')

    # Binary coverage (threshold: >= 3 keyword hits across all pages)
    binary_coverage = (coverage >= 3).astype(int)

    return coverage, binary_coverage, all_subcats, subcat_to_major


def compute_gap_analysis(coverage, binary_coverage, all_subcats, subcat_to_major, companies):
    """Compute disclosure gaps and recommendations per company."""
    print("Computing disclosure gap analysis...")

    gap_rows = []
    for company in companies:
        covered = [s for s in all_subcats if binary_coverage.at[company, s] == 1]
        missing = [s for s in all_subcats if binary_coverage.at[company, s] == 0]

        # Compute depth scores: how many keyword mentions per covered topic (normalized)
        depths = coverage.loc[company, covered]
        avg_depth = depths.mean() if len(depths) > 0 else 0

        # Coverage breadth
        breadth_score = len(covered) / len(all_subcats) * 100

        # E/S/G pillar breakdown
        e_covered = [s for s in covered if subcat_to_major.get(s) == 'Environmental']
        s_covered = [s for s in covered if subcat_to_major.get(s) == 'Social']
        g_covered = [s for s in covered if subcat_to_major.get(s) == 'Governance']

        e_total = sum(1 for s in all_subcats if subcat_to_major.get(s) == 'Environmental')
        s_total = sum(1 for s in all_subcats if subcat_to_major.get(s) == 'Social')
        g_total = sum(1 for s in all_subcats if subcat_to_major.get(s) == 'Governance')

        gap_rows.append({
            'company': company,
            'topics_covered': len(covered),
            'topics_missing': len(missing),
            'breadth_score': round(breadth_score, 1),
            'avg_depth': round(avg_depth, 1),
            'env_coverage_pct': round(len(e_covered) / e_total * 100, 1),
            'soc_coverage_pct': round(len(s_covered) / s_total * 100, 1),
            'gov_coverage_pct': round(len(g_covered) / g_total * 100, 1),
            'missing_topics': '; '.join(missing),
            'covered_topics': '; '.join(covered),
            'recommendation': _generate_recommendation(missing, subcat_to_major),
        })

    df_gaps = pd.DataFrame(gap_rows)
    out_path = DATA_DIR / "disclosure_gaps.csv"
    df_gaps.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved gap analysis → {out_path}")
    return df_gaps


def _generate_recommendation(missing, subcat_to_major):
    """Generate textual recommendation based on missing topics."""
    if len(missing) == 0:
        return "Comprehensive coverage - consider deepening existing disclosures."

    # Prioritize by ESG pillar gaps
    e_miss = [s for s in missing if subcat_to_major.get(s) == 'Environmental']
    s_miss = [s for s in missing if subcat_to_major.get(s) == 'Social']
    g_miss = [s for s in missing if subcat_to_major.get(s) == 'Governance']

    parts = []
    if g_miss:
        parts.append(f"Strengthen Governance disclosure: {', '.join(g_miss[:2])}")
    if e_miss:
        parts.append(f"Add Environmental topics: {', '.join(e_miss[:2])}")
    if s_miss:
        parts.append(f"Expand Social coverage: {', '.join(s_miss[:2])}")

    return '. '.join(parts[:2])


def plot_coverage_heatmap(coverage, subcat_to_major):
    """Heatmap showing disclosure intensity per sub-topic per company."""
    # Normalize by company (column-wise within company)
    norm_coverage = coverage.div(coverage.sum(axis=1) + 1, axis=0) * 100

    # Sort columns by ESG pillar
    sorted_cols = sorted(coverage.columns, key=lambda x: (
        0 if subcat_to_major.get(x) == 'Environmental' else
        1 if subcat_to_major.get(x) == 'Social' else 2
    ))
    norm_coverage = norm_coverage[sorted_cols]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Create color-coded column labels
    pillar_colors = {'Environmental': '#2ecc71', 'Social': '#3498db', 'Governance': '#9b59b6'}
    col_colors = [pillar_colors.get(subcat_to_major.get(c, ''), '#95a5a6') for c in sorted_cols]

    im = ax.imshow(norm_coverage.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(sorted_cols)))
    ax.set_xticklabels(sorted_cols, rotation=60, ha='right', fontsize=9)
    ax.set_yticks(range(len(coverage.index)))
    ax.set_yticklabels(coverage.index, fontsize=10)

    # Color the x-axis labels by pillar
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(col_colors[i])
        label.set_fontweight('bold')

    # Add value annotations
    for i in range(len(coverage.index)):
        for j in range(len(sorted_cols)):
            val = coverage.iloc[i][sorted_cols[j]]
            if val > 0:
                ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=7,
                        color='white' if norm_coverage.iloc[i][sorted_cols[j]] > 5 else 'black')

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('Relative Disclosure Intensity (%)')
    ax.set_title('ESG Disclosure Coverage & Intensity by Company\n'
                  '(Green=Environmental, Blue=Social, Purple=Governance)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = OUTPUT_DIR / "disclosure_gap_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_radar_chart(df_gaps):
    """Multi-company radar chart for ESG coverage dimensions."""
    categories = ['Breadth Score', 'Env Coverage', 'Soc Coverage', 'Gov Coverage', 'Avg Depth']
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(df_gaps)))

    for idx, (_, row) in enumerate(df_gaps.iterrows()):
        # Normalize values to 0-100 scale
        values = [
            row['breadth_score'],
            row['env_coverage_pct'],
            row['soc_coverage_pct'],
            row['gov_coverage_pct'],
            min(row['avg_depth'] / df_gaps['avg_depth'].max() * 100, 100),
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=1.5, label=row['company'],
                color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.05, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_title('ESG Disclosure Multi-Dimensional Profile', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "disclosure_radar.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_cls = pd.read_csv(DATA_DIR / "page_classification.csv")
    with open(CONFIG_DIR / "esg_keywords.json", 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)

    companies = sorted(df_cls['company'].unique())

    coverage, binary_coverage, all_subcats, subcat_to_major = build_coverage_matrix(df_cls, taxonomy)
    df_gaps = compute_gap_analysis(coverage, binary_coverage, all_subcats, subcat_to_major, companies)

    plot_coverage_heatmap(coverage, subcat_to_major)
    plot_radar_chart(df_gaps)

    # Print insights
    print("\n=== Disclosure Gap Analysis Insights ===")
    print(f"\nESG Framework: {len(all_subcats)} sub-topics across 3 pillars")
    print(f"\nCoverage Breadth Ranking:")
    for _, row in df_gaps.sort_values('breadth_score', ascending=False).iterrows():
        print(f"  {row['company']}: {row['breadth_score']}% ({row['topics_covered']}/{row['topics_covered']+row['topics_missing']} topics)")

    print(f"\nCompanies with Governance gaps:")
    for _, row in df_gaps[df_gaps['gov_coverage_pct'] < 100].iterrows():
        missing_g = [t for t in row['missing_topics'].split('; ') if subcat_to_major.get(t) == 'Governance']
        if missing_g:
            print(f"  {row['company']}: missing {', '.join(missing_g)}")

    print("\nStep 15 Complete!")


if __name__ == "__main__":
    main()
