"""
Step 13: Feature Engineering & ESG Composite Scoring
Input: All previous step outputs
Output: data/company_features_matrix.csv (Final feature matrix for Obj 5)
"""

import pandas as pd
import numpy as np
import textstat
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def extract_advanced_page_features(df_raw, df_clean):
    print("Extracting advanced text features...")
    df = pd.merge(df_raw[['company', 'year', 'page_num', 'raw_text', 'is_low_text']],
                  df_clean[['company', 'year', 'page_num', 'token_count']], 
                  on=['company', 'year', 'page_num'])
    
    df_content = df[~df['is_low_text']].copy()
    
    quants = []
    for text in df_content['raw_text']:
        t = str(text)
        quant_score = (t.count('%') + t.count('$') + sum(1 for c in t if c.isdigit())) / max(1, len(t.split()))
        quants.append(quant_score)
        
    df_content['quant_disclosure_score'] = quants
    
    return df_content.groupby('company').agg(
        avg_token_count=('token_count', 'mean'),
        avg_quant_score=('quant_disclosure_score', 'mean')
    ).reset_index()

def assemble_master_features():
    print("Iteratively merging parts into Master Feature Matrix...")
    
    # 1. Step 5 ESG scores
    OUTPUT_DIR = Path(__file__).parent.parent / "output"
    df_esg = pd.read_csv(OUTPUT_DIR / "esg_scores_summary.csv")
    master = df_esg[['company', 'env_score', 'soc_score', 'gov_score', 'total_content_pages']]
    
    # 2. Step 6 Sentiment
    if (DATA_DIR / "sentiment_by_major_category.csv").exists():
        df_sent = pd.read_csv(DATA_DIR / "sentiment_by_major_category.csv")
        master = pd.merge(master, df_sent, on='company', how='left')
        master = master.rename(columns={'Environmental': 'sent_env', 'Social': 'sent_soc', 'Governance':'sent_gov'})
        
    # 3. Step 9 Greenwashing
    if (DATA_DIR / "greenwashing_risk.csv").exists():
        df_gw = pd.read_csv(DATA_DIR / "greenwashing_risk.csv")
        master = pd.merge(master, df_gw[['company', 'avg_fog_index', 'avg_vagueness']], on='company', how='left')
        
    # 4. Step 10 LDA Topics
    if (DATA_DIR / "lda_topic_distributions.csv").exists():
        df_lda = pd.read_csv(DATA_DIR / "lda_topic_distributions.csv")
        topic_cols = [c for c in df_lda.columns if c.startswith('topic_')]
        lda_agg = df_lda.groupby('company')[topic_cols].mean().reset_index()
        master = pd.merge(master, lda_agg, on='company', how='left')

    # 5. Advanced page features
    df_raw = pd.read_csv(DATA_DIR / "raw_pages.csv")
    df_clean = pd.read_csv(DATA_DIR / "cleaned_pages.csv")
    df_adv = extract_advanced_page_features(df_raw, df_clean)
    master = pd.merge(master, df_adv, on='company', how='left')

    # 6. Step 14 Forward/Backward language scores
    if (DATA_DIR / "forward_backward_scores.csv").exists():
        df_fb = pd.read_csv(DATA_DIR / "forward_backward_scores.csv")
        master = pd.merge(master, df_fb[['company', 'fwd_bwd_balance']], on='company', how='left')

    # 7. Step 14 Commitment counts
    if (DATA_DIR / "esg_commitments.csv").exists():
        df_commit = pd.read_csv(DATA_DIR / "esg_commitments.csv")
        commit_counts = df_commit.groupby('company').size().reset_index(name='num_commitments')
        master = pd.merge(master, commit_counts, on='company', how='left')

    # 8. Step 15 Disclosure gap features
    if (DATA_DIR / "disclosure_gaps.csv").exists():
        df_gaps = pd.read_csv(DATA_DIR / "disclosure_gaps.csv")
        master = pd.merge(master, df_gaps[['company', 'breadth_score', 'avg_depth',
                                            'env_coverage_pct', 'soc_coverage_pct', 'gov_coverage_pct']],
                          on='company', how='left')

    return master

def compute_composite_score(master):
    # Normalize needed cols
    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    score = (
        norm(master['env_score']) * 0.3 + 
        norm(master['soc_score']) * 0.3 + 
        norm(master['gov_score']) * 0.1 + 
        norm(master.get('sent_env', 0)) * 0.1 +
        norm(master['avg_quant_score']) * 0.2
    )
    
    if 'avg_vagueness' in master:
        score -= norm(master['avg_vagueness']) * 0.1 # Penalty
        
    master['composite_esg_score'] = score * 100 # 0-100 scale
    return master

def main():
    master = assemble_master_features()
    master = compute_composite_score(master)
    
    out_path = DATA_DIR / "company_features_matrix.csv"
    master.fillna(0).to_csv(out_path, index=False)
    print(f"\nFinal Feature Matrix saved to {out_path}")
    print(f"Shape: {master.shape}")
    print("This is ready for Objective 5 Prediction Models!")

if __name__ == "__main__":
    main()
