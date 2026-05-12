"""
Step 7: Advanced Data Mining - Feature Correlation Analysis
Input:  data/keywords_tfidf.csv, data/page_sentiment.csv
Output: data/correlation_matrix.csv, output/correlation_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

TFIDF_PATH = DATA_DIR / "keywords_tfidf.csv"
SENTIMENT_PATH = DATA_DIR / "page_sentiment.csv"

# Target high-frequency ESG keywords to analyze for correlation
TARGET_KEYWORDS = [
    'energy', 'employee', 'emission', 'climate', 
    'carbon', 'risk', 'water', 'safety', 'diversity'
]

def extract_keyword_scores(df_tfidf: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the string representations of TF-IDF keywords and scores 
    into structured numerical columns for target keywords.
    """
    print("Parsing TF-IDF features for correlation mining...")
    
    # Initialize columns for our target keywords with 0.0
    for kw in TARGET_KEYWORDS:
        df_tfidf[f"kw_{kw}"] = 0.0
        
    # Extract numerical scores for target keywords page by page
    for idx, row in df_tfidf.iterrows():
        kw_str = str(row['tfidf_keywords']).split('; ')
        score_str = str(row['tfidf_scores']).split('; ')
        
        if len(kw_str) == len(score_str) and kw_str[0] != 'nan':
            kw_dict = dict(zip(kw_str, score_str))
            for kw in TARGET_KEYWORDS:
                if kw in kw_dict:
                    df_tfidf.at[idx, f"kw_{kw}"] = float(kw_dict[kw])
                    
    return df_tfidf

def run_correlation_mining():
    print("Loading TF-IDF and Sentiment data...")
    df_tfidf = pd.read_csv(TFIDF_PATH)
    df_sentiment = pd.read_csv(SENTIMENT_PATH)
    
    # Parse the TF-IDF strings into actual numerical columns
    df_tfidf_parsed = extract_keyword_scores(df_tfidf)
    
    # Merge on page identifiers
    print("Merging datasets for cross-analysis...")
    df_merged = pd.merge(
        df_sentiment[['company', 'year', 'page_num', 'compound', 'major_category']], 
        df_tfidf_parsed[['company', 'year', 'page_num'] + [f"kw_{k}" for k in TARGET_KEYWORDS]],
        on=['company', 'year', 'page_num'], 
        how='inner'
    )
    
    # Select numerical columns for correlation
    corr_cols = ['compound'] + [f"kw_{k}" for k in TARGET_KEYWORDS]
    df_corr_data = df_merged[corr_cols].copy()
    
    # Rename columns for better visualization readability
    rename_dict = {'compound': 'Overall Sentiment'}
    rename_dict.update({f"kw_{k}": k.capitalize() for k in TARGET_KEYWORDS})
    df_corr_data.rename(columns=rename_dict, inplace=True)
    
    # Calculate Pearson Correlation Matrix
    print("Calculating Pearson Correlation matrix...")
    corr_matrix = df_corr_data.corr(method='pearson')
    
    # Save raw matrix to CSV
    out_csv = DATA_DIR / "correlation_matrix.csv"
    corr_matrix.to_csv(out_csv, encoding="utf-8-sig")
    print(f"Saved numerical correlation matrix to {out_csv}")
    
    # --- Generate Advanced Visualization ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use seaborn to create a highly professional heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) 
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=0.3, 
        vmin=-0.3, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True, 
        fmt=".2f",
        ax=ax
    )
    
    ax.set_title("Feature Correlation: Sentiment vs ESG Keywords", fontsize=14, pad=20)
    plt.tight_layout()
    
    out_plot = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap to {out_plot}")
    
    print("\nCorrelation Analysis Complete!")

if __name__ == "__main__":
    run_correlation_mining()