"""
Step 9: Advanced Data Mining - Greenwashing Detection & Readability

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

RAW_PAGES_PATH = DATA_DIR / "raw_pages.csv"
SENTIMENT_PATH = DATA_DIR / "page_sentiment.csv"

# Lexicons for identifying vague corporate speak vs concrete measurable metrics
VAGUE_BUZZWORDS = [
    'committed', 'strive', 'believe', 'future', 'eco-friendly', 
    'green', 'sustainable', 'sustainability', 'initiative', 'focus',
    'journey', 'vision', 'aim', 'endeavor', 'promote'
]

CONCRETE_METRICS = [
    '%', 'tons', 'kwh', 'mwh', 'reduction', 'decrease', 'scope 1', 
    'scope 2', 'scope 3', 'invested', 'million', 'billion', 'target',
    'achieved', 'baseline', 'metric'
]

def calculate_vagueness_ratio(text: str) -> float:
    """
    Calculate the ratio of vague buzzwords to concrete metrics.
    High ratio means the text is "fluffy" (lots of talk, few numbers).
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
        
    text_lower = text.lower()
    
    vague_count = sum(text_lower.count(word) for word in VAGUE_BUZZWORDS)
    concrete_count = sum(text_lower.count(word) for word in CONCRETE_METRICS)
    
    # Add 1 to denominator to prevent division by zero
    ratio = vague_count / (concrete_count + 1)
    return ratio

def run_greenwashing_analysis():
    print("Loading raw text for readability analysis...")
    if not RAW_PAGES_PATH.exists() or not SENTIMENT_PATH.exists():
        print("  Missing required data files. Ensure Step 1 and Step 6 are complete.")
        return

    df_raw = pd.read_csv(RAW_PAGES_PATH)
    df_sentiment = pd.read_csv(SENTIMENT_PATH)
    
    # Focus only on content pages, drop low-text covers/charts
    df_content = df_raw[~df_raw['is_low_text']].copy()
    print(f"Analyzing readability and vagueness for {len(df_content)} pages. This may take a moment...")
    
    fog_indices = []
    vagueness_ratios = []
    
    for _, row in df_content.iterrows():
        text = str(row['raw_text'])
        
        if len(text.split()) < 20:
            fog_indices.append(np.nan)
            vagueness_ratios.append(np.nan)
            continue
            
        # Gunning Fog Index estimates the years of formal education needed to understand the text on a first reading.
        # Higher index = Harder to read (often used to obscure lack of real action).
        try:
            fog = textstat.gunning_fog(text)
        except:
            fog = np.nan
            
        ratio = calculate_vagueness_ratio(text)
        
        fog_indices.append(fog)
        vagueness_ratios.append(ratio)
        
    df_content['fog_index'] = fog_indices
    df_content['vagueness_ratio'] = vagueness_ratios
    
    # Merge with sentiment data to see if companies are masking negativity with complex positive buzzwords
    df_merged = pd.merge(
        df_content, 
        df_sentiment[['company', 'year', 'page_num', 'compound', 'major_category']], 
        on=['company', 'year', 'page_num'], 
        how='inner'
    )
    
    # Aggregate to Company Level
    print("Aggregating Greenwashing Risk Metrics by Company...")
    company_risk = df_merged.groupby('company').agg(
        avg_fog_index=('fog_index', 'mean'),
        avg_vagueness=('vagueness_ratio', 'mean'),
        avg_sentiment=('compound', 'mean')
    ).reset_index()
    
    out_csv = DATA_DIR / "greenwashing_risk.csv"
    company_risk.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved Greenwashing metrics to {out_csv}")
    
    # --- Generate Greenwashing Risk Quadrant Plot ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = company_risk['avg_fog_index']
    y = company_risk['avg_vagueness']
    sizes = (company_risk['avg_sentiment'] - company_risk['avg_sentiment'].min() + 0.1) * 1000 # Bubble size = Sentiment positivity
    
    scatter = ax.scatter(x, y, s=sizes, alpha=0.6, c=y, cmap='coolwarm', edgecolors='black', linewidth=1.5)
    
    # Draw quadrant lines based on medians
    median_x = x.median()
    median_y = y.median()
    ax.axvline(median_x, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(median_y, color='gray', linestyle='--', alpha=0.7)
    
    # Add company labels
    for i, txt in enumerate(company_risk['company']):
        ax.annotate(txt, (x[i], y[i]), xytext=(8, 0), textcoords='offset points', fontsize=9, fontweight='bold')
        
    # Annotate Quadrants
    ax.text(x.max(), y.max(), 'High Risk: Fluffy & Complex\n(Potential Greenwashing)', 
            horizontalalignment='right', verticalalignment='top', fontsize=11, color='darkred', alpha=0.8, weight='bold')
    ax.text(x.min(), y.min(), 'Low Risk: Clear & Concrete\n(Transparent Reporting)', 
            horizontalalignment='left', verticalalignment='bottom', fontsize=11, color='darkgreen', alpha=0.8, weight='bold')
            
    ax.set_title("ESG Greenwashing Risk Analysis (Quadrant Map)", fontsize=16, pad=15)
    ax.set_xlabel("Readability Complexity (Gunning Fog Index - Higher is Harder to Read)", fontsize=12)
    ax.set_ylabel("Vagueness Ratio (Buzzwords vs. Concrete Metrics - Higher is Fluffier)", fontsize=12)
    
    # Legend for bubble size
    # Create dummy scatter points for legend
    l1 = plt.scatter([],[], s=100, edgecolors='black', color='gray', alpha=0.6)
    l2 = plt.scatter([],[], s=500, edgecolors='black', color='gray', alpha=0.6)
    ax.legend([l1, l2], ['Lower Sentiment', 'Higher Positive Sentiment'], title="Bubble Size", loc="upper left")
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    out_plot = OUTPUT_DIR / "greenwashing_quadrant.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved Greenwashing Quadrant Map to {out_plot}")
    
    print("\nGreenwashing Pre-warning Analysis Complete!")

if __name__ == "__main__":
    run_greenwashing_analysis()