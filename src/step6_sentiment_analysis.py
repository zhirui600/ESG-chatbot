"""
Step 6: Advanced Sentiment Analysis for ESG Topics
Input:  data/page_classification.csv, data/cleaned_pages.csv
Output: data/page_sentiment.csv, data/sentiment_by_major_category.csv, output/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path

# Ensure the VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

PAGES_CLS_PATH = DATA_DIR / "page_classification.csv"
CLEANED_PAGES_PATH = DATA_DIR / "cleaned_pages.csv"

# Custom ESG/Business context lexicon adjustments
# Positive values indicate positive sentiment, negative values indicate negative sentiment
ESG_CUSTOM_LEXICON = {
    "risk": 0.0,             
    "mitigate": 1.5,         
    "mitigation": 1.5,
    "sustainable": 2.0,      
    "sustainability": 2.0,
    "emission": -1.0,        
    "pollution": -2.0,       
    "reduce": 1.0,           
    "reduction": 1.0,
    "compliance": 1.0,       
    "violation": -2.5,       
    "diversity": 1.5,        
    "inclusion": 1.5,        
    "transparent": 1.5,      
    "greenwashing": -3.0     
}

def analyze_sentiment():
    """
    Merge text with classification labels and calculate sentiment scores.
    """
    print("Loading text and classification data...")
    cls_df = pd.read_csv(PAGES_CLS_PATH)
    text_df = pd.read_csv(CLEANED_PAGES_PATH)
    
    # Merge classification results with cleaned text
    df = pd.merge(cls_df, text_df[['company', 'year', 'page_num', 'cleaned_text']], 
                  on=['company', 'year', 'page_num'], how='left')
    
    # Filter out pages with insufficient content or unclassified topics
    content_df = df[~df['major_category'].isin(['LowText', 'Unclassified'])].copy()
    print(f"Analyzing sentiment for {len(content_df)} pages using custom ESG lexicon...")

    # Initialize VADER and inject the custom business lexicon
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(ESG_CUSTOM_LEXICON)
    
    sentiments = []
    for _, row in content_df.iterrows():
        text = str(row['cleaned_text'])
        if text.strip() == "" or text == 'nan':
            sentiments.append({'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0})
            continue
            
        # Calculate sentiment polarity scores
        score = sia.polarity_scores(text)
        sentiments.append(score)
        
    sent_df = pd.DataFrame(sentiments, index=content_df.index)
    result_df = pd.concat([content_df, sent_df], axis=1)
    
    # Assign discrete sentiment labels based on compound score
    result_df['sentiment_label'] = result_df['compound'].apply(
        lambda x: 'Positive' if x >= 0.1 else ('Negative' if x <= -0.1 else 'Neutral')
    )

    # Save the granular page-level sentiment data
    out_page = DATA_DIR / "page_sentiment.csv"
    result_df.to_csv(out_page, index=False, encoding="utf-8-sig")
    print(f"Saved page-level sentiments to {out_page}")
    
    return result_df

def generate_insights_and_plots(result_df):
    """
    Aggregate sentiment data and generate visualizations.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating advanced sentiment cross-analysis visualizations...")

    # --- Aggregation 1: Average Sentiment by Company and Major Category ---
    major_summary = result_df.groupby(['company', 'major_category'])['compound'].mean().unstack(fill_value=0)
    
    out_summary1 = DATA_DIR / "sentiment_by_major_category.csv"
    major_summary.to_csv(out_summary1, encoding="utf-8-sig")
    
    # Visualization: Bar chart for Major Categories
    fig, ax = plt.subplots(figsize=(12, 6))
    major_summary.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
    ax.set_title('Average Sentiment Score by ESG Pillar & Company', fontsize=14)
    ax.set_ylabel('Average Compound Sentiment Score (-1 to 1)')
    ax.set_xlabel('Company')
    plt.xticks(rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=1) 
    plt.legend(title='ESG Pillar')
    plt.tight_layout()
    
    plot_path1 = OUTPUT_DIR / "sentiment_bar_major.png"
    fig.savefig(plot_path1, dpi=150)
    plt.close(fig)

    # --- Aggregation 2: Sentiment Heatmap for Sub-categories ---
    top_subs = result_df['sub_category'].value_counts().head(10).index
    sub_df = result_df[result_df['sub_category'].isin(top_subs)]
    
    sub_summary = sub_df.groupby(['company', 'sub_category'])['compound'].mean().unstack(fill_value=np.nan)
    
    # Visualization: Heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    im = ax2.imshow(sub_summary.values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.9)
    
    ax2.set_xticks(range(len(sub_summary.columns)))
    ax2.set_xticklabels(sub_summary.columns, rotation=45, ha="right", fontsize=10)
    ax2.set_yticks(range(len(sub_summary.index)))
    ax2.set_yticklabels(sub_summary.index, fontsize=10)
    ax2.set_title("ESG Sub-topic Sentiment Heatmap by Company (Green=Positive, Red=Negative)", fontsize=14)
    
    fig2.colorbar(im, ax=ax2, label="Sentiment Score")
    plt.tight_layout()
    
    plot_path2 = OUTPUT_DIR / "sentiment_heatmap_subtopics.png"
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)

    print("Saved advanced visualizations to output/ directory.")

def main():
    result_df = analyze_sentiment()
    generate_insights_and_plots(result_df)
    print("\nSentiment Analysis Complete!")

if __name__ == "__main__":
    main()