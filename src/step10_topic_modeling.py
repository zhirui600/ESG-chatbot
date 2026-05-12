"""
Step 10: Topic Modeling (LDA vs NMF)
Input: data/cleaned_pages.csv
Output: data/lda_topic_distributions.csv, output/lda_topic_wordclouds.png, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

def train_models(df_content, n_topics=8):
    print(f"Training LDA and NMF with K={n_topics}...")
    
    # BoW for LDA
    tf_vectorizer = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(df_content['cleaned_text'])
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    
    # TF-IDF for NMF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(df_content['cleaned_text'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
    lda_W = lda.fit_transform(tf)
    
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvd')
    nmf_W = nmf.fit_transform(tfidf)
    
    return lda, lda_W, tf_feature_names, nmf, nmf_W, tfidf_feature_names

def plot_wordclouds(model, feature_names, n_topics, title="LDA Topics"):
    fig, axes = plt.subplots(2, (n_topics+1)//2, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        freqs = {feature: weight for feature, weight in zip(top_features, weights)}
        wc = WordCloud(width=400, height=300, background_color="white", colormap="viridis")
        wc.generate_from_frequencies(freqs)
        
        ax = axes[topic_idx]
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Topic {topic_idx}", fontsize=12)
        ax.axis("off")
        
    for i in range(topic_idx + 1, len(axes)):
        axes[i].axis('off')
        
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "lda_topic_wordclouds.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved wordclouds to {out_path}")

def generate_features(df_content, lda_W, n_topics):
    topic_cols = [f"topic_{i}" for i in range(n_topics)]
    topic_df = pd.DataFrame(lda_W, columns=topic_cols, index=df_content.index)
    
    # Entropy as a measure of topic diversity
    topic_df['topic_entropy'] = -np.sum(lda_W * np.log(lda_W + 1e-9), axis=1)
    topic_df['dominant_topic'] = np.argmax(lda_W, axis=1)
    
    res = pd.concat([df_content[['company', 'year', 'page_num']], topic_df], axis=1)
    out_path = DATA_DIR / "lda_topic_distributions.csv"
    res.to_csv(out_path, index=False)
    print(f"Saved topic distributions to {out_path}")
    return res

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "cleaned_pages.csv")
    df_content = df[df['cleaned_text'].notna() & (df['token_count'] > 10)].copy()
    
    n_topics = 6 # Fixed for simplicity, enough for high level ESG groupings
    lda, lda_W, tf_fn, nmf, nmf_W, tfidf_fn = train_models(df_content, n_topics)
    
    plot_wordclouds(lda, tf_fn, n_topics)
    generate_features(df_content, lda_W, n_topics)
    
    print("Topic Modeling Complete!")

if __name__ == "__main__":
    main()
