"""
Step 11: BERT Embeddings & Clustering Comparison
Input: data/cleaned_pages.csv
Output: Page embeddings, UMAP plots, clustering metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
from sklearn.cluster import KMeans, AgglomerativeClustering
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

def encode_text(df_content):
    print("Encoding texts with Sentence-BERT (this takes a moment on GPU)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_content['cleaned_text'].tolist(), show_progress_bar=True)
    np.save(DATA_DIR / "page_embeddings.npy", embeddings)
    return embeddings

def evaluate_clustering(embeddings):
    print("Evaluating K-Means, Agglomerative, and HDBSCAN...")
    metrics = []
    
    # KMeans
    km = KMeans(n_clusters=6, random_state=42)
    km_labels = km.fit_predict(embeddings)
    metrics.append({
        'Method': 'K-Means (k=6)',
        'Silhouette': silhouette_score(embeddings, km_labels),
        'Calinski-Harabasz': calinski_harabasz_score(embeddings, km_labels),
        'Davies-Bouldin': davies_bouldin_score(embeddings, km_labels),
        'Labels': km_labels
    })
    
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=6)
    agg_labels = agg.fit_predict(embeddings)
    metrics.append({
        'Method': 'Agglomerative (k=6)',
        'Silhouette': silhouette_score(embeddings, agg_labels),
        'Calinski-Harabasz': calinski_harabasz_score(embeddings, agg_labels),
        'Davies-Bouldin': davies_bouldin_score(embeddings, agg_labels),
        'Labels': agg_labels
    })

    # HDBSCAN
    hdb = hdbscan.HDBSCAN(min_cluster_size=15)
    hdb_labels = hdb.fit_predict(embeddings)
    # Exclude noise for metrics if possible
    mask = hdb_labels != -1
    if mask.sum() > 0 and len(np.unique(hdb_labels[mask])) > 1:
        metrics.append({
            'Method': 'HDBSCAN',
            'Silhouette': silhouette_score(embeddings[mask], hdb_labels[mask]),
            'Calinski-Harabasz': calinski_harabasz_score(embeddings[mask], hdb_labels[mask]),
            'Davies-Bouldin': davies_bouldin_score(embeddings[mask], hdb_labels[mask]),
            'Labels': hdb_labels
        })
    
    df_metrics = pd.DataFrame([{k:v for k,v in m.items() if k!='Labels'} for m in metrics])
    
    # Plot metrics
    df_metrics.set_index('Method').plot(kind='bar', subplots=True, layout=(1,3), figsize=(15,4), title="Clustering Metrics")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clustering_metrics.png", dpi=150)
    plt.close()
    
    return metrics

def plot_umap(embeddings, labels, title, filename):
    print(f"Generating UMAP for {title}...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter)
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    
def company_similarity(df_content, embeddings):
    df_content['emb_idx'] = range(len(embeddings))
    comp_embs = []
    companies = df_content['company'].unique()
    
    for c in companies:
        idx = df_content[df_content['company'] == c]['emb_idx'].values
        comp_embs.append(np.mean(embeddings[idx], axis=0))
        
    sim_matrix = cosine_similarity(comp_embs)
    sim_df = pd.DataFrame(sim_matrix, index=companies, columns=companies)
    sim_df.to_csv(DATA_DIR / "company_similarity_matrix.csv")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, annot=True, cmap="YlGnBu", fmt='.2f')
    plt.title("Company Semantic Similarity (Cosine)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "company_similarity_heatmap.png", dpi=150)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / "cleaned_pages.csv")
    df_content = df[df['cleaned_text'].notna() & (df['token_count'] > 10)].copy()
    
    embeddings = encode_text(df_content)
    metrics = evaluate_clustering(embeddings)
    
    # Save best labels (K-Means) for feature matrix
    best_labels = metrics[0]['Labels'] 
    df_content['bert_cluster'] = best_labels
    df_content[['company', 'year', 'page_num', 'bert_cluster']].to_csv(DATA_DIR / "bert_clusters.csv", index=False)
    
    plot_umap(embeddings, best_labels, "UMAP with K-Means Clusters", "umap_kmeans.png")
    company_similarity(df_content, embeddings)
    
    print("BERT Clustering Complete!")

if __name__ == "__main__":
    main()
