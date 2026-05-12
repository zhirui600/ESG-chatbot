"""
Step 8: Advanced Data Mining - Network Graph & Clustering (Objective 4)
Input:  data/cleaned_pages.csv, data/report_summary.csv, data/sentiment_by_major_category.csv
Output: output/keyword_network.png, output/company_clusters.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

CLEANED_PAGES_PATH = DATA_DIR / "cleaned_pages.csv"
REPORT_SUM_PATH = DATA_DIR / "report_summary.csv"
SENTIMENT_PATH = DATA_DIR / "sentiment_by_major_category.csv"

# Target ESG keywords for the network
NETWORK_KEYWORDS = [
    'energy', 'employee', 'emission', 'climate', 'carbon', 
    'risk', 'water', 'safety', 'diversity', 'supplier',
    'community', 'waste', 'governance', 'technology'
]

def build_cooccurrence_network():
    """
    Build a network graph showing which ESG keywords frequently appear together.
    """
    print("Building Keyword Co-occurrence Network...")
    df = pd.read_csv(CLEANED_PAGES_PATH)
    content_df = df.dropna(subset=['cleaned_text']).copy()
    
    # Initialize co-occurrence matrix
    co_matrix = pd.DataFrame(0, index=NETWORK_KEYWORDS, columns=NETWORK_KEYWORDS)
    
    for text in content_df['cleaned_text']:
        words = set(str(text).split())
        # Find which target keywords are in this page
        found_kws = [kw for kw in NETWORK_KEYWORDS if kw in words]
        
        # Add 1 to the matrix for every pair found together
        for kw1, kw2 in itertools.combinations(found_kws, 2):
            co_matrix.at[kw1, kw2] += 1
            co_matrix.at[kw2, kw1] += 1
            
    # Create NetworkX Graph
    G = nx.Graph()
    for kw in NETWORK_KEYWORDS:
        G.add_node(kw)
        
    for i in range(len(NETWORK_KEYWORDS)):
        for j in range(i+1, len(NETWORK_KEYWORDS)):
            kw1 = NETWORK_KEYWORDS[i]
            kw2 = NETWORK_KEYWORDS[j]
            weight = co_matrix.at[kw1, kw2]
            if weight > 20: # Threshold to filter out weak connections
                G.add_edge(kw1, kw2, weight=weight)
                
    # Plot the network
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    # Node sizes based on total occurrences (degree)
    degrees = dict(G.degree(weight='weight'))
    node_sizes = [degrees.get(node, 0) * 2 for node in G.nodes()]
    
    # Edge widths based on co-occurrence weight
    edges = G.edges()
    weights = [G[u][v]['weight'] / 50 for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgreen', edgecolors='black', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", font_weight='bold', ax=ax)
    
    ax.set_title("ESG Keyword Co-occurrence Network", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "keyword_network.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved network graph to {out_path}")

def perform_company_clustering():
    """
    Cluster companies based on their ESG topic distribution and sentiment using K-Means.
    """
    print("\nPerforming K-Means Clustering on Company ESG Profiles...")
    
    if not REPORT_SUM_PATH.exists() or not SENTIMENT_PATH.exists():
        print("  Missing required data files. Run Step 4 and Step 6 first.")
        return
        
    df_topics = pd.read_csv(REPORT_SUM_PATH)
    df_sentiment = pd.read_csv(SENTIMENT_PATH)
    
    # Merge features
    df_features = pd.merge(df_topics, df_sentiment, on=['company'], how='inner')
    
    # Select columns for clustering (Percentages and Sentiments)
    topic_cols = [c for c in df_features.columns if c.endswith('_pct')]
    sentiment_cols = [c for c in df_features.columns if c.endswith('_sentiment')]
    feature_cols = topic_cols + sentiment_cols
    
    X = df_features[feature_cols].fillna(0)
    companies = df_features['company'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means (Assuming 3 strategic clusters)
    num_clusters = min(3, len(companies)) 
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Use PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=200, alpha=0.8, edgecolors='black')
    
    # Annotate points with company names
    for i, company in enumerate(companies):
        ax.annotate(company, (X_pca[i, 0], X_pca[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=10)
        
    ax.set_title("Company ESG Strategic Clusters (PCA Projection)", fontsize=14)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    
    # Add a custom legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "company_clusters.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved clustering scatter plot to {out_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_cooccurrence_network()
    perform_company_clustering()
    print("\nAdvanced Data Mining Complete!")

if __name__ == "__main__":
    main()