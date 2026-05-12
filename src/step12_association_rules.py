"""
Step 12: Association Rule Mining (Apriori)
Input: data/page_classification.csv, data/keywords_tfidf.csv
Output: data/association_rules.csv, output/association_rules_scatter.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

def prepare_transactions(df_cls, df_kw):
    df = pd.merge(df_cls[['company', 'year', 'page_num', 'all_matches']], 
                  df_kw[['company', 'year', 'page_num', 'tfidf_keywords']],
                  on=['company', 'year', 'page_num'], how='inner')
    
    transactions = []
    for _, row in df.iterrows():
        t = []
        # Add subcategories
        matches = str(row['all_matches'])
        if matches != 'nan' and matches != '':
            for m in matches.split('; '):
                cat = m.split('(')[0].strip()
                t.append(f"CAT:{cat}")
                
        # Add top keywords
        kws = str(row['tfidf_keywords'])
        if kws != 'nan':
            for kw in kws.split('; ')[:3]: # Take top 3 keywords to avoid explosion
                t.append(f"KW:{kw}")
        
        if len(t) > 0:
            transactions.append(t)
            
    return transactions

def run_apriori(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)
    if len(frequent_itemsets) == 0:
        print("No frequent itemsets found.")
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    rules = rules[rules['lift'] > 1.2]
    
    # Store rules cleanly
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    rules.to_csv(DATA_DIR / "association_rules.csv", index=False)
    print(f"Found {len(rules)} valid rules.")
    return rules

def plot_rules(rules):
    if len(rules) == 0: return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', 
                s=rules['lift']*50, alpha=0.7, edgecolors='k')
    plt.colorbar(label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules (Bubble Size/Color = Lift)')
    plt.savefig(OUTPUT_DIR / "association_rules_scatter.png", dpi=150)
    plt.close()
    print("Saved association_rules_scatter.png")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data for Association Rules...")
    df_cls = pd.read_csv(DATA_DIR / "page_classification.csv")
    df_kw = pd.read_csv(DATA_DIR / "keywords_tfidf.csv")
    
    transactions = prepare_transactions(df_cls, df_kw)
    rules = run_apriori(transactions)
    plot_rules(rules)
    
    print("Association Rule Mining Complete!")

if __name__ == "__main__":
    main()
