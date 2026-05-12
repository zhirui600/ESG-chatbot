"""
Step 17: Chatbot Knowledge Base Builder
- Auto-generate structured Q&A pairs from all analysis outputs
- Create searchable ESG fact sheets per company
- Build the knowledge base that the chatbot (Obj 6) will query

Input:  All previous step outputs
Output: data/chatbot_knowledge_base.json, data/chatbot_qa_pairs.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build_qa_pairs(df_feat, df_gaps, df_fb, df_commit, df_sent, df_bench):
    """Auto-generate question-answer pairs for the chatbot."""
    print("Generating Q&A pairs for chatbot knowledge base...")

    qa_pairs = []
    companies = df_feat['company'].unique()

    for company in companies:
        feat = df_feat[df_feat['company'] == company].iloc[0]

        # ── Basic ESG Profile Questions ──
        qa_pairs.append({
            'company': company,
            'category': 'overview',
            'question': f"What is {company}'s ESG profile?",
            'answer': (
                f"{company}'s ESG report allocates approximately "
                f"{feat['env_score']:.1f}% to Environmental topics, "
                f"{feat['soc_score']:.1f}% to Social topics, and "
                f"{feat['gov_score']:.1f}% to Governance topics."
            )
        })

        # ── Environmental Focus ──
        qa_pairs.append({
            'company': company,
            'category': 'environmental',
            'question': f"How much does {company} focus on environmental issues?",
            'answer': (
                f"Environmental topics account for {feat['env_score']:.1f}% of {company}'s "
                f"ESG report content. The average sentiment toward environmental issues is "
                f"{'positive' if feat.get('sent_env', 0) > 0.5 else 'neutral to negative'} "
                f"(score: {feat.get('sent_env', 0):.2f})."
            )
        })

        # ── Disclosure Quality ──
        qa_pairs.append({
            'company': company,
            'category': 'transparency',
            'question': f"How transparent is {company}'s ESG reporting?",
            'answer': (
                f"{company}'s report has a readability complexity (Gunning Fog) of "
                f"{feat.get('avg_fog_index', 0):.1f} and a vagueness ratio of "
                f"{feat.get('avg_vagueness', 0):.2f}. "
                f"{'Higher complexity and vagueness may indicate less transparent reporting.' if feat.get('avg_vagueness', 0) > 2 else 'The reporting is relatively clear and concrete.'}"
            )
        })

        # ── Disclosure Gaps ──
        if df_gaps is not None:
            gap_row = df_gaps[df_gaps['company'] == company]
            if len(gap_row) > 0:
                gap = gap_row.iloc[0]
                qa_pairs.append({
                    'company': company,
                    'category': 'gaps',
                    'question': f"What ESG topics does {company} not cover well?",
                    'answer': (
                        f"{company} covers {gap['topics_covered']} out of {gap['topics_covered'] + gap['topics_missing']} "
                        f"ESG sub-topics (breadth: {gap['breadth_score']:.1f}%). "
                        f"Missing topics include: {str(gap['missing_topics'])[:200] if pd.notna(gap['missing_topics']) and gap['missing_topics'] else 'None - comprehensive coverage'}."
                    )
                })

                qa_pairs.append({
                    'company': company,
                    'category': 'recommendation',
                    'question': f"What should {company} improve in its ESG reporting?",
                    'answer': gap['recommendation']
                })

        # ── Forward-looking Commitments ──
        if df_fb is not None:
            fb_row = df_fb[df_fb['company'] == company]
            if len(fb_row) > 0:
                fb = fb_row.iloc[0]
                fb_label = 'forward-looking (commitment-oriented)' if fb['fwd_bwd_balance'] > 0.5 else 'backward-looking (achievement-focused)'
                qa_pairs.append({
                    'company': company,
                    'category': 'language',
                    'question': f"Is {company}'s ESG report more about commitments or achievements?",
                    'answer': (
                        f"{company}'s ESG language is primarily {fb_label}. "
                        f"The forward-to-backward balance ratio is {fb['fwd_bwd_balance']:.2f} "
                        f"(1.0 = all forward-looking, 0.0 = all backward-looking)."
                    )
                })

        # ── Specific Commitments ──
        if df_commit is not None and len(df_commit) > 0:
            company_commits = df_commit[df_commit['company'] == company]
            if len(company_commits) > 0:
                commit_texts = []
                for _, c in company_commits.head(5).iterrows():
                    ctx = str(c.get('context', ''))[:150]
                    commit_texts.append(f"- {ctx}")
                qa_pairs.append({
                    'company': company,
                    'category': 'commitments',
                    'question': f"What are {company}'s key ESG commitments?",
                    'answer': (
                        f"{company} has made {len(company_commits)} identified ESG commitments/targets. "
                        f"Key ones include:\n" + '\n'.join(commit_texts)
                    )
                })

    # ── Cross-company Comparison Questions ──
    top_env = df_feat.nlargest(3, 'env_score')['company'].tolist()
    top_soc = df_feat.nlargest(3, 'soc_score')['company'].tolist()
    top_gov = df_feat.nlargest(3, 'gov_score')['company'].tolist()

    qa_pairs.append({
        'company': 'ALL',
        'category': 'comparison',
        'question': "Which companies have the strongest environmental focus?",
        'answer': f"The top 3 companies by environmental coverage are: {', '.join(top_env)}."
    })
    qa_pairs.append({
        'company': 'ALL',
        'category': 'comparison',
        'question': "Which companies have the strongest social responsibility focus?",
        'answer': f"The top 3 companies by social coverage are: {', '.join(top_soc)}."
    })
    qa_pairs.append({
        'company': 'ALL',
        'category': 'comparison',
        'question': "Which companies have the strongest governance?",
        'answer': f"The top 3 companies by governance coverage are: {', '.join(top_gov)}."
    })

    # Overall ranking
    top_overall = df_feat.nlargest(3, 'composite_esg_score')
    qa_pairs.append({
        'company': 'ALL',
        'category': 'ranking',
        'question': "Which company has the best overall ESG score?",
        'answer': (
            f"Based on our composite ESG scoring model, the top 3 companies are: "
            + ', '.join([f"{r['company']} ({r['composite_esg_score']:.1f})" for _, r in top_overall.iterrows()])
            + "."
        )
    })

    return qa_pairs


def build_knowledge_base(qa_pairs, profiles):
    """Combine Q&A pairs and profiles into a single knowledge base."""
    print("Building unified knowledge base...")

    kb = {
        'metadata': {
            'version': '1.0',
            'companies_analyzed': len(profiles),
            'total_qa_pairs': len(qa_pairs),
            'data_sources': [
                'ESG annual reports (10 companies)',
                'TF-IDF keyword analysis',
                'KeyBERT semantic keywords',
                'LDA topic modeling',
                'BERT embeddings & clustering',
                'VADER sentiment analysis',
                'Association rule mining',
                'Disclosure gap analysis',
                'Forward/backward language analysis',
                'Quantitative commitment extraction'
            ]
        },
        'company_profiles': profiles,
        'qa_pairs': qa_pairs,
    }

    return kb


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

    df_sent = None
    if (DATA_DIR / "sentiment_by_major_category.csv").exists():
        df_sent = pd.read_csv(DATA_DIR / "sentiment_by_major_category.csv")

    df_bench = None
    if (DATA_DIR / "esg_benchmarks.csv").exists():
        df_bench = pd.read_csv(DATA_DIR / "esg_benchmarks.csv")

    # Load company profiles if available
    profiles = {}
    if (DATA_DIR / "company_profiles.json").exists():
        with open(DATA_DIR / "company_profiles.json", 'r', encoding='utf-8') as f:
            profiles = json.load(f)

    # Generate Q&A pairs
    qa_pairs = build_qa_pairs(df_feat, df_gaps, df_fb, df_commit, df_sent, df_bench)

    # Save Q&A as CSV for easy review
    qa_df = pd.DataFrame(qa_pairs)
    qa_df.to_csv(DATA_DIR / "chatbot_qa_pairs.csv", index=False, encoding='utf-8-sig')
    print(f"  Saved {len(qa_pairs)} Q&A pairs → data/chatbot_qa_pairs.csv")

    # Build full knowledge base
    kb = build_knowledge_base(qa_pairs, profiles)
    with open(DATA_DIR / "chatbot_knowledge_base.json", 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"  Saved knowledge base → data/chatbot_knowledge_base.json")

    # Print sample Q&A
    print("\n=== Sample Q&A Pairs for Chatbot ===")
    for qa in qa_pairs[:6]:
        print(f"\n  Q: {qa['question']}")
        print(f"  A: {qa['answer'][:200]}...")

    print(f"\n\nTotal Q&A pairs: {len(qa_pairs)}")
    print(f"Q&A categories: {qa_df['category'].value_counts().to_dict()}")
    print("\nStep 17 Complete!")


if __name__ == "__main__":
    main()
