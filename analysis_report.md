# ESG Text Analysis — Detailed Analysis Report

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data & Preprocessing](#2-data--preprocessing)
3. [Objective 1: Keyword Extraction](#3-objective-1-keyword-extraction)
4. [Objective 2: Topic Classification](#4-objective-2-topic-classification)
5. [Objective 3: Sentiment Analysis](#5-objective-3-sentiment-analysis)
6. [Objective 4: Advanced Data Mining & Text Analysis](#6-objective-4-advanced-data-mining--text-analysis)
   - 6.1 [LDA Topic Modeling & NMF Comparison](#61-lda-topic-modeling--nmf-comparison)
   - 6.2 [BERT Semantic Embeddings & Multi-Method Clustering Comparison](#62-bert-semantic-embeddings--multi-method-clustering-comparison)
   - 6.3 [Association Rule Mining (Apriori)](#63-association-rule-mining-apriori)
   - 6.4 [ESG Commitment & Target Extraction](#64-esg-commitment--target-extraction)
   - 6.5 [ESG Disclosure Gap Analysis](#65-esg-disclosure-gap-analysis)
   - 6.6 [Peer Benchmarking & Company Profiling](#66-peer-benchmarking--company-profiling)
   - 6.7 [Feature Engineering & Composite Scoring](#67-feature-engineering--composite-scoring)
   - 6.8 [Chatbot Knowledge Base Construction](#68-chatbot-knowledge-base-construction)
7. [Model Comparison Summary](#7-model-comparison-summary)
8. [Key Findings & Insights](#8-key-findings--insights)

---

## 1. Project Overview

This project analyzes ESG (Environmental, Social, and Governance) annual reports from 10 major companies using a multi-stage text mining and data mining pipeline. The goal is to extract actionable ESG insights, compare companies, and build a structured knowledge base for an ESG chatbot.

**Companies analyzed**: Adobe, Amazon, Apple, Broadcom, Caterpillar, Deere & Co, General Electric, Honeywell, NVIDIA, United Parcel Service.

**Data scale**: 672 pages extracted, 633 content pages analyzed, ~5.2 MB of cleaned text.

---

## 2. Data & Preprocessing

### 2.1 PDF Text Extraction (Step 1)

- **Tool**: `pdfplumber` library
- **Process**: Page-by-page extraction from 10 PDF reports
- **Output**: 672 raw text pages with metadata (company, year, page number, word count)
- **Low-text filtering**: Pages with < 50 words flagged as charts/covers (39 pages, 5.8%)

### 2.2 Text Cleaning & NLP Preprocessing (Step 2)

- **Tool**: `spaCy` (en_core_web_sm model)
- **Pipeline**:
  1. Remove CID patterns (embedded font artifacts), URLs, emails
  2. Lowercasing
  3. Tokenization with spaCy
  4. Stop word removal (English + custom domain stop words including company names)
  5. Lemmatization
- **Result**: Average 269 tokens per content page after preprocessing

---

## 3. Objective 1: Keyword Extraction

### 3.1 TF-IDF Keyword Extraction (Step 3)

- **Model**: `TfidfVectorizer` from scikit-learn
- **Parameters**: ngram_range=(1,3), max_features=5000, min_df=2, max_df=0.85
- **Output**: Top 15 keywords per page with TF-IDF scores

### 3.2 KeyBERT Semantic Keyword Extraction (Step 3)

- **Model**: `KeyBERT` with default Sentence-BERT backbone (all-MiniLM-L6-v2)
- **Parameters**: keyphrase_ngram_range=(1,2), top_n=10
- **Advantage over TF-IDF**: Captures semantic meaning, not just frequency

### 3.3 Top Keywords Result

The top 10 keywords across all reports:

| Rank | Keyword | Page Frequency |
|------|---------|----------------|
| 1 | energy | 318 |
| 2 | employee | 287 |
| 3 | emission | 245 |
| 4 | climate | 221 |
| 5 | carbon | 198 |
| 6 | supplier | 176 |
| 7 | water | 165 |
| 8 | safety | 159 |
| 9 | renewable | 143 |
| 10 | sustainability | 138 |

### 3.4 ESG Performance Metric Detection

- **Method**: Dictionary-based matching against 13 ESG sub-categories with 100+ curated keywords
- **Result**: ESG terms identified per page, enabling structured classification

---

## 4. Objective 2: Topic Classification

### 4.1 Rule-Based ESG Classification (Step 4)

- **Method**: Keyword taxonomy matching
- **Taxonomy**: 3 major categories → 13 sub-categories → 100+ keywords defined in `esg_keywords.json`
- **Assignment logic**: Each page is assigned to the category with the highest keyword hit count; ties broken by taxonomy priority

### 4.2 Classification Results

| Category | Pages | Percentage |
|----------|-------|------------|
| Environmental | 320 | 50.6% |
| Social | 259 | 40.9% |
| Governance | 54 | 8.5% |

**Top 5 Sub-categories**:
1. Human Capital — 159 pages
2. Climate Change — 154 pages
3. Energy Management — 102 pages
4. Community & Social Impact — 42 pages
5. Board & Corporate Governance — 39 pages

### 4.3 Company-Level Distribution

| Company | Env % | Social % | Gov % | Content Pages |
|---------|-------|----------|-------|---------------|
| Apple Inc | 77.5 | 18.0 | 0.9 | 111 |
| General Electric Co | 59.5 | 33.9 | 5.0 | 121 |
| Deere & Co | 47.7 | 45.5 | 2.3 | 44 |
| Amazon.com Inc | 46.4 | 48.2 | 3.6 | 56 |
| Caterpillar Inc | 44.9 | 31.5 | 21.3 | 89 |
| NVIDIA Corp | 41.5 | 46.3 | 9.8 | 41 |
| Adobe Inc | 23.6 | 63.6 | 10.9 | 55 |
| Honeywell International Inc | 35.7 | 48.6 | 14.3 | 70 |
| Broadcom Inc | 32.1 | 57.1 | 8.9 | 56 |
| United Parcel Service Inc | 40.0 | 60.0 | 0.0 | 5 |

**Insight**: Apple's report is overwhelmingly Environmental-focused (77.5%), reflecting its strong climate/sustainability brand. Adobe and Broadcom prioritize Social topics (employee/diversity), which aligns with their tech workforce focus. Caterpillar stands out with the highest Governance coverage (21.3%).

---

## 5. Objective 3: Sentiment Analysis

### 5.1 VADER Sentiment Analysis with Custom ESG Lexicon (Step 6)

- **Model**: VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK
- **Customization**: Injected domain-specific sentiment adjustments:
  - Positive: "sustainable" (+2.0), "mitigation" (+1.5), "diversity" (+1.5)
  - Negative: "pollution" (−2.0), "violation" (−2.5), "greenwashing" (−3.0)
  - Neutral override: "risk" set to 0.0 (business-neutral in ESG context)
- **Output**: Compound score (−1 to +1) per page, with discrete labels (Positive/Neutral/Negative)

### 5.2 Sentiment by ESG Pillar

| Company | Env Sentiment | Social Sentiment | Gov Sentiment |
|---------|--------------|-----------------|---------------|
| Amazon.com Inc | 0.988 | 0.996 | 0.999 |
| Deere & Co | 0.994 | 0.897 | 0.997 |
| Apple Inc | 0.912 | 0.955 | 0.996 |
| NVIDIA Corp | 0.877 | 0.992 | 0.986 |
| General Electric Co | 0.957 | 0.984 | 0.993 |
| Caterpillar Inc | 0.887 | 0.940 | 0.971 |
| Honeywell International Inc | 0.847 | 0.964 | 0.893 |
| Adobe Inc | 0.686 | 0.950 | 0.925 |
| Broadcom Inc | 0.547 | 0.883 | 0.787 |
| United Parcel Service Inc | 0.982 | 0.872 | 0.000 |

**Insight**: All ESG reports skew strongly positive (self-reporting bias). Broadcom has the lowest Environmental sentiment (0.547), potentially indicating more candid risk disclosure. UPS has 0.000 Governance sentiment due to lack of governance content in its short brochure.

### 5.3 Keyword-Sentiment Correlation (Step 7)

- **Method**: Pearson correlation between TF-IDF keyword scores and VADER compound scores
- **Key finding**: "Emission" has the strongest negative correlation with sentiment (r = −0.255), meaning pages heavy on emission data tend to be more neutral/negative — they contain factual GHG data rather than promotional language

### 5.4 Greenwashing Risk Detection (Step 9)

- **Method**: Gunning Fog readability index + Vagueness ratio (buzzwords vs. concrete metrics)
- **Greenwashing risk metric**: A company with high readability complexity AND high vagueness ratio is at higher risk of greenwashing

| Company | Fog Index | Vagueness | Risk Level |
|---------|-----------|-----------|------------|
| Caterpillar Inc | 23.6 | 2.82 | ⬆ Higher |
| Apple Inc | 31.8 | 2.62 | ⬆ Higher |
| General Electric Co | 20.6 | 2.75 | ⬆ Higher |
| NVIDIA Corp | 20.1 | 2.40 | Medium |
| Honeywell International Inc | 26.2 | 2.39 | Medium |
| Amazon.com Inc | 20.5 | 1.79 | Medium |
| Broadcom Inc | 26.8 | 1.59 | Lower |
| Deere & Co | 19.6 | 1.55 | Lower |
| Adobe Inc | 21.6 | 1.29 | Lower |
| United Parcel Service Inc | 16.2 | 0.31 | ✅ Lowest |

**Insight**: Apple's report has the highest Fog Index (31.8), meaning it is the most complex to read. Combined with a high vagueness ratio (2.62), this suggests their environmental narrative may be more aspirational than concrete. UPS, with the shortest report, has the clearest and most concrete language.

---

## 6. Objective 4: Advanced Data Mining & Text Analysis

### 6.1 LDA Topic Modeling & NMF Comparison

#### Method

- **LDA (Latent Dirichlet Allocation)**: A probabilistic generative model that represents documents as mixtures of topics, where each topic is characterized by a probability distribution over words. Unlike the rule-based classifier (Step 4), LDA discovers topics from the data itself without predefined categories.
  - Implementation: `sklearn.decomposition.LatentDirichletAllocation`
  - Input: Bag-of-Words matrix from `CountVectorizer` (max_df=0.85, min_df=2)
  - K = 6 topics

- **NMF (Non-negative Matrix Factorization)**: A linear algebra approach that factorizes the TF-IDF matrix into two non-negative matrices (W and H), where columns of H represent topics. NMF tends to produce more interpretable and sparser topic representations.
  - Implementation: `sklearn.decomposition.NMF` with `init='nndsvd'`
  - Input: TF-IDF matrix

#### Results: LDA Topic Discovery

The 6 discovered topics align well with key ESG themes:

**Company × Topic Distribution (average probability)**:

| Company | Topic 0 | Topic 1 | Topic 2 | Topic 3 | Topic 4 | Topic 5 |
|---------|---------|---------|---------|---------|---------|---------|
| Adobe Inc | **0.471** | 0.131 | 0.053 | 0.266 | 0.059 | 0.019 |
| Amazon.com Inc | 0.060 | 0.219 | 0.094 | 0.034 | **0.505** | 0.088 |
| Apple Inc | 0.038 | 0.053 | 0.065 | 0.089 | **0.737** | 0.019 |
| Broadcom Inc | 0.244 | **0.379** | 0.044 | 0.258 | 0.055 | 0.020 |
| Caterpillar Inc | 0.117 | 0.274 | 0.065 | 0.242 | 0.026 | **0.275** |
| Deere & Co | 0.026 | 0.060 | **0.632** | 0.041 | 0.050 | 0.191 |
| General Electric Co | **0.462** | 0.165 | 0.042 | 0.051 | 0.046 | 0.235 |
| Honeywell International Inc | 0.158 | **0.460** | 0.036 | 0.211 | 0.077 | 0.059 |
| NVIDIA Corp | 0.163 | 0.227 | 0.118 | **0.329** | 0.132 | 0.032 |
| United Parcel Service Inc | **0.374** | 0.001 | 0.086 | 0.167 | 0.162 | 0.209 |

**Insight**: The LDA model successfully differentiated companies by their dominant ESG narrative:
- **Topic 4** (Environment/Climate) dominates Apple (73.7%) and Amazon (50.5%)
- **Topic 0** (Corporate Social Impact) dominates Adobe (47.1%) and GE (46.2%)
- **Topic 2** (Agriculture/Industry) uniquely dominates Deere & Co (63.2%), reflecting its agricultural sector focus
- **Topic 1** (Workforce/Governance) is dominant for Honeywell (46.0%) and Broadcom (37.9%)

#### LDA vs NMF Comparison

| Aspect | LDA | NMF |
|--------|-----|-----|
| **Model Type** | Probabilistic (Bayesian) | Matrix factorization |
| **Interpretability** | Moderate (soft assignments) | Higher (sparser) |
| **Handles polysemy** | Better (probabilistic) | Weaker (linear) |
| **Speed** | Slower | Faster |
| **Output** | Probability distributions | Weight vectors |
| **Best suited for** | Discovering latent themes | Clear topic separation |

Both models were trained with K=6 and produced largely consistent topic groupings. LDA was selected as the primary model for downstream features due to its probabilistic nature, which provides more meaningful topic diversity (entropy) measures.

---

### 6.2 BERT Semantic Embeddings & Multi-Method Clustering Comparison

#### Method

**Stage 1: Document Encoding**
- **Model**: Sentence-BERT (`all-MiniLM-L6-v2`) — a fine-tuned BERT model optimized for semantic similarity tasks
- **Output**: 384-dimensional dense vector per page (633 pages → 633×384 matrix)
- **Advantage over BoW/TF-IDF**: Captures contextual meaning, synonyms, and semantic relationships

**Stage 2: Dimensionality Reduction**
- **UMAP** (Uniform Manifold Approximation and Projection): Non-linear dimensionality reduction for 2D visualization
  - Parameters: n_components=2, random_state=42

**Stage 3: Clustering Comparison — 3 algorithms compared**:

| Algorithm | Type | Key Property | Parameters |
|-----------|------|-------------|------------|
| **K-Means** | Centroid-based (course content) | Requires predefined K, assumes spherical clusters | k=6, random_state=42 |
| **HDBSCAN** | Density-based (beyond course) | Finds clusters of varying density, identifies noise points | min_cluster_size=15 |
| **Agglomerative** | Hierarchical (course content) | Builds tree of merges, produces dendrogram | n_clusters=6 |

#### Clustering Evaluation Results

| Metric | K-Means | Agglomerative | HDBSCAN |
|--------|---------|---------------|---------|
| **Silhouette Score** ↑ | Best | Second | Third |
| **Calinski-Harabasz** ↑ | Best | Second | Third |
| **Davies-Bouldin** ↓ | Best | Second | Third |

**K-Means performed best** across all three metrics on BERT embeddings. This is expected because BERT embeddings naturally form roughly spherical clusters in high-dimensional space, matching K-Means' assumptions.

**HDBSCAN's value**: While scoring lowest on metrics, HDBSCAN identified 23 noise points (pages that don't fit any cluster), which correspond to transitional/mixed-topic pages. This is an advantage that K-Means and Agglomerative cannot provide.

#### BERT Cluster Distribution by Company

| Company | C0 | C1 | C2 | C3 | C4 | C5 |
|---------|----|----|----|----|----|----|
| Apple Inc | 0 | 0 | 1 | 4 | 39 | **66** |
| Caterpillar Inc | 1 | **78** | 1 | 6 | 2 | 0 |
| General Electric Co | 10 | 1 | 32 | 18 | **59** | 0 |
| Adobe Inc | **34** | 0 | 6 | 6 | 7 | 2 |
| Amazon.com Inc | 0 | 0 | 12 | 9 | 19 | **15** |

**Insight**: BERT clustering reveals semantic cohesion within industries:
- **Cluster 5** is dominated by Apple (66 pages) — their heavily environment-focused report forms a distinct semantic space
- **Cluster 1** is almost entirely Caterpillar (78 pages) — industrial/manufacturing language is semantically unique
- **Cluster 0** maps to corporate social responsibility / employee-focused content (Adobe dominant)

#### Company Semantic Similarity (Cosine Similarity)

Top 3 most similar pairs:
1. **Amazon ↔ Apple**: 0.922 (both tech companies with strong environmental narratives)
2. **GE ↔ Honeywell**: 0.891 (both industrial conglomerates)
3. **NVIDIA ↔ Amazon**: 0.881 (tech sector similar ESG language)

Least similar: **Adobe ↔ Deere**: 0.724 (tech creative vs. agriculture — very different ESG concerns)

---

### 6.3 Association Rule Mining (Apriori)

#### Method

- **Algorithm**: Apriori from `mlxtend`
- **Transaction definition**: Each page is a "basket" containing the ESG sub-categories and top-3 TF-IDF keywords present on that page
- **Parameters**: min_support=0.05, min_confidence=0.3
- **Filtering**: Rules with lift > 1.2 retained

#### Key Metrics Explained

| Metric | Meaning |
|--------|---------|
| **Support** | How frequently the items appear together (proportion of all pages) |
| **Confidence** | Given antecedent appears, how often consequent appears (conditional probability) |
| **Lift** | How much more likely items appear together vs. independently (>1 = positive association) |

#### Results

- **Total rules discovered**: 16,712
- **Rules with lift > 5**: 287 (very strong associations)

**Top association patterns** (simplified, highest-lift single-antecedent rules):

| When a page discusses... | It also discusses... | Confidence | Lift |
|--------------------------|---------------------|------------|------|
| Biodiversity | Waste Management + Water Management + D&I + Community Impact | 43.1% | 5.01 |
| Biodiversity | Supply Chain + Water + D&I + Waste | 43.1% | 4.84 |

**Business Insight**: Biodiversity pages are the strongest "connector" topic — whenever a company discusses biodiversity, they almost always also address water, waste, supply chain, and community impact in the same page. This suggests biodiversity is treated as a cross-cutting theme rather than a standalone topic. Companies that lack biodiversity coverage may be missing this important integrative lens.

---

### 6.4 ESG Commitment & Target Extraction

#### Method

- **Approach**: Regex-based Named Entity Recognition tailored to ESG commitments
- **4 extraction patterns**:
  1. `"reduce/achieve X% by YEAR"` — action + value + target year
  2. `"X% reduction by YEAR"` — value + action + target year
  3. `"net zero / carbon neutral by YEAR"` — categorical targets
  4. `"target/goal of X by YEAR"` — stated targets
- **Forward/Backward Language Analysis**: Lexicon-based scoring of forward-looking (commitment) vs backward-looking (achievement) language

#### Commitment Extraction Results

**Total commitments extracted: 74**

| Company | Commitments | Most Common Target |
|---------|-------------|-------------------|
| General Electric Co | 33 | 2050 |
| Amazon.com Inc | 12 | 2030 |
| Apple Inc | 10 | 2030 |
| Honeywell International Inc | 6 | 2035 |
| Adobe Inc | 5 | 2025 |
| United Parcel Service Inc | 3 | 2050 |
| Broadcom Inc | 3 | 2050 |
| Caterpillar Inc | 1 | 2030 |
| Deere & Co | 1 | 2030 |
| NVIDIA Corp | 0 | — |

**Target Year Distribution**: 2050 (31 mentions) > 2030 (17) > 2035 (6)

**Insight**: GE has by far the most explicit commitments (33), reflecting its detailed 2022 Sustainability Report. NVIDIA, despite being a major tech company, has zero extracted quantitative commitments — its report focuses on qualitative narratives rather than measurable targets. This is a significant finding for ESG evaluation.

#### Forward-Looking vs Backward-Looking Language

| Company | Forward % | Backward % | Balance |
|---------|-----------|------------|---------|
| Honeywell International Inc | 62.6% | 11.9% | 0.840 |
| General Electric Co | 70.5% | 16.7% | 0.826 |
| NVIDIA Corp | 50.8% | 18.6% | 0.814 |
| Deere & Co | 63.9% | 18.8% | 0.804 |
| Adobe Inc | 41.1% | 16.5% | 0.761 |
| Caterpillar Inc | 51.3% | 20.7% | 0.741 |
| Broadcom Inc | 43.7% | 24.6% | 0.702 |
| United Parcel Service Inc | 38.6% | 16.8% | 0.667 |
| Amazon.com Inc | 54.5% | 36.8% | 0.581 |
| Apple Inc | 38.2% | 27.1% | 0.553 |

**Insight**: Apple has the most balanced (lowest forward ratio) language — its reports emphasize **past achievements** (e.g., "reduced emissions by 60% since 2015") rather than future promises. Honeywell and GE are the most forward-looking, with language dominated by future targets and commitments.

---

### 6.5 ESG Disclosure Gap Analysis

#### Method

- **Framework**: Map each company's report content against the full 13 ESG sub-category taxonomy
- **Coverage threshold**: A sub-category is considered "covered" if ≥ 3 keyword hits across all pages
- **Metrics**: Breadth (how many topics covered) + Depth (average keyword hits per covered topic)

#### Results

| Company | Topics Covered | Breadth % | Missing Topics |
|---------|---------------|-----------|----------------|
| Adobe Inc | 13/13 | 100% | — |
| Amazon.com Inc | 13/13 | 100% | — |
| Apple Inc | 13/13 | 100% | — |
| Broadcom Inc | 13/13 | 100% | — |
| Caterpillar Inc | 13/13 | 100% | — |
| General Electric Co | 13/13 | 100% | — |
| Honeywell International Inc | 13/13 | 100% | — |
| NVIDIA Corp | 13/13 | 100% | — |
| Deere & Co | 12/13 | 92.3% | Data Privacy & Security |
| **United Parcel Service Inc** | **4/13** | **30.8%** | Water, Waste, Biodiversity, D&I, Supply Chain, Data Privacy, Board Governance, Ethics, Risk Management |

**Insight**: UPS's short brochure (only 5 content pages) has critical gaps — it misses all 3 Governance sub-categories and most Social topics. Deere is missing Data Privacy coverage, which is notable given increasing agricultural technology digitization. All other 8 companies achieve full 13/13 topic coverage, though coverage depth varies significantly.

---

### 6.6 Peer Benchmarking & Company Profiling

#### Method

- **Z-Score normalization**: Each metric is standardized to mean = 0, std = 1 across all 10 companies
- **Percentile ranking**: Companies ranked 0-100 on each dimension
- **Strength/Weakness identification**: Z > 1.0 = ★ Strength, Z < −1.0 = ▼ Weakness
- **Company Profile Cards**: Structured JSON profiles generated for chatbot integration

#### Composite ESG Score Ranking

| Rank | Company | Score |
|------|---------|-------|
| 1 | United Parcel Service Inc | 66.5 |
| 2 | Adobe Inc | 43.2 |
| 3 | Apple Inc | 43.0 |
| 4 | NVIDIA Corp | 42.5 |
| 5 | Amazon.com Inc | 39.9 |
| 6 | Deere & Co | 37.8 |
| 7 | Honeywell International Inc | 34.7 |
| 8 | Broadcom Inc | 32.6 |
| 9 | Caterpillar Inc | 32.2 |
| 10 | General Electric Co | 32.2 |

> **Note**: UPS's high composite score reflects its concise, concrete reporting style with very low vagueness — but should be interpreted cautiously given its extremely short report (5 pages). The composite formula weighs quantitative rigor and transparency, which favors concise, data-heavy reporting.

---

### 6.7 Feature Engineering & Composite Scoring

#### Master Feature Matrix

The final company-level feature matrix (`company_features_matrix.csv`) contains **10 rows × 27 columns**:

| Feature Group | Features | Source |
|---------------|----------|--------|
| ESG page proportions | `env_score`, `soc_score`, `gov_score` | Step 5 |
| ESG content volume | `total_content_pages` | Step 5 |
| Sentiment by pillar | `sent_env`, `sent_soc`, `sent_gov` | Step 6 |
| Readability | `avg_fog_index` | Step 9 |
| Vagueness | `avg_vagueness` | Step 9 |
| LDA topic probabilities | `topic_0` through `topic_5` | Step 10 |
| Topic diversity | `topic_entropy` | Step 10 |
| Text richness | `avg_token_count` | Step 13 |
| Quantitative disclosure | `avg_quant_score` | Step 13 |
| Language direction | `fwd_bwd_balance` | Step 14 |
| Commitment intensity | `num_commitments` | Step 14 |
| Disclosure coverage | `breadth_score`, `avg_depth` | Step 15 |
| Pillar coverage | `env_coverage_pct`, `soc_coverage_pct`, `gov_coverage_pct` | Step 15 |
| Overall score | `composite_esg_score` | Step 13 |

#### Composite ESG Score Formula

```
Score = 0.3 × norm(env_score) 
      + 0.3 × norm(soc_score) 
      + 0.1 × norm(gov_score) 
      + 0.1 × norm(sentiment_env) 
      + 0.2 × norm(quant_score) 
      − 0.1 × norm(vagueness)    [penalty]
```

This formula rewards companies with broad ESG coverage, positive environmental sentiment, concrete quantitative data, and penalizes vague/buzzword-heavy reporting.

---

### 6.8 Chatbot Knowledge Base Construction

#### Method

- **Auto Q&A Generation**: Programmatically generate question-answer pairs from all analysis outputs
- **Template categories**: overview, environmental, transparency, gaps, recommendation, language, commitments, comparison, ranking
- **Output format**: CSV (for review) + JSON (for chatbot integration)

#### Results

- **73 Q&A pairs** generated across 9 categories
- **10 company profile cards** with strengths, weaknesses, and key commitments
- **Knowledge base structure**:
  ```json
  {
    "metadata": { "companies_analyzed": 10, "total_qa_pairs": 73 },
    "company_profiles": { ... },
    "qa_pairs": [ ... ]
  }
  ```

**Sample Q&A**:
> **Q**: What is Apple Inc's ESG profile?
> **A**: Apple Inc's ESG report allocates approximately 77.5% to Environmental topics, 18.0% to Social topics, and 0.9% to Governance topics.

> **Q**: What should United Parcel Service Inc improve in its ESG reporting?
> **A**: Strengthen Governance disclosure: Board & Corporate Governance, Ethics & Compliance. Add Environmental topics: Water Management, Waste Management.

---

## 7. Model Comparison Summary

### Text Representation Methods Compared

| Method | Dimensionality | Captures Semantics | Captures Frequency | Used In |
|--------|---------------|-------------------|-------------------|---------|
| Bag-of-Words (Count) | ~5000 | ❌ | ✅ | LDA |
| TF-IDF | ~5000 | ❌ | ✅ (weighted) | NMF, Keywords, Classification |
| KeyBERT Embeddings | 384 | ✅ | ❌ | Keyword extraction |
| Sentence-BERT | 384 | ✅ | ❌ | Clustering, Similarity |

### Topic Discovery: Rule-Based vs LDA vs NMF

| Aspect | Rule-Based (Step 4) | LDA (Step 10) | NMF (Step 10) |
|--------|---------------------|---------------|---------------|
| Supervision | Supervised (curated keywords) | Unsupervised | Unsupervised |
| Categories | Fixed 13 sub-categories | Discovered K topics | Discovered K topics |
| Interpretability | Highest (human-defined) | Medium | High |
| Coverage | Limited by keyword list | Full vocabulary | Full vocabulary |
| Novel discovery | ❌ | ✅ (finds unexpected themes) | ✅ |

### Clustering: K-Means vs HDBSCAN vs Hierarchical

| Aspect | K-Means | HDBSCAN | Agglomerative |
|--------|---------|---------|---------------|
| Requires predefined K | ✅ | ❌ | ✅ |
| Handles noise | ❌ | ✅ | ❌ |
| Cluster shape | Spherical | Arbitrary | Arbitrary |
| Silhouette Score | **Best** | Lowest | Second |
| Unique value | Simple, fast | Noise detection | Dendrogram |
| Best for | Dense, well-separated clusters | Unknown cluster count | Hierarchical insight |

### Sentiment Analysis: VADER

| Aspect | VADER (Used) |
|--------|-------------|
| Speed | Very fast |
| Domain adaptation | Custom lexicon injected |
| Limitation | Rule-based, struggles with complex negation |
| Alternative (not used) | Transformer-based sentiment (would be slower but more accurate) |

---

## 8. Key Findings & Insights

### Finding 1: Industry Determines ESG Focus
Apple (tech/consumer) is 77.5% environmental. Adobe (tech/creative) is 63.6% social. Caterpillar (industrial) has the most balanced ESG profile. BERT embeddings confirm this — companies in the same industry cluster together semantically.

### Finding 2: Governance Is Universally Underreported
Only 8.5% of all pages address governance topics. Even among companies with "full coverage," governance depth is significantly lower. This is a systemic gap across the ESG reporting landscape.

### Finding 3: Forward-Looking Language Dominates
Most companies are forward-looking (commitments > achievements). Apple is the notable exception, emphasizing what it has already achieved. GE and Honeywell make the most future promises.

### Finding 4: Biodiversity Is a Cross-Cutting Connector
Association rules show that biodiversity discussion is the strongest predictor of comprehensive ESG coverage — companies that discuss biodiversity also discuss water, waste, supply chain, and community in the same content.

### Finding 5: Report Length ≠ Report Quality
UPS has only 5 content pages but the lowest vagueness ratio (0.31) and highest quantitative density. GE has 121 pages but higher vagueness (2.75). Concise, data-driven reporting may be more effective than lengthy narrative reports.

### Finding 6: Quantitative Commitment Gaps
NVIDIA has zero extracted quantitative commitments despite being a top tech company. Caterpillar and Deere each have only 1. This suggests room for improvement in measurable ESG target-setting.

---

*Report generated as part of STAT8017 ESG Chatbot Project — Objective 4: Advanced Data Mining & Text Analysis*
