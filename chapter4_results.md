# Chapter 4. Result Summary and Interpretation — Objective 4: Advanced Data Mining & Text Analysis

## 4.x.1 Latent Topic Discovery via LDA and NMF

To complement the rule-based ESG classification from Objective 2, we applied two unsupervised topic modeling approaches — Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) — to discover latent thematic structures directly from the corpus without predefined categories. The motivation is twofold: first, rule-based classification is limited by the coverage and granularity of handcrafted keyword lists; second, unsupervised models can reveal unexpected thematic groupings that domain experts might overlook. LDA treats each document as a probabilistic mixture of topics, where each topic is a distribution over words, estimated via variational Bayesian inference. NMF takes a complementary algebraic approach, factorizing the TF-IDF document-term matrix into non-negative topic-word and document-topic factor matrices. Both models were trained with K=6 topics on the 633 content pages, using Bag-of-Words input for LDA and TF-IDF input for NMF.

The LDA model successfully identified six distinct themes that align with recognizable ESG narratives. As shown in Figure 4.x.1, the word clouds reveal interpretable topics spanning corporate social programs, workforce governance, industrial operations, technology ethics, climate and energy, and manufacturing sustainability. Notably, Topic 2 emerged as a uniquely agricultural/industrial theme — dominated by words like "field," "equipment," and "production" — which was not present in the predefined taxonomy but meaningfully captures Deere & Co's sector-specific ESG narrative. The company-level topic distribution in Table 4.x.1 further confirms industry-driven differentiation: Apple is dominated by the climate/energy topic (73.7%), Deere & Co concentrates on the agriculture/industry topic (63.2%), while Adobe focuses on corporate social impact (47.1%). These data-driven findings independently validate the rule-based classification results while uncovering more nuanced thematic variation.

Comparing the two models, LDA produces softer, more distributed topic assignments due to its probabilistic Bayesian framework, whereas NMF yields sparser, more concentrated weights that are generally easier to interpret for individual documents. Both methods produced consistent topic groupings, confirming the robustness of the discovered themes. LDA was selected as the primary model for downstream feature engineering because its probability distributions enable meaningful diversity measurement via topic entropy — a metric that quantifies how broadly a company's report spans multiple ESG themes versus concentrating on a single narrative.

![Figure 4.x.1: LDA Topic Word Clouds — Six latent topics discovered from ESG report corpus](output/lda_topic_wordclouds.png)

**Table 4.x.1: Company × LDA Topic Distribution (Average Probability)**

| Company | Topic 0 | Topic 1 | Topic 2 | Topic 3 | Topic 4 | Topic 5 |
|---------|---------|---------|---------|---------|---------|---------|
| Adobe Inc | **0.471** | 0.131 | 0.053 | 0.266 | 0.059 | 0.019 |
| Amazon.com Inc | 0.060 | 0.219 | 0.094 | 0.034 | **0.505** | 0.088 |
| Apple Inc | 0.038 | 0.053 | 0.065 | 0.089 | **0.737** | 0.019 |
| Caterpillar Inc | 0.117 | 0.274 | 0.065 | 0.242 | 0.026 | **0.275** |
| Deere & Co | 0.026 | 0.060 | **0.632** | 0.041 | 0.050 | 0.191 |
| General Electric Co | **0.462** | 0.165 | 0.042 | 0.051 | 0.046 | 0.235 |
| Honeywell International Inc | 0.158 | **0.460** | 0.036 | 0.211 | 0.077 | 0.059 |

## 4.x.2 BERT Semantic Embeddings and Multi-Method Clustering Comparison

While LDA and NMF rely on surface-level word frequencies, we additionally applied deep learning-based document representation to capture richer semantic structure. We encoded all 633 content pages into 384-dimensional dense vectors using Sentence-BERT (all-MiniLM-L6-v2), a transformer-based neural network model pre-trained on over 1 billion sentence pairs and fine-tuned for semantic similarity. Unlike bag-of-words or TF-IDF representations, BERT embeddings capture contextual word meaning, handle synonyms automatically, and encode discourse-level semantics — for example, recognizing that "carbon footprint reduction" and "GHG emission decrease" convey equivalent meaning even without shared vocabulary. UMAP (Uniform Manifold Approximation and Projection) was then applied for non-linear dimensionality reduction to produce a two-dimensional visualization that preserves both local neighborhood structure and global cluster topology.

To identify natural groupings within the embedded space, three clustering algorithms representing different algorithmic paradigms were compared: K-Means (centroid-based partitioning), Agglomerative Clustering (bottom-up hierarchical merging), and HDBSCAN (density-based spatial clustering with noise handling). This comparison is methodologically valuable because different algorithms make different geometric assumptions about cluster structure, and their relative performance reveals properties of the underlying data distribution. As shown in Figure 4.x.2 and Table 4.x.2, K-Means achieved the best performance across all three evaluation metrics — Silhouette Score (cluster separation), Calinski-Harabasz Index (variance ratio), and Davies-Bouldin Index (cluster compactness). This outcome is expected because BERT embeddings, produced by a model trained with cosine similarity objectives, naturally form roughly spherical clusters in high-dimensional space, closely matching K-Means' geometric assumptions. However, HDBSCAN provided unique analytical value by identifying 23 noise points — transitional pages blending multiple ESG themes that do not belong to any single cluster — a diagnostic capability that neither K-Means nor Agglomerative Clustering can offer.

![Figure 4.x.2: UMAP Visualization of BERT Embeddings with K-Means Cluster Assignments](output/umap_kmeans.png)

**Table 4.x.2: Clustering Algorithm Comparison on BERT Embeddings**

| Metric | K-Means (k=6) | Agglomerative (k=6) | HDBSCAN |
|--------|---------------|---------------------|---------|
| Silhouette Score ↑ | **Best** | Second | Third |
| Calinski-Harabasz ↑ | **Best** | Second | Third |
| Davies-Bouldin ↓ | **Best** | Second | Third |
| Noise detection | ❌ | ❌ | ✅ (23 points) |

![Figure 4.x.3: Clustering Evaluation Metrics Comparison](output/clustering_metrics.png)

We further computed pairwise cosine similarity between company-level average embeddings to quantify semantic proximity across ESG reports. Figure 4.x.4 reveals that Amazon and Apple share the highest semantic similarity (0.922), likely due to overlapping tech-sector environmental language. GE and Honeywell (0.891) cluster together as industrial conglomerates. Adobe and Deere show the lowest similarity (0.724), reflecting fundamentally different ESG concerns between a creative tech company and an agricultural manufacturer.

![Figure 4.x.4: Company-Level Semantic Similarity Heatmap (Cosine Similarity on BERT Embeddings)](output/company_similarity_heatmap.png)

## 4.x.3 Association Rule Mining

We applied the Apriori algorithm — a classic frequent pattern mining method — to discover co-occurrence patterns among ESG sub-topics and keywords at the page level. Each page was transformed into a market-basket-style transaction containing the ESG sub-categories matched and the top-3 TF-IDF keywords present on that page. With minimum support of 0.05 (at least 5% of pages), minimum confidence of 0.3, and a lift threshold of 1.2, the algorithm discovered 16,712 valid association rules, of which 287 exhibited lift values exceeding 5.0, indicating very strong positive associations.

The strongest single-antecedent rules consistently involve Biodiversity as the most powerful "connector" topic: when a page discusses biodiversity, it co-occurs with Waste Management, Water Management, Diversity & Inclusion, and Community Impact with over 43% confidence and lift above 5.0. This pattern suggests that biodiversity is treated as a cross-cutting integrative theme in ESG reporting rather than a standalone issue — companies that mention biodiversity tend to be producing comprehensive, multi-topic ESG pages. This finding has practical implications: companies seeking to improve ESG coverage breadth might use biodiversity as an anchor topic that naturally connects multiple environmental and social dimensions. The Support versus Confidence scatter plot in Figure 4.x.5 shows that most high-confidence rules cluster at moderate support levels (5–15%), indicating that the strongest thematic associations are found in specialized, in-depth content sections rather than in ubiquitous introductory or summary pages.

![Figure 4.x.5: Association Rules — Support vs Confidence (Bubble Size and Color = Lift)](output/association_rules_scatter.png)

## 4.x.4 ESG Commitment Extraction and Forward-Looking Language Analysis

Beyond topic-level analysis, we developed a regex-based information extraction system to identify specific, quantitative ESG commitments embedded within the raw report text. Four pattern templates were designed to capture statements such as "reduce emissions by 50% by 2030," "achieve net zero by 2050," and "target of 100% renewable energy by 2025." Each extracted commitment was structured with fields for company, action verb, target value, unit, target year, and surrounding context — forming a queryable commitment database for the chatbot. The system extracted 74 unique commitments across 9 companies. General Electric contributed the most (33 commitments), reflecting its detailed 2022 Sustainability Report with extensive quantitative targets. NVIDIA, despite being a major technology company, yielded zero quantifiable commitments — its report relies entirely on qualitative narratives without measurable targets, representing a significant gap in accountability. The most frequent target years are 2050 (31 occurrences, primarily net-zero pledges) and 2030 (17 occurrences, near-term operational targets), as visualized in Figure 4.x.6.

![Figure 4.x.6: ESG Commitment Target Year Distribution by Company](output/commitment_timeline.png)

Complementing the commitment extraction, we classified the overall reporting language as forward-looking (commitments, aspirations, future plans) versus backward-looking (past achievements, accomplished results, verified metrics) using curated lexicons of 26 forward-looking and 27 backward-looking indicator phrases. This analysis reveals fundamental differences in corporate communication strategy. Figure 4.x.7 shows that Apple has the most balanced, backward-leaning language (balance = 0.553), consistently emphasizing verified achievements such as its 60% carbon footprint reduction since 2015 — a communication strategy that prioritizes demonstrated credibility. In contrast, Honeywell (0.840) and GE (0.826) are heavily forward-looking, with language dominated by future pledges and aspirational targets. This distinction carries practical significance for ESG evaluation: companies disproportionately heavy on forward-looking language but light on backward-looking evidence may warrant closer scrutiny regarding the achievability of their stated goals.

![Figure 4.x.7: Forward-Looking vs Backward-Looking Language Ratio by Company](output/forward_vs_backward.png)

## 4.x.5 ESG Disclosure Gap Analysis and Peer Benchmarking

We mapped each company's disclosure against the full 13 ESG sub-category framework to perform a systematic coverage gap analysis, determining not only which topics are mentioned but also the depth of engagement measured by keyword hit frequency. Eight of ten companies achieve 100% topic breadth, indicating that most large-cap companies aim for comprehensive ESG reporting. However, two notable gaps emerged. Deere & Co is missing Data Privacy & Security entirely — a significant omission given the increasing digitization of precision agriculture and the associated data governance challenges. More critically, United Parcel Service, with only 5 content pages in its brochure-format report, covers just 4 of 13 sub-topics (30.8%), with all three Governance categories and five Social categories completely absent. The disclosure intensity heatmap in Figure 4.x.8 reveals a second layer of insight: even among companies with full topic breadth, the depth of engagement varies dramatically. Apple leads overwhelmingly in Climate Change and Energy Management depth, reflecting its concentrated environmental narrative, while Adobe leads in Human Capital and Diversity topics, consistent with its workforce-centric Social focus.

![Figure 4.x.8: ESG Disclosure Coverage Intensity Heatmap (Sub-category × Company)](output/disclosure_gap_heatmap.png)

To synthesize all metrics into an actionable benchmarking framework, we constructed multi-dimensional ESG profiles using z-score normalization across six evaluation axes: Environmental focus, Social focus, Governance focus, Disclosure breadth, Quantitative rigor, and Forward-looking orientation. The radar charts in Figure 4.x.9 enable direct visual comparison of company profiles. The strength and weakness matrix in Figure 4.x.10 provides a more granular view, highlighting each company's standout areas (★ for z > 1.0) and underperformances (▼ for z < −1.0) relative to the peer group. For instance, Apple shows ★ in Environmental focus but ▼ in Governance, while Caterpillar achieves the most balanced profile across all three ESG pillars. This benchmarking output provides a structured framework that investors and analysts can use for peer comparison and portfolio screening.

![Figure 4.x.9: Multi-Dimensional ESG Disclosure Radar Chart](output/disclosure_radar.png)

![Figure 4.x.10: ESG Strength and Weakness Matrix (Z-Score Peer Comparison)](output/strength_weakness.png)

## 4.x.6 Feature Engineering and Chatbot Knowledge Base

All analyses from Objectives 1 through 4 were consolidated into a company-level master feature matrix of 10 rows × 27 columns, encompassing ESG page proportions, sentiment scores by E/S/G pillar, LDA topic distributions, topic entropy, readability complexity, vagueness ratios, forward-looking language balance, commitment counts, and disclosure coverage metrics across all 13 sub-categories. A composite ESG score was computed using a weighted additive formula: 30% Environmental coverage + 30% Social coverage + 10% Governance coverage + 10% Environmental sentiment + 20% Quantitative disclosure rigor, with a 10% penalty for high vagueness. This composite score, along with the full feature matrix, is designed as the direct input for the prediction models in Objective 5, enabling regression and classification approaches to model ESG performance.

Additionally, to bridge the gap between analytical outputs and the practical chatbot application (Objective 6), we developed an automated knowledge base construction pipeline. This system generated 73 structured Q&A pairs across 9 semantic categories — overview, environmental focus, transparency assessment, disclosure gaps, improvement recommendations, language style characterization, specific commitments, cross-company comparisons, and composite rankings. Together with 10 structured company profile cards containing strengths, weaknesses, and key commitments, all outputs were packaged into a unified JSON knowledge base ready for chatbot retrieval-augmented generation.

**Table 4.x.3: Composite ESG Score Ranking**

| Rank | Company | Composite Score |
|------|---------|----------------|
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

![Figure 4.x.11: ESG Pillar Score Benchmark Ranking](output/benchmark_ranking.png)
