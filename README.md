# ESG Report Text Analysis Pipeline

## Objective

对 10 份企业 ESG/可持续发展年度报告（PDF），完成完整的数据挖掘与文本分析闭环：数据预处理 → 关键词提取（Obj 1）→ 主题分类（Obj 2）→ 情感分析（Obj 3）→ 高级数据挖掘与文本分析（Obj 4）→ 特征工程。

## Data

10 份英文 PDF 报告，来自以下公司：

| Company | Year | Report Title |
|---------|------|-------------|
| Adobe Inc | 2023 | Corporate Social Responsibility Report |
| Amazon.com Inc | 2024 | Sustainability Report |
| Apple Inc | 2024 | Environmental Progress Report |
| Broadcom Inc | 2023 | ESG Report |
| Caterpillar Inc | 2024 | Sustainability Report |
| Deere & Co | 2024 | Business Impact Report |
| General Electric Co | 2022 | Sustainability Report |
| Honeywell International Inc | 2024 | Impact Report |
| NVIDIA Corp | 2024 | Sustainability Report |
| United Parcel Service Inc | 2023 | Sustainability and Social Impact Highlights Brochure |

共提取 **672 页**，其中 **633 页** 有实质性文本内容可用于分析。

## Pipeline

按顺序运行以下脚本：

```bash
# === Objective 1 & 2: 数据预处理、关键词提取、主题分类 ===
python src/step1_extract_pdf.py        # PDF 文本提取
python src/step2_preprocess.py         # 文本清洗与预处理
python src/step3_keywords.py           # 关键词提取 (TF-IDF + KeyBERT)
python src/step4_classify.py           # ESG 主题分类 (Rule-based)
python src/step5_visualize.py          # 可视化与汇总

# === Objective 3: 情感分析 ===
python src/step6_sentiment_analysis.py # VADER 情感分析
python src/step7_correlation_analysis.py # 关键词-情感相关性
python src/step8_advanced_mining.py    # 共现网络 + K-Means 聚类
python src/step9_greenwashing_analysis.py # 洗绿风险检测

# === Objective 4: 高级数据挖掘与文本分析 ===
python src/step10_topic_modeling.py    # LDA 主题建模 + NMF 对比
python src/step11_bert_clustering.py   # BERT 语义嵌入 + 多方法聚类对比
python src/step12_association_rules.py # Apriori 关联规则挖掘
python src/step14_commitment_extraction.py # ESG 承诺/目标提取 + 前瞻性语言分析
python src/step15_disclosure_gap.py    # ESG 披露差距分析
python src/step16_benchmarking.py      # 同行对标基准 + 公司画像
python src/step17_chatbot_kb.py        # 聊天机器人知识库构建
python src/step13_feature_engineering.py # 特征工程（汇总所有特征，最后运行）
```

## Outputs

### Data Files (`data/`)

| File | Description | Source |
|------|-------------|--------|
| `raw_pages.csv` | 逐页提取的原始文本 | Step 1 |
| `cleaned_pages.csv` | 清洗后的文本（分词、去停用词、词形还原） | Step 2 |
| `keywords_tfidf.csv` | TF-IDF 关键词（每页 top-15） | Step 3 |
| `keywords_keybert.csv` | KeyBERT 语义关键词（每页 top-10） | Step 3 |
| `esg_metrics.csv` | 每页命中的 ESG 术语列表 | Step 3 |
| `page_classification.csv` | 每页的 ESG 主题分类结果 | Step 4 |
| `report_summary.csv` | 每份报告的 ESG 类别分布汇总 | Step 4 |
| `page_sentiment.csv` | 每页 VADER 情感得分 | Step 6 |
| `sentiment_by_major_category.csv` | 按 E/S/G 类别聚合的情感得分 | Step 6 |
| `correlation_matrix.csv` | 关键词与情感相关性矩阵 | Step 7 |
| `greenwashing_risk.csv` | 各公司洗绿风险指标 | Step 9 |
| `lda_topic_distributions.csv` | 每页 LDA 主题概率分布 | Step 10 |
| `page_embeddings.npy` | 每页 384 维 BERT 语义嵌入向量 | Step 11 |
| `bert_clusters.csv` | 每页 K-Means 聚类标签 | Step 11 |
| `company_similarity_matrix.csv` | 公司间余弦语义相似度 | Step 11 |
| `association_rules.csv` | ESG 关联规则（16,712 条） | Step 12 |
| `esg_commitments.csv` | 提取的 74 条 ESG 量化承诺/目标 | Step 14 |
| `forward_backward_scores.csv` | 前瞻性 vs 回顾性语言得分 | Step 14 |
| `disclosure_coverage_matrix.csv` | 公司 × ESG 子议题覆盖矩阵 | Step 15 |
| `disclosure_gaps.csv` | 各公司 ESG 披露差距分析 | Step 15 |
| `esg_benchmarks.csv` | Z-score 百分位基准排名 | Step 16 |
| `company_profiles.json` | 结构化公司 ESG 画像（供 Chatbot） | Step 16 |
| `chatbot_qa_pairs.csv` | 自动生成的 73 条 Q&A 对 | Step 17 |
| `chatbot_knowledge_base.json` | 统一的聊天机器人知识库 | Step 17 |
| `company_features_matrix.csv` | **Master 特征矩阵**（10×27）供 Obj 5 | Step 13 |

### Visualizations (`output/`)

| File | Description | Source |
|------|-------------|--------|
| `esg_distribution_by_company.png` | 各公司 ESG 页面数量堆叠图 | Step 5 |
| `esg_proportion_by_company.png` | 各公司 ESG 类别占比对比 | Step 5 |
| `top20_keywords.png` | 全部报告 Top 20 关键词频率 | Step 5 |
| `subcategory_heatmap.png` | 各公司 × ESG 子类别热力图 | Step 5 |
| `sentiment_bar_major.png` | ESG 柱面情感得分柱状图 | Step 6 |
| `sentiment_heatmap_subtopics.png` | ESG 子类别情感热力图 | Step 6 |
| `correlation_heatmap.png` | 关键词-情感相关性热力图 | Step 7 |
| `keyword_network.png` | 关键词共现网络图 | Step 8 |
| `company_clusters.png` | K-Means 公司聚类 PCA 图 | Step 8 |
| `greenwashing_quadrant.png` | 洗绿风险象限图 | Step 9 |
| `lda_topic_wordclouds.png` | LDA 主题词云 | Step 10 |
| `umap_kmeans.png` | BERT UMAP 聚类散点图 | Step 11 |
| `clustering_metrics.png` | 多方法聚类评估指标对比 | Step 11 |
| `company_similarity_heatmap.png` | 公司语义相似度热力图 | Step 11 |
| `association_rules_scatter.png` | 关联规则 Support-Confidence 气泡图 | Step 12 |
| `commitment_timeline.png` | ESG 承诺目标年份分布图 | Step 14 |
| `forward_vs_backward.png` | 前瞻性 vs 回顾性语言对比图 | Step 14 |
| `disclosure_gap_heatmap.png` | ESG 披露覆盖度热力图 | Step 15 |
| `disclosure_radar.png` | 多维 ESG 披露雷达图 | Step 15 |
| `benchmark_ranking.png` | E/S/G 三柱得分排名 | Step 16 |
| `strength_weakness.png` | 公司优势/短板 Z-score 矩阵 | Step 16 |

### Config (`config/`)

| File | Description |
|------|-------------|
| `esg_keywords.json` | ESG 分类关键词词表（3 大类 13 子类） |

## Key Results

**关键词提取 (Objective 1)**: Top 关键词：energy, employee, emission, climate, carbon, supplier, water, safety, renewable, sustainability

**主题分类 (Objective 2)**: Environmental 50.6% | Social 40.9% | Governance 8.5%

**情感分析 (Objective 3)**: 所有公司 ESG 情感总体偏正面，Broadcom 最低（0.76），Amazon 最高（0.99）

**高级数据挖掘 (Objective 4)**: LDA 发现 6 个潜在主题 · BERT 聚类对比 3 种方法 · 16,712 条关联规则 · 74 条 ESG 承诺被提取 · 73 条 Q&A 整合为 Chatbot 知识库

详见 `analysis_report.md` 获取完整分析报告。

## Environment

```bash
conda env: textmining
Python 3.12
```

See `requirements.txt` for full dependency list.
