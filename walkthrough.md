# Objective 4: Advanced Data Mining — Final Walkthrough

## Summary

共实现 **8 个分析脚本** (step10-step17)，产出 **22 张可视化图表** 和 **25 个数据文件**，覆盖课内外多种方法。

---

## Scripts & Methods Overview

| Script | 分析模块 | 方法 | 课程关联 |
|--------|---------|------|---------|
| `step10` | LDA 主题建模 | LDA + NMF | 无监督学习（超纲加分） |
| `step11` | BERT 语义聚类 | Sentence-BERT + K-Means/DBSCAN/Hierarchical | 聚类分析 + 神经网络 |
| `step12` | 关联规则挖掘 | Apriori | **课程核心：Association Rules** |
| `step13` | 特征工程与评分 | Feature aggregation + composite scoring | 特征选择 |
| `step14` | ESG 承诺提取 | Regex NER + Forward/Backward 语言分析 | 文本挖掘（超纲加分） |
| `step15` | 披露差距分析 | Coverage mapping + gap identification | 数据挖掘（超纲加分） |
| `step16` | 同行对标基准 | Z-score percentile + strength/weakness | 分类对比（超纲加分） |
| `step17` | 聊天机器人知识库 | Auto Q&A generation + structured KB | 为 Obj 6 直接服务 |

---

## Key Results & Insights

### 1. LDA Topic Discovery (Step 10)
- 发现 6 个潜在主题：涵盖气候/碳排放、人力资源/多样性、能源管理、社区影响等
- Apple 和 Amazon 显示出强环境主题聚焦，Adobe 偏重社会/员工主题

### 2. BERT Semantic Clustering (Step 11)
- K-Means 在 Silhouette Score 上表现最优
- 公司余弦相似度揭示：GE 和 Caterpillar 的报告在语义上最相似（同为工业制造业）
- UMAP 可视化清晰展示 E/S/G 三类话题的语义边界

### 3. Association Rules (Step 12)
- 发现 **16,712 条** 关联规则
- 最强规则（lift > 9）：当页面同时讨论 Energy Management + Supply Chain + Biodiversity + Human Capital 时，几乎一定也会讨论 Waste Management + Water Management + D&I（置信度 > 80%）
- 揭示了 ESG 议题之间的深层联动关系

### 4. ESG Commitment Extraction (Step 14)
- 从原始文本中提取出 **74 条** 具体 ESG 承诺/目标
- GE 最多（33条），Apple（10条），Amazon（12条）
- 最常见目标年份：2050（31次），2030（17次）
- Apple 的语言最偏向已完成成果（backward-looking），Honeywell 最偏向承诺（forward-looking）

### 5. Disclosure Gap Analysis (Step 15)
- 8/10 家公司覆盖了全部 13 个 ESG 子议题
- UPS 覆盖率最低（仅 30.8%，4/13 子议题），缺失全部治理类别
- Deere & Co 缺少 Biodiversity 议题

### 6. Peer Benchmarking (Step 16)
- 综合 ESG 得分排名：UPS > Adobe > Apple > NVIDIA > Amazon
- Z-score 矩阵清晰标注每家公司的 ★ 优势和 ▼ 短板
- 生成了 10 家公司的结构化 profile cards (JSON) 可供 chatbot 直接查询

### 7. Chatbot Knowledge Base (Step 17)
- 自动生成 **73 条 Q&A 对**，覆盖：概况、环境、透明度、差距、建议、语言风格、承诺、对比、排名
- 输出为 `chatbot_knowledge_base.json`，可直接集成到 Obj 6

---

## Final Feature Matrix (交给 Obj 5)

`data/company_features_matrix.csv` — **10 rows × 27 columns**:

| 特征组 | 列数 | 具体特征 |
|-------|------|---------|
| ESG 页面占比 | 3 | env_score, soc_score, gov_score |
| 情感得分 | 3 | sent_env, sent_soc, sent_gov |
| 文本可读性 | 2 | avg_fog_index, avg_vagueness |
| LDA 主题分布 | 7 | topic_0~5, topic_entropy |
| 文本量化特征 | 2 | avg_token_count, avg_quant_score |
| 前瞻性语言 | 1 | fwd_bwd_balance |
| 承诺数量 | 1 | num_commitments |
| 披露覆盖度 | 5 | breadth_score, avg_depth, env/soc/gov_coverage_pct |
| 综合评分 | 1 | composite_esg_score (可做预测目标) |
| 基础信息 | 2 | company, total_content_pages |

---

## All New Visualizations (output/)

| File | Description |
|------|-------------|
| `lda_topic_wordclouds.png` | LDA 发现的 6 个主题词云 |
| `umap_kmeans.png` | BERT 嵌入 UMAP 散点图 (K-Means 着色) |
| `clustering_metrics.png` | 3 种聚类方法指标对比 |
| `company_similarity_heatmap.png` | 公司间语义相似度热力图 |
| `association_rules_scatter.png` | 关联规则 Support vs Confidence 气泡图 |
| `commitment_timeline.png` | ESG 承诺目标年份分布 |
| `forward_vs_backward.png` | 前瞻性 vs 回顾性语言对比图 |
| `disclosure_gap_heatmap.png` | 披露覆盖度热力图 (带子类别 × 公司) |
| `disclosure_radar.png` | 多维 ESG 雷达图 |
| `benchmark_ranking.png` | E/S/G 三柱得分排名对比 |
| `strength_weakness.png` | 优势/短板 Z-score 矩阵 |

---

## Execution Commands

```bash
python src/step10_topic_modeling.py
python src/step11_bert_clustering.py
python src/step12_association_rules.py
python src/step14_commitment_extraction.py
python src/step15_disclosure_gap.py
python src/step16_benchmarking.py
python src/step17_chatbot_kb.py
python src/step13_feature_engineering.py   # Run last to collect all features
```
