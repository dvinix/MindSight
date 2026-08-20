# Dreaddit Dataset: Exploratory Data Analysis & Quality Summary

## 1. Dataset Dimensions & Completeness
- **Training Set**: `2,838` posts, `116` columns.
- **Testing Set**: `715` posts, `116` columns.
- **Combined Total**: `3,553` posts.
- **Missing Values**:
  - Training Set: `0` missing entries.
  - Testing Set: `0` missing entries.

---

## 2. Class Balance Analysis
The target variable `label` represents psychological stress classification:
- `1`: Stressed
- `0`: Not Stressed (Neutral / Non-stress context)

| Split | Not Stressed (0) | Stressed (1) | Total | Stressed % |
|---|---|---|---|---|
| **Train** | 1,350 (47.57%) | 1,488 (52.43%) | 2,838 | **52.43%** |
| **Test** | 346 (48.39%) | 369 (51.61%) | 715 | **51.61%** |
| **Overall** | 1,696 | 1,857 | 3,553 | **52.27%** |

> **Finding**: The dataset is well-balanced (~51.5% - 52.4% stressed), eliminating the immediate need for severe class-reweighting or extreme oversampling (SMOTE).

---

## 3. Text Length Characteristics (Word Counts)

| Statistic | Stressed (1) | Not Stressed (0) | Overall |
|---|---|---|---|
| **Count** | 1,857 | 1,696 | 3,553 |
| **Mean Words** | 88.78 | 82.17 | 85.63 |
| **Std Dev** | 34.24 | 29.15 | 32.08 |
| **Median (50%)** | 83.0 | 78.0 | 80.0 |
| **Min Words** | 1 | 1 | 1 |
| **Max Words** | 310 | 255 | 310 |

> **Finding**: Conversational Reddit posts are relatively dense paragraphs (averaging ~85-90 words, max over 300 words), making contextual transformers (BERT/RoBERTa) with a max sequence length of 128 to 256 tokens ideal.

---

## 4. Subreddit Diversity
The dataset spans 10 mental health and support-oriented subreddits:
- `r/ptsd`: 711 posts (20.0%)
- `r/relationships`: 694 posts (19.5%)
- `r/anxiety`: 650 posts (18.3%)
- `r/domesticviolence`: 388 posts (10.9%)
- `r/assistance`: 355 posts (10.0%)
- `r/survivorsofabuse`: 315 posts (8.9%)
- `r/homeless`: 220 posts (6.2%)
- `r/almosthomeless`: 99 posts (2.8%)
- `r/stress`: 78 posts (2.2%)
- `r/food_pantry`: 43 posts (1.2%)

---

## 5. Conversational & Platform Artifacts Detected

| Artifact Type | Train Count | Test Count | Combined | Preprocessing Strategy |
|---|---|---|---|---|
| **URLs / Web Links** | 141 | 30 | 171 | Replace with `<URL>` token or strip |
| **Subreddit Mentions (`r/`)** | 43 | 7 | 50 | Normalize to avoid subreddit bias |
| **User Mentions (`u/`)** | 7 | 0 | 7 | Anonymize to `<USER>` for privacy |
| **Deleted / Removed Tags** | 0 | 0 | 0 | Strip or filter out empty entries |

---

## 6. Generated Visualizations
The following high-resolution figures are generated in `reports/figures/`:
1. `class_distribution.png`: Comparative bar chart of stress vs. non-stress splits.
2. `text_length_distribution.png`: Histogram, KDE curves, and boxplot of post lengths.
3. `top_words_comparison.png`: Word-level frequency distribution for stressed vs non-stressed posts.
