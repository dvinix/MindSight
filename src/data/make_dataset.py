"""
MindSight: AI-Powered Mental Health Risk Detection Through Conversational Analytics
Module: Data Ingestion and Exploratory Data Analysis (EDA) Pipeline
File: src/data/make_dataset.py
"""

import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import seaborn as sns


def setup_directories(base_dir: Path):
    """Ensure required output directories exist."""
    (base_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (base_dir / "reports" / "figures").mkdir(parents=True, exist_ok=True)


def load_raw_data(raw_data_dir: Path):
    """Load Dreaddit train and test CSV datasets."""
    train_path = raw_data_dir / "dreaddit-train.csv"
    test_path = raw_data_dir / "dreaddit-test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"Loaded Train data: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    print(f"Loaded Test data: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    return df_train, df_test


def compute_text_statistics(df: pd.DataFrame, text_col: str = "text"):
    """Compute character length and word count statistics."""
    df_stats = df.copy()
    df_stats["char_length"] = df_stats[text_col].astype(str).apply(len)
    df_stats["word_count"] = df_stats[text_col].astype(str).apply(lambda x: len(x.split()))
    return df_stats


def scan_reddit_artifacts(df: pd.DataFrame, text_col: str = "text"):
    """Count URLs, subreddits (r/), usernames (u/), and deleted/removed markers."""
    texts = df[text_col].astype(str)

    url_regex = re.compile(r"https?://\S+|www\.\S+|<url>", re.IGNORECASE)
    subreddit_regex = re.compile(r"\br/[A-Za-z0-9_]+", re.IGNORECASE)
    user_regex = re.compile(r"\bu/[A-Za-z0-9_-]+", re.IGNORECASE)
    deleted_regex = re.compile(r"\[deleted\]|\[removed\]", re.IGNORECASE)

    url_count = texts.apply(lambda t: len(url_regex.findall(t))).sum()
    subreddit_count = texts.apply(lambda t: len(subreddit_regex.findall(t))).sum()
    user_count = texts.apply(lambda t: len(user_regex.findall(t))).sum()
    deleted_count = texts.apply(lambda t: len(deleted_regex.findall(t))).sum()

    return {
        "urls": int(url_count),
        "subreddit_mentions": int(subreddit_count),
        "user_mentions": int(user_count),
        "deleted_or_removed_tags": int(deleted_count),
    }


def generate_eda_figures(df_train: pd.DataFrame, df_test: pd.DataFrame, output_dir: Path):
    """Generate and save 3 distinct EDA visual figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    # -------------------------------------------------------------
    # Plot 1: Class Distribution (Train vs Test)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    train_counts = df_train["label"].value_counts().sort_index()
    test_counts = df_test["label"].value_counts().sort_index()
    labels = ["Not Stressed (0)", "Stressed (1)"]
    palette = ["#4A90E2", "#E94E77"]

    # Train plot
    bars1 = axes[0].bar(labels, train_counts, color=palette, edgecolor="black", alpha=0.85)
    axes[0].set_title(f"Train Set Class Distribution (N={len(df_train)})", fontsize=13, weight="bold")
    axes[0].set_ylabel("Number of Posts")
    axes[0].set_ylim(0, max(train_counts) * 1.18)
    for bar in bars1:
        yval = bar.get_height()
        pct = (yval / len(df_train)) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 20, f"{yval:,} ({pct:.1f}%)", ha="center", weight="bold")

    # Test plot
    bars2 = axes[1].bar(labels, test_counts, color=palette, edgecolor="black", alpha=0.85)
    axes[1].set_title(f"Test Set Class Distribution (N={len(df_test)})", fontsize=13, weight="bold")
    axes[1].set_ylabel("Number of Posts")
    axes[1].set_ylim(0, max(test_counts) * 1.18)
    for bar in bars2:
        yval = bar.get_height()
        pct = (yval / len(df_test)) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval:,} ({pct:.1f}%)", ha="center", weight="bold")

    plt.tight_layout()
    fig_path_1 = output_dir / "class_distribution.png"
    plt.savefig(fig_path_1, dpi=300)
    plt.close()
    print(f"Saved: {fig_path_1}")

    # -------------------------------------------------------------
    # Plot 2: Text Length Distribution (Word Count & Character Count)
    # -------------------------------------------------------------
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_combined = compute_text_statistics(df_combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        data=df_combined,
        x="word_count",
        hue="label",
        kde=True,
        element="step",
        bins=40,
        palette={0: "#4A90E2", 1: "#E94E77"},
        ax=axes[0],
    )
    axes[0].set_title("Word Count Distribution by Stress Label", fontsize=13, weight="bold")
    axes[0].set_xlabel("Word Count per Post")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(title="Class", labels=["Stressed (1)", "Not Stressed (0)"])

    sns.boxplot(
        data=df_combined,
        x="label",
        y="word_count",
        hue="label",
        palette={0: "#4A90E2", 1: "#E94E77"},
        legend=False,
        ax=axes[1],
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black"},
    )
    axes[1].set_title("Word Count Spread & Outliers", fontsize=13, weight="bold")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Not Stressed (0)", "Stressed (1)"])
    axes[1].set_xlabel("Class Label")
    axes[1].set_ylabel("Word Count")

    plt.tight_layout()
    fig_path_2 = output_dir / "text_length_distribution.png"
    plt.savefig(fig_path_2, dpi=300)
    plt.close()
    print(f"Saved: {fig_path_2}")

    # -------------------------------------------------------------
    # Plot 3: Top 20 Distinctive / Frequent Words (Stressed vs Not Stressed)
    # -------------------------------------------------------------
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    # Add common conversational/social stop words
    custom_stops = {"im", "dont", "like", "get", "go", "would", "one", "also", "even", "ive", "really", "cant", "got", "know", "said"}
    all_stops = stop_words.union(custom_stops)

    def extract_top_words(texts, n=20):
        words = []
        for text in texts:
            clean = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
            words.extend([w for w in clean if w not in all_stops])
        return Counter(words).most_common(n)

    stressed_words = extract_top_words(df_combined[df_combined["label"] == 1]["text"], n=20)
    non_stressed_words = extract_top_words(df_combined[df_combined["label"] == 0]["text"], n=20)

    df_sw = pd.DataFrame(stressed_words, columns=["word", "count"]).sort_values("count", ascending=True)
    df_nsw = pd.DataFrame(non_stressed_words, columns=["word", "count"]).sort_values("count", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].barh(df_sw["word"], df_sw["count"], color="#E94E77", edgecolor="black", alpha=0.85)
    axes[0].set_title("Top 20 Frequent Words in Stressed Posts", fontsize=13, weight="bold")
    axes[0].set_xlabel("Frequency Count")

    axes[1].barh(df_nsw["word"], df_nsw["count"], color="#4A90E2", edgecolor="black", alpha=0.85)
    axes[1].set_title("Top 20 Frequent Words in Non-Stressed Posts", fontsize=13, weight="bold")
    axes[1].set_xlabel("Frequency Count")

    plt.tight_layout()
    fig_path_3 = output_dir / "top_words_comparison.png"
    plt.savefig(fig_path_3, dpi=300)
    plt.close()
    print(f"Saved: {fig_path_3}")


def export_eda_report(df_train: pd.DataFrame, df_test: pd.DataFrame, report_path: Path):
    """Write statistical findings into reports/eda_summary.md."""
    df_comb = pd.concat([df_train, df_test], ignore_index=True)
    df_comb = compute_text_statistics(df_comb)

    train_c = df_train["label"].value_counts().to_dict()
    test_c = df_test["label"].value_counts().to_dict()

    train_p0 = (train_c.get(0, 0) / len(df_train)) * 100
    train_p1 = (train_c.get(1, 0) / len(df_train)) * 100
    test_p0 = (test_c.get(0, 0) / len(df_test)) * 100
    test_p1 = (test_c.get(1, 0) / len(df_test)) * 100

    stats_s = df_comb[df_comb["label"] == 1]["word_count"].describe()
    stats_ns = df_comb[df_comb["label"] == 0]["word_count"].describe()

    artifacts_train = scan_reddit_artifacts(df_train)
    artifacts_test = scan_reddit_artifacts(df_test)

    missing_train = df_train.isnull().sum().sum()
    missing_test = df_test.isnull().sum().sum()

    subreddits = df_comb["subreddit"].value_counts().to_dict()

    md_content = f"""# Dreaddit Dataset: Exploratory Data Analysis & Quality Summary

## 1. Dataset Dimensions & Completeness
- **Training Set**: `{df_train.shape[0]:,}` posts, `{df_train.shape[1]}` columns.
- **Testing Set**: `{df_test.shape[0]:,}` posts, `{df_test.shape[1]}` columns.
- **Combined Total**: `{len(df_comb):,}` posts.
- **Missing Values**:
  - Training Set: `{missing_train}` missing entries.
  - Testing Set: `{missing_test}` missing entries.

---

## 2. Class Balance Analysis
The target variable `label` represents psychological stress classification:
- `1`: Stressed
- `0`: Not Stressed (Neutral / Non-stress context)

| Split | Not Stressed (0) | Stressed (1) | Total | Stressed % |
|---|---|---|---|---|
| **Train** | {train_c.get(0, 0):,} ({train_p0:.2f}%) | {train_c.get(1, 0):,} ({train_p1:.2f}%) | {len(df_train):,} | **{train_p1:.2f}%** |
| **Test** | {test_c.get(0, 0):,} ({test_p0:.2f}%) | {test_c.get(1, 0):,} ({test_p1:.2f}%) | {len(df_test):,} | **{test_p1:.2f}%** |
| **Overall** | {(train_c.get(0, 0) + test_c.get(0, 0)):,} | {(train_c.get(1, 0) + test_c.get(1, 0)):,} | {len(df_comb):,} | **{((train_c.get(1, 0) + test_c.get(1, 0)) / len(df_comb) * 100):.2f}%** |

> **Finding**: The dataset is well-balanced (~51.5% - 52.4% stressed), eliminating the immediate need for severe class-reweighting or extreme oversampling (SMOTE).

---

## 3. Text Length Characteristics (Word Counts)

| Statistic | Stressed (1) | Not Stressed (0) | Overall |
|---|---|---|---|
| **Count** | {int(stats_s['count']):,} | {int(stats_ns['count']):,} | {len(df_comb):,} |
| **Mean Words** | {stats_s['mean']:.2f} | {stats_ns['mean']:.2f} | {df_comb['word_count'].mean():.2f} |
| **Std Dev** | {stats_s['std']:.2f} | {stats_ns['std']:.2f} | {df_comb['word_count'].std():.2f} |
| **Median (50%)** | {stats_s['50%']:.1f} | {stats_ns['50%']:.1f} | {df_comb['word_count'].median():.1f} |
| **Min Words** | {int(stats_s['min'])} | {int(stats_ns['min'])} | {int(df_comb['word_count'].min())} |
| **Max Words** | {int(stats_s['max'])} | {int(stats_ns['max'])} | {int(df_comb['word_count'].max())} |

> **Finding**: Conversational Reddit posts are relatively dense paragraphs (averaging ~85-90 words, max over 300 words), making contextual transformers (BERT/RoBERTa) with a max sequence length of 128 to 256 tokens ideal.

---

## 4. Subreddit Diversity
The dataset spans 10 mental health and support-oriented subreddits:
"""
    for sub, cnt in subreddits.items():
        md_content += f"- `r/{sub}`: {cnt:,} posts ({cnt / len(df_comb) * 100:.1f}%)\n"

    md_content += f"""
---

## 5. Conversational & Platform Artifacts Detected

| Artifact Type | Train Count | Test Count | Combined | Preprocessing Strategy |
|---|---|---|---|---|
| **URLs / Web Links** | {artifacts_train['urls']} | {artifacts_test['urls']} | {artifacts_train['urls'] + artifacts_test['urls']} | Replace with `<URL>` token or strip |
| **Subreddit Mentions (`r/`)** | {artifacts_train['subreddit_mentions']} | {artifacts_test['subreddit_mentions']} | {artifacts_train['subreddit_mentions'] + artifacts_test['subreddit_mentions']} | Normalize to avoid subreddit bias |
| **User Mentions (`u/`)** | {artifacts_train['user_mentions']} | {artifacts_test['user_mentions']} | {artifacts_train['user_mentions'] + artifacts_test['user_mentions']} | Anonymize to `<USER>` for privacy |
| **Deleted / Removed Tags** | {artifacts_train['deleted_or_removed_tags']} | {artifacts_test['deleted_or_removed_tags']} | {artifacts_train['deleted_or_removed_tags'] + artifacts_test['deleted_or_removed_tags']} | Strip or filter out empty entries |

---

## 6. Generated Visualizations
The following high-resolution figures are generated in `reports/figures/`:
1. `class_distribution.png`: Comparative bar chart of stress vs. non-stress splits.
2. `text_length_distribution.png`: Histogram, KDE curves, and boxplot of post lengths.
3. `top_words_comparison.png`: Word-level frequency distribution for stressed vs non-stressed posts.
"""

    report_path.write_text(md_content, encoding="utf-8")
    print(f"Saved EDA report: {report_path}")


def save_processed_copies(df_train: pd.DataFrame, df_test: pd.DataFrame, processed_dir: Path):
    """Save clean processed copies in data/processed."""
    train_clean_path = processed_dir / "dreaddit_train.csv"
    test_clean_path = processed_dir / "dreaddit_test.csv"
    df_train.to_csv(train_clean_path, index=False)
    df_test.to_csv(test_clean_path, index=False)
    print(f"Saved processed train data -> {train_clean_path}")
    print(f"Saved processed test data -> {test_clean_path}")


def run_pipeline(base_dir: Path = None):
    """Execute complete data preparation and EDA pipeline."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    raw_dir = base_dir / "data" / "raw"
    proc_dir = base_dir / "data" / "processed"
    fig_dir = base_dir / "reports" / "figures"
    report_path = base_dir / "reports" / "eda_summary.md"

    setup_directories(base_dir)

    print("=" * 70)
    print("MindSight: Running Data Ingestion & EDA Pipeline")
    print("=" * 70)

    # 1. Load Data
    df_train, df_test = load_raw_data(raw_dir)

    # 2. EDA Statistics & Output to Console
    df_comb = pd.concat([df_train, df_test], ignore_index=True)
    df_comb_stats = compute_text_statistics(df_comb)

    print("\n--- Class Distribution (Train) ---")
    print(df_train["label"].value_counts())
    print(df_train["label"].value_counts(normalize=True) * 100)

    print("\n--- Class Distribution (Test) ---")
    print(df_test["label"].value_counts())
    print(df_test["label"].value_counts(normalize=True) * 100)

    print("\n--- Text Length Summary (Combined Word Counts) ---")
    print(df_comb_stats.groupby("label")["word_count"].describe())

    print("\n--- Missing Values Check ---")
    print("Train missing count:\n", df_train[["text", "label", "subreddit", "confidence", "social_timestamp"]].isnull().sum())
    print("Test missing count:\n", df_test[["text", "label", "subreddit", "confidence", "social_timestamp"]].isnull().sum())

    print("\n--- Example Posts (Label 1: Stressed) ---")
    stressed_samples = df_train[df_train["label"] == 1]["text"].head(5)
    for idx, post in enumerate(stressed_samples, 1):
        print(f"[{idx}] {post[:160]}...\n")

    print("--- Example Posts (Label 0: Not Stressed) ---")
    non_stressed_samples = df_train[df_train["label"] == 0]["text"].head(5)
    for idx, post in enumerate(non_stressed_samples, 1):
        print(f"[{idx}] {post[:160]}...\n")

    # 3. Generate EDA Figures
    print("\nGenerating EDA figures...")
    generate_eda_figures(df_train, df_test, fig_dir)

    # 4. Export Summary Report
    print("Exporting summary report...")
    export_eda_report(df_train, df_test, report_path)

    # 5. Save Processed Copies
    save_processed_copies(df_train, df_test, proc_dir)

    print("\n" + "=" * 70)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
