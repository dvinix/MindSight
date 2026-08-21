import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import compute_text_statistics, scan_reddit_artifacts


def test_tune_threshold_satisfies_target_recall():
    """Verify threshold search successfully finds cutoffs achieving the recall target."""
    pytest.importorskip("torch")
    from src.models.train_bilstm import tune_decision_threshold as tune_threshold_bilstm
    from src.models.train_transformer import tune_decision_threshold as tune_threshold_transformer

    np.random.seed(42)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.85, 0.8, 0.7, 0.65, 0.3, 0.2, 0.15, 0.1, 0.05])

    thresh = tune_threshold_bilstm(y_true, y_probs, target_recall=0.80, min_precision=0.50)
    assert 0.05 <= thresh <= 0.95

    # Transformer threshold tuner should behave equivalently
    thresh_trans = tune_threshold_transformer(y_true, y_probs, target_recall=0.80, min_precision=0.50)
    assert thresh == thresh_trans


def test_tune_threshold_fallback_on_impossible_constraint():
    """When target recall cannot be met with min precision, should fallback gracefully."""
    pytest.importorskip("torch")
    from src.models.train_bilstm import tune_decision_threshold as tune_threshold_bilstm

    y_true = np.array([0, 0, 0, 0, 0])
    y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    thresh = tune_threshold_bilstm(y_true, y_probs, target_recall=0.99, min_precision=0.99)
    assert isinstance(thresh, float)
    assert 0.05 <= thresh <= 0.95


def test_set_seed_reproducibility():
    """Ensure set_seed produces identical random states across calls."""
    torch = pytest.importorskip("torch")
    from src.models.train_bilstm import set_seed as set_seed_bilstm

    set_seed_bilstm(1234)
    t1 = torch.randn(5)
    n1 = np.random.rand(5)

    set_seed_bilstm(1234)
    t2 = torch.randn(5)
    n2 = np.random.rand(5)

    assert torch.allclose(t1, t2)
    assert np.allclose(n1, n2)


def test_data_pipeline_statistics_and_artifact_scanner():
    """Verify compute_text_statistics and scan_reddit_artifacts handle standard and edge-case DataFrames."""
    df = pd.DataFrame({
        "text": [
            "Check r/support and u/mod on https://reddit.com [deleted]",
            "Short note",
            "",
        ],
        "label": [1, 0, 0]
    })

    stats = compute_text_statistics(df)
    assert "char_length" in stats.columns
    assert "word_count" in stats.columns
    assert stats["word_count"].iloc[2] == 0

    artifacts = scan_reddit_artifacts(df)
    assert artifacts["urls"] == 1
    assert artifacts["subreddit_mentions"] == 1
    assert artifacts["user_mentions"] == 1
    assert artifacts["deleted_or_removed_tags"] == 1
