import numpy as np
import pytest

from src.api.main import compute_linguistic_signals
from src.app.client import MindSightClient


# ---------------------------------------------------------------------------
# BiLSTM Architecture & Numerical Stability Checks (Requires PyTorch)
# ---------------------------------------------------------------------------


def test_attention_pooling_forward_and_mask():
    """Verify AttentionPooling computes valid normalized weights summing to 1.0 and respects masking."""
    torch = pytest.importorskip("torch")
    from src.models.bilstm import AttentionPooling

    batch_size = 4
    seq_len = 16
    hidden_dim = 64

    attn = AttentionPooling(hidden_dim=hidden_dim)
    lstm_out = torch.randn(batch_size, seq_len, hidden_dim)

    # Valid mask where second half of sequence is padded for batch item 0
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 8:] = False

    context, weights = attn(lstm_out, mask=mask)

    assert context.shape == (batch_size, hidden_dim)
    assert weights.shape == (batch_size, seq_len)

    # Weights must sum to 1.0 across sequence dimension
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Masked positions in item 0 should have close to 0 weight
    assert torch.all(weights[0, 8:] < 1e-4)


def test_bilstm_forward_shapes_and_batch_sizes():
    """Test MentalHealthBiLSTM handles various batch sizes and sequence lengths."""
    torch = pytest.importorskip("torch")
    from src.models.bilstm import MentalHealthBiLSTM

    vocab_size = 100
    model = MentalHealthBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=1,
        dropout=0.1,
    )
    model.eval()

    # Standard batch
    input_ids = torch.randint(1, vocab_size, (8, 24))
    logits = model(input_ids)
    assert logits.shape == (8,)

    # Single sample batch (edge case)
    single_input = torch.randint(1, vocab_size, (1, 10))
    single_logits = model(single_input)
    assert single_logits.shape == (1,)

    # All-padding tokens (edge case)
    pad_input = torch.zeros((2, 15), dtype=torch.long)
    pad_logits = model(pad_input)
    assert pad_logits.shape == (2,)
    assert not torch.isnan(pad_logits).any()


def test_bilstm_predict_proba_bounds():
    """Ensure predicted probabilities are strictly bounded between [0.0, 1.0]."""
    torch = pytest.importorskip("torch")
    from src.models.bilstm import MentalHealthBiLSTM

    vocab_size = 50
    model = MentalHealthBiLSTM(vocab_size=vocab_size, embedding_dim=16, hidden_dim=16, num_layers=1)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (12, 30))
    probs = model.predict_proba(input_ids)

    assert probs.shape == (12,)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    assert not torch.isnan(probs).any()


# ---------------------------------------------------------------------------
# Transformer Dataset Edge Cases
# ---------------------------------------------------------------------------


class DummyTokenizer:
    def __call__(self, text, padding, truncation, max_length, return_tensors):
        torch = pytest.importorskip("torch")
        return {
            "input_ids": torch.randint(0, 100, (1, max_length)),
            "attention_mask": torch.ones((1, max_length)),
        }


def test_transformer_dataset_edge_cases():
    """Verify TransformerMentalHealthDataset handles empty text, varying lengths, and optional labels."""
    pytest.importorskip("torch")
    from src.models.transformer_classifier import TransformerMentalHealthDataset

    texts = ["", "Short text", "A " * 200]
    labels = np.array([0, 1, 0])
    tokenizer = DummyTokenizer()

    # With labels
    ds_labeled = TransformerMentalHealthDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_length=32)
    assert len(ds_labeled) == 3
    item0 = ds_labeled[0]
    assert "input_ids" in item0
    assert "attention_mask" in item0
    assert "labels" in item0
    assert item0["labels"].item() == 0

    # Without labels
    ds_unlabeled = TransformerMentalHealthDataset(texts=texts, labels=None, tokenizer=tokenizer, max_length=32)
    item_unlabeled = ds_unlabeled[1]
    assert "labels" not in item_unlabeled


# ---------------------------------------------------------------------------
# Linguistic Model Monotonicity & Sensitivity Regressions
# ---------------------------------------------------------------------------


def test_monotonic_risk_increase_on_distress_injection():
    """Adding distress markers into neutral text should monotonically increase risk confidence."""
    base_text = "I am preparing my notes for work tomorrow morning."
    base_score = compute_linguistic_signals(base_text)["confidence"]

    stressed_text_1 = base_text + " I feel anxious and stressed."
    score_1 = compute_linguistic_signals(stressed_text_1)["confidence"]

    stressed_text_2 = stressed_text_1 + " I feel hopeless, depressed, and overwhelmed with panic."
    score_2 = compute_linguistic_signals(stressed_text_2)["confidence"]

    assert score_1 > base_score
    assert score_2 > score_1


def test_monotonic_risk_decrease_on_protective_injection():
    """Adding positive recovery markers should decrease distress confidence."""
    base_text = "I am struggling with my routine."
    base_score = compute_linguistic_signals(base_text)["confidence"]

    positive_text = base_text + " But I feel calm, happy, grateful and supported today."
    positive_score = compute_linguistic_signals(positive_text)["confidence"]

    assert positive_score < base_score


def test_casing_and_punctuation_invariance():
    """Heuristic scoring should be case-insensitive."""
    lower_text = "i feel anxious, hopeless and deeply depressed every single day"
    upper_text = "I FEEL ANXIOUS, HOPELESS AND DEEPLY DEPRESSED EVERY SINGLE DAY"

    res_lower = compute_linguistic_signals(lower_text)
    res_upper = compute_linguistic_signals(upper_text)

    assert res_lower["confidence"] == res_upper["confidence"]
    assert res_lower["features"]["negation_count"] == res_upper["features"]["negation_count"]


def test_extreme_text_payloads_stability():
    """Regression test: massive text payload (10,000 words) executes without memory error or freeze."""
    large_text = "I feel stressed and anxious. " * 2000
    res = compute_linguistic_signals(large_text)
    assert res["confidence"] > 0.50
    assert res["features"]["word_count"] > 1000
    assert len(res["explanation"]) <= 10  # Top 10 capped


def test_foreign_and_emoji_characters():
    """Handles emojis, symbols, and non-ASCII characters gracefully."""
    text = "Feeling so stressed 😭😭😭 焦虑 overwhelmed 💔"
    res = compute_linguistic_signals(text)
    assert res["confidence"] >= 0.0
    assert isinstance(res["features"]["sentiment_valence"], float)
