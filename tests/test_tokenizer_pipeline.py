import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.tokenizer import TextTokenizer


def test_tokenizer_clean_text_edge_cases():
    """Verify clean_text handles None, blank text, URLs, subreddits, usernames, and punctuation."""
    assert TextTokenizer.clean_text("") == []
    assert TextTokenizer.clean_text("   \n\t  ") == []
    assert TextTokenizer.clean_text(None) == []

    # Social and platform tags normalized to word tokens
    tokens = TextTokenizer.clean_text("Check r/anxiety and u/user123 on https://example.com/help")
    assert "subreddit" in tokens
    assert "user" in tokens
    assert "url" in tokens

    # Punctuation tokens
    punct_tokens = TextTokenizer.clean_text("Panic attack! Are you okay?")
    assert "!" in punct_tokens
    assert "?" in punct_tokens
    assert "panic" in punct_tokens


def test_tokenizer_fit_vocab_building_and_constraints():
    """Verify vocabulary building adheres to min_freq and max_vocab_size limits."""
    corpus = [
        "anxious anxious anxious panic panic calm",
        "anxious stress stress calm happy",
        "rareword1 rareword2",
    ]

    tokenizer = TextTokenizer(max_vocab_size=10, min_freq=2, max_seq_len=20)
    tokenizer.fit(corpus)

    # Special tokens must exist
    assert tokenizer.PAD_TOKEN in tokenizer.word2idx
    assert tokenizer.UNK_TOKEN in tokenizer.word2idx
    assert tokenizer.word2idx[tokenizer.PAD_TOKEN] == 0
    assert tokenizer.word2idx[tokenizer.UNK_TOKEN] == 1

    # High frequency words should be in vocabulary
    assert "anxious" in tokenizer.word2idx
    assert "calm" in tokenizer.word2idx

    # Single-occurrence words should be filtered out by min_freq=2
    assert "rareword1" not in tokenizer.word2idx
    assert "rareword2" not in tokenizer.word2idx

    # Vocab size must not exceed max_vocab_size
    assert tokenizer.vocab_size <= 10


def test_tokenizer_encode_padding_and_truncation():
    """Ensure sequences are properly padded or truncated to max_seq_len with correct unk mapping."""
    tokenizer = TextTokenizer(max_seq_len=8, min_freq=1)
    tokenizer.fit(["knownword1 knownword2 knownword3"])

    # Short sentence -> padded
    seq, length = tokenizer.encode("knownword1")
    assert len(seq) == 8
    assert seq[0] == tokenizer.word2idx["knownword1"]
    assert seq[1:] == [0] * 7
    assert length == 1

    # Long sentence -> truncated
    long_text = "knownword1 " * 20
    seq_long, length_long = tokenizer.encode(long_text)
    assert len(seq_long) == 8
    assert length_long == 8

    # OOV tokens mapped to UNK (1)
    seq_oov, _ = tokenizer.encode("completely_unknown_token_xyz")
    assert seq_oov[0] == 1


def test_tokenizer_save_and_load_roundtrip():
    """Verify serialization to JSON and reload maintains identical dictionary mappings."""
    tokenizer = TextTokenizer(max_seq_len=16, min_freq=1)
    tokenizer.fit(["depression anxiety panic support therapy recovery"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        vocab_path = Path(tmp_dir) / "vocab.json"
        tokenizer.save_vocab(vocab_path)
        assert vocab_path.exists()

        loaded_tokenizer = TextTokenizer.load_vocab(vocab_path)
        assert loaded_tokenizer.max_seq_len == tokenizer.max_seq_len
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        assert loaded_tokenizer.word2idx == tokenizer.word2idx
        assert loaded_tokenizer.idx2word == tokenizer.idx2word


def test_mental_health_dataset_and_dataloaders():
    """Verify PyTorch dataset and dataloaders return well-formed batches with correct dtypes."""
    torch = pytest.importorskip("torch")
    from src.features.tokenizer import create_dataloaders

    tokenizer = TextTokenizer(max_seq_len=12, min_freq=1)
    df_train = pd.DataFrame({
        "text": ["I feel overwhelmed with panic.", "Today was a good day.", "Exhausted and tired."],
        "label": [1, 0, 1]
    })
    df_val = pd.DataFrame({
        "text": ["Calm and relaxed.", "Stressed about tests."],
        "label": [0, 1]
    })
    tokenizer.fit(df_train["text"])

    train_loader, val_loader = create_dataloaders(df_train, df_val, tokenizer, batch_size=2)

    for batch in train_loader:
        assert "input_ids" in batch
        assert "length" in batch
        assert "label" in batch
        assert batch["input_ids"].shape[1] == 12
        assert batch["input_ids"].dtype == torch.long
        assert batch["label"].dtype == torch.float32
        break

    for batch in val_loader:
        assert batch["input_ids"].shape[0] <= 2
        break
