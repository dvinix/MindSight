"""
MindSight: AI-Powered Mental Health Risk Detection Through Conversational Analytics
Module: Sequential Tokenizer, Vocabulary Builder & PyTorch Dataset Pipeline
File: src/features/tokenizer.py
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object
    DataLoader = None


class TextTokenizer:
    """
    Vocabulary builder and sequence tokenizer for PyTorch sequential models (BiLSTM).
    Handles special tokens, frequency filtering, truncation, and padding.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        max_vocab_size: int = 15000,
        min_freq: int = 2,
        max_seq_len: int = 160,
    ):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len

        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.idx2word: Dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }
        self.vocab_size: int = 2

    @staticmethod
    def clean_text(text: str) -> List[str]:
        """Tokenize and standardize text for sequential modeling."""
        if pd.isna(text) or not str(text).strip():
            return []
        text_str = str(text).lower()
        # Standardize URLs, subreddits, users
        text_str = re.sub(r"https?://\S+|www\.\S+|<url>", " <url> ", text_str)
        text_str = re.sub(r"\br/[A-Za-z0-9_]+", " <subreddit> ", text_str)
        text_str = re.sub(r"\bu/[A-Za-z0-9_-]+", " <user> ", text_str)
        # Tokenize words and retain meaningful punctuation tokens
        tokens = re.findall(r"\b\w+\b|[!?]", text_str)
        return tokens

    def fit(self, texts: Union[List[str], pd.Series]) -> "TextTokenizer":
        """Build vocabulary from a collection of training texts."""
        token_counter = Counter()
        for text in texts:
            tokens = self.clean_text(text)
            token_counter.update(tokens)

        # Filter by minimum frequency and take top max_vocab_size
        filtered_words = [
            word
            for word, freq in token_counter.most_common(self.max_vocab_size - 2)
            if freq >= self.min_freq
        ]

        for word in filtered_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary successfully built: {self.vocab_size:,} unique tokens")
        return self

    def encode(self, text: str) -> Tuple[List[int], int]:
        """Convert raw text to fixed-length integer index sequence."""
        tokens = self.clean_text(text)
        unk_idx = self.word2idx[self.UNK_TOKEN]
        pad_idx = self.word2idx[self.PAD_TOKEN]

        indices = [self.word2idx.get(t, unk_idx) for t in tokens]
        original_length = min(len(indices), self.max_seq_len)

        # Truncate or Pad to max_seq_len
        if len(indices) < self.max_seq_len:
            indices = indices + [pad_idx] * (self.max_seq_len - len(indices))
        else:
            indices = indices[: self.max_seq_len]

        return indices, max(original_length, 1)

    def save_vocab(self, file_path: Union[str, Path]):
        """Serialize vocabulary mappings to JSON."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "max_seq_len": self.max_seq_len,
                    "vocab_size": self.vocab_size,
                },
                f,
                indent=2,
            )
        print(f"Saved vocabulary -> {file_path}")

    @classmethod
    def load_vocab(cls, file_path: Union[str, Path]) -> "TextTokenizer":
        """Load vocabulary from JSON file."""
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(max_seq_len=data["max_seq_len"])
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(idx): word for word, idx in tokenizer.word2idx.items()}
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for conversational text sequences and risk labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TextTokenizer,
        text_col: str = "text",
        label_col: Optional[str] = "label",
    ):
        self.texts = df[text_col].tolist()
        self.has_labels = label_col in df.columns if label_col else False
        self.labels = df[label_col].values.astype(np.float32) if self.has_labels else None
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        seq, length = self.tokenizer.encode(text)

        item = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
        if self.has_labels:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: TextTokenizer,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation splits."""
    train_ds = MentalHealthDataset(train_df, tokenizer)
    val_ds = MentalHealthDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
