"""
MindSight: AI-Powered Mental Health Risk Detection Through Conversational Analytics
Module: Contextual Transformer Sequence Classifier & Tokenizer Helpers
File: src/models/transformer_classifier.py
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase


class TransformerMentalHealthDataset(Dataset):
    """PyTorch Dataset for transformer tokenized text sequences and risk labels."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[Union[List[int], np.ndarray]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 160,
    ):
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.labels = np.array(labels, dtype=np.int64) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_transformer_pipeline(
    model_name_or_path: str = "distilbert-base-uncased",
    num_labels: int = 2,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load pretrained transformer model and tokenizer for binary mental health risk classification.
    
    Compatible backbones:
    - 'distilbert-base-uncased' (Fast, lightweight, highly efficient)
    - 'mental/mental-bert-base-uncased' (Domain-specific mental health pretrained BERT)
    - 'roberta-base' (Robust contextual language model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )
    return model, tokenizer


def create_transformer_dataloaders(
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = 160,
    batch_size: int = 16,
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create batched DataLoaders for fine-tuning transformer models."""
    # Allow both train_df / df_train and val_df / df_val
    if train_df is None and "df_train" in kwargs:
        train_df = kwargs["df_train"]
    if val_df is None and "df_val" in kwargs:
        val_df = kwargs["df_val"]

    if train_df is None or val_df is None:
        raise ValueError("Both train_df and val_df must be provided.")

    train_ds = TransformerMentalHealthDataset(
        texts=train_df[text_col].tolist(),
        labels=train_df[label_col].values,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_ds = TransformerMentalHealthDataset(
        texts=val_df[text_col].tolist(),
        labels=val_df[label_col].values,
        tokenizer=tokenizer,
        max_length=max_length,
    )

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
