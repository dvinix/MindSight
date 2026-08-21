"""
MindSight: AI-Powered Mental Health Risk Detection Through Conversational Analytics
Module: Bidirectional LSTM (BiLSTM) with Self-Attention Mechanism
File: src/models/bilstm.py
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Self-Attention pooling mechanism that computes a learned weighted average
    of hidden states across the sequence, focusing on high-salience tokens.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(
        self,
        lstm_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] boolean mask (True for valid tokens, False for padding)
        Returns:
            context_vector: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len]
        """
        # Calculate attention energy: [batch_size, seq_len, 1]
        energy = self.projection(lstm_outputs)
        energy = energy.squeeze(-1)  # [batch_size, seq_len]

        if mask is not None:
            energy = energy.masked_fill(~mask, -1e9)

        attention_weights = F.softmax(energy, dim=-1)  # [batch_size, seq_len]
        # Weighted sum: [batch_size, hidden_dim]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)

        return context_vector, attention_weights


class MentalHealthBiLSTM(nn.Module):
    """
    Deep Bidirectional LSTM Architecture with Attention Pooling for conversational distress detection.
    Combines:
    1. Word Embedding with Dropout.
    2. 2-layer Bidirectional LSTM capturing forward & backward context.
    3. Hybrid Attention + Global Max Pooling to capture both contextual narrative and sharp local crisis tokens.
    4. Dense MLP Classification Head with Layer Normalization and Dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.embed_dropout = nn.Dropout(dropout)

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2  # Bidirectional (forward + backward)

        # 3. Attention Pooling
        self.attention = AttentionPooling(hidden_dim=lstm_output_dim)

        # 4. Multi-Layer Perceptron Classification Head
        # Concatenates Attention Context + Global Max Pooling (2 * lstm_output_dim)
        classifier_input_dim = lstm_output_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: [batch_size, seq_len] tensor of word indices.
            lengths: [batch_size] tensor of actual unpadded sequence lengths.
        Returns:
            logits: [batch_size] raw unnormalized risk score logits.
        """
        # Create padding mask: [batch_size, seq_len] (True for valid tokens)
        mask = input_ids != 0

        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embed_dropout(self.embedding(input_ids))

        # LSTM: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(embedded)

        # 1. Attention-pooled representation: [batch_size, hidden_dim * 2]
        attn_context, _ = self.attention(lstm_out, mask=mask)

        # 2. Global Max-pooled representation: [batch_size, hidden_dim * 2]
        masked_lstm = lstm_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pooled, _ = torch.max(masked_lstm, dim=1)

        # Combine both representations: [batch_size, hidden_dim * 4]
        combined = torch.cat([attn_context, max_pooled], dim=-1)

        # Final classification logits: [batch_size, 1] -> [batch_size]
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Helper to get calibrated sigmoid probability scores [0.0, 1.0]."""
        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = torch.sigmoid(logits)
        return probs
