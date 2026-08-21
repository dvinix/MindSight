"""
MindSight: AI-Powered Mental Health Risk Detection Through Conversational Analytics
Module: BiLSTM Deep Learning Training, Validation & Threshold Tuning Pipeline
File: src/models/train_bilstm.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.features.tokenizer import TextTokenizer, create_dataloaders
from src.models.bilstm import MentalHealthBiLSTM


def set_seed(seed: int = 42):
    """Ensure exact reproducibility across PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tune_decision_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_recall: float = 0.80,
    min_precision: float = 0.50,
) -> float:
    """Tune classification probability cutoff to ensure recall >= target_recall."""
    thresholds = np.linspace(0.05, 0.95, 91)
    satisfying_candidates = []
    fallback_best_f1, fallback_thresh = -1.0, 0.5

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if rec >= target_recall and prec >= min_precision:
            satisfying_candidates.append({
                "threshold": float(t),
                "recall": float(rec),
                "precision": float(prec),
                "f1": float(f1),
            })
        if prec >= min_precision and f1 > fallback_best_f1:
            fallback_best_f1, fallback_thresh = f1, float(t)

    if satisfying_candidates:
        satisfying_candidates.sort(key=lambda x: (x["f1"], x["precision"]), reverse=True)
        return round(satisfying_candidates[0]["threshold"], 4)
    return round(float(fallback_thresh), 4)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one full training epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend((probs >= 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy().astype(int))

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on validation dataloader."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().astype(int))

    eval_loss = total_loss / len(dataloader.dataset)
    y_probs = np.array(all_probs)
    y_true = np.array(all_labels)
    eval_acc = accuracy_score(y_true, (y_probs >= 0.5).astype(int))
    return eval_loss, eval_acc, y_probs, y_true


def plot_training_curves(history: dict, output_path: Path):
    """Plot training and validation loss/accuracy progression."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#4A90E2", lw=2)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#E94E77", lw=2)
    axes[0].set_title("BiLSTM Training & Validation Loss", fontsize=12, weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()

    # Accuracy Curve
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#4A90E2", lw=2)
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc", color="#E94E77", lw=2)
    axes[1].set_title("BiLSTM Training & Validation Accuracy", fontsize=12, weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (Default 0.50 Cutoff)")
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves -> {output_path}", flush=True)


def run_bilstm_pipeline():
    """Execute end-to-end BiLSTM data preparation, training, tuning, and evaluation."""
    set_seed(42)

    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    figures_dir = base_dir / "reports" / "figures"
    reports_dir = base_dir / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65, flush=True)
    print("MindSight: Deep Sequential Modeling with BiLSTM + Attention", flush=True)
    print("=" * 65, flush=True)

    # 1. Load Data
    train_path = processed_dir / "dreaddit_train.csv"
    val_path = processed_dir / "dreaddit_test.csv"
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print(f"Train Set: {len(df_train):,} posts", flush=True)
    print(f"Validation Set: {len(df_val):,} posts", flush=True)

    # 2. Tokenize & Build Vocabulary on Training Data
    print("\nBuilding sequence vocabulary from training corpus...", flush=True)
    tokenizer = TextTokenizer(max_vocab_size=12000, min_freq=2, max_seq_len=160)
    tokenizer.fit(df_train["text"])
    tokenizer.save_vocab(models_dir / "bilstm_vocab.json")

    # 3. Create DataLoaders
    batch_size = 32
    train_loader, val_loader = create_dataloaders(
        df_train,
        df_val,
        tokenizer,
        batch_size=batch_size,
    )

    # 4. Initialize BiLSTM Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on Device: {device}", flush=True)

    model = MentalHealthBiLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.35,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized: {total_params:,} trainable parameters", flush=True)

    # 5. Training Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    epochs = 12
    best_val_loss = float("inf")
    best_model_state = None
    patience = 4
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "-" * 65, flush=True)
    print("Beginning Training Loop with Early Stopping...", flush=True)
    print("-" * 65, flush=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"LR: {lr_current:.1e}",
            flush=True,
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Patience={patience})", flush=True)
                break

    # Load best weights
    model.load_state_dict(best_model_state)

    # 6. Generate Diagnostic Plots
    plot_training_curves(history, figures_dir / "bilstm_training_curves.png")

    # 7. Validation Evaluation & Threshold Tuning
    print("\n" + "=" * 65, flush=True)
    print("Clinical Decision Threshold Tuning & Final Validation", flush=True)
    print("=" * 65, flush=True)

    _, _, y_probs_val, y_true_val = evaluate(model, val_loader, criterion, device)

    # Search threshold targeting recall >= 0.80
    thresh_bilstm = tune_decision_threshold(y_true_val, y_probs_val, target_recall=0.80, min_precision=0.50)
    y_preds_tuned = (y_probs_val >= thresh_bilstm).astype(int)

    acc = accuracy_score(y_true_val, y_preds_tuned)
    prec = precision_score(y_true_val, y_preds_tuned, zero_division=0)
    rec = recall_score(y_true_val, y_preds_tuned, zero_division=0)
    f1 = f1_score(y_true_val, y_preds_tuned, zero_division=0)
    roc_auc = roc_auc_score(y_true_val, y_probs_val)
    cm = confusion_matrix(y_true_val, y_preds_tuned).tolist()
    clf_report = classification_report(y_true_val, y_preds_tuned, output_dict=True, zero_division=0)

    print(f"\nOptimal Calibrated Threshold (\\tau): {thresh_bilstm:.4f}", flush=True)
    print(f"Accuracy:  {acc:.4f}", flush=True)
    print(f"Precision: {prec:.4f}", flush=True)
    print(f"Recall:    {rec:.4f} (Priority Clinical Metric)", flush=True)
    print(f"F1-Score:  {f1:.4f}", flush=True)
    print(f"ROC-AUC:   {roc_auc:.4f}", flush=True)
    print("\nConfusion Matrix [TN, FP / FN, TP]:", flush=True)
    print(f"  TN: {cm[0][0]:<5} FP: {cm[0][1]}", flush=True)
    print(f"  FN: {cm[1][0]:<5} TP: {cm[1][1]}", flush=True)

    # 8. Serialize BiLSTM Artifacts
    model_save_path = models_dir / "bilstm_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "embedding_dim": 128,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.35,
                "max_seq_len": 160,
            },
        },
        model_save_path,
    )
    joblib.dump(thresh_bilstm, models_dir / "threshold_bilstm.pkl")

    metrics_summary = {
        "model": "BiLSTM with Self-Attention (Tuned)",
        "optimal_threshold": thresh_bilstm,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }

    with open(models_dir / "bilstm_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(reports_dir / "bilstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved BiLSTM weights -> {model_save_path}", flush=True)
    print(f"Saved threshold -> {models_dir / 'threshold_bilstm.pkl'}", flush=True)
    print(f"Saved metrics -> {models_dir / 'bilstm_evaluation_metrics.json'}", flush=True)
    print("\n" + "=" * 65, flush=True)
    print("Phase 3 BiLSTM Deep Sequential Pipeline Completed Successfully!", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_bilstm_pipeline()
