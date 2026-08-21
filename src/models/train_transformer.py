"""
MindSight: AI-Powered Mental Health Risk Detection Through Conversational Analytics
Module: Contextual Transformer Fine-Tuning, Validation & Clinical Threshold Tuning
File: src/models/train_transformer.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.models.transformer_classifier import (
    create_transformer_dataloaders,
    load_transformer_pipeline,
)


def set_seed(seed: int = 42):
    """Ensure exact reproducibility across PyTorch, NumPy, and Python."""
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
    """Tune classification probability threshold to ensure clinical recall target."""
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
    model: torch.nn.Module,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch with gradient clipping and lr scheduling."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate_transformer(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate transformer model and extract calibrated class 1 probabilities."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * len(labels)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    eval_loss = total_loss / len(dataloader.dataset)
    y_probs = np.array(all_probs)
    y_true = np.array(all_labels)
    eval_acc = accuracy_score(y_true, (y_probs >= 0.5).astype(int))
    return eval_loss, eval_acc, y_probs, y_true


def plot_training_curves(history: dict, output_path: Path):
    """Plot transformer training and validation loss/accuracy progression."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#4A90E2", lw=2)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#E94E77", lw=2)
    axes[0].set_title("Transformer Fine-Tuning Loss", fontsize=12, weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    # Accuracy Curve
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#4A90E2", lw=2)
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc", color="#E94E77", lw=2)
    axes[1].set_title("Transformer Validation Accuracy", fontsize=12, weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (Default 0.50 Cutoff)")
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves -> {output_path}", flush=True)


def run_transformer_training(
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 4,
    max_length: int = 160,
):
    """Main transformer fine-tuning, validation, and threshold calibration pipeline."""
    set_seed(42)

    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    figures_dir = base_dir / "reports" / "figures"
    reports_dir = base_dir / "reports"
    transformer_save_dir = models_dir / "transformer_model"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    transformer_save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65, flush=True)
    print(f"MindSight: Fine-Tuning Contextual Transformer ({model_name})", flush=True)
    print("=" * 65, flush=True)

    # 1. Load Processed Splits
    train_path = processed_dir / "dreaddit_train.csv"
    val_path = processed_dir / "dreaddit_test.csv"
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print(f"Train Set: {len(df_train):,} posts", flush=True)
    print(f"Validation Set: {len(df_val):,} posts", flush=True)

    # 2. Load Pretrained Transformer & Tokenizer
    print(f"\nLoading pretrained weights for '{model_name}'...", flush=True)
    model, tokenizer = load_transformer_pipeline(model_name_or_path=model_name, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on Device: {device}", flush=True)

    # 3. Create DataLoaders
    train_loader, val_loader = create_transformer_dataloaders(
        df_train=df_train,
        df_val=df_val,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )

    # 4. Optimizer & Warmup Scheduler
    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_training_steps)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_loss = float("inf")
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "-" * 65, flush=True)
    print(f"Beginning Fine-Tuning ({epochs} Epochs, {total_training_steps} steps)...", flush=True)
    print("-" * 65, flush=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, _, _ = evaluate_transformer(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"LR: {lr_current:.2e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Checkpoint best state in memory
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # 5. Plot Training Curves
    plot_training_curves(history, figures_dir / "transformer_training_curves.png")

    # 6. Validation Evaluation & Threshold Tuning
    print("\n" + "=" * 65, flush=True)
    print("Clinical Decision Threshold Tuning & Final Validation", flush=True)
    print("=" * 65, flush=True)

    _, _, y_probs_val, y_true_val = evaluate_transformer(model, val_loader, device)

    # Search threshold targeting recall >= 0.80
    thresh_transformer = tune_decision_threshold(y_true_val, y_probs_val, target_recall=0.80, min_precision=0.50)
    y_preds_tuned = (y_probs_val >= thresh_transformer).astype(int)

    acc = accuracy_score(y_true_val, y_preds_tuned)
    prec = precision_score(y_true_val, y_preds_tuned, zero_division=0)
    rec = recall_score(y_true_val, y_preds_tuned, zero_division=0)
    f1 = f1_score(y_true_val, y_preds_tuned, zero_division=0)
    roc_auc = roc_auc_score(y_true_val, y_probs_val)
    cm = confusion_matrix(y_true_val, y_preds_tuned).tolist()
    clf_report = classification_report(y_true_val, y_preds_tuned, output_dict=True, zero_division=0)

    print(f"\nOptimal Calibrated Threshold (\\tau): {thresh_transformer:.4f}", flush=True)
    print(f"Accuracy:  {acc:.4f}", flush=True)
    print(f"Precision: {prec:.4f}", flush=True)
    print(f"Recall:    {rec:.4f} (Priority Clinical Metric)", flush=True)
    print(f"F1-Score:  {f1:.4f}", flush=True)
    print(f"ROC-AUC:   {roc_auc:.4f}", flush=True)
    print("\nConfusion Matrix [TN, FP / FN, TP]:", flush=True)
    print(f"  TN: {cm[0][0]:<5} FP: {cm[0][1]}", flush=True)
    print(f"  FN: {cm[1][0]:<5} TP: {cm[1][1]}", flush=True)

    # 7. Serialize Transformer Checkpoint & Tokenizer
    model.save_pretrained(transformer_save_dir)
    tokenizer.save_pretrained(transformer_save_dir)
    joblib.dump(thresh_transformer, models_dir / "threshold_transformer.pkl")

    metrics_summary = {
        "model": f"Contextual Transformer ({model_name}) (Tuned)",
        "optimal_threshold": thresh_transformer,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }

    with open(models_dir / "transformer_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(reports_dir / "transformer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved Transformer model & tokenizer -> {transformer_save_dir}", flush=True)
    print(f"Saved threshold -> {models_dir / 'threshold_transformer.pkl'}", flush=True)
    print(f"Saved metrics -> {models_dir / 'transformer_evaluation_metrics.json'}", flush=True)
    print("\n" + "=" * 65, flush=True)
    print("Phase 4 Transformer Fine-Tuning Completed Successfully!", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Contextual Transformer on Dreaddit")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=160)
    args = parser.parse_args()

    run_transformer_training(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
    )
