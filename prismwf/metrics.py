"""Evaluation metrics used by the PrismWF experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Return the fraction of relevant labels among the top-k predictions."""
    top_k = np.argsort(y_score, axis=1)[:, -k:]
    hits = np.take_along_axis(y_true, top_k, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Return MAP@k following the multi-label protocol used by ARES."""
    return float(np.mean([precision_at_k(y_true, y_score, i) for i in range(1, k + 1)]))


def fixed_k_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict[str, float]:
    return {
        "AUC": float(roc_auc_score(y_true, y_score, average="macro")),
        f"P@{k}": precision_at_k(y_true, y_score, k),
        f"MAP@{k}": map_at_k(y_true, y_score, k),
    }


def threshold_predictions(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (y_score >= threshold).astype(np.int64)


def unknown_k_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = threshold_predictions(y_score, threshold)
    return {
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_set_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "label_count_mae": float(np.mean(np.abs(y_true.sum(axis=1) - y_pred.sum(axis=1)))),
    }


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Select one global threshold by validation micro-F1."""
    if candidates is None:
        candidates = np.linspace(0.01, 0.99, 99)
    scored = [(float(t), unknown_k_metrics(y_true, y_score, float(t))) for t in candidates]
    return max(scored, key=lambda item: (item[1]["micro_f1"], -item[0]))
