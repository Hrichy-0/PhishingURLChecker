"""Model construction, threshold selection, and evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phishguard.features import URLFeatureExtractor


def build_model(random_state: int = 42) -> Pipeline:
    """Build an interpretable URL feature and logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("features", URLFeatureExtractor()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.90,
) -> float:
    """Choose the best-F1 threshold while meeting a phishing-recall target."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    eligible = recall >= minimum_recall

    if eligible.any():
        eligible_indices = np.flatnonzero(eligible)
        best_index = eligible_indices[np.argmax(f1[eligible])]
    else:
        best_index = int(np.argmax(f1))

    return float(thresholds[best_index])


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate probabilities at a selected operating threshold."""
    predictions = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }
