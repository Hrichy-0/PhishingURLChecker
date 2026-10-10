"""Native-training helpers that do not require scikit-learn's binary extensions."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from phishguard.domains import get_registered_domain
from phishguard.lexical import FEATURE_NAMES, extract_url_features

TRAINING_FEATURES = tuple(name for name in FEATURE_NAMES if name != "slash_count")


def url_matrix(urls: Iterable[str]) -> np.ndarray:
    rows = []
    for url in urls:
        values = extract_url_features(str(url))
        rows.append([values[name] for name in TRAINING_FEATURES])
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(TRAINING_FEATURES))


def domain_partition(data: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Assign registered domains by stable SHA-256 buckets, approximately 80/10/10."""
    domains = data["url"].map(get_registered_domain)
    buckets = {
        domain: int.from_bytes(
            hashlib.sha256(f"{seed}:{domain}".encode()).digest()[:8], "big"
        )
        / 2**64
        for domain in domains.unique()
    }
    bucket = domains.map(buckets)
    masks = {
        "train": bucket < 0.8,
        "validation": (bucket >= 0.8) & (bucket < 0.9),
        "test": bucket >= 0.9,
    }
    splits = {}
    for name, mask in masks.items():
        frame = data.loc[mask].reset_index(drop=True)
        if set(frame["is_phishing"].unique()) != {0, 1}:
            raise ValueError(f"{name}: both classes are required")
        splits[name] = frame
    domain_sets = [set(domains.loc[mask]) for mask in masks.values()]
    if any(
        not group.isdisjoint(other)
        for index, group in enumerate(domain_sets)
        for other in domain_sets[index + 1 :]
    ):
        raise ValueError("Registered-domain leakage between splits")
    return splits


def ranking_curve(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or labels.shape != probabilities.shape
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Require aligned binary labels and finite scores in [0, 1]")
    order = np.argsort(-probabilities, kind="stable")
    sorted_scores = probabilities[order]
    # Evaluate at the end of each tied-score group: alert iff score >= threshold.
    ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    indices = np.flatnonzero(ends)
    tp = np.cumsum(labels[order])[indices]
    fp = indices + 1 - tp
    precision = tp / (indices + 1)
    recall = tp / labels.sum()
    fpr = fp / (len(labels) - labels.sum())
    return sorted_scores[indices], precision, recall, fpr


def choose_threshold(labels, probabilities, minimum_recall: float = 0.90) -> float:
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    thresholds, precision, recall, _ = ranking_curve(labels, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    eligible = np.flatnonzero(recall >= minimum_recall)
    # Descending thresholds favor fewer alerts when F1 is tied.
    return float(thresholds[eligible[np.argmax(f1[eligible])]])


def evaluate_probabilities(labels, probabilities, threshold: float) -> dict:
    _, precision_curve, recall_curve, fpr_curve = ranking_curve(labels, probabilities)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predicted = probabilities >= threshold
    tp = int(((labels == 1) & predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    tn = int(((labels == 0) & ~predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / (tp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": float(np.trapezoid(np.r_[0, recall_curve], np.r_[0, fpr_curve])),
        "average_precision": float(
            (np.diff(np.r_[0, recall_curve]) * precision_curve).sum()
        ),
        "false_positive_rate": fp / (fp + tn),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }
