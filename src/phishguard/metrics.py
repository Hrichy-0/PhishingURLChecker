"""Count-based metrics, Wilson intervals and component-clustered bootstrap draws.

``native.evaluate_probabilities`` (frozen) remains the score-based metric source;
these helpers work from confusion counts so binary flags and resamples share them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd

COUNT_METRICS = ("accuracy", "precision", "recall", "f1", "false_positive_rate")


def count_metrics(counts) -> dict:
    """Metrics from (true negative, false positive, false negative, true positive)."""
    tn, fp, fn, tp = (int(value) for value in counts)
    return {
        "accuracy": (tn + tp) / max(1, tn + fp + fn + tp),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float] | None:
    """Wilson score interval for a proportion; None when there are no trials."""
    if total == 0:
        return None
    rate = successes / total
    centre = rate + z * z / (2 * total)
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    denominator = 1 + z * z / total
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def component_strata(frame: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    """Component index per row, and the components belonging to each outer fold."""
    if (frame.groupby("component_id").fold.nunique() != 1).any():
        raise ValueError("A component crosses outer folds")
    components, groups = np.unique(frame.component_id, return_inverse=True)
    folds = np.zeros(len(components), dtype=int)
    folds[groups] = frame.fold
    return groups, [np.flatnonzero(folds == fold) for fold in np.unique(folds)]


def cluster_bootstrap(
    counts: np.ndarray, strata: list[np.ndarray], draws: int, seed: int
) -> Iterator[np.ndarray]:
    """Resample whole components within each stratum and yield summed counts.

    ``counts`` has one leading row per component and confusion counts (length 4)
    as its last axis. Draws where any count table lacks a class are skipped: they
    cannot estimate both false-positive rate and recall.
    """
    if draws < 1:
        raise ValueError("Require positive bootstrap draws")
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        selected = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        totals = counts[selected].sum(axis=0)
        if any(
            row[0] + row[1] == 0 or row[2] + row[3] == 0
            for row in totals.reshape(-1, 4)
        ):
            continue
        yield totals
