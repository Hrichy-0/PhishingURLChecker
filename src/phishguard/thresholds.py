"""Validation-only cutoff selection shared by training and evaluation scripts.

Every function here picks a cutoff on validation scores; none may see test data.
The legacy joblib model keeps its own ``modeling.choose_threshold`` (ties pick the
lower cutoff) so rollback thresholds stay reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phishguard.native import choose_threshold, ranking_curve

__all__ = [
    "ABSTAIN",
    "choose_threshold",
    "max_recall_at_fpr",
    "per_source_max_recall_at_fpr",
]

# Above every score in [0, 1] even after a float32 cast; float64 nextafter(1.0)
# rounds back to 1.0 against float32 XGBoost scores and would alert on 1.0.
ABSTAIN = float(np.nextafter(np.float32(1), np.float32(np.inf)))


def max_recall_at_fpr(labels, scores, maximum_fpr: float = 0.01) -> float:
    """Maximize validation recall within an empirical false-alarm budget.

    The budget is NOT a guarantee on new data. Ties prefer the higher cutoff
    (fewer alerts). Returns ``ABSTAIN`` when no phishing row fits the budget.
    """
    if not 0 <= maximum_fpr < 1:
        raise ValueError("maximum_fpr must be at least 0 and less than 1")
    thresholds, _, recalls, fprs = ranking_curve(labels, scores)
    eligible = np.flatnonzero(fprs <= maximum_fpr)
    if not len(eligible) or recalls[eligible].max() == 0:
        return ABSTAIN
    return float(thresholds[eligible[np.argmax(recalls[eligible])]])


def per_source_max_recall_at_fpr(
    validation: pd.DataFrame, scores: np.ndarray, maximum_fpr: float
) -> float:
    """Meet the budget pooled and in each two-class validation source."""
    labels = validation["is_phishing"].to_numpy()
    thresholds = [max_recall_at_fpr(labels, scores, maximum_fpr)]
    for source in validation["source"].unique():
        mask = validation["source"].eq(source).to_numpy()
        if set(labels[mask]) == {0, 1}:
            thresholds.append(
                max_recall_at_fpr(labels[mask], scores[mask], maximum_fpr)
            )
    return max(thresholds)
