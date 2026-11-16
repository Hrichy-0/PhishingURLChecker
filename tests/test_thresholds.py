"""Validation-only cutoff selection: ties, abstention and budgets."""

import numpy as np
import pandas as pd
import pytest

from phishguard.native import evaluate_probabilities
from phishguard.thresholds import (
    ABSTAIN,
    max_recall_at_fpr,
    per_source_max_recall_at_fpr,
)


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ([0, 0, 1, 1], [0.1, 0.6, 0.6, 0.9], 0.9),  # tie with a negative is excluded
        ([0, 1, 0, 1], [0.8, 0.8, 0.1, 0.9], 0.9),
        ([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.95], 0.95),
    ],
)
def test_zero_budget_keeps_highest_clean_cutoff(labels, scores, expected):
    assert max_recall_at_fpr(labels, scores, 0) == expected
    assert evaluate_probabilities(labels, scores, expected)["false_positive_rate"] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_abstains_above_every_score(dtype):
    scores = np.ones(4, dtype=dtype)
    cutoff = max_recall_at_fpr([0, 0, 1, 1], scores, 0)
    assert cutoff == ABSTAIN > 1
    assert not (scores >= np.asarray(cutoff, dtype=dtype)).any()


def test_abstains_rather_than_alerting_only_on_legitimate_rows():
    # A 50% budget admits the top legitimate row, but it catches no phishing.
    assert max_recall_at_fpr([0, 1, 0], [0.9, 0.5, 0.1], 0.5) == 0.5
    assert max_recall_at_fpr([0, 0, 1], [0.9, 0.8, 0.1], 0.5) == ABSTAIN


@pytest.mark.parametrize("budget", [-0.01, 1.0])
def test_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="maximum_fpr"):
        max_recall_at_fpr([0, 1], [0.1, 0.9], budget)


def test_per_source_cutoff_meets_budget_in_each_source():
    frame = pd.DataFrame({"source": ["a", "a", "b", "b"], "is_phishing": [0, 1, 0, 1]})
    scores = np.array([0.2, 0.6, 0.8, 0.9])
    threshold = per_source_max_recall_at_fpr(frame, scores, 0)
    assert threshold == 0.9
    for source in ("a", "b"):
        mask = frame["source"].eq(source).to_numpy()
        metrics = evaluate_probabilities(
            frame.loc[mask, "is_phishing"].to_numpy(), scores[mask], threshold
        )
        assert metrics["false_positive_rate"] == 0
