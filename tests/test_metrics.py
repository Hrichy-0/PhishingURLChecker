"""Count metrics, Wilson intervals and component-clustered bootstrap draws."""

import numpy as np
import pandas as pd
import pytest

from phishguard.metrics import (
    cluster_bootstrap,
    component_strata,
    count_metrics,
    wilson,
)


def test_count_metrics_guards_empty_classes():
    assert count_metrics([90, 10, 20, 80])["false_positive_rate"] == 0.1
    assert count_metrics([0, 0, 0, 0])["recall"] == 0


def test_wilson_matches_known_interval():
    assert wilson(0, 0) is None
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)


def test_strata_keep_components_inside_folds():
    frame = pd.DataFrame({"component_id": [7, 7, 9, 4], "fold": [1, 1, 2, 2]})
    groups, strata = component_strata(frame)
    assert groups[0] == groups[1]
    assert [sorted(s) for s in strata] == [[groups[0]], sorted({groups[2], groups[3]})]
    frame.loc[1, "fold"] = 2
    with pytest.raises(ValueError, match="crosses outer folds"):
        component_strata(frame)


def test_bootstrap_skips_single_class_draws_and_is_seeded():
    counts = np.array([[5, 0, 0, 0], [0, 0, 0, 5]])  # one all-benign, one all-phish
    strata = [np.array([0, 1])]
    draws = list(cluster_bootstrap(counts, strata, 50, seed=3))
    assert 0 < len(draws) < 50
    assert all(d[0] + d[1] and d[2] + d[3] for d in draws)
    again = list(cluster_bootstrap(counts, strata, 50, seed=3))
    assert all(np.array_equal(x, y) for x, y in zip(draws, again, strict=True))
    with pytest.raises(ValueError, match="positive"):
        list(cluster_bootstrap(counts, strata, 0, seed=3))
