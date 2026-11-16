"""The shared native candidate search records every depth and keeps the best."""

import numpy as np
import xgboost as xgb

from phishguard.native_training import candidate_parameters, fit_native_candidates


def matrices(seed=0, rows=200):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 4)).astype(np.float32)
    labels = (values[:, 0] + 0.5 * rng.normal(size=rows) > 0).astype(int)
    return xgb.DMatrix(values, label=labels, feature_names=list("abcd"))


def test_search_keeps_best_validation_f1_with_early_stopped_trees():
    best, params, search = fit_native_candidates(
        matrices(0), matrices(1), trees=60, threads=1, seed=7, depths=(2, 4)
    )
    assert [entry["parameters"]["max_depth"] for entry in search] == [2, 4]
    chosen = max(search, key=lambda entry: entry["validation_metrics"]["f1"])
    assert params == chosen["parameters"]
    assert best.num_boosted_rounds() == chosen["trees"] <= 60


def test_parameters_are_fixed_except_depth():
    shallow = candidate_parameters(3, device="cpu", threads=2, seed=1)
    deep = candidate_parameters(7, device="cpu", threads=2, seed=1)
    assert (shallow["min_child_weight"], deep["min_child_weight"]) == (10, 5)
    changed = {key for key in shallow if shallow[key] != deep[key]}
    assert changed == {"max_depth", "min_child_weight"}
