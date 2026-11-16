"""The fixed native XGBoost candidate search shared by the training scripts."""

from __future__ import annotations

import json
from typing import Any

import xgboost as xgb

from phishguard.native import choose_threshold, evaluate_probabilities

DEPTHS = (3, 5, 7)
EARLY_STOPPING_ROUNDS = 40


def candidate_parameters(
    depth: int, *, device: str, threads: int, seed: int
) -> dict[str, Any]:
    """Deliberately small, regularized search space; only depth varies."""
    return {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": device,
        "nthread": threads,
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": depth,
        "min_child_weight": 10 if depth == 3 else 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 5.0,
        "alpha": 0.1,
        "seed": seed,
    }


def fit_native_candidates(
    train: xgb.DMatrix,
    validation: xgb.DMatrix,
    *,
    trees: int,
    threads: int,
    seed: int,
    device: str = "cpu",
    depths: tuple[int, ...] = DEPTHS,
) -> tuple[xgb.Booster, dict[str, Any], list[dict[str, Any]]]:
    """Fit each depth with early stopping and keep the best validation F1.

    Each candidate keeps only its early-stopped trees and is scored at its own
    validation cutoff (F1 with at least 90% recall). Test data is never seen.
    Returns the selected booster, its parameters and the full search record.
    """
    labels = validation.get_label().astype(int)
    best, best_f1, best_params, search = None, -1.0, None, []
    for depth in depths:
        params = candidate_parameters(depth, device=device, threads=threads, seed=seed)
        print(f"  Candidate depth={depth} on {device}, {threads} threads", flush=True)
        model = xgb.train(
            params,
            train,
            num_boost_round=trees,
            evals=[(validation, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=50,
        )
        config = json.loads(model.save_config())
        if device == "cuda" and not config["learner"]["generic_param"][
            "device"
        ].startswith("cuda"):
            raise RuntimeError("CUDA was requested but XGBoost fell back to CPU")
        selected = model[: model.best_iteration + 1]
        scores = selected.predict(validation)
        metrics = evaluate_probabilities(
            labels, scores, choose_threshold(labels, scores)
        )
        search.append(
            {
                "parameters": params,
                "trees": selected.num_boosted_rounds(),
                "validation_metrics": metrics,
            }
        )
        if metrics["f1"] > best_f1:
            best, best_f1, best_params = selected, metrics["f1"], params
    assert best is not None
    return best, best_params, search
