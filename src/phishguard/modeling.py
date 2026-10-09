"""Model construction, threshold selection, and evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

from phishguard.features import FEATURE_NAMES, URLFeatureExtractor
from phishguard.text_features import URLTextNormalizer

TRAINING_FEATURES = tuple(name for name in FEATURE_NAMES if name != "slash_count")

# A small, reproducible validation search. Test data never chooses the winner.
XGBOOST_CANDIDATES = (
    {"max_depth": 3, "min_child_weight": 10},
    {"max_depth": 5, "min_child_weight": 5},
    {"max_depth": 7, "min_child_weight": 5},
)


def build_model(
    random_state: int = 42,
    algorithm: str = "xgboost",
    feature_names: tuple[str, ...] = TRAINING_FEATURES,
    **classifier_params: Any,
) -> Pipeline:
    """Build a dataset-independent URL transformer and binary classifier."""
    if algorithm == "char_logistic":
        params = {
            "C": 1.0,
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 1_000,
            "random_state": random_state,
        }
        params.update(classifier_params)
        return Pipeline(
            [
                ("normalize", URLTextNormalizer()),
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=100_000,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                ("classifier", LogisticRegression(**params)),
            ]
        )
    features = URLFeatureExtractor(feature_names=feature_names)
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "reg_alpha": 0.1,
            "n_jobs": 4,
            "random_state": random_state,
        }
        params.update(classifier_params)
        return Pipeline(
            [("features", features), ("classifier", XGBClassifier(**params))]
        )
    if algorithm != "logistic":
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return Pipeline(
        steps=[
            ("features", features),
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


def fit_xgboost_candidates(
    train_urls: Any,
    train_labels: np.ndarray,
    validation_urls: Any,
    validation_labels: np.ndarray,
    minimum_recall: float = 0.90,
    random_state: int = 42,
) -> tuple[Pipeline, list[dict[str, Any]]]:
    """Fit/stop trees on validation logloss, then select by validation F1."""
    extractor = URLFeatureExtractor(feature_names=TRAINING_FEATURES)
    train_features = extractor.fit_transform(train_urls)
    validation_features = extractor.transform(validation_urls)
    best_model: Pipeline | None = None
    best_f1 = -1.0
    search_results = []

    for params in XGBOOST_CANDIDATES:
        model = build_model(
            random_state=random_state, early_stopping_rounds=40, **params
        )
        classifier = model.named_steps["classifier"]
        classifier.fit(
            train_features,
            train_labels,
            eval_set=[(validation_features, validation_labels)],
            verbose=False,
        )
        # eval_set expects transformed numbers, not URL strings. Attach the
        # fitted transformer to the fitted classifier for identical inference.
        model.steps[0] = ("features", extractor)
        probabilities = classifier.predict_proba(validation_features)[:, 1]
        threshold = choose_threshold(
            validation_labels, probabilities, minimum_recall=minimum_recall
        )
        metrics = evaluate_probabilities(validation_labels, probabilities, threshold)
        search_results.append(
            {
                "params": params,
                "best_iteration": int(classifier.best_iteration),
                "validation_metrics": metrics,
            }
        )
        print(
            f"depth={params['max_depth']}: "
            f"best round={classifier.best_iteration + 1}, "
            f"validation F1={metrics['f1']:.4f}",
            flush=True,
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model

    assert best_model is not None
    return best_model, search_results


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
        "false_positive_rate": float(
            false_positive / max(1, true_negative + false_positive)
        ),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }
