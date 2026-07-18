"""Train, evaluate, and package the PhishGuard model."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from phishguard import __version__
from phishguard.config import (
    ARTIFACTS_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
)
from phishguard.features import FEATURE_NAMES
from phishguard.modeling import (
    build_model,
    choose_threshold,
    evaluate_probabilities,
)


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/split_data.py first."
        )
    return pd.read_csv(path)


def print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(f"\n{name.upper()}")
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ):
        print(f"{metric}: {float(metrics[metric]):.4f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")


def main() -> None:
    train_data = load_split("train")
    validation_data = load_split("validation")
    test_data = load_split("test")

    train_urls = train_data["url"]
    train_labels = train_data["is_phishing"].to_numpy()
    validation_labels = validation_data["is_phishing"].to_numpy()
    test_labels = test_data["is_phishing"].to_numpy()

    print(f"Training rows: {len(train_data):,}")
    print("Fitting dummy baseline...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train_labels), 1)), train_labels)
    dummy_probabilities = dummy.predict_proba(np.zeros((len(test_labels), 1)))[
        :, list(dummy.classes_).index(1)
    ]
    dummy_metrics = evaluate_probabilities(test_labels, dummy_probabilities, 0.5)

    print("Fitting logistic-regression pipeline...")
    model = build_model()
    model.fit(train_urls, train_labels)

    validation_probabilities = model.predict_proba(validation_data["url"])[:, 1]
    threshold = choose_threshold(validation_labels, validation_probabilities)
    validation_metrics = evaluate_probabilities(
        validation_labels, validation_probabilities, threshold
    )

    test_probabilities = model.predict_proba(test_data["url"])[:, 1]
    test_metrics = evaluate_probabilities(test_labels, test_probabilities, threshold)

    print_metrics("dummy baseline (test)", dummy_metrics)
    print(f"\nSelected threshold from validation data: {threshold:.4f}")
    print_metrics("validation", validation_metrics)
    print_metrics("test", test_metrics)

    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_pairs = sorted(
        zip(FEATURE_NAMES, coefficients, strict=True), key=lambda item: item[1]
    )
    top_legitimate = [
        {"feature": name, "coefficient": float(value)}
        for name, value in coefficient_pairs[:5]
    ]
    top_phishing = [
        {"feature": name, "coefficient": float(value)}
        for name, value in reversed(coefficient_pairs[-5:])
    ]

    metadata = {
        "model_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "model_type": "StandardScaler + LogisticRegression",
        "label_convention": {"0": "legitimate", "1": "phishing"},
        "threshold": threshold,
        "minimum_validation_recall_target": 0.90,
        "dataset_rows": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "features": list(FEATURE_NAMES),
        "dummy_test_metrics": dummy_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "top_phishing_features": top_phishing,
        "top_legitimate_features": top_legitimate,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metadata: {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
