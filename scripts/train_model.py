"""Compare XGBoost with a logistic baseline using domain-disjoint URL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.dummy import DummyClassifier

from phishguard import __version__
from phishguard.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from phishguard.modeling import (
    TRAINING_FEATURES,
    build_model,
    choose_threshold,
    evaluate_probabilities,
    fit_xgboost_candidates,
)
from phishguard.split import validate_training_splits


def load_split(name: str, data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/split_data.py first."
        )
    return pd.read_csv(path)


def print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(f"\n{name.upper()}", flush=True)
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "false_positive_rate",
    ):
        print(f"{metric}: {float(metrics[metric]):.4f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR / "experiments" / "xgboost",
        help="Experiment output directory; activate separately with activate_model.py.",
    )
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.minimum_recall <= 1:
        parser.error("--minimum-recall must be greater than 0 and at most 1")

    splits = {
        name: load_split(name, args.data_dir)
        for name in ("train", "validation", "test")
    }
    validate_training_splits(splits)
    manifest_path = args.data_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    hashes = {
        name: hashlib.sha256((args.data_dir / f"{name}.csv").read_bytes()).hexdigest()
        for name in splits
    }
    if manifest and manifest["split_sha256"] != hashes:
        raise ValueError("Split files changed since their manifest was created")

    train_data = splits["train"]
    validation_data = splits["validation"]
    test_data = splits["test"]
    train_labels = train_data["is_phishing"].to_numpy()
    validation_labels = validation_data["is_phishing"].to_numpy()
    test_labels = test_data["is_phishing"].to_numpy()

    print(f"Training rows: {len(train_data):,}", flush=True)
    print("Fitting logistic-regression comparison baseline...", flush=True)
    logistic = build_model(algorithm="logistic", random_state=args.seed)
    logistic.fit(train_data["url"], train_labels)
    logistic_threshold = choose_threshold(
        validation_labels,
        logistic.predict_proba(validation_data["url"])[:, 1],
        minimum_recall=args.minimum_recall,
    )

    print(
        "Training XGBoost candidates; early stopping uses validation only...",
        flush=True,
    )
    model, search_results = fit_xgboost_candidates(
        train_data["url"],
        train_labels,
        validation_data["url"],
        validation_labels,
        minimum_recall=args.minimum_recall,
        random_state=args.seed,
    )
    validation_probabilities = model.predict_proba(validation_data["url"])[:, 1]
    threshold = choose_threshold(
        validation_labels,
        validation_probabilities,
        minimum_recall=args.minimum_recall,
    )
    validation_metrics = evaluate_probabilities(
        validation_labels, validation_probabilities, threshold
    )

    # The search is finished. Only now evaluate the selected model on test data.
    test_probabilities = model.predict_proba(test_data["url"])[:, 1]
    test_metrics = evaluate_probabilities(test_labels, test_probabilities, threshold)
    logistic_probabilities = logistic.predict_proba(test_data["url"])[:, 1]
    logistic_metrics = evaluate_probabilities(
        test_labels, logistic_probabilities, logistic_threshold
    )
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train_labels), 1)), train_labels)
    dummy_probabilities = dummy.predict_proba(np.zeros((len(test_labels), 1)))[:, 1]
    dummy_metrics = evaluate_probabilities(test_labels, dummy_probabilities, 0.5)

    print_metrics("logistic baseline (same test split)", logistic_metrics)
    print_metrics("xgboost validation", validation_metrics)
    print_metrics("xgboost test", test_metrics)

    source_metrics = {}
    if "source" in test_data:
        for source in sorted(test_data["source"].unique()):
            mask = test_data["source"].eq(source).to_numpy()
            if set(test_labels[mask]) == {0, 1}:
                source_metrics[source] = {
                    "rows": int(mask.sum()),
                    "xgboost": evaluate_probabilities(
                        test_labels[mask], test_probabilities[mask], threshold
                    ),
                    "logistic": evaluate_probabilities(
                        test_labels[mask],
                        logistic_probabilities[mask],
                        logistic_threshold,
                    ),
                }

    classifier = model.named_steps["classifier"]
    importances = sorted(
        zip(TRAINING_FEATURES, classifier.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    metadata = {
        "model_version": f"{__version__}-xgboost",
        "created_at": datetime.now(UTC).isoformat(),
        "model_type": "URLFeatureExtractor + XGBClassifier",
        "label_convention": {"0": "legitimate", "1": "phishing"},
        "threshold": threshold,
        "minimum_validation_recall_target": args.minimum_recall,
        "selection_method": (
            "highest validation F1; early stopping on validation logloss"
        ),
        "seed": args.seed,
        "dataset_rows": {name: len(frame) for name, frame in splits.items()},
        "data_dir": str(args.data_dir.resolve()),
        "split_sha256": hashes,
        "split_manifest": manifest,
        "features": list(TRAINING_FEATURES),
        "dropped_features": ["slash_count"],
        "best_iteration": int(classifier.best_iteration),
        "parameters": classifier.get_params(),
        "validation_search": search_results,
        "dummy_test_metrics": dummy_metrics,
        "logistic_test_metrics": logistic_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_source": source_metrics,
        "feature_importances": [
            {"feature": name, "gain_importance": float(value)}
            for name, value in importances
        ],
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "phishguard_model.joblib"
    metadata_path = args.artifacts_dir / "model_metadata.json"
    joblib.dump(model, model_path)
    joblib.dump(logistic, args.artifacts_dir / "logistic_baseline.joblib")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    # Native XGBoost export for portability; the joblib file additionally
    # includes the Python feature transformer needed by the API.
    classifier.save_model(args.artifacts_dir / "xgboost_model.ubj")
    print(f"\nSaved pipeline: {model_path}")
    print(f"Saved metrics and provenance: {metadata_path}")


if __name__ == "__main__":
    main()
