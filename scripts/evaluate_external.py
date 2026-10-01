"""Evaluate the existing PhishGuard model on an external URL dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishguard.config import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
)
from phishguard.modeling import evaluate_probabilities


DATASET_PATH = Path("data/external/balanced_urls.csv")

RESULTS_PATH = Path("artifacts/external_evaluation.json")
FALSE_POSITIVES_PATH = Path("data/external/false_positives.csv")
FALSE_NEGATIVES_PATH = Path("data/external/false_negatives.csv")

BATCH_SIZE = 20_000


def load_external_dataset() -> pd.DataFrame:
    """Load, validate, clean, and label the external dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"url", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    print(f"Original rows: {len(df):,}")

    # Keep only the fields needed for evaluation.
    df = df[["url", "label"]].copy()

    # Remove missing values.
    df = df.dropna(subset=["url", "label"])

    # Remove exact duplicate URLs.
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["url"]).copy()

    removed_duplicates = before_duplicates - len(df)

    print(f"Duplicate URLs removed: {removed_duplicates:,}")
    print(f"Rows after deduplication: {len(df):,}")

    # Convert external labels to the project's convention:
    # 0 = legitimate
    # 1 = phishing
    label_mapping = {
        "legitimate": 0,
        "phishing": 1,
    }

    df["is_phishing"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(label_mapping)
    )

    if df["is_phishing"].isna().any():
        unknown_labels = df.loc[
            df["is_phishing"].isna(), "label"
        ].unique()

        raise ValueError(
            f"Unknown labels found in external dataset: {unknown_labels}"
        )

    df["is_phishing"] = df["is_phishing"].astype(int)

    return df


def remove_original_dataset_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Remove URLs appearing in the original train/validation/test sets."""
    original_urls: set[str] = set()

    for split_name in ("train", "validation", "test"):
        split_path = PROCESSED_DATA_DIR / f"{split_name}.csv"

        if not split_path.exists():
            print(
                f"Warning: {split_path} not found. "
                "Skipping overlap check for this split."
            )
            continue

        split = pd.read_csv(split_path, usecols=["url"])

        original_urls.update(
            split["url"]
            .dropna()
            .astype(str)
            .tolist()
        )

    if not original_urls:
        print("\nNo original dataset URLs were loaded.")
        print("External evaluation will continue without overlap removal.")
        return df

    overlap_mask = df["url"].astype(str).isin(original_urls)
    overlap_count = int(overlap_mask.sum())

    print(f"\nURLs overlapping original dataset: {overlap_count:,}")

    df = df.loc[~overlap_mask].copy()

    print(f"External-only rows remaining: {len(df):,}")

    return df


def predict_in_batches(model, urls: pd.Series) -> np.ndarray:
    """Generate phishing scores without loading all features at once."""
    probabilities: list[np.ndarray] = []

    total = len(urls)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch = urls.iloc[start:end]

        batch_probabilities = model.predict_proba(batch)[:, 1]
        probabilities.append(batch_probabilities)

        print(
            f"Processed {end:,} / {total:,} URLs",
            end="\r",
            flush=True,
        )

    print()

    return np.concatenate(probabilities)


def main() -> None:
    print("=" * 60)
    print("PHISHGUARD EXTERNAL DATASET EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load external dataset
    # ---------------------------------------------------------

    df = load_external_dataset()

    print("\nLABEL DISTRIBUTION")

    print(
        df["is_phishing"]
        .value_counts()
        .rename(index={0: "legitimate", 1: "phishing"})
    )

    # ---------------------------------------------------------
    # 2. Remove URLs already seen in PhiUSIIL
    # ---------------------------------------------------------

    df = remove_original_dataset_overlap(df)

    if df.empty:
        raise RuntimeError(
            "No URLs remain after overlap removal."
        )

    # ---------------------------------------------------------
    # 3. Load frozen model
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run scripts/train_model.py first."
        )

    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {MODEL_METADATA_PATH}"
        )

    print("\nLoading existing PhishGuard model...")

    model = joblib.load(MODEL_PATH)

    metadata = json.loads(
        MODEL_METADATA_PATH.read_text()
    )

    threshold = float(metadata["threshold"])

    print(f"Model: {MODEL_PATH}")
    print(f"Frozen threshold: {threshold:.4f}")

    # ---------------------------------------------------------
    # 4. Run predictions
    # ---------------------------------------------------------

    print("\nRunning external predictions...")

    probabilities = predict_in_batches(
        model,
        df["url"].astype(str),
    )

    labels = df["is_phishing"].to_numpy()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # 5. Evaluate
    # ---------------------------------------------------------

    metrics = evaluate_probabilities(
        labels,
        probabilities,
        threshold,
    )

    print("\n" + "=" * 60)
    print("EXTERNAL TEST RESULTS")
    print("=" * 60)

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ):
        print(
            f"{metric}: "
            f"{float(metrics[metric]):.4f}"
        )

    print("\nCONFUSION MATRIX")

    for name, value in metrics["confusion_matrix"].items():
        print(f"{name}: {value:,}")

    # ---------------------------------------------------------
    # 6. Save incorrect predictions
    # ---------------------------------------------------------

    df["phishing_score"] = probabilities
    df["predicted_is_phishing"] = predictions

    false_positives = df[
        (df["is_phishing"] == 0)
        & (df["predicted_is_phishing"] == 1)
    ].copy()

    false_negatives = df[
        (df["is_phishing"] == 1)
        & (df["predicted_is_phishing"] == 0)
    ].copy()

    FALSE_POSITIVES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    false_positives.to_csv(
        FALSE_POSITIVES_PATH,
        index=False,
    )

    false_negatives.to_csv(
        FALSE_NEGATIVES_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # 7. Save evaluation report
    # ---------------------------------------------------------

    results = {
        "dataset": str(DATASET_PATH),
        "rows_evaluated": len(df),
        "model_path": str(MODEL_PATH),
        "threshold": threshold,
        "metrics": metrics,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2) + "\n"
    )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(f"Evaluation: {RESULTS_PATH}")
    print(f"False positives: {FALSE_POSITIVES_PATH}")
    print(f"False negatives: {FALSE_NEGATIVES_PATH}")


if __name__ == "__main__":
    main()