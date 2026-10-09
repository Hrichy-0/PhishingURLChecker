"""Activate an evaluated model for the local API, archiving the previous one."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2, move

import joblib

from phishguard.config import ARTIFACTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifacts_dir.resolve()
    if source == ARTIFACTS_DIR.resolve():
        raise ValueError("This model is already in the active artifacts directory")
    model_path = source / "phishguard_model.joblib"
    metadata_path = source / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("activation_eligible") is False:
        raise ValueError(
            "This is a development candidate, not an approved deployment model. "
            "Complete and review independent final evaluation before activation."
        )
    if "test_metrics" not in metadata or "threshold" not in metadata:
        raise ValueError("Model metadata must contain a completed evaluation")
    model = joblib.load(model_path)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected binary labels: 0=legitimate, 1=phishing")
    model.predict_proba(["https://example.com/"])
    files = ["phishguard_model.joblib", "model_metadata.json"]
    backup = ARTIFACTS_DIR / "archive" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup.mkdir(parents=True, exist_ok=False)
    for name in files:
        previous = ARTIFACTS_DIR / name
        if previous.exists():
            copy2(previous, backup / name)
    for name in files:
        copy2(source / name, ARTIFACTS_DIR / name)
    # Reports for the previous model must not appear to describe the new one.
    for name in (
        "external_evaluation.json",
        "false_positives.csv",
        "false_negatives.csv",
    ):
        previous_report = ARTIFACTS_DIR / name
        if previous_report.exists():
            move(str(previous_report), str(backup / name))
    print(f"Activated: {metadata['model_type']}")
    print(f"Previous active model archived: {backup}")
    print("Restart the API and reload the Chrome extension to use this model.")


if __name__ == "__main__":
    main()
