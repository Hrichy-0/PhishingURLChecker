"""Save, load and batch-score native XGBoost model packages with integrity checks.

A package directory holds ``xgboost_model.ubj`` and ``model_metadata.json``.
Loading verifies the model checksum, feature order and best-iteration tree range,
and scoring always uses only those trees on CPU. ``prediction.NativeURLService``
keeps its own equivalent loader because the frozen URL benchmark hashes it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

MODEL_FILE = "xgboost_model.ubj"
METADATA_FILE = "model_metadata.json"


@dataclass(frozen=True)
class NativeModel:
    booster: xgb.Booster
    features: tuple[str, ...]
    rounds: int
    sha256: str

    def scores(self, matrix) -> np.ndarray:
        """Phishing scores for every row, using only the selected trees."""
        data = xgb.DMatrix(matrix, feature_names=list(self.features), nthread=4)
        return self.booster.predict(data, iteration_range=(0, self.rounds))


def open_model(path: Path, features, rounds: int, sha256: str) -> NativeModel:
    """Load one UBJ file after checking it against independently recorded facts."""
    if not sha256:
        raise ValueError("A recorded model checksum is required")
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha256:
        raise ValueError("Model checksum does not match its record")
    booster = xgb.Booster()
    booster.load_model(bytearray(payload))
    booster.set_param({"device": "cpu", "nthread": 4})
    features = tuple(features)
    if booster.num_features() != len(features) or booster.feature_names not in (
        None,
        list(features),
    ):
        raise ValueError("Model feature order does not match its record")
    if not 0 < rounds <= booster.num_boosted_rounds():
        raise ValueError("Model record has an invalid prediction tree range")
    return NativeModel(booster, features, int(rounds), actual)


def load(
    directory: Path, features=None, expected_sha256: str | None = None
) -> tuple[NativeModel, dict[str, Any]]:
    """Load a package; ``features`` and ``expected_sha256`` add caller-side checks."""
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if features is not None and list(features) != metadata["features"]:
        raise ValueError("Saved metadata feature schema differs")
    recorded = metadata.get("model_sha256")
    if recorded and expected_sha256 and recorded != expected_sha256:
        raise ValueError("Model checksum does not match its record")
    if not 0 <= float(metadata["threshold"]) <= 1:
        raise ValueError("Model threshold must be between 0 and 1")
    model = open_model(
        directory / MODEL_FILE,
        metadata["features"],
        int(metadata["best_iteration"]) + 1,
        recorded or expected_sha256,
    )
    return model, metadata


def save(booster: xgb.Booster, directory: Path, metadata: dict[str, Any]) -> dict:
    """Write the model and its metadata, recording the checksum and tree range."""
    directory = Path(directory)
    payload = bytes(booster.save_raw(raw_format="ubj"))
    (directory / MODEL_FILE).write_bytes(payload)
    metadata = {
        **metadata,
        "best_iteration": booster.num_boosted_rounds() - 1,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
