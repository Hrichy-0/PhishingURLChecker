"""Load a packaged model and turn probabilities into user-facing results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from phishguard.config import MODEL_METADATA_PATH, MODEL_PATH
from phishguard.lexical import extract_url_features


def prediction_result(url: str, probability: float, metadata: dict, **identity) -> dict:
    """Apply the saved cutoff consistently for legacy and experimental models."""
    threshold = float(metadata["threshold"])
    is_phishing = probability >= threshold
    risk_level = (
        "high"
        if is_phishing
        else "medium"
        if probability >= min(0.5, threshold * 0.7)
        else "low"
    )
    return {
        "url": url,
        "is_phishing": is_phishing,
        "phishing_probability": probability,
        "risk_level": risk_level,
        "threshold": threshold,
        "reasons": build_reasons(url),
        "model_version": str(metadata["model_version"]),
        "model_type": metadata.get("model_type", "URL model"),
        "feature_count": len(metadata.get("features", [])) or None,
        **identity,
    }


def build_reasons(url: str) -> list[str]:
    """Return understandable lexical signals, without claiming causality."""
    features = extract_url_features(url)
    reasons: list[str] = []

    if features["has_ip_hostname"]:
        reasons.append("The hostname is an IP address rather than a named domain.")
    if features["has_at_symbol"]:
        reasons.append(
            "The URL contains an @ symbol, which can obscure its destination."
        )
    if features["suspicious_word_count"]:
        reasons.append(
            "The URL contains account, login, payment, or verification terms."
        )
    if features["url_length"] >= 100:
        reasons.append("The URL is unusually long.")
    if features["subdomain_count"] >= 3:
        reasons.append("The hostname contains many subdomains.")
    if features["percent_encoding_count"] >= 2:
        reasons.append("The URL contains multiple encoded characters.")
    if not features["uses_https"]:
        reasons.append("The URL does not use HTTPS.")
    if features["character_entropy"] >= 4.8:
        reasons.append("The URL has an unusually varied character pattern.")

    if not reasons:
        reasons.append("No strong human-readable lexical warning was identified.")
    return reasons[:4]


class ModelService:
    """In-memory prediction service shared by the API endpoints."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
    ) -> None:
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run scripts/train_model.py first."
            )
        self.model = joblib.load(model_path)
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text())
        self.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.experimental = False

    def predict(self, url: str) -> dict[str, Any]:
        probability = float(self.model.predict_proba([url])[0][1])
        return prediction_result(
            url,
            probability,
            self.metadata,
            experimental=False,
            model_sha256=getattr(self, "model_sha256", None),
        )


class NativeURLService:
    """Serve the main local native URL model without rewriting research artifacts."""

    schema_error = "URL endpoint requires the URL-only model"

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        from phishguard.native import TRAINING_FEATURES

        return TRAINING_FEATURES

    def __init__(self, model_dir: Path) -> None:
        import xgboost as xgb

        features = self.feature_names()
        model_path = model_dir / "xgboost_model.ubj"
        self.metadata = json.loads((model_dir / "model_metadata.json").read_text())
        if tuple(self.metadata["features"]) != features:
            raise ValueError(self.schema_error)
        payload = model_path.read_bytes()
        self.model_sha256 = hashlib.sha256(payload).hexdigest()
        if self.model_sha256 != self.metadata.get("model_sha256"):
            raise ValueError("Native model checksum does not match its metadata")
        threshold = float(self.metadata["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError("Native model threshold must be between 0 and 1")
        self.model = xgb.Booster()
        self.model.load_model(bytearray(payload))
        self.model.set_param({"device": "cpu", "nthread": 4})
        if self.model.num_features() != len(features) or (
            self.model.feature_names != list(features)
        ):
            raise ValueError("Native model feature order does not match training")
        self.rounds = int(self.metadata["best_iteration"]) + 1
        if not 0 < self.rounds <= self.model.num_boosted_rounds():
            raise ValueError("Native model has an invalid prediction tree range")
        self.experimental = True

    def score(self, values) -> float:
        import xgboost as xgb

        return float(
            self.model.predict(
                xgb.DMatrix(values, feature_names=list(self.feature_names())),
                iteration_range=(0, self.rounds),
            )[0]
        )

    def predict(self, url: str) -> dict[str, Any]:
        from phishguard.native import url_matrix

        return prediction_result(
            url,
            self.score(url_matrix([url])),
            self.metadata,
            experimental=True,
            model_sha256=self.model_sha256,
        )


class NativeURLHTMLService(NativeURLService):
    """Research-only URL+HTML model for page content the browser already loaded."""

    schema_error = "Content endpoint requires the URL+HTML model"

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        from phishguard.native_html import HYBRID_FEATURE_NAMES

        return HYBRID_FEATURE_NAMES

    def predict(self, url: str, html: str) -> dict[str, Any]:
        from phishguard.native_html import snapshot_vector

        values = snapshot_vector(url, html, max_html_bytes=self.max_html_bytes)
        truncated_index = self.feature_names().index("html_truncated")
        return prediction_result(
            url,
            self.score(values),
            self.metadata,
            experimental=True,
            model_sha256=self.model_sha256,
            html_truncated=bool(values[0, truncated_index]),
        )

    @property
    def max_html_bytes(self) -> int:
        return int(self.metadata["max_html_bytes"])
