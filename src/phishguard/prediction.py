"""Load a packaged model and turn probabilities into user-facing results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from phishguard.config import MODEL_METADATA_PATH, MODEL_PATH
from phishguard.features import extract_url_features


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

    def predict(self, url: str) -> dict[str, Any]:
        probability = float(self.model.predict_proba([url])[0][1])
        threshold = float(self.metadata["threshold"])
        is_phishing = probability >= threshold

        if is_phishing:
            risk_level = "high"
        elif probability >= min(0.5, threshold * 0.7):
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "url": url,
            "is_phishing": is_phishing,
            "phishing_probability": probability,
            "risk_level": risk_level,
            "threshold": threshold,
            "reasons": build_reasons(url),
            "model_version": str(self.metadata["model_version"]),
        }
