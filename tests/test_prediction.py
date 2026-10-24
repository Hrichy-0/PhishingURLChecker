import hashlib
import json

import numpy as np
import pytest
import xgboost as xgb

from phishguard.native import TRAINING_FEATURES, url_matrix
from phishguard.prediction import ModelService, NativeURLService, build_reasons


class FakeModel:
    def predict_proba(self, urls: list[str]) -> list[list[float]]:
        return [[0.15, 0.85] for _ in urls]


def test_build_reasons_detects_obvious_lexical_signals() -> None:
    reasons = build_reasons("http://user@192.168.1.5/login/verify")

    assert any("IP address" in reason for reason in reasons)
    assert any("@ symbol" in reason for reason in reasons)
    assert any("HTTPS" in reason for reason in reasons)


def test_model_service_applies_saved_threshold() -> None:
    service = object.__new__(ModelService)
    service.model = FakeModel()
    service.metadata = {"threshold": 0.55, "model_version": "test"}

    result = service.predict("https://example.test/login")

    assert result["is_phishing"] is True
    assert result["risk_level"] == "high"
    assert result["model_version"] == "test"


def test_native_service_preserves_saved_predictions_and_checks_checksum(tmp_path):
    urls = ["https://example.test/", "http://example.test/login"]
    data = xgb.DMatrix(
        url_matrix(urls), label=[0, 1], feature_names=list(TRAINING_FEATURES)
    )
    booster = xgb.train(
        {"objective": "binary:logistic", "min_child_weight": 0, "nthread": 1},
        data,
        num_boost_round=2,
    )
    path = tmp_path / "xgboost_model.ubj"
    booster.save_model(path)
    metadata = {
        "features": list(TRAINING_FEATURES),
        "threshold": 0.55,
        "best_iteration": 1,
        "model_version": "test-native",
        "model_type": "Native XGBoost: url_only",
        "activation_eligible": False,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))
    service = NativeURLService(tmp_path)
    for url, score in zip(urls, booster.predict(data), strict=True):
        result = service.predict(url)
        assert np.isclose(result["phishing_probability"], score)
        assert result["is_phishing"] == bool(score >= metadata["threshold"])
        assert result["experimental"] is True
        assert result["model_sha256"] == metadata["model_sha256"]
        assert result["feature_count"] == 21
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        NativeURLService(tmp_path)


def test_content_service_scores_html_and_rejects_url_only_schema(tmp_path):
    from phishguard.native_html import HYBRID_FEATURE_NAMES, snapshot_vector
    from phishguard.prediction import NativeURLHTMLService

    pages = [
        ("https://example.test/", "<html><body><p>Hello</p></body></html>"),
        ("http://example.test/login", '<form action="https://other.test/"></form>'),
    ]
    values = np.vstack([snapshot_vector(url, html) for url, html in pages])
    data = xgb.DMatrix(values, label=[0, 1], feature_names=list(HYBRID_FEATURE_NAMES))
    booster = xgb.train(
        {"objective": "binary:logistic", "min_child_weight": 0, "nthread": 1},
        data,
        num_boost_round=2,
    )
    path = tmp_path / "xgboost_model.ubj"
    booster.save_model(path)
    metadata = {
        "features": list(HYBRID_FEATURE_NAMES),
        "threshold": 0.5,
        "best_iteration": 1,
        "max_html_bytes": 1_000_000,
        "model_version": "test-url-html",
        "model_type": "Native XGBoost: url_html",
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))
    service = NativeURLHTMLService(tmp_path)
    for (url, html), score in zip(pages, booster.predict(data), strict=True):
        result = service.predict(url, html)
        assert np.isclose(result["phishing_probability"], score)
        assert result["feature_count"] == 38
        assert result["html_truncated"] is False
    assert service.predict(pages[0][0], "a" * 1_000_001)["html_truncated"] is True

    metadata["features"] = list(TRAINING_FEATURES)
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="URL\\+HTML model"):
        NativeURLHTMLService(tmp_path)


def test_snapshot_script_matches_service_tree_range(tmp_path):
    from scripts.predict_native_snapshot import score_snapshot

    urls = ["https://example.test/", "http://example.test/login"]
    data = xgb.DMatrix(
        url_matrix(urls), label=[0, 1], feature_names=list(TRAINING_FEATURES)
    )
    booster = xgb.train(
        {"objective": "binary:logistic", "min_child_weight": 0, "nthread": 1},
        data,
        num_boost_round=5,
    )
    path = tmp_path / "xgboost_model.ubj"
    booster.save_model(path)
    metadata = {
        "features": list(TRAINING_FEATURES),
        "threshold": 0.5,
        "best_iteration": 1,
        "model_version": "test-native",
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))
    service = NativeURLService(tmp_path)
    for url in urls:
        values = url_matrix([url])
        script = score_snapshot(booster, values, TRAINING_FEATURES, metadata)
        assert np.isclose(script, service.predict(url)["phishing_probability"])
        full = float(
            booster.predict(xgb.DMatrix(values, feature_names=list(TRAINING_FEATURES)))[
                0
            ]
        )
        assert not np.isclose(script, full)
    with pytest.raises(ValueError, match="tree range"):
        score_snapshot(booster, values, TRAINING_FEATURES, {"best_iteration": 9})
