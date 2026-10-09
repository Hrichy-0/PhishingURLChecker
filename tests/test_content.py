"""Offline HTML is data, never executable content or a definitive verdict."""

import json
import socket

import joblib
import numpy as np
import pandas as pd
import pytest

from phishguard.content import (
    CONTENT_FEATURE_NAMES,
    HYBRID_FEATURE_NAMES,
    MAX_HTML_BYTES,
    HybridFeatureExtractor,
    extract_content_features,
)
from phishguard.experiments import run_benchmark
from scripts.benchmark_hybrid import load_snapshots


def test_content_aggregates_without_retaining_values_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline feature extraction must not use the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    html = """<title>PayPal sign in</title>
        <form action="https://collector.invalid/send">
          <input type="password" value="sensitive-password">
          <input type="hidden" value="sensitive-token">
        </form><a href="/about">About</a><a href="https://other.test/">Other</a>
        <iframe src="https://evil.invalid/"></iframe>
        <script src="https://evil.invalid/run.js">
          throw new Error('do not run')
        </script>"""
    features = extract_content_features("https://example.test/login", html)
    assert set(features) == set(CONTENT_FEATURE_NAMES)
    assert features["password_field_count"] == 1
    assert features["hidden_field_count"] == 1
    assert features["external_form_ratio"] == 1
    assert features["external_link_ratio"] == 0.5
    assert features["external_script_ratio"] == 1
    assert features["title_brand_domain_mismatch_count"] == 1
    assert "sensitive" not in json.dumps(features)


def test_legitimate_brand_login_not_automatically_flagged():
    features = extract_content_features(
        "https://www.paypal.com/signin",
        '<title>PayPal sign in</title><form action="/signin">'
        '<input type="password"></form>',
    )
    assert features["password_field_count"] == 1
    assert features["title_brand_domain_mismatch_count"] == 0
    assert features["external_form_ratio"] == 0
    assert "is_phishing" not in features


def test_base_element_and_empty_form_action():
    features = extract_content_features(
        "https://example.test/",
        '<base href="https://other.test/"><form action="send"></form><form></form>',
    )
    assert features["external_form_ratio"] == 0.5


def test_malformed_and_non_http_destinations_do_not_crash():
    features = extract_content_features(
        "https://example.test/",
        '<base href="http://["><form action="http://["></form>'
        '<form action="javascript:alert(1)"></form><a href="http://[">X</a>',
    )
    assert features["unsupported_form_ratio"] == 1
    assert features["external_link_ratio"] == 0


@pytest.mark.parametrize(
    "url,html",
    [
        ("file:///etc/passwd", "<html></html>"),
        ("https://example.test/", ""),
        ("https://example.test/", "é" * (MAX_HTML_BYTES // 2 + 1)),
    ],
)
def test_rejects_invalid_content_input(url, html):
    with pytest.raises(ValueError):
        extract_content_features(url, html)


def test_hybrid_transformer_schema():
    frame = pd.DataFrame({"url": ["https://example.test/"], "html": ["<html></html>"]})
    transformer = HybridFeatureExtractor()
    values = transformer.fit_transform(frame)
    assert values.shape == (1, len(HYBRID_FEATURE_NAMES))
    assert np.isfinite(values).all()
    assert "uses_https" not in transformer.get_feature_names_out()
    with pytest.raises(ValueError, match="DataFrame"):
        transformer.transform(frame["url"])


def test_snapshot_cleaning_explicit_labels_and_missing_html(tmp_path):
    path = tmp_path / "snapshots.csv"
    pd.DataFrame(
        {
            "URL": ["https://a.test/", "https://b.test/", "https://c.test/"],
            "HTML": ["<html>safe</html>", "<html>phish</html>", None],
            "class": ["benign", "phish", "phish"],
        }
    ).to_csv(path, index=False)
    frame, report = load_snapshots(path, "URL", "HTML", "class", "phish", "benign")
    assert len(frame) == 2
    assert frame["is_phishing"].tolist() == [0, 1]
    assert report["missing_or_blank_html_rows"] == 1
    with pytest.raises(ValueError, match="must differ"):
        load_snapshots(path, "URL", "HTML", "class", "phish", "phish")


def test_optional_hybrid_benchmark_is_packaged_separately(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "url": f"https://site{domain}.test/{label}",
                "is_phishing": label,
                "html": '<form><input type="password"></form>'
                if label
                else "<title>About</title>",
            }
            for domain in range(20)
            for label in (0, 1)
        ]
    )
    run_benchmark(
        frame, tmp_path / "hybrid", variants=("hybrid_xgboost",), folds=2, trees=8
    )
    artifact = tmp_path / "hybrid" / "hybrid_xgboost"
    model = joblib.load(artifact / "phishguard_model.joblib")
    assert model.predict_proba(frame[["url", "html"]].iloc[:2]).shape == (2, 2)
    metadata = json.loads((artifact / "model_metadata.json").read_text())
    assert metadata["activation_eligible"] is False
