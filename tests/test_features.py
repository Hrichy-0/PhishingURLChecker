import numpy as np

from phishguard.features import (
    FEATURE_NAMES,
    URLFeatureExtractor,
    extract_url_features,
    shannon_entropy,
)


def test_extract_url_features_for_normal_url() -> None:
    features = extract_url_features("https://login.example.com/reset?id=42")

    assert tuple(features) == FEATURE_NAMES
    assert features["uses_https"] == 1
    assert features["subdomain_count"] == 1
    assert features["query_parameter_count"] == 1
    assert features["digit_count"] == 2
    assert features["suspicious_word_count"] == 2


def test_extract_url_features_detects_ip_and_at_symbol() -> None:
    features = extract_url_features("http://user@192.168.1.10:8080/login")

    assert features["has_ip_hostname"] == 1
    assert features["has_at_symbol"] == 1
    assert features["has_explicit_port"] == 1
    assert features["uses_https"] == 0


def test_feature_transformer_has_stable_shape() -> None:
    transformer = URLFeatureExtractor()
    transformed = transformer.fit_transform(
        ["https://example.com", "http://example.test/login"]
    )

    assert transformed.shape == (2, len(FEATURE_NAMES))
    assert transformed.dtype == np.float64


def test_root_slash_does_not_change_features() -> None:
    without_slash = extract_url_features("https://example.com")
    with_slash = extract_url_features("https://example.com/")

    assert without_slash == with_slash


def test_entropy_handles_empty_and_repeated_text() -> None:
    assert shannon_entropy("") == 0
    assert shannon_entropy("aaaa") == 0
