"""Explainable lexical features computed without visiting a URL."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SUSPICIOUS_WORDS = frozenset(
    {
        "account",
        "bank",
        "confirm",
        "credential",
        "invoice",
        "login",
        "password",
        "payment",
        "recover",
        "reset",
        "secure",
        "signin",
        "support",
        "suspend",
        "update",
        "verify",
        "wallet",
    }
)

FEATURE_NAMES = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "digit_count",
    "digit_ratio",
    "subdomain_count",
    "query_parameter_count",
    "percent_encoding_count",
    "special_character_count",
    "has_ip_hostname",
    "uses_https",
    "has_at_symbol",
    "has_explicit_port",
    "suspicious_word_count",
    "character_entropy",
    "longest_token_length",
    "tld_length",
)

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
SPECIAL_CHARACTERS = "@?=&%_~:+!$,'()*;[]"


def shannon_entropy(text: str) -> float:
    """Return Shannon entropy in bits per character."""
    if not text:
        return 0.0
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length) for count in Counter(text).values()
    )


def extract_url_features(url: str) -> dict[str, float]:
    """Extract deterministic URL-only features for one absolute URL."""
    parsed = urlsplit(url)
    # Browsers commonly add a root slash even when training data omits it.
    # Treat https://example.com and https://example.com/ as the same URL shape.
    if parsed.path == "/":
        parsed = parsed._replace(path="")
        url = urlunsplit(parsed)
    hostname = (parsed.hostname or "").lower()
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(url)]
    digits = sum(character.isdigit() for character in url)

    try:
        has_ip = int(bool(hostname) and ip_address(hostname) is not None)
    except ValueError:
        has_ip = 0

    hostname_parts = [part for part in hostname.split(".") if part]
    subdomain_count = max(0, len(hostname_parts) - 2)
    tld_length = len(hostname_parts[-1]) if len(hostname_parts) >= 2 else 0
    query_parameter_count = 0
    if parsed.query:
        query_parameter_count = parsed.query.count("&") + 1

    try:
        has_explicit_port = int(parsed.port is not None)
    except ValueError:
        has_explicit_port = 1

    values = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "underscore_count": url.count("_"),
        "slash_count": url.count("/"),
        "digit_count": digits,
        "digit_ratio": digits / max(1, len(url)),
        "subdomain_count": subdomain_count,
        "query_parameter_count": query_parameter_count,
        "percent_encoding_count": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "special_character_count": sum(url.count(char) for char in SPECIAL_CHARACTERS),
        "has_ip_hostname": has_ip,
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_at_symbol": int("@" in url),
        "has_explicit_port": has_explicit_port,
        "suspicious_word_count": sum(token in SUSPICIOUS_WORDS for token in tokens),
        "character_entropy": shannon_entropy(url),
        "longest_token_length": max((len(token) for token in tokens), default=0),
        "tld_length": tld_length,
    }
    return {name: float(values[name]) for name in FEATURE_NAMES}


class URLFeatureExtractor(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer that converts URLs into lexical features."""

    def fit(self, urls: Iterable[str], y: object = None) -> URLFeatureExtractor:
        return self

    def transform(self, urls: Iterable[str]) -> np.ndarray:
        rows = [extract_url_features(str(url)) for url in urls]
        return np.asarray(
            [[row[name] for name in FEATURE_NAMES] for row in rows],
            dtype=np.float64,
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)
