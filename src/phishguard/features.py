"""Scikit-learn adapter for shared, deterministic lexical URL features."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from phishguard.lexical import (
    FEATURE_NAMES as FEATURE_NAMES,
)
from phishguard.lexical import (
    SPECIAL_CHARACTERS as SPECIAL_CHARACTERS,
)
from phishguard.lexical import (
    SUSPICIOUS_WORDS as SUSPICIOUS_WORDS,
)
from phishguard.lexical import (
    TOKEN_PATTERN as TOKEN_PATTERN,
)
from phishguard.lexical import (
    extract_url_features as extract_url_features,
)
from phishguard.lexical import (
    shannon_entropy as shannon_entropy,
)


class URLFeatureExtractor(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer that converts URLs into lexical features."""

    def __init__(
        self,
        feature_names: tuple[str, ...] | None = None,
        normalize_protocol: bool = False,
    ) -> None:
        self.feature_names = feature_names
        self.normalize_protocol = normalize_protocol

    def _selected_names(self) -> tuple[str, ...]:
        # Artifacts created before feature selection have no feature_names field.
        return getattr(self, "feature_names", None) or FEATURE_NAMES

    def fit(self, urls: Iterable[str], y: object = None) -> URLFeatureExtractor:
        return self

    def transform(self, urls: Iterable[str]) -> np.ndarray:
        rows = []
        for value in urls:
            url = str(value)
            if getattr(self, "normalize_protocol", False):
                # Constant protocol also removes its effects on length/entropy.
                # This does not visit the site or depend on labels.
                url = urlunsplit(urlsplit(url)._replace(scheme="https"))
            rows.append(extract_url_features(url))
        return np.asarray(
            [[row[name] for name in self._selected_names()] for row in rows],
            dtype=np.float64,
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self._selected_names(), dtype=object)
