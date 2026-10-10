"""Legacy scikit-learn adapter for the shared offline HTML extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from phishguard.html_features import (
    CONTENT_FEATURE_NAMES as CONTENT_FEATURE_NAMES,
)
from phishguard.html_features import (
    MAX_HTML_BYTES as MAX_HTML_BYTES,
)
from phishguard.html_features import (
    extract_content_features as extract_content_features,
)
from phishguard.lexical import extract_url_features
from phishguard.modeling import TRAINING_FEATURES

HYBRID_URL_FEATURES = tuple(n for n in TRAINING_FEATURES if n != "uses_https")
HYBRID_FEATURE_NAMES = HYBRID_URL_FEATURES + CONTENT_FEATURE_NAMES


class HybridFeatureExtractor(TransformerMixin, BaseEstimator):
    """Combine URL and saved-HTML signals using the same code at inference."""

    def fit(self, frames: pd.DataFrame, y: object = None) -> HybridFeatureExtractor:
        return self

    def transform(self, frames: pd.DataFrame) -> np.ndarray:
        if not isinstance(frames, pd.DataFrame) or not {"url", "html"}.issubset(frames):
            raise ValueError("Hybrid input must be a DataFrame with url and html")
        rows = []
        for url, html in frames[["url", "html"]].itertuples(index=False, name=None):
            values = {
                **extract_url_features(url),
                **extract_content_features(url, html),
            }
            rows.append([values[name] for name in HYBRID_FEATURE_NAMES])
        return np.asarray(rows, dtype=np.float64).reshape(-1, len(HYBRID_FEATURE_NAMES))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(HYBRID_FEATURE_NAMES, dtype=object)
