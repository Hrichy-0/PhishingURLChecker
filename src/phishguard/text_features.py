"""Stateless URL normalization for character-based models; never visits URLs."""

from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class URLTextNormalizer(TransformerMixin, BaseEstimator):
    """Omit protocol/fragment and normalize the optional root slash.

    Removing http(s) prevents this representation from directly learning the
    protocol shortcut. It does not remove all collection artifacts or leakage.
    """

    def fit(self, urls: Iterable[str], y: object = None) -> "URLTextNormalizer":
        return self

    def transform(self, urls: Iterable[str]) -> np.ndarray:
        normalized = []
        for url in urls:
            parsed = urlsplit(str(url))
            path = "" if parsed.path == "/" else parsed.path
            normalized.append(
                urlunsplit(("", parsed.netloc, path, parsed.query, "")).removeprefix(
                    "//"
                )
            )
        return np.asarray(normalized, dtype=object)
