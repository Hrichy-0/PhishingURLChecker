"""Reusable cleaning rules for phishing URL datasets."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

EXPECTED_LABELS = {0, 1}


def combine_sources(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove all contradictory cross-source URLs, then deduplicate consistently."""
    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby("url")["is_phishing"].nunique()
    conflicts = counts[counts > 1].index
    mask = combined["url"].isin(conflicts)
    consistent = combined.loc[~mask]
    report = {
        "conflicting_urls_excluded": len(conflicts),
        "conflicting_rows_excluded": int(mask.sum()),
        "cross_source_duplicates_removed": int(consistent.duplicated("url").sum()),
    }
    return consistent.drop_duplicates("url").reset_index(drop=True), report


# Source-label -> project-label mappings. Inside this project, 1 always means
# phishing and 0 always means legitimate.
PHIUSIIL_LABEL_MAPPING = {0: 1, 1: 0}
PROJECT_LABEL_MAPPING = {0: 0, 1: 1}
TEXT_LABEL_MAPPING = {"legitimate": 0, "phishing": 1}


def is_valid_http_url(value: str) -> bool:
    """Validate a URL locally, without resolving or opening its hostname."""
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and not any(character.isspace() for character in value)
            and (parsed.port is None or 0 < parsed.port <= 65535)
        )
    except ValueError:
        return False


def _normalize_label_value(value: Any) -> str:
    """Normalize textual and numeric source labels to comparable strings."""
    if isinstance(value, Real):
        return f"{value:g}"
    return str(value).strip().lower()


def clean_url_data(
    raw_data: pd.DataFrame,
    *,
    url_column: str = "URL",
    label_column: str = "label",
    label_mapping: Mapping[Any, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean a URL dataset and standardize it to ``url``/``is_phishing``.

    ``label_mapping`` maps labels in the source dataset to the project
    convention: 0 = legitimate and 1 = phishing. It defaults to PhiUSIIL's
    reversed numeric convention so the original workflow remains compatible.
    """
    if url_column == label_column:
        raise ValueError("URL and label columns must be different")

    required_columns = {url_column, label_column}
    missing_columns = required_columns - set(raw_data.columns)

    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")

    mapping = dict(PHIUSIIL_LABEL_MAPPING if label_mapping is None else label_mapping)
    invalid_targets = set(mapping.values()) - EXPECTED_LABELS
    if invalid_targets:
        raise ValueError("Label mapping values must be 0 (legitimate) or 1 (phishing)")

    normalized_mapping = {}
    for source, target in mapping.items():
        key = _normalize_label_value(source)
        if key in normalized_mapping and normalized_mapping[key] != target:
            raise ValueError("Label mapping has conflicting normalized source labels")
        normalized_mapping[key] = target

    data = raw_data.loc[:, [url_column, label_column]].copy()
    data.columns = ["url", "source_label"]

    report = {
        "rows_before": len(data),
        "missing_urls_removed": int(data["url"].isna().sum()),
        "missing_labels_removed": int(data["source_label"].isna().sum()),
    }

    data = data.dropna(subset=["url", "source_label"])
    data["url"] = data["url"].astype("string").str.strip()

    blank_url_mask = data["url"].eq("")
    report["blank_urls_removed"] = int(blank_url_mask.sum())
    data = data.loc[~blank_url_mask].copy()

    valid_url_mask = data["url"].map(is_valid_http_url)
    report["invalid_urls_removed"] = int((~valid_url_mask).sum())
    data = data.loc[valid_url_mask].copy()

    normalized_labels = data["source_label"].map(_normalize_label_value)
    data["is_phishing"] = normalized_labels.map(normalized_mapping)

    unknown_label_mask = data["is_phishing"].isna()
    if unknown_label_mask.any():
        unknown_labels = sorted(
            {str(value) for value in data.loc[unknown_label_mask, "source_label"]}
        )
        raise ValueError(f"Unexpected labels were found: {unknown_labels}")

    data["is_phishing"] = data["is_phishing"].astype("int8")

    label_counts_per_url = data.groupby("url")["is_phishing"].nunique()
    conflicting_urls = label_counts_per_url[label_counts_per_url > 1]

    if not conflicting_urls.empty:
        raise ValueError(f"Found {len(conflicting_urls)} URLs with conflicting labels")

    duplicate_url_mask = data.duplicated(subset=["url"], keep="first")
    report["duplicate_urls_removed"] = int(duplicate_url_mask.sum())
    data = data.loc[~duplicate_url_mask, ["url", "is_phishing"]].reset_index(drop=True)

    report["rows_after"] = len(data)

    assert data["url"].notna().all()
    assert not data["url"].duplicated().any()
    assert set(data["is_phishing"].unique()) <= EXPECTED_LABELS

    return data, report
