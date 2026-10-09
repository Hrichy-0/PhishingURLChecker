import pandas as pd
import pytest

from phishguard.data import TEXT_LABEL_MAPPING, clean_url_data


# Test label conversion and duplicate removal
def test_clean_url_data_normalizes_and_deduplicates() -> None:
    raw_data = pd.DataFrame(
        {
            "URL": [
                " https://legitimate.example ",
                "https://legitimate.example",
                "http://phishing.example/login",
            ],
            "label": [1, 1, 0],
            "unused_column": [100, 200, 300],
        }
    )

    cleaned_data, report = clean_url_data(raw_data)

    assert cleaned_data.to_dict(orient="records") == [
        {
            "url": "https://legitimate.example",
            "is_phishing": 0,
        },
        {
            "url": "http://phishing.example/login",
            "is_phishing": 1,
        },
    ]

    assert report["rows_before"] == 3
    assert report["duplicate_urls_removed"] == 1
    assert report["rows_after"] == 2


# Test conflicting labels
def test_clean_url_data_rejects_conflicting_labels() -> None:
    raw_data = pd.DataFrame(
        {
            "URL": [
                "https://example.com",
                "https://example.com",
            ],
            "label": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        clean_url_data(raw_data)


# Test missing columns
def test_clean_url_data_requires_url_and_label() -> None:
    raw_data = pd.DataFrame(
        {
            "URL": ["https://example.com"],
        }
    )

    with pytest.raises(ValueError, match="Required columns"):
        clean_url_data(raw_data)


def test_clean_url_data_supports_text_labels_and_lowercase_url_column() -> None:
    raw_data = pd.DataFrame(
        {
            "url": [
                " https://legitimate.example/article ",
                "http://phishing.example/verify",
            ],
            "label": [" Legitimate ", "PHISHING"],
        }
    )

    cleaned_data, report = clean_url_data(
        raw_data,
        url_column="url",
        label_mapping=TEXT_LABEL_MAPPING,
    )

    assert cleaned_data.to_dict(orient="records") == [
        {
            "url": "https://legitimate.example/article",
            "is_phishing": 0,
        },
        {
            "url": "http://phishing.example/verify",
            "is_phishing": 1,
        },
    ]
    assert report["rows_after"] == 2


def test_clean_url_data_rejects_unknown_text_labels() -> None:
    raw_data = pd.DataFrame(
        {
            "url": ["https://example.com"],
            "label": ["unknown"],
        }
    )

    with pytest.raises(ValueError, match="Unexpected labels"):
        clean_url_data(
            raw_data,
            url_column="url",
            label_mapping=TEXT_LABEL_MAPPING,
        )


def test_clean_url_data_excludes_malformed_urls() -> None:
    raw_data = pd.DataFrame(
        {
            "url": [
                "a123hash",
                "https://[broken",
                "https://x.test:bad/",
                "https://x.test/",
            ],
            "label": ["phishing", "phishing", "phishing", "legitimate"],
        }
    )
    cleaned, report = clean_url_data(
        raw_data, url_column="url", label_mapping=TEXT_LABEL_MAPPING
    )
    assert cleaned["url"].tolist() == ["https://x.test/"]
    assert report["invalid_urls_removed"] == 3


def test_custom_numeric_mapping_handles_csv_float_labels() -> None:
    raw = pd.DataFrame(
        {
            "address": ["https://benign.test/", "http://phish.test/"],
            "status": [0.0, 1.0],
        }
    )
    cleaned, _ = clean_url_data(
        raw,
        url_column="address",
        label_column="status",
        label_mapping={"0": 0, "1": 1},
    )
    assert cleaned["is_phishing"].tolist() == [0, 1]


def test_custom_label_mapping_rejects_normalization_conflicts() -> None:
    raw = pd.DataFrame({"URL": ["https://example.com"], "label": ["safe"]})
    with pytest.raises(ValueError, match="conflicting normalized"):
        clean_url_data(raw, label_mapping={"safe": 0, " SAFE ": 1})
