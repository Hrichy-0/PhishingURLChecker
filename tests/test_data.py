import pandas as pd
import pytest

from phishguard.data import clean_url_data


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
