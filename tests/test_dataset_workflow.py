"""Regression checks for combining sources and filtering previously used data."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from phishguard.split import validate_training_splits
from scripts.evaluate_external import (
    apply_missing_scheme,
    guard_final_dataset,
    load_external_dataset,
    remove_original_dataset_overlap,
    validation_threshold_comparison,
)
from scripts.split_data import main as split_main


def test_multi_source_split_deduplicates_and_keeps_domains_together(
    tmp_path, monkeypatch
):
    paths = []
    for source in range(2):
        rows = [
            {
                "url": f"https://site{domain}.test/source{source}/{label}",
                "is_phishing": label,
            }
            for domain in range(40)
            for label in (0, 1)
        ]
        rows.append({"url": "https://conflict.test/", "is_phishing": source})
        rows.append({"url": "https://duplicate.test/", "is_phishing": 0})
        path = tmp_path / f"source{source}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths.append(path)
    destination = tmp_path / "splits"
    monkeypatch.setattr(
        "sys.argv",
        [
            "split_data.py",
            "--input",
            *(str(path) for path in paths),
            "--output-dir",
            str(destination),
        ],
    )
    split_main()
    splits = {
        name: pd.read_csv(destination / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    validate_training_splits(splits)
    merged = pd.concat(splits.values(), ignore_index=True)
    assert len(merged) == 161
    assert not merged["url"].duplicated().any()
    assert "https://conflict.test/" not in set(merged["url"])
    manifest = json.loads((destination / "split_manifest.json").read_text())
    assert manifest["conflicting_urls_excluded"] == 1
    assert manifest["cross_source_duplicates_removed"] == 1


def test_external_filter_requires_all_reference_splits(tmp_path):
    data = pd.DataFrame({"url": ["https://example.test/"], "is_phishing": [0]})
    data.to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(FileNotFoundError, match="Missing reference split"):
        remove_original_dataset_overlap(data, tmp_path)


def test_explicit_missing_scheme_does_not_use_embedded_query_protocol():
    urls = pd.Series(
        [
            "example.test/login?next=https://other.test/",
            "https://known.test/",
            "ftp://unsupported.test/",
            None,
            "  ",
            "//relative.test/login",
        ]
    )
    values, count = apply_missing_scheme(urls, "http")
    assert count == 2
    assert values.iloc[0] == "http://example.test/login?next=https://other.test/"
    assert values.iloc[1] == "https://known.test/"
    assert values.iloc[2] == "ftp://unsupported.test/"
    assert pd.isna(values.iloc[3])
    assert values.iloc[4] == ""
    assert values.iloc[5] == "http://relative.test/login"


def test_missing_scheme_default_preserves_strings_without_guessing():
    values, count = apply_missing_scheme(
        pd.Series(["example.test/", "https://x.test/"])
    )
    assert count == 0
    assert values.iloc[0] == "example.test/"


def test_external_conflict_exclusion_removes_both_labels(tmp_path):
    path = tmp_path / "new.csv"
    pd.DataFrame(
        {
            "url": [
                "https://conflict.test/",
                "https://conflict.test/",
                "https://safe.test/",
                "https://risk.test/",
            ],
            "label": ["good", "bad", "good", "bad"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting labels"):
        load_external_dataset(path, label_mapping={"good": 0, "bad": 1})
    data = load_external_dataset(
        path, label_mapping={"good": 0, "bad": 1}, exclude_conflicting_urls=True
    )
    assert len(data) == 2
    assert data.attrs["conflicting_rows_excluded"] == 2
    assert data.attrs["conflicting_urls_excluded"] == 1
    assert set(data["is_phishing"]) == {0, 1}


def test_final_dataset_guard_rejects_development_and_old_evaluations(tmp_path):
    dataset = tmp_path / "urls.csv"
    dataset.write_text("url,label\nhttps://example.test/,0\n")
    fingerprint = hashlib.sha256(dataset.read_bytes()).hexdigest()
    metadata = {"development_sources": [{"sha256": fingerprint}]}
    with pytest.raises(ValueError, match="untouched final test"):
        guard_final_dataset(dataset, metadata, tmp_path / "no_history")
    history = tmp_path / "history"
    history.mkdir()
    (history / "external_evaluation.json").write_text(
        json.dumps({"dataset_sha256": fingerprint})
    )
    with pytest.raises(ValueError, match="untouched final test"):
        guard_final_dataset(dataset, {}, history)
    guard_final_dataset(dataset, {}, tmp_path / "no_history")


def test_validation_threshold_comparison_respects_budgets_and_preserves_input():
    data = pd.DataFrame({"is_phishing": [0, 0, 1, 1], "source": ["a"] * 4})
    scores = np.array([0.1, 0.8, 0.7, 0.9])
    original = scores.copy()
    results = validation_threshold_comparison(data, scores, 0.5)
    assert results[0]["metrics"]["false_positive_rate"] == 0.5
    for row in results[1:3]:
        assert row["metrics"]["false_positive_rate"] <= row["maximum_validation_fpr"]
        assert row["metrics"]["recall"] == 0.5
        assert row["by_source"]["a"] == row["metrics"]
    np.testing.assert_array_equal(scores, original)


def test_validation_threshold_comparison_rejects_misaligned_scores():
    with pytest.raises(ValueError, match="equal lengths"):
        validation_threshold_comparison(
            pd.DataFrame({"is_phishing": [0, 1]}), np.array([0.5]), 0.5
        )
