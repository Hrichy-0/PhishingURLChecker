"""Offline extraction, paired comparisons, cache integrity, and leakage checks."""

import hashlib
import json
import socket

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from phishguard.html_features import CONTENT_FEATURE_NAMES, extract_content_features
from phishguard.native import evaluate_probabilities
from phishguard.native_html import (
    HTML_FEATURE_NAMES,
    HYBRID_FEATURE_NAMES,
    VARIANT_FEATURES,
    content_partition,
    fingerprint,
    prepare_snapshot_cache,
    purge_publisher_test,
    snapshot_features,
    snapshot_vector,
)
from scripts.train_native_html_model import low_fpr_threshold, train_comparison


def snapshot_frame(prefix, count=160):
    rows = []
    for index in range(count):
        label = index % 2
        url = f"https://{prefix}{index}.com/login"
        html = (
            f'<title>PayPal {index}</title><form action="https://other.test/send">'
            '<input type="password"></form>'
            if label
            else f"<title>About {index}</title><a href='/about'>About</a>"
        )
        features = snapshot_vector(url, html)[0]
        rows.append(
            {
                "url": url,
                "domain": f"{prefix}{index}.com",
                "html_sha256": hashlib.sha256((prefix + html).encode()).hexdigest(),
                "date": "2025-01-01",
                "is_phishing": label,
                **dict(zip(HYBRID_FEATURE_NAMES, features, strict=True)),
            }
        )
    return pd.DataFrame(rows)


def test_html_parser_does_not_visit_links_or_retain_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network calls in offline HTML parsing")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    html = (
        '<title>PayPal Login</title><form action="https://other.test/send">'
        '<input type="password" value="secret-password">'
        '<input type="hidden" value="secret-token"></form>'
        '<script src="https://other.test/run.js">throw new Error("no");</script>'
    )
    values = snapshot_features("https://example.test/", html)
    assert values["password_field_count"] == values["hidden_field_count"] == 1
    assert values["external_form_ratio"] == 1
    assert values["title_brand_domain_mismatch_count"] == 1
    assert values["external_script_ratio"] == 1
    assert "secret" not in json.dumps(values)
    assert snapshot_vector("https://example.test/", html).shape == (1, 38)
    assert len(HTML_FEATURE_NAMES) == 17
    legitimate = extract_content_features(
        "https://www.paypal.com/login",
        '<title>PayPal Login</title><form action="/login">'
        '<input type="password"></form>',
    )
    assert legitimate["title_brand_domain_mismatch_count"] == 0
    assert legitimate["external_form_ratio"] == 0


def test_missing_html_is_unknown_and_multibyte_truncation_is_bounded():
    missing = snapshot_features("https://example.test/", None)
    assert missing["html_available"] == 0
    assert all(np.isnan(missing[name]) for name in CONTENT_FEATURE_NAMES)
    html = "<title>éééééééééééé</title><input type='password'>"
    truncated = snapshot_features("https://example.test/", html, max_html_bytes=20)
    assert truncated["html_available"] == truncated["html_truncated"] == 1
    assert truncated["html_parse_error"] == 0
    assert truncated["password_field_count"] == 0
    complete = snapshot_features("https://example.test/", html)
    assert complete["password_field_count"] == 1
    assert complete["html_truncated"] == 0
    with pytest.raises(ValueError, match="absolute"):
        snapshot_features("file:///local", html)


def test_form_base_and_unsupported_destinations():
    values = snapshot_features(
        "https://example.test/",
        '<base href="https://other.test/"><form action="send"></form>'
        '<form></form><form action="javascript:alert(1)"></form>',
    )
    assert values["external_form_ratio"] == 0.5
    assert values["unsupported_form_ratio"] == pytest.approx(1 / 3)


def test_transitive_html_and_domain_components_cannot_cross_splits():
    frame = snapshot_frame("domain")
    # Bridge three different domains through two different exact snapshots.
    frame.loc[0:1, "html_sha256"] = "shared-one"
    frame.loc[2, "domain"] = frame.loc[1, "domain"]
    frame.loc[2:3, "html_sha256"] = "shared-two"
    first = content_partition(frame)
    reordered = content_partition(frame.sample(frac=1, random_state=7))
    for name, part in first.items():
        assert set(part["url"]) == set(reordered[name]["url"])
        assert len(set(part["url"]) & set(frame.loc[:3, "url"])) in {0, 4}
    assert sum(len(part) for part in first.values()) == len(frame)


def test_publisher_purge_removes_domain_and_html_overlap():
    development = snapshot_frame("train")
    test = snapshot_frame("test", 30)
    test.loc[0, "domain"] = development.loc[0, "domain"]
    test.loc[1, "html_sha256"] = development.loc[1, "html_sha256"]
    remaining, report = purge_publisher_test(test, development)
    assert len(remaining) == 28
    assert report["excluded_rows_union"] == 2


def test_cache_resumes_and_rejects_changed_original_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    entries = []
    for split in ("train", "test"):
        path = raw / f"{split}-000.parquet"
        with path.open("wb") as stream:
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "url": f"https://{split}{label}.test/",
                            "label": "phish" if label else "benign",
                            "date": "2025-01-01",
                            "html": "<form><input type='password'></form>"
                            if label
                            else None,
                            "target": "annotation-must-not-be-a-feature",
                            "sha256": "annotation-hash-not-trusted",
                        }
                        for label in (0, 1)
                    ]
                ),
                stream,
            )
        entries.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": fingerprint(path),
            }
        )
    (raw / "download_manifest.json").write_text(
        json.dumps({"complete_dataset": True, "revision": "fixture", "files": entries})
    )
    cache = tmp_path / "cache"
    first, _ = prepare_snapshot_cache(raw, cache, batch_rows=1)
    path = cache / "train-000.parquet"
    original_mtime = path.stat().st_mtime_ns
    second, _ = prepare_snapshot_cache(raw, cache)
    pd.testing.assert_frame_equal(first["train"], second["train"])
    assert path.stat().st_mtime_ns == original_mtime
    assert "target" not in first["train"]
    assert "sha256" not in first["train"]
    path.write_bytes(b"broken cache")
    repaired, _ = prepare_snapshot_cache(raw, cache)
    pd.testing.assert_frame_equal(first["train"], repaired["train"])
    source = raw / "train-000.parquet"
    source.write_bytes(b"x" * source.stat().st_size)
    with pytest.raises(ValueError, match="checksum"):
        prepare_snapshot_cache(raw, cache)


def test_low_fpr_threshold_uses_validation_and_can_choose_no_alerts():
    threshold = low_fpr_threshold([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.95], 0)
    assert threshold == 0.95
    tied = np.asarray([1.0, 1.0], dtype=np.float32)
    threshold = low_fpr_threshold([0, 1], tied, 0)
    assert threshold > 1.0
    metrics = evaluate_probabilities([0, 1], tied, threshold)
    assert metrics["false_positive_rate"] == metrics["recall"] == 0


def test_small_native_html_training_is_paired_and_reloadable(tmp_path):
    from scripts.train_native_html_model import protected_hashes

    before = protected_hashes()
    report = train_comparison(
        snapshot_frame("train"),
        snapshot_frame("external", 40),
        tmp_path / "experiment",
        provenance={"fixture": True},
        trees=12,
        threads=2,
        depths=(2,),
    )
    assert report["status"] == "completed"
    assert report["active_and_previous_url_models_unchanged"]
    assert before == protected_hashes()
    row_counts = []
    for variant, names in VARIANT_FEATURES.items():
        folder = tmp_path / "experiment" / variant
        metadata = json.loads((folder / "model_metadata.json").read_text())
        row_counts.append(metadata["dataset_rows"])
        assert metadata["features"] == list(names)
        assert metadata["activation_eligible"] is False
        model = xgb.Booster()
        model.load_model(bytearray((folder / "xgboost_model.ubj").read_bytes()))
        assert model.num_features() == len(names)
        values = snapshot_vector("https://new.test/login", None, names)
        scores = model.predict(xgb.DMatrix(values, feature_names=list(names)))
        assert np.isfinite(scores).all()
    assert row_counts[0] == row_counts[1] == row_counts[2]
    with pytest.raises(ValueError, match="empty experiment"):
        train_comparison(
            snapshot_frame("train"),
            snapshot_frame("test"),
            tmp_path / "experiment",
            provenance={},
        )
