"""Leakage safeguards, thresholds, aggregation and a tiny real CV smoke test."""

import numpy as np
import pandas as pd
import pytest

from phishguard.native import TRAINING_FEATURES, url_matrix
from phishguard.native_html import HYBRID_FEATURE_NAMES
from scripts.cross_validate_native import (
    POLICIES,
    clustered_intervals,
    compare_runs,
    connected_groups,
    count_metrics,
    low_fpr_threshold,
    make_partitions,
    run_cv,
)


def development():
    urls = [
        f"https://site{domain}.test/{'login-verify-account' if label else 'about'}"
        for domain in range(100)
        for label in (0, 1)
    ]
    frame = pd.DataFrame(url_matrix(urls), columns=TRAINING_FEATURES)
    frame["url"] = urls
    frame["domain"] = [f"site{domain}.test" for domain in range(100) for _ in (0, 1)]
    frame["is_phishing"] = [0, 1] * 100
    frame["html_sha256"] = [f"digest{index}" for index in range(200)]
    frame["original_split"] = "train"
    return frame


def test_components_are_transitive_and_ignore_empty_hashes():
    frame = pd.DataFrame(
        {
            "domain": ["a.test", "b.test", "b.test", "c.test", "d.test", "e.test"],
            "html_sha256": ["one", "one", "two", "two", "", ""],
        }
    )
    groups = connected_groups(frame)
    assert len(set(groups[:4])) == 1
    assert len(set(groups)) == 3
    shuffled = frame.sample(frac=1, random_state=7)
    shuffled_groups = connected_groups(shuffled)
    assert dict(zip(shuffled.index, shuffled_groups, strict=True)) == dict(
        enumerate(groups)
    )


def test_every_role_is_group_disjoint_and_outer_rows_covered_once():
    frame = development()
    frame.loc[2, "html_sha256"] = frame.loc[0, "html_sha256"]
    groups = connected_groups(frame)
    partitions = make_partitions(frame, groups)
    assert len(partitions) == 5
    for fit, inner, outer in partitions:
        assert set(groups[fit]).isdisjoint(groups[inner])
        assert set(groups[fit]).isdisjoint(groups[outer])
        assert set(groups[inner]).isdisjoint(groups[outer])
    np.testing.assert_array_equal(
        np.sort(np.concatenate([p[2] for p in partitions])), np.arange(len(frame))
    )


def test_low_fpr_ties_and_no_alert_fallback():
    assert low_fpr_threshold([0, 1, 0, 1], [0.8, 0.8, 0.1, 0.9], 0.0) == 0.9
    scores = np.asarray([1, 1], dtype=np.float32)
    cutoff = low_fpr_threshold([0, 1], scores, 0.0)
    assert cutoff > 1
    assert not (scores >= cutoff).any()
    assert count_metrics([90, 10, 20, 80])["false_positive_rate"] == 0.1


def test_cluster_bootstrap_keeps_perfect_predictions_and_rejects_split_components():
    frame = pd.DataFrame(
        {
            "component_id": np.repeat(np.arange(10), 2),
            "fold": np.repeat([1, 2], 10),
            "is_phishing": [0, 1] * 10,
            "score": [0.1, 0.9] * 10,
            **{f"threshold_{policy}": 0.5 for policy in POLICIES},
        }
    )
    result = clustered_intervals(frame, draws=20)
    assert result["draws_valid"] == 20
    assert result["intervals"][POLICIES[0]]["recall"] == [1, 1]
    assert result["intervals"][POLICIES[0]]["false_positive_rate"] == [0, 0]
    frame.loc[0, "fold"] = 2
    with pytest.raises(ValueError, match="component crosses"):
        clustered_intervals(frame)


def test_cv_smoke_outputs_best_round_models_and_one_oof_score_per_row(tmp_path):
    frame = development()
    output = tmp_path / "cv"
    parameters = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "max_depth": 2,
        "min_child_weight": 0,
    }
    report = run_cv(
        frame, output, parameters, 0.49, folds=3, trees=5, threads=1, bootstrap_draws=20
    )
    assert report["status"] == "completed"
    assert report["features"] == list(TRAINING_FEATURES)
    assert len(report["folds"]) == 3
    oof = pd.read_parquet(output / "out_of_fold.parquet")
    assert len(oof) == len(frame)
    assert oof.row_index.is_unique
    assert (oof.score.between(0, 1)).all()
    for record in report["folds"]:
        assert record["trees"] <= 5
        assert record["parameters"]["device"] == "cpu"
    for policy in POLICIES:
        assert sum(report["pooled"][policy]["confusion_matrix"].values()) == len(frame)
    with pytest.raises(ValueError, match="existing runs"):
        run_cv(frame, output, parameters, 0.49)


def test_content_candidate_reuses_identical_roles_and_never_includes_group_metadata(
    tmp_path,
):
    frame = development()
    for name in HYBRID_FEATURE_NAMES:
        if name not in frame:
            frame[name] = 0.0
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "max_depth": 2,
    }
    outputs = [tmp_path / name for name in ("url", "combined")]
    for output, features in zip(
        outputs, (TRAINING_FEATURES, HYBRID_FEATURE_NAMES), strict=True
    ):
        report = run_cv(
            frame,
            output,
            params,
            0.49,
            folds=2,
            trees=2,
            threads=1,
            bootstrap_draws=5,
            feature_names=features,
            provenance={"split_sha256": {"train": "fixture"}},
        )
        assert report["features"] == list(features)
        assert "html_sha256" not in report["features"]
        assert "domain" not in report["features"]
    pd.testing.assert_frame_equal(
        *[pd.read_parquet(p / "role_assignments.parquet") for p in outputs]
    )
    comparison = compare_runs(outputs[0], outputs[0], draws=5)
    assert comparison["same_records_roles_components"] is True
    for metrics in comparison["policy_differences"].values():
        for result in metrics.values():
            assert result["point"] == 0
            assert result["percentile_95_interval"] == [0, 0]
    mismatched = pd.read_parquet(outputs[1] / "out_of_fold.parquet")
    mismatched.loc[0, "is_phishing"] = 1 - mismatched.loc[0, "is_phishing"]
    mismatched.to_parquet(outputs[1] / "out_of_fold.parquet", index=False)
    with pytest.raises(AssertionError):
        compare_runs(*outputs, draws=5)
