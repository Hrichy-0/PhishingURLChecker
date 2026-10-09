"""Keep model selection and threshold selection away from outer test folds."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from phishguard.experiments import (
    build_variant,
    choose_low_fpr_threshold,
    grouped_fit_validation_indices,
    run_benchmark,
    run_source_tuning,
    source_class_weights,
    source_partitions,
    source_validation_threshold,
)
from phishguard.modeling import build_model, evaluate_probabilities
from phishguard.split import validate_training_splits
from phishguard.text_features import URLTextNormalizer
from scripts.activate_model import main as activate_main
from scripts.compare_url_models import compare_url


def sample_data(domains=30):
    return pd.DataFrame(
        [
            {
                "url": (
                    f"https://site{domain}.test/"
                    f"{'verify-account' if label else 'about'}"
                ),
                "is_phishing": label,
                "source": f"source{domain % 2}",
            }
            for domain in range(domains)
            for label in (0, 1)
        ]
    )


def test_text_normalization_omits_protocol_fragment_and_root_slash():
    urls = [
        "http://example.test/",
        "https://example.test/#private",
        "https://example.test/login?a=b#secret",
    ]
    assert URLTextNormalizer().fit_transform(urls).tolist() == [
        "example.test",
        "example.test",
        "example.test/login?a=b",
    ]


def test_character_pipeline_roundtrip_and_no_validation_vocabulary(tmp_path):
    frame = sample_data()
    model = build_model(algorithm="char_logistic")
    model.fit(frame["url"], frame["is_phishing"])
    original_vocabulary = dict(model.named_steps["tfidf"].vocabulary_)
    probabilities = model.predict_proba(["https://never-seen.test/zzzzzzzzz"])
    assert model.named_steps["tfidf"].vocabulary_ == original_vocabulary
    assert "zzzzz" not in original_vocabulary
    assert probabilities.shape == (1, 2)
    joblib.dump(model, tmp_path / "char.joblib")
    np.testing.assert_allclose(
        probabilities,
        joblib.load(tmp_path / "char.joblib").predict_proba(
            ["https://never-seen.test/zzzzzzzzz"]
        ),
    )


def test_low_false_positive_threshold_handles_ties():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.6, 0.9])
    threshold = choose_low_fpr_threshold(labels, scores, 0)
    metrics = evaluate_probabilities(labels, scores, threshold)
    assert threshold == 0.9
    assert metrics["false_positive_rate"] == 0
    assert metrics["recall"] == 0.5


def test_threshold_can_choose_no_alerts():
    labels = np.array([0, 0, 1, 1])
    scores = np.ones(4)
    threshold = choose_low_fpr_threshold(labels, scores, 0)
    assert threshold > 1
    assert evaluate_probabilities(labels, scores, threshold)["recall"] == 0


@pytest.mark.parametrize("budget", [-0.01, 1.0])
def test_threshold_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="maximum_fpr"):
        choose_low_fpr_threshold(np.array([0, 1]), np.array([0.1, 0.9]), budget)


def test_inner_validation_is_domain_disjoint():
    frame = sample_data()
    fit, validation = grouped_fit_validation_indices(frame, seed=42)
    validate_training_splits(
        {"fit": frame.iloc[fit], "validation": frame.iloc[validation]}
    )


def test_https_ablation_does_not_change_original_feature_schema():
    assert (
        "uses_https"
        in build_variant("xgboost").named_steps["features"].get_feature_names_out()
    )
    assert (
        "uses_https"
        not in build_variant("xgboost_no_https")
        .named_steps["features"]
        .get_feature_names_out()
    )


def test_benchmark_saves_both_models_with_identical_fold_roles(tmp_path):
    report = run_benchmark(
        sample_data(),
        tmp_path / "experiment",
        variants=("xgboost_no_https", "char_logistic"),
        folds=2,
        trees=8,
    )
    assert report["status"] == "complete"
    assert report["protocol"]["final_test"] == "not used by this benchmark"
    xgb = report["variants"]["xgboost_no_https"]["folds"]
    char = report["variants"]["char_logistic"]["folds"]
    assert [r["role_sha256"] for r in xgb] == [r["role_sha256"] for r in char]
    for variant in ("xgboost_no_https", "char_logistic"):
        destination = tmp_path / "experiment" / variant
        predictions = pd.read_csv(destination / "out_of_fold_predictions.csv")
        assert len(predictions) == len(sample_data())
        assert predictions["url"].is_unique
        metadata = json.loads((destination / "model_metadata.json").read_text())
        assert metadata["activation_eligible"] is False
        model = joblib.load(destination / "phishguard_model.joblib")
        assert model.predict_proba(["https://example.test/"]).shape == (1, 2)
    with pytest.raises(ValueError, match="never overwrite"):
        run_benchmark(sample_data(), tmp_path / "experiment", folds=2)


def test_local_comparison_returns_scores_without_activation(tmp_path):
    run_benchmark(sample_data(), tmp_path / "compare", folds=2, trees=8)
    result = compare_url("https://example.test/", tmp_path / "compare")
    assert result["experimental"] is True
    assert set(result["results"]) == {"xgboost", "xgboost_no_https", "char_logistic"}
    assert all(0 <= r["uncalibrated_score"] <= 1 for r in result["results"].values())
    assert not (tmp_path / "compare" / "phishguard_model.joblib").exists()


def test_local_comparison_rejects_non_http_url(tmp_path):
    with pytest.raises(ValueError, match="HTTP"):
        compare_url("file:///etc/passwd", tmp_path)


def test_source_class_weights_equalize_pairs_using_fit_only():
    frame = sample_data()
    frame = pd.concat(
        [frame, frame.loc[frame["source"].eq("source0")]], ignore_index=True
    )
    weights = source_class_weights(frame)
    totals = (
        frame.assign(weight=weights).groupby(["source", "is_phishing"])["weight"].sum()
    )
    np.testing.assert_allclose(totals.to_numpy(), np.repeat(len(frame) / 4, 4))
    assert weights.mean() == pytest.approx(1)


def test_source_partitions_purge_shared_domains():
    frame = pd.concat(
        [
            sample_data(),
            pd.DataFrame(
                [
                    {
                        "url": f"https://shared.test/source{source}/{label}",
                        "source": f"source{source}",
                        "is_phishing": label,
                    }
                    for source in (0, 1)
                    for label in (0, 1)
                ]
            ),
        ],
        ignore_index=True,
    )
    partitions = source_partitions(frame, seed=42)
    for source, fit, validation, test, excluded in partitions:
        assert excluded == 2
        assert not fit["url"].str.contains("shared.test", regex=False).any()
        assert not validation["url"].str.contains("shared.test", regex=False).any()
        assert set(test["source"]) == {source}
        assert source not in set(fit["source"]) | set(validation["source"])


def test_source_threshold_applies_budget_to_each_source():
    frame = pd.DataFrame({"source": ["a", "a", "b", "b"], "is_phishing": [0, 1, 0, 1]})
    scores = np.array([0.2, 0.6, 0.8, 0.9])
    threshold = source_validation_threshold(frame, scores, 0)
    assert threshold == 0.9
    for source in ("a", "b"):
        mask = frame["source"].eq(source).to_numpy()
        assert (
            evaluate_probabilities(
                frame.loc[mask, "is_phishing"].to_numpy(), scores[mask], threshold
            )["false_positive_rate"]
            == 0
        )


def test_source_tuning_saves_only_selected_and_comparison(tmp_path):
    candidates = (
        {
            "name": "baseline",
            "feature_policy": "standard",
            "weighted": False,
            "params": {},
        },
        {
            "name": "invariant",
            "feature_policy": "protocol_invariant",
            "weighted": True,
            "params": {},
        },
    )
    reserved = pd.DataFrame(
        {
            "url": ["https://reserved.test/a", "https://reserved.test/b"],
            "is_phishing": [0, 1],
        }
    )
    destination = tmp_path / "tune"
    report = run_source_tuning(
        sample_data(), destination, reserved, trees=8, candidates=candidates
    )
    assert report["status"] == "complete"
    assert len(report["trials"]) == 3  # two hypotheses plus character baseline
    assert report["selected_xgboost"] in {"baseline", "invariant"}
    trials = list(report["trials"].values())
    assert all(
        [r["role_sha256"] for r in trial["held_sources"]]
        == [r["role_sha256"] for r in trials[0]["held_sources"]]
        for trial in trials
    )
    for variant in ("xgboost", "char_logistic"):
        metadata = json.loads(
            (destination / variant / "model_metadata.json").read_text()
        )
        assert metadata["activation_eligible"] is False
        assert "test_metrics" not in metadata
        model = joblib.load(destination / variant / "phishguard_model.joblib")
        assert model.predict_proba(["https://example.test/"]).shape == (1, 2)
    refs = {
        name: pd.read_csv(destination / "reference" / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    validate_training_splits(refs)
    with pytest.raises(ValueError, match="never overwrite"):
        run_source_tuning(
            sample_data(), destination, reserved, trees=8, candidates=candidates
        )


def test_development_candidate_cannot_be_activated(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model_metadata.json").write_text(
        json.dumps({"activation_eligible": False})
    )
    active = tmp_path / "active"
    monkeypatch.setattr("scripts.activate_model.ARTIFACTS_DIR", active)
    monkeypatch.setattr(
        "sys.argv", ["activate_model.py", "--artifacts-dir", str(candidate)]
    )
    with pytest.raises(ValueError, match="development candidate"):
        activate_main()
    assert not active.exists()
