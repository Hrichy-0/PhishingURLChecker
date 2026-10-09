import joblib
import numpy as np
import pytest

from phishguard.modeling import (
    build_model,
    choose_threshold,
    evaluate_probabilities,
    fit_xgboost_candidates,
)


@pytest.mark.parametrize("algorithm", ["logistic", "xgboost"])
def test_model_pipeline_fits_and_predicts_probabilities(algorithm, tmp_path) -> None:
    urls = np.array(
        [
            "https://example.com/about",
            "https://university.edu/research",
            "http://192.168.1.1/login/verify-account",
            "http://secure-update.example.test/password?id=12345",
        ]
    )
    labels = np.array([0, 0, 1, 1])
    kwargs = {"n_estimators": 10} if algorithm == "xgboost" else {}
    model = build_model(algorithm=algorithm, **kwargs)
    model.fit(urls, labels)

    probabilities = model.predict_proba(urls)[:, 1]
    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(
        joblib.load(path).predict_proba(urls)[:, 1], probabilities
    )
    assert "slash_count" not in model.named_steps["features"].get_feature_names_out()


def test_threshold_selection_respects_recall_target() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = choose_threshold(labels, probabilities, minimum_recall=1.0)
    metrics = evaluate_probabilities(labels, probabilities, threshold)

    assert metrics["recall"] == 1.0
    assert 0 <= threshold <= 1


def test_xgboost_validation_features_match_packaged_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        "phishguard.modeling.XGBOOST_CANDIDATES",
        ({"max_depth": 2, "n_estimators": 12, "min_child_weight": 1},),
    )
    urls = np.array(
        [
            "https://example.com/about",
            "https://university.edu/research",
            "http://192.0.2.10/login",
            "http://secure-account.test/verify",
        ]
        * 10
    )
    labels = np.array([0, 0, 1, 1] * 10)
    model, search = fit_xgboost_candidates(urls, labels, urls[:4], labels[:4])
    features = model.named_steps["features"].transform(urls[:4])
    np.testing.assert_allclose(
        model.predict_proba(urls[:4]),
        model.named_steps["classifier"].predict_proba(features),
    )
    assert len(search) == 1
    assert search[0]["best_iteration"] < 12
