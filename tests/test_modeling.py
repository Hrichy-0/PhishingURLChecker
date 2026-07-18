import numpy as np

from phishguard.modeling import (
    build_model,
    choose_threshold,
    evaluate_probabilities,
)


def test_model_pipeline_fits_and_predicts_probabilities() -> None:
    urls = np.array(
        [
            "https://example.com/about",
            "https://university.edu/research",
            "http://192.168.1.1/login/verify-account",
            "http://secure-update.example.test/password?id=12345",
        ]
    )
    labels = np.array([0, 0, 1, 1])
    model = build_model()
    model.fit(urls, labels)

    probabilities = model.predict_proba(urls)[:, 1]
    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_threshold_selection_respects_recall_target() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = choose_threshold(labels, probabilities, minimum_recall=1.0)
    metrics = evaluate_probabilities(labels, probabilities, threshold)

    assert metrics["recall"] == 1.0
    assert 0 <= threshold <= 1
