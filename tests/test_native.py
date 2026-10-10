import numpy as np
import pandas as pd
import pytest

from phishguard.lexical import extract_url_features
from phishguard.native import (
    TRAINING_FEATURES,
    choose_threshold,
    domain_partition,
    evaluate_probabilities,
    url_matrix,
)


def test_tied_scores_have_chance_auc_and_prevalence_average_precision():
    metrics = evaluate_probabilities([0, 1, 0, 1], [0.5] * 4, 0.5)
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == 1
    assert metrics["recall"] == 1


def test_ranking_metrics_for_known_order_and_threshold():
    metrics = evaluate_probabilities([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.35)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 2,
    }
    threshold = choose_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 1.0)
    assert threshold == pytest.approx(0.35)


@pytest.mark.parametrize("scores", [[0.1], [0.1, np.nan], [0.1, 1.1]])
def test_invalid_ranking_inputs_are_rejected(scores):
    with pytest.raises(ValueError):
        evaluate_probabilities([0, 1], scores, 0.5)


def test_domains_stay_together_and_split_is_independent_of_row_order():
    frame = pd.DataFrame(
        [
            {"url": f"https://www.site{domain}.com/{label}", "is_phishing": label}
            for domain in range(100)
            for label in (0, 1)
        ]
    )
    first = domain_partition(frame)
    second = domain_partition(frame.sample(frac=1, random_state=7))
    for name in first:
        assert set(first[name]["url"]) == set(second[name]["url"])
        for domain in range(100):
            count = (
                first[name]["url"].str.contains(f"site{domain}.com/", regex=False).sum()
            )
            assert count in {0, 2}
    assert sum(len(part) for part in first.values()) == len(frame)


def test_native_features_keep_training_order_and_root_slash_equivalence():
    matrix = url_matrix(
        [
            "https://example.com",
            "https://example.com/",
            "http://user@192.0.2.10:8080/login",
        ]
    )
    assert matrix.shape == (3, 21)
    assert "slash_count" not in TRAINING_FEATURES
    np.testing.assert_array_equal(matrix[0], matrix[1])
    features = extract_url_features("http://user@192.0.2.10:8080/login")
    assert features["has_ip_hostname"] == 1
    assert features["has_at_symbol"] == 1
    assert features["suspicious_word_count"] == 1


def test_native_booster_save_reload_preserves_predictions(tmp_path):
    import xgboost as xgb

    urls = [
        "https://example.test/about",
        "https://example.test/contact",
        "https://example.test/blog/article",
        "https://example.test/products",
        "http://192.0.2.1/login/verify-account",
        "http://192.0.2.2/login/verify-account",
        "http://192.0.2.3/login/verify-account",
        "http://192.0.2.4/login/verify-account",
    ]
    data = xgb.DMatrix(
        url_matrix(urls),
        label=[0, 0, 0, 0, 1, 1, 1, 1],
        feature_names=list(TRAINING_FEATURES),
    )
    model = xgb.train(
        {
            "objective": "binary:logistic",
            "device": "cpu",
            "nthread": 2,
            "max_depth": 2,
            "min_child_weight": 1,
            "seed": 42,
        },
        data,
        num_boost_round=10,
    )
    path = tmp_path / "model.ubj"
    path.write_bytes(model.save_raw(raw_format="ubj"))
    restored = xgb.Booster()
    restored.load_model(bytearray(path.read_bytes()))
    np.testing.assert_array_equal(model.predict(data), restored.predict(data))
    assert restored.num_features() == len(TRAINING_FEATURES)
    assert restored.predict(data)[:4].max() < restored.predict(data)[4:].min()
