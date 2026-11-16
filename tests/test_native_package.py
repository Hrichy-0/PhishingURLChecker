"""Native model packages: save/load round trip and every integrity refusal."""

import json

import numpy as np
import pytest
import xgboost as xgb

from phishguard.native import TRAINING_FEATURES, url_matrix
from phishguard.native_package import METADATA_FILE, MODEL_FILE, load, open_model, save

URLS = ["https://example.test/", "http://example.test/login", "http://a.test/x?y=1"]


def trained(rounds=6):
    data = xgb.DMatrix(
        url_matrix(URLS), label=[0, 1, 1], feature_names=list(TRAINING_FEATURES)
    )
    params = {"objective": "binary:logistic", "min_child_weight": 0, "nthread": 1}
    return xgb.train(params, data, num_boost_round=rounds)


def saved(tmp_path, **extra):
    booster = trained()
    metadata = save(booster, tmp_path, {"features": list(TRAINING_FEATURES), **extra})
    return booster, metadata


def test_round_trip_records_checksum_and_scores_every_row(tmp_path):
    booster, metadata = saved(tmp_path, threshold=0.5)
    model, loaded = load(tmp_path, TRAINING_FEATURES)
    assert loaded == metadata == json.loads((tmp_path / METADATA_FILE).read_text())
    assert model.sha256 == metadata["model_sha256"]
    assert model.rounds == booster.num_boosted_rounds()
    matrix = url_matrix(URLS)
    expected = booster.predict(
        xgb.DMatrix(matrix, feature_names=list(TRAINING_FEATURES))
    )
    np.testing.assert_allclose(model.scores(matrix), expected)


def test_scores_use_only_recorded_trees(tmp_path):
    booster, metadata = saved(tmp_path, threshold=0.5)
    model = open_model(
        tmp_path / MODEL_FILE, TRAINING_FEATURES, 2, metadata["model_sha256"]
    )
    matrix = url_matrix(URLS)
    data = xgb.DMatrix(matrix, feature_names=list(TRAINING_FEATURES))
    np.testing.assert_allclose(
        model.scores(matrix), booster.predict(data, iteration_range=(0, 2))
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m, p: p.write_bytes(p.read_bytes() + b" "), "checksum"),
        (lambda m, p: m.update(model_sha256=None), "checksum is required"),
        (lambda m, p: m.update(best_iteration=99), "tree range"),
        (lambda m, p: m.update(threshold=1.5), "threshold"),
        (lambda m, p: m.update(features=list(reversed(m["features"]))), "order"),
    ],
)
def test_load_refuses_inconsistent_packages(tmp_path, change, message):
    _, metadata = saved(tmp_path, threshold=0.5)
    change(metadata, tmp_path / MODEL_FILE)
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        load(tmp_path)


def test_load_checks_caller_schema_and_expected_checksum(tmp_path):
    saved(tmp_path, threshold=0.5)
    with pytest.raises(ValueError, match="schema"):
        load(tmp_path, TRAINING_FEATURES[:-1])
    with pytest.raises(ValueError, match="checksum"):
        load(tmp_path, expected_sha256="0" * 64)
