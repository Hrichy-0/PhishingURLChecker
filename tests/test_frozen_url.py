"""External test labels must be reviewed before the frozen model can be scored."""

import json
import socket

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_frozen_url import (
    REVIEW_FIELDS,
    history_domains,
    registered,
    score,
    select_candidates,
    sha,
    uncertainty,
    verified_review,
    write_json,
)


def candidates_and_review():
    rows, _ = select_candidates(
        "https://suspect.example/login\n",
        [(12, "organization.example")],
        set(),
    )
    candidates = pd.DataFrame(rows)
    review = candidates.copy()
    for field in REVIEW_FIELDS:
        review[field] = ""
    review["status"] = "accepted"
    review["review_label"] = ["1", "0"]
    review["reviewer"] = "test fixture, not a real adjudication"
    review["reviewed_at"] = "2026-10-19T10:00:00+00:00"
    review["evidence"] = "synthetic evidence for unit test only"
    review.loc[review.source == "Tranco", "url"] = "https://organization.example/about"
    return candidates, review


def test_candidates_exclude_known_domains_and_dont_guess_benign_url():
    rows, report = select_candidates(
        "https://old.example/login\nhttps://new.example/a\n"
        "http://new.example/b\nno-scheme.example/login\n",
        [(1, "old.example"), (2, "new.example"), (3, "ranked.example")],
        {"old.example"},
    )
    assert len(rows) == 2
    assert {r["domain"] for r in rows} == {"new.example", "ranked.example"}
    ranked = next(r for r in rows if r["source"] == "Tranco")
    assert ranked["url"] == "" and ranked["source_label"] == "unknown"
    assert report["feed_known_domain"] == 1
    assert report["feed_repeated_domain"] == 1
    assert (
        rows
        == select_candidates(
            "https://old.example/login\nhttps://new.example/a\n"
            "http://new.example/b\nno-scheme.example/login\n",
            [(1, "old.example"), (2, "new.example"), (3, "ranked.example")],
            {"old.example"},
        )[0]
    )


@pytest.mark.parametrize(
    "case",
    [
        "pending",
        "evidence",
        "label",
        "url",
        "source",
        "known",
        "missing",
        "duplicate",
        "timestamp",
    ],
)
def test_review_guardrails(case):
    candidates, review = candidates_and_review()
    known = set()
    if case == "pending":
        review.loc[0, "status"] = "pending"
    elif case == "evidence":
        review.loc[0, "evidence"] = ""
    elif case == "label":
        review.loc[0, "review_label"] = "phishing"
    elif case == "url":
        review.loc[0, "url"] = "https://replacement.example/"
    elif case == "source":
        review.loc[0, "source"] = "changed"
    elif case == "known":
        known.add("organization.example")
    elif case == "missing":
        review = review.iloc[:1]
    elif case == "duplicate":
        review = pd.concat([review, review.iloc[:1]])
    elif case == "timestamp":
        review.loc[0, "reviewed_at"] = "2026-10-19"
    with pytest.raises(ValueError):
        verified_review(candidates, review, known)


def test_review_accepts_labels_with_explicit_evidence_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must never visit URLs to score them")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    candidates, review = candidates_and_review()
    accepted, counts = verified_review(candidates, review, set())
    assert counts == {"accepted": 2}
    assert accepted.is_phishing.tolist() == [1, 0]


def test_uncertain_records_remain_in_attrition_count():
    candidates, review = candidates_and_review()
    review.loc[0, "status"] = "uncertain"
    review.loc[0, "review_label"] = ""
    accepted, counts = verified_review(candidates, review, set())
    assert counts == {"uncertain": 1, "accepted": 1}
    assert len(accepted) == 1


def test_history_includes_previous_tests_and_bare_domain_exclusions(tmp_path):
    raw, old, output = (
        tmp_path / "data/raw",
        tmp_path / "artifacts/old",
        tmp_path / "data/processed/fresh",
    )
    for directory in (raw, old, output):
        directory.mkdir(parents=True)
    pd.DataFrame({"URL": ["https://a.old-example.com/", "bare.example/path"]}).to_csv(
        raw / "source.csv", index=False
    )
    pd.DataFrame({"url": ["https://prior-test.example/"]}).to_parquet(
        old / "test.parquet", index=False
    )
    pd.DataFrame({"url": ["https://candidate.example/"]}).to_csv(
        output / "review.csv", index=False
    )
    domains, inventory = history_domains(tmp_path, output)
    assert domains == {
        "old-example.com",
        "bare.example",
        "prior-test.example",
    }
    assert "candidate.example" not in domains
    assert len(inventory) == 2


@pytest.mark.parametrize("tamper", [False, True])
def test_score_refuses_unreviewed_or_modified_inputs_before_model_load(
    tmp_path, monkeypatch, tamper
):
    candidates, review = candidates_and_review()
    review["status"] = "pending"
    candidates.to_parquet(tmp_path / "candidates.parquet", index=False)
    review.to_csv(tmp_path / "review.csv", index=False)
    (tmp_path / "excluded_domains.txt").write_text("previous.example\n")
    write_json(
        tmp_path / "protocol.json",
        {
            "software_sha256": {},
            "versions": {},
            "minimum_reviewed_domains_per_class": 30,
        },
    )
    write_json(
        tmp_path / "preparation.json",
        {
            "frozen_sha256": {
                name: sha(tmp_path / name)
                for name in (
                    "protocol.json",
                    "candidates.parquet",
                    "excluded_domains.txt",
                )
            },
        },
    )
    if tamper:
        payload = json.loads((tmp_path / "protocol.json").read_text())
        payload["threshold"] = 0.99
        write_json(tmp_path / "protocol.json", payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("No model may be loaded before review passes")

    monkeypatch.setattr("scripts.evaluate_frozen_url.NativeURLService", forbidden)
    with pytest.raises(ValueError, match="changed" if tamper else "incomplete"):
        score(tmp_path, tmp_path)
    assert not (tmp_path / "predictions.parquet").exists()


def test_missing_scheme_not_inferred_for_scoring():
    with pytest.raises(ValueError, match="explicit HTTP"):
        registered("organization.example/about")
    assert registered("organization.example/about", historical=True)


def test_zero_false_positives_does_not_mean_zero_uncertainty():
    labels = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    scores = labels.astype(float)
    ci = uncertainty(labels, scores, 0.5, 200, 42)
    assert ci["conditional_stratified_domain_bootstrap_95"]["false_positive_rate"] == [
        0,
        0,
    ]
    lower, upper = ci["wilson_95"]["false_positive_rate"]
    assert abs(lower) < 1e-10 and 0.03 < upper < 0.04
