"""Audit a completed HTML comparison and reproduce predictions on local CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phishguard.native import evaluate_probabilities
from phishguard.native_html import VARIANT_FEATURES
from phishguard.native_package import load
from phishguard.provenance import verify_split_files


def verify_results(experiment: Path) -> dict:
    manifest = json.loads((experiment / "split_manifest.json").read_text())
    comparison = json.loads((experiment / "comparison.json").read_text())
    assert comparison["status"] == "completed"
    splits = ("train", "validation", "test", "publisher_test")
    verify_split_files(experiment, manifest, splits)
    frames = {}
    for name in splits:
        path = experiment / f"{name}_rows.parquet"
        with path.open("rb") as stream:
            frames[name] = pd.read_parquet(stream)
        assert len(frames[name]) == manifest["cohorts"][name]["rows"], name
    for index, name in enumerate(frames):
        for other in list(frames)[index + 1 :]:
            assert set(frames[name]["domain"]).isdisjoint(frames[other]["domain"])
            hashes = set(frames[name]["html_sha256"]) - {""}
            assert hashes.isdisjoint(set(frames[other]["html_sha256"]) - {""})
    report = {
        "status": "verified",
        "domain_and_exact_html_disjoint": True,
        "split_checksums_verified": True,
        "cohort_rows": {name: len(frame) for name, frame in frames.items()},
        "prediction_device": "cpu",
        "models": {},
    }
    for variant, names in VARIANT_FEATURES.items():
        directory = experiment / variant
        model, metadata = load(directory, names)
        # Saved experiment packages keep only their best-iteration trees.
        assert model.booster.num_boosted_rounds() == model.rounds
        checked = {}
        for name in ("validation", "test", "publisher_test"):
            frame = frames[name]
            saved = pd.read_csv(experiment / f"{variant}_{name}_predictions.csv")
            labels = frame["is_phishing"].to_numpy()
            np.testing.assert_array_equal(labels, saved["is_phishing"])
            matrix = frame.loc[:, list(names)].to_numpy(dtype=np.float32)
            scores = model.scores(matrix)
            np.testing.assert_allclose(scores, saved["score"], rtol=1e-5, atol=1e-7)
            predicted = scores >= metadata["threshold"]
            saved_predicted = saved["predicted"].to_numpy(dtype=bool)
            metrics = evaluate_probabilities(labels, scores, metadata["threshold"])
            recorded = comparison["models"][variant]["metrics"][name]
            differences = int((predicted != saved_predicted).sum())
            low_fpr = metadata["validation_1_percent_fpr_operating_point"]
            conservative = evaluate_probabilities(labels, scores, low_fpr["threshold"])
            recorded_conservative = comparison["models"][variant][
                "validation_1_percent_fpr_metrics"
            ][name]
            checked[name] = {
                "rows": len(frame),
                "probabilities_match_with_float_tolerance": True,
                "cpu_vs_recorded_threshold_decision_differences": differences,
                "confusion_matrix_matches": metrics["confusion_matrix"]
                == recorded["confusion_matrix"],
                "cpu_accuracy": metrics["accuracy"],
                "low_fpr_confusion_matrix_matches": (
                    conservative["confusion_matrix"]
                    == recorded_conservative["confusion_matrix"]
                ),
                "low_fpr_cpu_metrics": conservative,
            }
        assert checked["publisher_test"]["confusion_matrix_matches"], variant
        report["models"][variant] = {
            "model_checksum_verified": True,
            "features": len(model.features),
            "trees": model.rounds,
            "cohorts": checked,
        }
        print(f"Verified saved {variant} on CPU", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_results(args.experiment)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
