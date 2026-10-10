"""Train and evaluate URL-only XGBoost locally through its supported native API."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from phishguard.config import PROJECT_ROOT
from phishguard.data import PROJECT_LABEL_MAPPING, clean_url_data, combine_sources
from phishguard.domains import get_registered_domain
from phishguard.native import (
    TRAINING_FEATURES,
    choose_threshold,
    domain_partition,
    evaluate_probabilities,
    url_matrix,
)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--external-test", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trees <= 0 or not 1 <= args.threads <= 16:
        parser.error("Require positive trees and 1–16 CPU threads")
    if (args.artifacts_dir / "model_metadata.json").exists():
        parser.error("Experiment already completed; choose a new artifacts directory")
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    active_hashes = {
        name: fingerprint(PROJECT_ROOT / "artifacts" / name)
        for name in ("phishguard_model.joblib", "model_metadata.json")
    }
    sources, frames = [], []
    for path in args.input:
        frame, cleaning = clean_url_data(
            pd.read_csv(path),
            url_column="url",
            label_column="is_phishing",
            label_mapping=PROJECT_LABEL_MAPPING,
        )
        frame["source"] = path.stem
        frames.append(frame)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": fingerprint(path),
                "cleaning": cleaning,
            }
        )
    combined, combined_report = combine_sources(frames)
    print(f"Combined usable URLs: {len(combined):,}; {combined_report}", flush=True)
    splits = domain_partition(combined, args.seed)
    manifest = {
        "seed": args.seed,
        "grouping": "registered domain including private suffixes",
        "split_method": "SHA-256(seed:domain), 80/10/10 buckets",
        "sources": sources,
        "phreshphish_provenance": json.loads(
            (args.external_test.parent / "download_manifest.json").read_text()
        ),
        **combined_report,
        "split_sha256": {},
    }
    matrices, labels = {}, {}
    for name, frame in splits.items():
        path = args.data_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["split_sha256"][name] = fingerprint(path)
        print(f"Extracting {name}: {len(frame):,} rows...", flush=True)
        matrices[name] = url_matrix(frame["url"])
        labels[name] = frame["is_phishing"].to_numpy()
        np.save(args.data_dir / f"{name}_features.npy", matrices[name])
    (args.data_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    datasets = {
        name: xgb.DMatrix(
            matrix, label=labels[name], feature_names=list(TRAINING_FEATURES)
        )
        for name, matrix in matrices.items()
    }
    best_model, best_f1, search, selected_params = None, -1.0, [], None
    for depth, child_weight in ((3, 10), (5, 5), (7, 5)):
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cpu",
            "eval_metric": "logloss",
            "eta": 0.05,
            "max_depth": depth,
            "min_child_weight": child_weight,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 5.0,
            "alpha": 0.1,
            "nthread": args.threads,
            "seed": args.seed,
        }
        print(
            f"Fitting depth={depth} on local CPU, {args.threads} threads...", flush=True
        )
        model = xgb.train(
            params,
            datasets["train"],
            num_boost_round=args.trees,
            evals=[(datasets["validation"], "validation")],
            early_stopping_rounds=40,
            verbose_eval=50,
        )
        # Persist only the trees selected by validation early stopping.
        selected = model[: model.best_iteration + 1]
        scores = selected.predict(datasets["validation"])
        threshold = choose_threshold(labels["validation"], scores)
        metrics = evaluate_probabilities(labels["validation"], scores, threshold)
        search.append(
            {
                "params": params,
                "trees": selected.num_boosted_rounds(),
                "validation_metrics": metrics,
            }
        )
        if metrics["f1"] > best_f1:
            best_model, best_f1, selected_params = selected, metrics["f1"], params
    assert best_model is not None
    validation_scores = best_model.predict(datasets["validation"])
    threshold = choose_threshold(labels["validation"], validation_scores)
    metrics = evaluate_probabilities(
        labels["test"], best_model.predict(datasets["test"]), threshold
    )
    gains = best_model.get_score(importance_type="gain")
    total_gain = sum(gains.values())
    by_source = {}
    scores = best_model.predict(datasets["test"])
    for source in splits["test"]["source"].unique():
        mask = splits["test"]["source"].eq(source).to_numpy()
        if set(labels["test"][mask]) == {0, 1}:
            by_source[source] = evaluate_probabilities(
                labels["test"][mask], scores[mask], threshold
            )
    model_path = args.artifacts_dir / "xgboost_model.ubj"
    model_path.write_bytes(best_model.save_raw(raw_format="ubj"))
    metadata = {
        "model_version": "0.1.0-native-xgboost-phreshphish",
        "model_type": "Shared URL lexical features + native XGBoost Booster",
        "created_at": datetime.now(UTC).isoformat(),
        "threshold": threshold,
        "parameters": selected_params,
        "features": list(TRAINING_FEATURES),
        "dropped_features": ["slash_count"],
        "best_iteration": best_model.num_boosted_rounds() - 1,
        "selection_method": (
            "validation F1, minimum 90% recall; ties prefer fewer alerts"
        ),
        "dataset_rows": {name: len(frame) for name, frame in splits.items()},
        "data_dir": str(args.data_dir.resolve()),
        "split_sha256": manifest["split_sha256"],
        "split_manifest": manifest,
        "validation_search": search,
        "validation_metrics": evaluate_probabilities(
            labels["validation"], validation_scores, threshold
        ),
        "test_metrics": metrics,
        "test_metrics_by_source": by_source,
        "feature_importances": [
            {"feature": name, "gain_importance": gains.get(name, 0) / total_gain}
            for name in TRAINING_FEATURES
        ],
        "runtime": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "activation_eligible": False,
        "artifact_format": "Native UBJ; not the scikit-learn joblib API package",
        "active_model_hashes_before": active_hashes,
    }
    metadata_path = args.artifacts_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("DOMAIN-DISJOINT MIXED HOLDOUT:", json.dumps(metrics), flush=True)
    # Independent publisher test is never used to choose model or threshold.
    external, cleaning = clean_url_data(
        pd.read_csv(args.external_test),
        url_column="url",
        label_column="label",
        label_mapping={"benign": 0, "phish": 1},
    )
    known_domains = set(combined["url"].map(get_registered_domain))
    overlap = external["url"].map(get_registered_domain).isin(known_domains)
    external = external.loc[~overlap].copy()
    if external.empty or set(external["is_phishing"]) != {0, 1}:
        raise ValueError("No two-class publisher test remains after domain exclusion")
    external_features = url_matrix(external["url"])
    external_scores = best_model.predict(
        xgb.DMatrix(external_features, feature_names=list(TRAINING_FEATURES))
    )
    external_report = {
        "dataset": str(args.external_test.resolve()),
        "dataset_sha256": fingerprint(args.external_test),
        "cleaning": cleaning,
        "known_domain_rows_excluded": int(overlap.sum()),
        "rows_evaluated": len(external),
        "class_counts": external["is_phishing"].value_counts().to_dict(),
        "metrics": evaluate_probabilities(
            external["is_phishing"].to_numpy(), external_scores, threshold
        ),
        "role": (
            "Publisher-held-out test; all development and internal-test domains purged"
        ),
        "independent_collection_verified": False,
    }
    previous_native = (
        PROJECT_ROOT / "artifacts/experiments/xgboost_mixed/xgboost_model.ubj"
    )
    if previous_native.exists():
        previous = xgb.Booster()
        previous.load_model(bytearray(previous_native.read_bytes()))
        previous_metadata = json.loads(
            (PROJECT_ROOT / "artifacts/model_metadata.json").read_text()
        )
        if list(TRAINING_FEATURES) != previous_metadata["features"]:
            raise ValueError("Previous model feature order differs")
        if previous.num_features() != len(TRAINING_FEATURES):
            raise ValueError("Previous model feature count differs")
        previous_scores = previous.predict(
            xgb.DMatrix(external_features),
            iteration_range=(0, previous_metadata["best_iteration"] + 1),
        )
        external_report["previous_model_same_rows"] = evaluate_probabilities(
            external["is_phishing"].to_numpy(),
            previous_scores,
            previous_metadata["threshold"],
        )
        external_report["previous_native_sha256"] = fingerprint(previous_native)
    (args.artifacts_dir / "phreshphish_external_evaluation.json").write_text(
        json.dumps(external_report, indent=2) + "\n"
    )
    print("PURGED PUBLISHER TEST:", json.dumps(external_report), flush=True)
    after = {
        name: fingerprint(PROJECT_ROOT / "artifacts" / name) for name in active_hashes
    }
    if active_hashes != after:
        raise ValueError("Active model files changed during the experiment")
    completed = {
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "model_sha256": fingerprint(model_path),
        "active_model_unchanged": True,
        "training_device": "local Windows CPU",
        "threads": args.threads,
    }
    (args.artifacts_dir / "run_status.json").write_text(
        json.dumps(completed, indent=2) + "\n"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    (args.artifacts_dir / "requirements-local.txt").write_text(freeze.stdout)
    print(f"Saved native experiment: {args.artifacts_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
