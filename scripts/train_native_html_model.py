"""Train paired URL, HTML-derived, and combined XGBoost models locally."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from phishguard.config import PROJECT_ROOT
from phishguard.html_features import MAX_HTML_BYTES
from phishguard.native import (
    choose_threshold,
    evaluate_probabilities,
    ranking_curve,
)
from phishguard.native_html import (
    VARIANT_FEATURES,
    content_partition,
    fingerprint,
    prepare_snapshot_cache,
    purge_publisher_test,
)


def low_fpr_threshold(labels, scores, maximum_fpr: float = 0.01) -> float:
    """Maximize validation recall under an FPR budget; ties prefer fewer alerts."""
    if not 0 <= maximum_fpr <= 1:
        raise ValueError("FPR budget must be in [0, 1]")
    thresholds, _, recalls, fprs = ranking_curve(labels, scores)
    eligible = np.flatnonzero(fprs <= maximum_fpr)
    if not len(eligible):
        # Scores are float32; a float64 epsilon can round back to 1 on comparison.
        return float(np.nextafter(np.float32(1), np.float32(np.inf)))
    return float(thresholds[eligible[np.argmax(recalls[eligible])]])


def protected_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "artifacts/phishguard_model.joblib",
        PROJECT_ROOT / "artifacts/model_metadata.json",
        PROJECT_ROOT
        / "artifacts/experiments/phreshphish_full_download_20261009"
        / "xgboost_model.ubj",
        PROJECT_ROOT
        / "artifacts/experiments/phreshphish_full_download_20261009"
        / "model_metadata.json",
    ]
    return {str(path): fingerprint(path) for path in paths if path.exists()}


def _cohort_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "class_counts": frame["is_phishing"].value_counts().to_dict(),
        "html_status_by_class": {
            str(label): {
                flag: int(part[flag].sum())
                for flag in ("html_available", "html_truncated", "html_parse_error")
            }
            for label, part in frame.groupby("is_phishing")
        },
        "date_min": min((date for date in frame["date"] if date), default=None),
        "date_max": max((date for date in frame["date"] if date), default=None),
    }


def train_comparison(
    development: pd.DataFrame,
    publisher_test: pd.DataFrame,
    output_dir: Path,
    *,
    provenance: dict,
    trees: int = 800,
    threads: int = 4,
    device: str = "cpu",
    seed: int = 42,
    depths: tuple[int, ...] = (3, 5, 7),
) -> dict:
    if trees < 1 or not 1 <= threads <= 16 or device not in {"cpu", "cuda"}:
        raise ValueError("Require positive trees, 1–16 threads, and cpu/cuda device")
    if not depths or any(not 1 <= depth <= 12 for depth in depths):
        raise ValueError("Depths must be between 1 and 12")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new empty experiment directory")
    before = protected_hashes()
    if any(Path(path).is_relative_to(output_dir) for path in before):
        raise ValueError("Experiment output contains a protected model")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    splits = content_partition(development, seed)
    external, exclusions = purge_publisher_test(publisher_test, development)
    manifest = {
        "seed": seed,
        "split_method": "80/10/10 SHA-256 buckets of domain/exact-HTML components",
        "domain_and_exact_html_disjoint": True,
        "publisher_test_exclusions": exclusions,
        "cohorts": {
            **{name: _cohort_summary(frame) for name, frame in splits.items()},
            "publisher_test": _cohort_summary(external),
        },
        "provenance": provenance,
        "split_sha256": {},
        "limitations": [
            "Near-duplicate HTML templates are not decontaminated",
            "Internal splits are domain/HTML grouped, not temporal",
            "Publisher-test dates are reported, not assumed to be strictly later",
            "Publisher URL-only results have previously been inspected",
            "HTML-derived relational features use the page URL as context",
            "Brand reference list is small and is not an allowlist",
        ],
    }
    all_frames = {**splits, "publisher_test": external}
    for name, frame in all_frames.items():
        path = output_dir / f"{name}_rows.parquet"
        with path.open("wb") as stream:
            frame.to_parquet(stream, index=False)
        manifest["split_sha256"][name] = fingerprint(path)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    comparison = {}
    # Fixed, equal search budget. No model or threshold is chosen on test results.
    for variant, names in VARIANT_FEATURES.items():
        print(
            f"Training {variant}: {len(names)} features on local {device}", flush=True
        )
        fit_started = time.monotonic()
        datasets = {
            name: xgb.DMatrix(
                frame.loc[:, list(names)].to_numpy(dtype=np.float32),
                label=frame["is_phishing"].to_numpy(),
                feature_names=list(names),
            )
            for name, frame in all_frames.items()
        }
        best, best_f1, selected_params, search = None, -1.0, None, []
        for depth in depths:
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": device,
                "nthread": threads,
                "eval_metric": "logloss",
                "eta": 0.05,
                "max_depth": depth,
                "min_child_weight": 10 if depth == 3 else 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambda": 5.0,
                "alpha": 0.1,
                "seed": seed,
            }
            print(f"  Candidate depth={depth}", flush=True)
            model = xgb.train(
                params,
                datasets["train"],
                num_boost_round=trees,
                evals=[(datasets["validation"], "validation")],
                early_stopping_rounds=40,
                verbose_eval=50,
            )
            actual_device = json.loads(model.save_config())["learner"]["generic_param"][
                "device"
            ]
            if device == "cuda" and not actual_device.startswith("cuda"):
                raise RuntimeError("CUDA was requested but XGBoost fell back to CPU")
            selected = model[: model.best_iteration + 1]
            scores = selected.predict(datasets["validation"])
            labels = splits["validation"]["is_phishing"].to_numpy()
            threshold = choose_threshold(labels, scores)
            metrics = evaluate_probabilities(labels, scores, threshold)
            search.append(
                {
                    "parameters": params,
                    "trees": selected.num_boosted_rounds(),
                    "validation_metrics": metrics,
                }
            )
            if metrics["f1"] > best_f1:
                best, best_f1, selected_params = selected, metrics["f1"], params
        assert best is not None
        validation_labels = splits["validation"]["is_phishing"].to_numpy()
        validation_scores = best.predict(datasets["validation"])
        threshold = choose_threshold(validation_labels, validation_scores)
        conservative = low_fpr_threshold(validation_labels, validation_scores)
        evaluations, conservative_evaluations = {}, {}
        for name, frame in all_frames.items():
            if name == "train":
                continue
            scores = best.predict(datasets[name])
            labels = frame["is_phishing"].to_numpy()
            evaluations[name] = evaluate_probabilities(labels, scores, threshold)
            conservative_evaluations[name] = evaluate_probabilities(
                labels, scores, conservative
            )
            pd.DataFrame(
                {
                    "is_phishing": labels,
                    "score": scores,
                    "predicted": scores >= threshold,
                }
            ).to_csv(output_dir / f"{variant}_{name}_predictions.csv", index=False)
        artifact_dir = output_dir / variant
        artifact_dir.mkdir()
        model_path = artifact_dir / "xgboost_model.ubj"
        model_path.write_bytes(best.save_raw(raw_format="ubj"))
        restored = xgb.Booster()
        restored.load_model(bytearray(model_path.read_bytes()))
        restored.set_param({"device": device, "nthread": threads})
        np.testing.assert_array_equal(
            best.predict(datasets["validation"]),
            restored.predict(datasets["validation"]),
        )
        gains = best.get_score(importance_type="gain")
        total = sum(gains.values())
        metadata = {
            "model_version": "0.1.0-native-html-experiment",
            "model_type": f"Native XGBoost: {variant}",
            "created_at": datetime.now(UTC).isoformat(),
            "features": list(names),
            "threshold": threshold,
            "best_iteration": best.num_boosted_rounds() - 1,
            "parameters": selected_params,
            "max_html_bytes": provenance.get("max_html_bytes", MAX_HTML_BYTES),
            "validation_search": search,
            "selection_method": "Validation F1 with >=90% validation recall",
            "validation_metrics": evaluations["validation"],
            "test_metrics": evaluations["test"],
            "publisher_test_metrics": evaluations["publisher_test"],
            "validation_1_percent_fpr_operating_point": {
                "threshold": conservative,
                "target_validation_fpr": 0.01,
                "metrics": conservative_evaluations,
                "note": "Test FPR is measured; 1% is not guaranteed on unseen data",
            },
            "dataset_rows": {name: len(frame) for name, frame in all_frames.items()},
            "split_manifest_sha256": fingerprint(output_dir / "split_manifest.json"),
            "feature_importances": [
                {
                    "feature": name,
                    "gain_importance": gains.get(name, 0) / total if total else 0,
                }
                for name in names
            ],
            "artifact_format": "Native UBJ experiment; not an API joblib package",
            "activation_eligible": False,
            "runtime": {
                "python": platform.python_version(),
                "xgboost": xgb.__version__,
            },
            "training_and_evaluation_seconds": time.monotonic() - fit_started,
            "model_sha256": fingerprint(model_path),
            "save_reload_predictions_identical": True,
        }
        (artifact_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        comparison[variant] = {
            "features": len(names),
            "trees": best.num_boosted_rounds(),
            "metrics": evaluations,
            "validation_1_percent_fpr_metrics": conservative_evaluations,
        }
        print(f"  Publisher test: {evaluations['publisher_test']}", flush=True)
    if before != protected_hashes():
        raise RuntimeError("A protected model artifact changed during training")
    report = {
        "status": "completed",
        "paired_comparison": True,
        "models": comparison,
        "active_and_previous_url_models_unchanged": True,
        "elapsed_seconds_excluding_extraction": time.monotonic() - started,
        "device": device,
        "limitations": manifest["limitations"],
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "run_status.json").write_text(
        json.dumps(
            {key: value for key, value in report.items() if key != "models"}, indent=2
        )
        + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/raw/phreshphish")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/processed/html_features_v1")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=16)
    parser.add_argument("--max-html-bytes", type=int, default=MAX_HTML_BYTES)
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if args.trees < 1 or not 1 <= args.threads <= 16:
        parser.error("Require positive trees and 1–16 threads")
    if (
        not 1 <= args.batch_rows <= 256
        or not 1 <= args.max_html_bytes <= MAX_HTML_BYTES
    ):
        parser.error("Require 1–256 batch rows and 1–1,000,000 HTML bytes")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.extract_only
    ):
        parser.error("Choose a new empty experiment directory")
    started = time.monotonic()
    frames, provenance = prepare_snapshot_cache(
        args.dataset_dir,
        args.cache_dir,
        batch_rows=args.batch_rows,
        max_html_bytes=args.max_html_bytes,
    )
    provenance["extraction_and_verification_seconds"] = time.monotonic() - started
    if args.extract_only:
        print("Feature extraction complete. Training not started.", flush=True)
        return
    train_comparison(
        frames["train"],
        frames["test"],
        args.output_dir,
        provenance=provenance,
        trees=args.trees,
        threads=args.threads,
        device=args.device,
        seed=args.seed,
    )
    print(f"Saved paired HTML experiment: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
