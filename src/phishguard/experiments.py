"""Leakage-aware, reproducible experiments, separate from the active model."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from phishguard.domains import get_registered_domain
from phishguard.features import URLFeatureExtractor
from phishguard.modeling import TRAINING_FEATURES, build_model, evaluate_probabilities
from phishguard.split import validate_training_splits

URL_VARIANTS = ("xgboost", "xgboost_no_https", "char_logistic")
# Deliberately small, explicit search: six hypotheses, not a large hidden HPO run.
REGULARIZED = {
    "max_depth": 3,
    "min_child_weight": 15,
    "gamma": 0.5,
    "reg_alpha": 1.0,
    "reg_lambda": 15.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
}
SOURCE_CANDIDATES = (
    {"name": "baseline", "feature_policy": "standard", "weighted": False, "params": {}},
    {
        "name": "source_balanced",
        "feature_policy": "standard",
        "weighted": True,
        "params": {},
    },
    {
        "name": "regularized",
        "feature_policy": "standard",
        "weighted": True,
        "params": REGULARIZED,
    },
    {
        "name": "protocol_invariant",
        "feature_policy": "protocol_invariant",
        "weighted": True,
        "params": {},
    },
    {
        "name": "protocol_regularized",
        "feature_policy": "protocol_invariant",
        "weighted": True,
        "params": REGULARIZED,
    },
    {
        "name": "reduced_shape",
        "feature_policy": "reduced_shape",
        "weighted": True,
        "params": REGULARIZED,
    },
)
METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "false_positive_rate",
    "false_alarms_per_1000_benign",
)


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint selected records in their actual training order."""
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def choose_low_fpr_threshold(
    labels: np.ndarray, probabilities: np.ndarray, maximum_fpr: float = 0.01
) -> float:
    """Maximize validation recall within an empirical false-alarm budget.

    The budget is NOT a guarantee on new data. A threshold above 1 is an
    intentional abstaining operating point when no positive is affordable.
    """
    if not 0 <= maximum_fpr < 1:
        raise ValueError("maximum_fpr must be at least 0 and less than 1")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold validation requires both classes")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Expected finite scores between 0 and 1")
    fpr, recall, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= maximum_fpr)
    # roc_curve thresholds descend: ties favor fewer alerts (higher threshold).
    best = eligible[np.argmax(recall[eligible])]
    threshold = float(thresholds[best])
    return threshold if np.isfinite(threshold) else float(np.nextafter(1.0, 2.0))


def grouped_fit_validation_indices(
    frame: pd.DataFrame, seed: int, validation_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Find a domain-disjoint inner validation split containing both classes."""
    groups = frame["url"].map(get_registered_domain)
    splitter = GroupShuffleSplit(
        n_splits=20, test_size=validation_size, random_state=seed
    )
    for fit, validation in splitter.split(frame, groups=groups):
        if all(
            set(frame.iloc[indices]["is_phishing"].unique()) == {0, 1}
            for indices in (fit, validation)
        ):
            return fit, validation
    raise ValueError("Could not form two-class domain-disjoint inner splits")


def build_variant(
    variant: str,
    seed: int = 42,
    trees: int = 400,
    params: dict[str, Any] | None = None,
    feature_policy: str = "standard",
) -> Any:
    if variant == "char_logistic":
        return build_model(algorithm="char_logistic", random_state=seed)
    if variant == "hybrid_xgboost":
        from phishguard.content import HybridFeatureExtractor

        model = build_model(
            random_state=seed, n_estimators=trees, early_stopping_rounds=30
        )
        model.steps[0] = ("features", HybridFeatureExtractor())
        return model
    if variant not in URL_VARIANTS:
        raise ValueError(f"Unknown experiment variant: {variant}")
    features = TRAINING_FEATURES
    if variant == "xgboost_no_https":
        features = tuple(name for name in features if name != "uses_https")
    if feature_policy not in {"standard", "protocol_invariant", "reduced_shape"}:
        raise ValueError(f"Unknown feature policy: {feature_policy}")
    if feature_policy != "standard":
        dropped = {"uses_https"}
        if feature_policy == "reduced_shape":
            dropped.update(
                {"url_length", "hostname_length", "path_length", "query_length"}
            )
        features = tuple(name for name in features if name not in dropped)
    classifier_params = {"n_estimators": trees, "early_stopping_rounds": 30}
    classifier_params.update(params or {})
    model = build_model(
        random_state=seed,
        feature_names=features,
        **classifier_params,
    )
    if feature_policy != "standard":
        model.steps[0] = (
            "features",
            URLFeatureExtractor(features, normalize_protocol=True),
        )
    return model


def model_input(frame: pd.DataFrame, variant: str) -> Any:
    return frame[["url", "html"]] if variant == "hybrid_xgboost" else frame["url"]


def fit_variant(
    variant: str,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int = 42,
    trees: int = 400,
    params: dict[str, Any] | None = None,
    feature_policy: str = "standard",
    balance_sources: bool = False,
) -> Any:
    """Fit learned transforms only on fit rows, never on outer evaluation rows."""
    model = build_variant(variant, seed, trees, params, feature_policy)
    fit_input = model_input(fit, variant)
    validation_input = model_input(validation, variant)
    weights = source_class_weights(fit) if balance_sources else None
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if variant == "char_logistic":
            options = (
                {"classifier__sample_weight": weights} if weights is not None else {}
            )
            model.fit(fit_input, fit["is_phishing"].to_numpy(), **options)
        else:
            extractor = model.named_steps["features"]
            fit_features = extractor.fit_transform(fit_input)
            validation_features = extractor.transform(validation_input)
            model.named_steps["classifier"].fit(
                fit_features,
                fit["is_phishing"].to_numpy(),
                eval_set=[(validation_features, validation["is_phishing"].to_numpy())],
                verbose=False,
                sample_weight=weights,
            )
    return model


def source_class_weights(fit: pd.DataFrame) -> np.ndarray:
    """Give each observed source/class pair equal total weight, using fit only."""
    counts = fit.groupby(["source", "is_phishing"])["url"].transform("size")
    pairs = fit[["source", "is_phishing"]].drop_duplicates().shape[0]
    return len(fit) / (pairs * counts.to_numpy(dtype=float))


def source_validation_threshold(
    validation: pd.DataFrame,
    scores: np.ndarray,
    maximum_fpr: float,
) -> float:
    """Meet the empirical budget in each two-class inner validation source."""
    thresholds = [
        choose_low_fpr_threshold(
            validation["is_phishing"].to_numpy(), scores, maximum_fpr
        )
    ]
    for source in validation["source"].unique():
        mask = validation["source"].eq(source).to_numpy()
        if set(validation.loc[mask, "is_phishing"].unique()) == {0, 1}:
            thresholds.append(
                choose_low_fpr_threshold(
                    validation.loc[mask, "is_phishing"].to_numpy(),
                    scores[mask],
                    maximum_fpr,
                )
            )
    return max(thresholds)


def metrics_with_sources(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    metrics = evaluate_probabilities(
        frame["is_phishing"].to_numpy(), probabilities, threshold
    )
    metrics["false_alarms_per_1000_benign"] = metrics["false_positive_rate"] * 1000
    sources = {}
    if "source" in frame:
        for source in sorted(frame["source"].unique()):
            mask = frame["source"].eq(source).to_numpy()
            subset = frame.loc[mask]
            sources[str(source)] = {
                "rows": len(subset),
                "class_counts": {
                    str(k): int(v)
                    for k, v in subset["is_phishing"].value_counts().items()
                },
                "metrics": (
                    evaluate_probabilities(
                        subset["is_phishing"].to_numpy(), probabilities[mask], threshold
                    )
                    if set(subset["is_phishing"].unique()) == {0, 1}
                    else None
                ),
            }
    return {"metrics": metrics, "by_source": sources}


def summarize_folds(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold SD describes variability, not a statistical confidence interval."""
    return {
        name: {
            "mean": float(np.mean([r["metrics"][name] for r in records])),
            "std": float(np.std([r["metrics"][name] for r in records], ddof=1)),
        }
        for name in METRIC_NAMES
    }


def save_package(model: Any, destination: Path, metadata: dict[str, Any]) -> None:
    """One packaging implementation shared by CV and source tuning."""
    destination.mkdir(parents=True, exist_ok=True)
    parameters = model.named_steps["classifier"].get_params()
    names = (
        ("C", "solver", "class_weight", "max_iter", "random_state")
        if "tfidf" in model.named_steps
        else (
            "n_estimators",
            "max_depth",
            "min_child_weight",
            "learning_rate",
            "gamma",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "n_jobs",
            "random_state",
            "early_stopping_rounds",
            "tree_method",
            "eval_metric",
        )
    )
    metadata["classifier_parameters"] = {name: parameters[name] for name in names}
    if "tfidf" in model.named_steps:
        metadata["representation"] = {
            "scheme_removed": True,
            "fragment_removed": True,
            "analyzer": "char",
            "ngram_range": [3, 5],
            "max_features": 100000,
            "learned_vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        }
    else:
        extractor = model.named_steps["features"]
        metadata["features"] = list(extractor.get_feature_names_out())
        metadata["normalize_protocol"] = getattr(extractor, "normalize_protocol", False)
        metadata["best_iteration"] = int(model.named_steps["classifier"].best_iteration)
    joblib.dump(model, destination / "phishguard_model.joblib")
    (destination / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def run_benchmark(
    development: pd.DataFrame,
    output_dir: Path,
    variants: tuple[str, ...] = URL_VARIANTS,
    folds: int = 5,
    seed: int = 42,
    maximum_fpr: float = 0.01,
    trees: int = 400,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Nested holdout threshold selection within domain-grouped outer CV.

    Hyperparameters are fixed, not searched. The inner validation split chooses
    early stopping and the threshold; outer scores must never choose those.
    Final packages retain an inner validation set instead of refitting on it.
    """
    if folds < 2 or trees < 1 or not variants or len(set(variants)) != len(variants):
        raise ValueError("Require >=2 folds, positive trees, and unique variants")
    if not 0 <= maximum_fpr < 1:
        raise ValueError("maximum_fpr must be at least 0 and less than 1")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new empty experiment directory; never overwrite a run")
    development = development.reset_index(drop=True).copy()
    if development.empty or development["url"].duplicated().any():
        raise ValueError("Development data must be nonempty with unique URLs")
    if development[["url", "is_phishing"]].isna().any().any():
        raise ValueError("Missing development URLs or labels")
    if set(development["is_phishing"].unique()) != {0, 1}:
        raise ValueError("Development labels must contain both binary classes")
    groups = development["url"].map(get_registered_domain)
    if groups.nunique() < folds:
        raise ValueError("Not enough distinct domains for requested folds")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    partitions = []
    for fold, (outer_train, outer_test) in enumerate(
        splitter.split(development, development["is_phishing"], groups), start=1
    ):
        outer = development.iloc[outer_train]
        fit_idx, validation_idx = grouped_fit_validation_indices(outer, seed + fold)
        fit_idx, validation_idx = outer_train[fit_idx], outer_train[validation_idx]
        validate_training_splits(
            {
                "fit": development.iloc[fit_idx],
                "validation": development.iloc[validation_idx],
                "outer_test": development.iloc[outer_test],
            }
        )
        partitions.append((fit_idx, validation_idx, outer_test))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "seed": seed,
        "fold_count": folds,
        "development_rows": len(development),
        "development_domains": int(groups.nunique()),
        "development_sha256": frame_fingerprint(development),
        "class_counts": {
            str(k): int(v) for k, v in development["is_phishing"].value_counts().items()
        },
        "provenance": provenance or {},
        "protocol": {
            "outer_cv": (
                "StratifiedGroupKFold by registered domain including private suffixes"
            ),
            "inner_validation": (
                "20% of outer-training domains; early stopping + threshold"
            ),
            "maximum_inner_validation_fpr": maximum_fpr,
            "comparison_metric": "mean outer-fold average_precision (PR-AUC summary)",
            "hyperparameters": "fixed; no outer-fold tuning",
            "score_caution": "uncalibrated model scores, not verified probabilities",
            "final_test": "not used by this benchmark",
            "uncertainty": "fold standard deviation; not a confidence interval",
        },
        "runtime": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        "variants": {},
    }
    summary_path = output_dir / "benchmark.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for variant in variants:
        records = []
        predictions = []
        for fold, (fit_idx, val_idx, test_idx) in enumerate(partitions, start=1):
            start = time.perf_counter()
            fit, validation, test = (
                development.iloc[idx] for idx in (fit_idx, val_idx, test_idx)
            )
            model = fit_variant(variant, fit, validation, seed + fold, trees)
            val_scores = model.predict_proba(model_input(validation, variant))[:, 1]
            threshold = choose_low_fpr_threshold(
                validation["is_phishing"].to_numpy(), val_scores, maximum_fpr
            )
            scores = model.predict_proba(model_input(test, variant))[:, 1]
            result = metrics_with_sources(test, scores, threshold)
            record = {
                "fold": fold,
                "threshold": threshold,
                "rows": {
                    "fit": len(fit),
                    "validation": len(validation),
                    "test": len(test),
                },
                "role_sha256": {
                    name: frame_fingerprint(frame)
                    for name, frame in (
                        ("fit", fit),
                        ("validation", validation),
                        ("test", test),
                    )
                },
                "seconds": time.perf_counter() - start,
                "validation_metrics": evaluate_probabilities(
                    validation["is_phishing"].to_numpy(), val_scores, threshold
                ),
                "fixed_0_5_metrics": evaluate_probabilities(
                    test["is_phishing"].to_numpy(), scores, 0.5
                ),
                **result,
            }
            records.append(record)
            prediction = test[["url", "is_phishing"]].copy()
            prediction["fold"] = fold
            prediction["score"] = scores
            prediction["threshold"] = threshold
            predictions.append(prediction)
            print(
                f"{variant} fold {fold}/{folds}: "
                f"AP={result['metrics']['average_precision']:.4f}, "
                f"recall={result['metrics']['recall']:.4f}, "
                f"FPR={result['metrics']['false_positive_rate']:.4f} "
                f"({record['seconds']:.1f}s)",
                flush=True,
            )
        destination = output_dir / variant
        destination.mkdir()
        pd.concat(predictions).to_csv(
            destination / "out_of_fold_predictions.csv", index=False
        )
        fit_idx, val_idx = grouped_fit_validation_indices(development, seed + 1000)
        fit, validation = development.iloc[fit_idx], development.iloc[val_idx]
        validate_training_splits({"fit": fit, "validation": validation})
        model = fit_variant(variant, fit, validation, seed, trees)
        val_scores = model.predict_proba(model_input(validation, variant))[:, 1]
        threshold = choose_low_fpr_threshold(
            validation["is_phishing"].to_numpy(), val_scores, maximum_fpr
        )
        metadata = {
            "model_version": f"0.2.0-experimental-{variant}",
            "model_type": variant,
            "label_convention": {"0": "legitimate", "1": "phishing"},
            "threshold": threshold,
            "created_at": datetime.now(UTC).isoformat(),
            "selection_method": "fixed parameters; inner-domain validation FPR budget",
            "maximum_validation_fpr": maximum_fpr,
            "data_dir": (provenance or {}).get("data_dir"),
            "split_sha256": (provenance or {}).get("split_sha256"),
            "development_sha256": summary["development_sha256"],
            "fit_sha256": frame_fingerprint(fit),
            "validation_sha256": frame_fingerprint(validation),
            "dataset_rows": {"fit": len(fit), "validation": len(validation)},
            "validation_metrics": evaluate_probabilities(
                validation["is_phishing"].to_numpy(), val_scores, threshold
            ),
            "cross_validation_summary": summarize_folds(records),
            "score_caution": "uncalibrated; advisory research experiment only",
            "activation_eligible": False,
        }
        save_package(model, destination, metadata)
        summary["variants"][variant] = {
            "folds": records,
            "summary": summarize_folds(records),
            "package": str(destination),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["highest_cv_average_precision"] = max(
        variants,
        key=lambda name: summary["variants"][name]["summary"]["average_precision"][
            "mean"
        ],
    )
    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def source_partitions(
    development: pd.DataFrame,
    seed: int,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]]:
    """Hold out whole sources; purge their domains before inner splitting."""
    if "source" not in development or development["source"].nunique() < 2:
        raise ValueError("Source tuning requires at least two identified sources")
    if development["url"].duplicated().any() or development["source"].isna().any():
        raise ValueError("Source data must have unique URLs and nonmissing sources")
    domains = development["url"].map(get_registered_domain)
    partitions = []
    for index, source in enumerate(sorted(development["source"].unique())):
        test_mask = development["source"].eq(source)
        test = development.loc[test_mask]
        if set(test["is_phishing"].unique()) != {0, 1}:
            raise ValueError(f"Source {source} must contain both binary classes")
        overlap = ~test_mask & domains.isin(set(domains.loc[test_mask]))
        outer_train = development.loc[~test_mask & ~overlap].reset_index(drop=True)
        if outer_train.empty or set(outer_train["is_phishing"].unique()) != {0, 1}:
            raise ValueError(
                f"Insufficient training data after purging domains for {source}"
            )
        fit_idx, val_idx = grouped_fit_validation_indices(outer_train, seed + index)
        fit, validation = outer_train.iloc[fit_idx], outer_train.iloc[val_idx]
        validate_training_splits(
            {"fit": fit, "validation": validation, "held_source": test}
        )
        partitions.append((str(source), fit, validation, test, int(overlap.sum())))
    return partitions


def run_source_tuning(
    development: pd.DataFrame,
    output_dir: Path,
    reserved_references: pd.DataFrame,
    seed: int = 42,
    maximum_fpr: float = 0.01,
    trees: int = 400,
    provenance: dict[str, Any] | None = None,
    candidates: tuple[dict[str, Any], ...] = SOURCE_CANDIDATES,
) -> dict[str, Any]:
    """Source-held-out DEVELOPMENT search; final independent test is separate.

    Save only the selected XGBoost and the character comparison, not every trial.
    Scores used to choose a candidate are selection-biased development evidence.
    """
    if trees < 1 or not 0 <= maximum_fpr < 1 or not candidates:
        raise ValueError("Require positive trees, candidates, and 0 <= FPR budget < 1")
    if len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new empty tuning directory; never overwrite a run")
    development = development.reset_index(drop=True)
    partitions = source_partitions(development, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "seed": seed,
        "development_rows": len(development),
        "development_sha256": frame_fingerprint(development),
        "provenance": provenance or {},
        "source_counts": development.groupby("source")["is_phishing"]
        .agg(["size", "sum"])
        .rename(columns={"size": "rows", "sum": "phishing_rows"})
        .to_dict("index"),
        "protocol": {
            "selection": (
                "highest macro held-source average precision; worst FPR breaks ties"
            ),
            "threshold": "inner validation only; configurable per-source FPR budget",
            "maximum_inner_validation_fpr": maximum_fpr,
            "domain_purge": "exclude ALL held-source registered domains before fitting",
            "weighting": "equal observed source/class total weight; fit labels only",
            "caution": (
                "development search, NOT unbiased final evaluation; uncalibrated scores"
            ),
            "final_test": "fresh independent dataset required; active model unchanged",
        },
        "trials": {},
    }
    path = output_dir / "tuning.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    comparison = {
        "name": "char_logistic",
        "feature_policy": "standard",
        "weighted": True,
        "params": {},
    }
    for candidate in (*candidates, comparison):
        records = []
        variant = "char_logistic" if candidate["name"] == "char_logistic" else "xgboost"
        for fold, (source, fit, validation, test, excluded) in enumerate(partitions):
            start = time.perf_counter()
            model = fit_variant(
                variant,
                fit,
                validation,
                seed + fold,
                trees,
                params=candidate["params"],
                feature_policy=candidate["feature_policy"],
                balance_sources=candidate["weighted"],
            )
            val_scores = model.predict_proba(validation["url"])[:, 1]
            threshold = source_validation_threshold(validation, val_scores, maximum_fpr)
            scores = model.predict_proba(test["url"])[:, 1]
            metrics = evaluate_probabilities(
                test["is_phishing"].to_numpy(), scores, threshold
            )
            records.append(
                {
                    "held_source": source,
                    "metrics": metrics,
                    "validation": metrics_with_sources(
                        validation, val_scores, threshold
                    ),
                    "shared_domain_training_rows_excluded": excluded,
                    "rows": {
                        "fit": len(fit),
                        "validation": len(validation),
                        "held_source": len(test),
                    },
                    "role_sha256": {
                        name: frame_fingerprint(frame)
                        for name, frame in (
                            ("fit", fit),
                            ("validation", validation),
                            ("held_source", test),
                        )
                    },
                    "majority_accuracy": float(
                        test["is_phishing"].value_counts(normalize=True).max()
                    ),
                    "chance_average_precision": float(test["is_phishing"].mean()),
                    "seconds": time.perf_counter() - start,
                }
            )
            print(
                f"{candidate['name']} -> {source}: "
                f"AP={metrics['average_precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"FPR={metrics['false_positive_rate']:.4f}",
                flush=True,
            )
        report["trials"][candidate["name"]] = {
            "configuration": candidate,
            "held_sources": records,
            "macro_average_precision": float(
                np.mean([r["metrics"]["average_precision"] for r in records])
            ),
            "macro_recall": float(np.mean([r["metrics"]["recall"] for r in records])),
            "worst_source_fpr": max(
                r["metrics"]["false_positive_rate"] for r in records
            ),
        }
        path.write_text(json.dumps(report, indent=2) + "\n")
    selected = max(
        candidates,
        key=lambda c: (
            report["trials"][c["name"]]["macro_average_precision"],
            -report["trials"][c["name"]]["worst_source_fpr"],
        ),
    )
    report["selected_xgboost"] = selected["name"]
    fit_idx, val_idx = grouped_fit_validation_indices(development, seed + 1000)
    fit, validation = development.iloc[fit_idx], development.iloc[val_idx]
    # Keep every tuning-used URL in fit/validation references. Add old known
    # domains not represented there to test.csv ONLY as an exclusion reference.
    used_domains = set(development["url"].map(get_registered_domain))
    reserved = reserved_references.loc[
        ~reserved_references["url"].map(get_registered_domain).isin(used_domains)
    ].copy()
    if not reserved.empty:
        validate_training_splits(
            {"fit": fit, "validation": validation, "reserved": reserved}
        )
    reference_dir = output_dir / "reference"
    reference_dir.mkdir()
    hashes = {}
    for name, frame in (("train", fit), ("validation", validation), ("test", reserved)):
        ref = reference_dir / f"{name}.csv"
        frame.to_csv(ref, index=False)
        hashes[name] = hashlib.sha256(ref.read_bytes()).hexdigest()
    for variant, candidate in (("xgboost", selected), ("char_logistic", comparison)):
        model = fit_variant(
            variant,
            fit,
            validation,
            seed,
            trees,
            params=candidate["params"],
            feature_policy=candidate["feature_policy"],
            balance_sources=candidate["weighted"],
        )
        scores = model.predict_proba(validation["url"])[:, 1]
        threshold = source_validation_threshold(validation, scores, maximum_fpr)
        metadata = {
            "model_version": f"0.3.0-source-tuned-{variant}",
            "model_type": variant,
            "created_at": datetime.now(UTC).isoformat(),
            "label_convention": {"0": "legitimate", "1": "phishing"},
            "threshold": threshold,
            "maximum_validation_fpr": maximum_fpr,
            "selected_configuration": candidate,
            "selection_method": "macro held-source development average precision",
            "data_dir": str(reference_dir),
            "split_sha256": hashes,
            "development_sha256": report["development_sha256"],
            "development_sources": (provenance or {}).get("development_sources", []),
            "dataset_rows": {"fit": len(fit), "validation": len(validation)},
            "fit_sha256": frame_fingerprint(fit),
            "validation_sha256": frame_fingerprint(validation),
            "validation": metrics_with_sources(validation, scores, threshold),
            "activation_eligible": False,
            "score_caution": (
                "uncalibrated development model; fresh final evaluation required"
            ),
            "reference_test_role": "previously known-domain exclusion only; not scored",
        }
        save_package(model, output_dir / variant, metadata)
    report["status"] = "complete"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
