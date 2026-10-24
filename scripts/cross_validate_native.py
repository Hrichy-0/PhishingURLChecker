"""Development-only grouped CV of native URL or URL+static-content XGBoost."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from phishguard.config import MAIN_URL_MODEL_DIR, PROJECT_ROOT
from phishguard.native import (
    TRAINING_FEATURES,
    choose_threshold,
    evaluate_probabilities,
    ranking_curve,
)
from phishguard.native_html import HYBRID_FEATURE_NAMES

POLICIES = ("saved_cutoff_reference", "inner_f1_recall90", "inner_fpr01", "inner_fpr02")


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def connected_groups(frame: pd.DataFrame) -> np.ndarray:
    """Transitive domain/exact-content groups; content hashes are NOT features."""
    if frame[["domain", "html_sha256"]].isna().any().any():
        raise ValueError("Missing grouping metadata")
    parents = {domain: domain for domain in frame.domain.unique()}

    def root(domain):
        while parents[domain] != domain:
            parents[domain] = parents[parents[domain]]
            domain = parents[domain]
        return domain

    owners = {}
    for domain, digest in frame[["domain", "html_sha256"]].itertuples(index=False):
        if not domain:
            raise ValueError("Empty domain")
        if digest:  # Empty hashes must not join all unavailable pages.
            previous = owners.setdefault(digest, domain)
            first, second = sorted((root(domain), root(previous)))
            parents[second] = first
    roots = frame.domain.map({domain: root(domain) for domain in parents})
    return pd.factorize(roots, sort=True)[0]


def make_partitions(frame, groups, folds=5, seed=42):
    labels = frame.is_phishing.to_numpy()
    if set(np.unique(labels)) != {0, 1} or len(np.unique(groups)) < folds:
        raise ValueError("Require two binary classes and enough components")
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    partitions = []
    for fold, (outer_train, outer_test) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups), start=1
    ):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
        fit, validation = next(inner.split(outer_train, groups=groups[outer_train]))
        parts = (outer_train[fit], outer_train[validation], outer_test)
        for index, indices in enumerate(parts):
            if set(np.unique(labels[indices])) != {0, 1}:
                raise ValueError(f"Fold {fold}: a role lacks both classes")
            for other in parts[index + 1 :]:
                for column in ("domain", "html_sha256"):
                    left = set(frame.iloc[indices][column]) - {""}
                    right = set(frame.iloc[other][column]) - {""}
                    if left & right:
                        raise ValueError(f"Fold {fold}: {column} leakage")
                if set(groups[indices]) & set(groups[other]):
                    raise ValueError("Component leakage")
        partitions.append(parts)
    tests = np.concatenate([parts[2] for parts in partitions])
    if not np.array_equal(np.sort(tests), np.arange(len(frame))):
        raise ValueError("Each development record must be evaluated exactly once")
    return partitions


def low_fpr_threshold(labels, scores, budget):
    thresholds, _, recalls, fprs = ranking_curve(labels, scores)
    eligible = np.flatnonzero(fprs <= budget)
    if not len(eligible):
        return float(np.nextafter(np.float32(1), np.float32(np.inf)))
    return float(thresholds[eligible[np.argmax(recalls[eligible])]])


def count_metrics(counts):
    tn, fp, fn, tp = (int(value) for value in counts)
    return {
        "accuracy": (tn + tp) / max(1, tn + fp + fn + tp),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def clustered_intervals(oof: pd.DataFrame, draws=1000, seed=42):
    """Percentile cluster bootstrap within folds, conditional on fitted models."""
    if draws < 1:
        raise ValueError("Require positive bootstrap draws")
    if (oof.groupby("component_id").fold.nunique() != 1).any():
        raise ValueError("A component crosses outer folds")
    components, groups = np.unique(oof.component_id, return_inverse=True)
    group_folds = np.zeros(len(components), dtype=int)
    group_folds[groups] = oof.fold
    labels = oof.is_phishing.to_numpy(dtype=int)
    counts = np.zeros((len(components), len(POLICIES), 4), dtype=np.int64)
    for index, policy in enumerate(POLICIES):
        predicted = oof.score.to_numpy() >= oof[f"threshold_{policy}"].to_numpy()
        np.add.at(counts[:, index, :], (groups, labels * 2 + predicted), 1)
    strata = [np.flatnonzero(group_folds == fold) for fold in np.unique(group_folds)]
    rng = np.random.default_rng(seed)
    samples = {policy: [] for policy in POLICIES}
    for _ in range(draws):
        selected = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        totals = counts[selected].sum(axis=0)
        # A replicate missing a class cannot estimate both FPR and recall.
        if any(row[0] + row[1] == 0 or row[2] + row[3] == 0 for row in totals):
            continue
        for policy, total in zip(POLICIES, totals, strict=True):
            samples[policy].append(count_metrics(total))
    intervals = {}
    for policy, records in samples.items():
        if not records:
            raise ValueError("No two-class bootstrap replicate")
        intervals[policy] = {
            metric: np.percentile([r[metric] for r in records], [2.5, 97.5]).tolist()
            for metric in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
            )
        }
    return {
        "draws_requested": draws,
        "draws_valid": len(samples[POLICIES[0]]),
        "method": "95% percentile; resample whole components within outer folds",
        "conditional_on_fitted_models": True,
        "intervals": intervals,
    }


def compare_runs(left: Path, right: Path, draws=1000, seed=42):
    """Paired cluster bootstrap of right-minus-left confusion-metric differences."""
    a, b = [pd.read_parquet(path / "out_of_fold.parquet") for path in (left, right)]
    identity = ["row_index", "component_id", "fold", "is_phishing", "original_split"]
    pd.testing.assert_frame_equal(a[identity], b[identity])
    pd.testing.assert_frame_equal(
        *[pd.read_parquet(path / "role_assignments.parquet") for path in (left, right)]
    )
    reports = [
        json.loads((path / "cv_results.json").read_text()) for path in (left, right)
    ]
    if any(report["status"] != "completed" for report in reports):
        raise ValueError("Require completed runs")
    if (
        reports[0]["provenance"]["split_sha256"]
        != reports[1]["provenance"]["split_sha256"]
    ):
        raise ValueError("Different source records")
    if draws < 1:
        raise ValueError("Require positive bootstrap draws")
    if (a.groupby("component_id").fold.nunique() != 1).any():
        raise ValueError("Component crosses folds")
    components, groups = np.unique(a.component_id, return_inverse=True)
    folds = np.zeros(len(components), dtype=int)
    folds[groups] = a.fold
    strata = [np.flatnonzero(folds == fold) for fold in np.unique(folds)]
    counts = np.zeros((len(components), 2, len(POLICIES), 4), dtype=np.int64)
    for model, frame in enumerate((a, b)):
        for index, policy in enumerate(POLICIES):
            predicted = (
                frame.score.to_numpy() >= frame[f"threshold_{policy}"].to_numpy()
            )
            np.add.at(
                counts[:, model, index, :],
                (groups, frame.is_phishing.to_numpy() * 2 + predicted),
                1,
            )
    metrics = ("accuracy", "precision", "recall", "f1", "false_positive_rate")
    samples = {policy: {metric: [] for metric in metrics} for policy in POLICIES}
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(draws):
        selected = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        totals = counts[selected].sum(axis=0)
        if any(
            row[0] + row[1] == 0 or row[2] + row[3] == 0
            for row in totals.reshape(-1, 4)
        ):
            continue
        valid += 1
        for index, policy in enumerate(POLICIES):
            first, second = [count_metrics(totals[model, index]) for model in (0, 1)]
            for metric in metrics:
                samples[policy][metric].append(second[metric] - first[metric])
    if not valid:
        raise ValueError("No valid paired bootstrap samples")
    return {
        "left": str(left),
        "right": str(right),
        "rows": len(a),
        "same_records_roles_components": True,
        "draws": valid,
        "seed": seed,
        "method": "Right minus left; paired whole-component resampling within folds",
        "conditional_on_fitted_models": True,
        "limitation": (
            "Development-only; excludes refitting, source and label uncertainty"
        ),
        "policy_differences": {
            policy: {
                metric: {
                    "point": reports[1]["pooled"][policy][metric]
                    - reports[0]["pooled"][policy][metric],
                    "percentile_95_interval": np.percentile(
                        values, [2.5, 97.5]
                    ).tolist(),
                }
                for metric, values in by_metric.items()
            }
            for policy, by_metric in samples.items()
        },
    }


def run_cv(
    frame,
    output,
    parameters,
    saved_cutoff,
    *,
    folds=5,
    seed=42,
    trees=800,
    threads=4,
    bootstrap_draws=1000,
    provenance=None,
    feature_names=TRAINING_FEATURES,
):
    if folds < 2 or trees < 1 or not 1 <= threads <= 16 or bootstrap_draws < 1:
        raise ValueError("Invalid fold, tree, thread or bootstrap settings")
    if output.exists():
        raise ValueError("Choose a new output directory; existing runs are immutable")
    if frame.url.isna().any() or frame.url.duplicated().any():
        raise ValueError("Missing or duplicate development URLs")
    feature_names = tuple(feature_names)
    if feature_names not in (TRAINING_FEATURES, HYBRID_FEATURE_NAMES):
        raise ValueError("Require the fixed URL or URL+content feature schema")
    matrix = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite model features")
    started = time.monotonic()
    groups = connected_groups(frame)
    partitions = make_partitions(frame, groups, folds, seed)
    sizes = np.bincount(groups)
    report = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "development_rows": len(frame),
        "development_domains": int(frame.domain.nunique()),
        "components": len(sizes),
        "largest_component_rows": int(sizes.max()),
        "component_size_percentiles": dict(
            zip(
                ("median", "p95", "p99"),
                np.percentile(sizes, [50, 95, 99]).tolist(),
                strict=True,
            )
        ),
        "class_counts": {
            str(k): int(v) for k, v in frame.is_phishing.value_counts().items()
        },
        "seed": seed,
        "features": list(feature_names),
        "runtime": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "architecture": platform.machine(),
        },
        "provenance": provenance or {},
        "folds": [],
        "protocol": {
            "outer": "StratifiedGroupKFold by domain/exact-content components",
            "inner": "GroupShuffleSplit: 20% outer-training components; seed + fold",
            "hyperparameters": "Fixed current configuration; no search",
            "maximum_rounds": trees,
            "early_stopping_rounds": 40,
            "prediction": "Only best_iteration + 1 trees; CPU",
            "test_and_publisher": "No fitting, early stopping, selection or scoring",
            "html_role": (
                "Hash for grouping only; no content features"
                if feature_names == TRAINING_FEATURES
                else "17 cached static-content features plus 21 URL features"
            ),
            "bootstrap": "Whole components within folds, conditional on fitted models",
        },
        "limitations": [
            "Development CV; hyperparameters previously selected on this collection",
            "Saved cutoff reference was selected using prior validation now inside CV",
            "Inner validation is reused for early stopping and threshold selection",
            "No strict temporal, near-template or source independence proof",
            "Pooled ranking scores come from different fitted models; descriptive only",
            "Bootstrap excludes full refitting and training-set uncertainty",
            "No new external real-world accuracy or browser-performance claim",
        ],
    }
    output.mkdir(parents=True)

    def save_report():
        (output / "cv_results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save_report()
    roles = pd.DataFrame({"row_index": np.arange(len(frame)), "component_id": groups})
    predictions = []
    labels = frame.is_phishing.to_numpy()
    for fold, (fit, validation, test) in enumerate(partitions, start=1):
        start = time.monotonic()
        params = {
            **parameters,
            "device": "cpu",
            "nthread": threads,
            "seed": seed + fold,
        }
        datasets = {
            name: xgb.DMatrix(
                matrix[indices],
                label=labels[indices],
                feature_names=list(feature_names),
                nthread=threads,
            )
            for name, indices in (
                ("fit", fit),
                ("validation", validation),
                ("outer", test),
            )
        }
        print(
            f"Fold {fold}/{folds}: fit={len(fit):,}, "
            f"inner={len(validation):,}, outer={len(test):,}",
            flush=True,
        )
        model = xgb.train(
            params,
            datasets["fit"],
            num_boost_round=trees,
            evals=[(datasets["validation"], "inner_validation")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        best = model[: model.best_iteration + 1]
        scores = {name: best.predict(data) for name, data in datasets.items()}
        cutoffs = dict(
            zip(
                POLICIES,
                (
                    saved_cutoff,
                    choose_threshold(labels[validation], scores["validation"]),
                    low_fpr_threshold(labels[validation], scores["validation"], 0.01),
                    low_fpr_threshold(labels[validation], scores["validation"], 0.02),
                ),
                strict=True,
            )
        )
        model_path = output / f"fold_{fold}.ubj"
        best.save_model(model_path)
        record = {
            "fold": fold,
            "trees": best.num_boosted_rounds(),
            "parameters": params,
            "model_sha256": fingerprint(model_path),
            "cutoffs": cutoffs,
            "roles": {},
            "metrics": {},
        }
        result = pd.DataFrame(
            {
                "row_index": test,
                "component_id": groups[test],
                "fold": fold,
                "is_phishing": labels[test],
                "score": scores["outer"],
            }
        )
        role = np.empty(len(frame), dtype=np.int8)
        for code, (name, indices) in enumerate(
            (("fit", fit), ("validation", validation), ("outer", test))
        ):
            role[indices] = code
            record["roles"][name] = {
                "rows": len(indices),
                "components": len(np.unique(groups[indices])),
                "class_counts": {
                    str(k): int(v)
                    for k, v in pd.Series(labels[indices]).value_counts().items()
                },
            }
            record["metrics"][name] = {
                policy: evaluate_probabilities(labels[indices], scores[name], cutoff)
                for policy, cutoff in cutoffs.items()
            }
        roles[f"fold_{fold}_role_0fit_1inner_2outer"] = role
        for policy, cutoff in cutoffs.items():
            result[f"threshold_{policy}"] = cutoff
        predictions.append(result)
        record["seconds"] = time.monotonic() - start
        report["folds"].append(record)
        save_report()
        metric = record["metrics"]["outer"]["inner_f1_recall90"]
        print(
            f"  {record['trees']} trees; recall={metric['recall']:.4f}, "
            f"FPR={metric['false_positive_rate']:.4f}; {record['seconds']:.1f}s",
            flush=True,
        )
    oof = pd.concat(predictions).sort_values("row_index").reset_index(drop=True)
    oof["original_split"] = frame.loc[oof.row_index, "original_split"].to_numpy()
    oof.to_parquet(output / "out_of_fold.parquet", index=False)
    roles.to_parquet(output / "role_assignments.parquet", index=False)
    report["pooled"] = {}
    for policy in POLICIES:
        predicted = oof.score.to_numpy() >= oof[f"threshold_{policy}"].to_numpy()
        counts = np.bincount(oof.is_phishing.to_numpy() * 2 + predicted, minlength=4)
        report["pooled"][policy] = count_metrics(counts)
    ranking = evaluate_probabilities(oof.is_phishing, oof.score, 0.5)
    report["pooled_ranking_descriptive"] = {
        key: ranking[key] for key in ("roc_auc", "average_precision")
    }
    report["fold_mean_and_sample_sd"] = {
        policy: {
            metric: {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
            }
            for metric in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "roc_auc",
                "average_precision",
            )
            for values in [
                [r["metrics"]["outer"][policy][metric] for r in report["folds"]]
            ]
        }
        for policy in POLICIES
    }
    print("Computing component-clustered confidence intervals...", flush=True)
    report["bootstrap"] = clustered_intervals(oof, bootstrap_draws, seed)
    report["output_sha256"] = {
        name: fingerprint(output / name)
        for name in ("out_of_fold.parquet", "role_assignments.parquet")
    }
    report["seconds"] = time.monotonic() - started
    report["status"] = "completed"
    save_report()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=MAIN_URL_MODEL_DIR.parent)
    parser.add_argument(
        "--variant", choices=("url_only", "url_html"), default="url_only"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument(
        "--compare-runs",
        type=Path,
        nargs=2,
        help="Compare two completed runs without fitting; output is JSON file",
    )
    args = parser.parse_args()
    if args.compare_runs:
        if args.output.exists():
            parser.error("Comparison output exists; choose a new filename")
        report = compare_runs(*args.compare_runs, draws=args.bootstrap_draws)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Paired comparison: {args.output}")
        return
    manifest = json.loads((args.experiment / "split_manifest.json").read_text())
    metadata = json.loads(
        (args.experiment / args.variant / "model_metadata.json").read_text()
    )
    feature_names = (
        TRAINING_FEATURES if args.variant == "url_only" else HYBRID_FEATURE_NAMES
    )
    if metadata["features"] != list(feature_names):
        raise ValueError("Model feature definitions do not match")
    if (
        fingerprint(args.experiment / args.variant / "xgboost_model.ubj")
        != metadata["model_sha256"]
    ):
        raise ValueError("Current model checksum mismatch")
    frames, hashes = {}, {}
    for name in ("train", "validation", "test", "publisher_test"):
        path = args.experiment / f"{name}_rows.parquet"
        hashes[name] = fingerprint(path)
        if hashes[name] != manifest["split_sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
        # Holdout labels and feature values are never loaded by this command.
        columns = ["url", "domain", "html_sha256"]
        if name in {"train", "validation"}:
            columns += ["is_phishing", *feature_names]
        frames[name] = pd.read_parquet(path, columns=columns)
        if len(frames[name]) != manifest["cohorts"][name]["rows"]:
            raise ValueError("Manifest row count mismatch")
    for index, name in enumerate(frames):
        for other in list(frames)[index + 1 :]:
            for column in ("url", "domain", "html_sha256"):
                if (set(frames[name][column]) - {""}) & (
                    set(frames[other][column]) - {""}
                ):
                    raise ValueError(f"Cross-split {column} leakage")
    development = pd.concat(
        [frames[name].assign(original_split=name) for name in ("train", "validation")],
        ignore_index=True,
    )
    protected = [
        MAIN_URL_MODEL_DIR / name
        for name in ("xgboost_model.ubj", "model_metadata.json")
    ]
    protected += [
        PROJECT_ROOT / "artifacts" / name
        for name in ("phishguard_model.joblib", "model_metadata.json")
    ]
    protected += [
        args.experiment / args.variant / name
        for name in ("xgboost_model.ubj", "model_metadata.json")
    ]
    before = {str(path): fingerprint(path) for path in protected}
    provenance = {
        "variant": args.variant,
        "split_sha256": hashes,
        "manifest_sha256": fingerprint(args.experiment / "split_manifest.json"),
        "current_model_sha256": metadata["model_sha256"],
        "protected_artifacts_before": before,
        "source_code_sha256": {
            name: fingerprint(PROJECT_ROOT / name)
            for name in (
                "scripts/cross_validate_native.py",
                "src/phishguard/native.py",
                "src/phishguard/lexical.py",
            )
        },
    }
    report = run_cv(
        development,
        args.output,
        metadata["parameters"],
        metadata["threshold"],
        trees=args.trees,
        threads=args.threads,
        bootstrap_draws=args.bootstrap_draws,
        provenance=provenance,
        feature_names=feature_names,
    )
    unchanged = before == {str(path): fingerprint(path) for path in protected}
    if not unchanged:
        raise RuntimeError("Protected model changed during CV")
    report["protected_artifacts_unchanged"] = True
    (args.output / "cv_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"Completed; report: {args.output / 'cv_results.json'}", flush=True)


if __name__ == "__main__":
    main()
