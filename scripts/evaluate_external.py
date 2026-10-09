"""Evaluate frozen models or compare thresholds on their saved validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishguard.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from phishguard.data import (
    PHIUSIIL_LABEL_MAPPING,
    PROJECT_LABEL_MAPPING,
    TEXT_LABEL_MAPPING,
    _normalize_label_value,
    clean_url_data,
)
from phishguard.domains import get_registered_domain
from phishguard.experiments import choose_low_fpr_threshold
from phishguard.modeling import evaluate_probabilities

DATASET_PATH = PROJECT_ROOT / "data/external/balanced_urls.csv"
BATCH_SIZE = 20_000
SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def guard_final_dataset(
    dataset_path: Path,
    metadata: dict,
    history_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Reject known-used file identities. This cannot prove independent provenance."""
    fingerprint = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    known = set((metadata.get("split_sha256") or {}).values())
    sources = [
        *metadata.get("development_sources", []),
        *(metadata.get("split_manifest") or {}).get("sources", []),
    ]
    known.update(source["sha256"] for source in sources if "sha256" in source)
    for path in history_dir.rglob("*evaluation.json"):
        record = json.loads(path.read_text())
        if record.get("dataset_sha256"):
            known.add(record["dataset_sha256"])
    if fingerprint in known:
        raise ValueError(
            "This dataset was already used for development/evaluation or is a "
            "known split. It cannot be advertised as an untouched final test."
        )


def apply_missing_scheme(
    urls: pd.Series, policy: str = "reject"
) -> tuple[pd.Series, int]:
    """Make missing-protocol assumptions explicit; never infer from labels."""
    if policy not in {"reject", "http", "https"}:
        raise ValueError("Missing-scheme policy must be reject, http, or https")
    values = urls.astype("string").str.strip()
    missing = (
        values.notna() & values.ne("") & ~values.str.match(SCHEME_PATTERN, na=False)
    )
    if policy == "reject":
        return values, 0
    values.loc[missing] = policy + "://" + values.loc[missing].str.removeprefix("//")
    return values, int(missing.sum())


def load_external_dataset(
    dataset_path: Path = DATASET_PATH,
    url_column: str = "url",
    label_column: str = "label",
    label_mapping: dict | None = None,
    missing_scheme: str = "reject",
    exclude_conflicting_urls: bool = False,
) -> pd.DataFrame:
    raw_data = pd.read_csv(dataset_path)
    raw_data[url_column], assumed = apply_missing_scheme(
        raw_data[url_column], missing_scheme
    )
    mapping = TEXT_LABEL_MAPPING if label_mapping is None else label_mapping
    conflict_urls = 0
    conflict_rows = 0
    if exclude_conflicting_urls:
        normalized_mapping = {
            _normalize_label_value(key): value for key, value in mapping.items()
        }
        targets = (
            raw_data[label_column].map(_normalize_label_value).map(normalized_mapping)
        )
        counts = targets.groupby(raw_data[url_column]).nunique()
        conflicts = counts[counts > 1].index
        mask = raw_data[url_column].isin(conflicts)
        conflict_urls, conflict_rows = len(conflicts), int(mask.sum())
        raw_data = raw_data.loc[~mask].copy()
    clean_data, report = clean_url_data(
        raw_data,
        url_column=url_column,
        label_column=label_column,
        label_mapping=mapping,
    )
    print("Cleaning:", report, flush=True)
    print(f"Missing-scheme policy: {missing_scheme}; assumed rows: {assumed:,}")
    clean_data.attrs["cleaning_report"] = report
    clean_data.attrs["scheme_assumed_rows"] = assumed
    clean_data.attrs["conflicting_urls_excluded"] = conflict_urls
    clean_data.attrs["conflicting_rows_excluded"] = conflict_rows
    print(
        f"Conflicting labels excluded: {conflict_urls:,} URLs / {conflict_rows:,} rows"
    )
    return clean_data


def remove_original_dataset_overlap(
    df: pd.DataFrame, reference_dir: Path = PROCESSED_DATA_DIR
) -> pd.DataFrame:
    """Require all reference splits; exclude all previously used exact URLs."""
    original_urls: set[str] = set()
    for name in ("train", "validation", "test"):
        path = reference_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing reference split: {path}")
        original_urls.update(pd.read_csv(path, usecols=["url"])["url"].str.strip())
    overlap_mask = df["url"].isin(original_urls)
    print(f"Previously used exact URLs excluded: {int(overlap_mask.sum()):,}")
    return df.loc[~overlap_mask].copy()


def predict_in_batches(model: object, urls: pd.Series) -> np.ndarray:
    probabilities = []
    for start in range(0, len(urls), BATCH_SIZE):
        batch = urls.iloc[start : start + BATCH_SIZE]
        probabilities.append(model.predict_proba(batch)[:, 1])
    return np.concatenate(probabilities)


def validation_threshold_comparison(
    data: pd.DataFrame, scores: np.ndarray, current_threshold: float
) -> list[dict]:
    """Compare fixed cutoffs and validation-only empirical false-alarm budgets."""
    labels = data["is_phishing"].to_numpy()
    if len(scores) != len(data):
        raise ValueError("Validation rows and scores must have equal lengths")
    budgets = (0.02, 0.01)
    thresholds = [("current", current_threshold, None)]
    thresholds.extend(
        (
            f"validation_fpr_{budget:.0%}",
            choose_low_fpr_threshold(labels, scores, budget),
            budget,
        )
        for budget in budgets
    )
    thresholds.extend(
        (f"fixed_{threshold:g}", threshold, None)
        for threshold in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    )
    results = []
    for name, threshold, budget in thresholds:
        result = {
            "name": name,
            "maximum_validation_fpr": budget,
            "metrics": evaluate_probabilities(labels, scores, threshold),
            "by_source": {},
        }
        if "source" in data:
            for source in sorted(data["source"].unique()):
                mask = data["source"].eq(source).to_numpy()
                if set(labels[mask]) == {0, 1}:
                    result["by_source"][str(source)] = evaluate_probabilities(
                        labels[mask], scores[mask], threshold
                    )
        results.append(result)
    return results


def analyze_validation_thresholds(
    model_path: Path, metadata: dict, reference_dir: Path, output_dir: Path
) -> None:
    """Leave the model/metadata untouched and never load test labels or scores."""
    if not metadata.get("split_sha256"):
        raise ValueError("Threshold analysis requires saved split fingerprints")
    validation_path = reference_dir / "validation.csv"
    data = pd.read_csv(validation_path)
    train_urls = pd.read_csv(reference_dir / "train.csv", usecols=["url"])["url"]
    if data["url"].duplicated().any() or data["url"].isna().any():
        raise ValueError("Validation URLs must be unique and nonmissing")
    train_domains = set(train_urls.map(get_registered_domain))
    if data["url"].map(get_registered_domain).isin(train_domains).any():
        raise ValueError("Training and validation domains overlap")
    model = joblib.load(model_path)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected class order 0=legitimate, 1=phishing")
    scores = predict_in_batches(model, data["url"])
    results = validation_threshold_comparison(
        data, scores, float(metadata["threshold"])
    )
    saved_metrics = metadata.get("validation_metrics")
    if saved_metrics:
        for metric in ("recall", "false_positive_rate"):
            if not np.isclose(
                results[0]["metrics"][metric],
                saved_metrics[metric],
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    "Loaded model does not reproduce saved validation metrics"
                )
    report = {
        "evaluation_role": "validation_threshold_analysis_only",
        "activation_eligible": False,
        "model_path": str(model_path.resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "validation_path": str(validation_path.resolve()),
        "validation_sha256": metadata["split_sha256"]["validation"],
        "rows": len(data),
        "class_counts": {
            str(k): int(v) for k, v in data["is_phishing"].value_counts().items()
        },
        "train_validation_domains_disjoint": True,
        "test_labels_or_scores_used": False,
        "caution": (
            "This validation set already informed early stopping/model selection. "
            "These are reused development results, not independent performance "
            "estimates. Pooled FPR budgets need not hold per source or on new data. "
            "Scores are uncalibrated. No active threshold was changed."
        ),
        "comparisons": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_thresholds.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        m = result["metrics"]
        print(
            f"{result['name']:22} threshold={m['threshold']:.6f} "
            f"FPR={m['false_positive_rate']:.2%} recall={m['recall']:.2%} "
            f"precision={m['precision']:.2%}",
            flush=True,
        )
    print(f"Saved validation-only analysis: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET_PATH)
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--label-convention", choices=["text", "phiusiil", "project"], default="text"
    )
    parser.add_argument("--phishing-label")
    parser.add_argument("--legitimate-label")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        help="Model training splits; defaults to metadata data_dir or data/processed.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--validation-thresholds",
        action="store_true",
        help="Compare cutoffs using saved validation.csv; ignore external input.",
    )
    parser.add_argument("--exclude-shared-domains", action="store_true")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Final-test request: strict domains/protocol; reject known-used files.",
    )
    parser.add_argument(
        "--exclude-conflicting-urls",
        action="store_true",
        help="Exclude ALL rows for contradictory URL labels; default is to fail.",
    )
    parser.add_argument(
        "--missing-scheme",
        choices=["reject", "http", "https"],
        default="reject",
        help=(
            "Explicit protocol assumption for incomplete URL strings; "
            "not real protocol evidence."
        ),
    )
    args = parser.parse_args()
    if args.validation_thresholds and args.final_test:
        parser.error("Validation threshold analysis cannot be a final test")
    mappings = {
        "text": TEXT_LABEL_MAPPING,
        "phiusiil": PHIUSIIL_LABEL_MAPPING,
        "project": PROJECT_LABEL_MAPPING,
    }
    label_mapping = mappings[args.label_convention]
    if args.phishing_label is not None or args.legitimate_label is not None:
        if args.phishing_label is None or args.legitimate_label is None:
            raise ValueError("Specify both custom source labels")
        if args.phishing_label.strip().lower() == args.legitimate_label.strip().lower():
            raise ValueError("Custom source labels must be different")
        label_mapping = {args.phishing_label: 1, args.legitimate_label: 0}
    model_path = args.artifacts_dir / "phishguard_model.joblib"
    metadata = json.loads((args.artifacts_dir / "model_metadata.json").read_text())
    if args.final_test:
        if args.missing_scheme != "reject":
            raise ValueError("Final evaluation cannot assume missing protocols")
        guard_final_dataset(args.input, metadata)
        args.exclude_shared_domains = True
    reference_dir = args.reference_data_dir or Path(
        metadata.get("data_dir", str(PROCESSED_DATA_DIR))
    )
    if metadata.get("data_dir") and (
        reference_dir.resolve() != Path(metadata["data_dir"]).resolve()
    ):
        raise ValueError("Reference directory does not match model training provenance")
    if metadata.get("split_sha256"):
        hashes = {
            name: hashlib.sha256(
                (reference_dir / f"{name}.csv").read_bytes()
            ).hexdigest()
            for name in ("train", "validation", "test")
        }
        if hashes != metadata["split_sha256"]:
            raise ValueError("Reference splits differ from saved training fingerprints")
    if args.validation_thresholds:
        analyze_validation_thresholds(
            model_path,
            metadata,
            reference_dir,
            args.output_dir or args.artifacts_dir / "threshold_analysis",
        )
        return
    df = load_external_dataset(
        args.input,
        args.url_column,
        args.label_column,
        label_mapping,
        args.missing_scheme,
        args.exclude_conflicting_urls,
    )
    cleaning_report = df.attrs["cleaning_report"]
    assumed_rows = df.attrs["scheme_assumed_rows"]
    conflicts = {
        key: df.attrs[key]
        for key in ("conflicting_urls_excluded", "conflicting_rows_excluded")
    }
    before = len(df)
    df = remove_original_dataset_overlap(df, reference_dir)
    reference_domains: set[str] = set()
    for name in ("train", "validation", "test"):
        reference_domains.update(
            pd.read_csv(reference_dir / f"{name}.csv", usecols=["url"])["url"].map(
                get_registered_domain
            )
        )
    shared = df["url"].map(get_registered_domain).isin(reference_domains)
    shared_count = int(shared.sum())
    print(f"Remaining URLs on previously used domains: {shared_count:,}")
    if args.exclude_shared_domains:
        df = df.loc[~shared].copy()
    if df.empty or set(df["is_phishing"].unique()) != {0, 1}:
        raise ValueError(
            "No valid two-class external set remains after overlap removal. "
            "This dataset may already have been used for training."
        )
    model = joblib.load(model_path)
    probabilities = predict_in_batches(model, df["url"])
    threshold = float(metadata["threshold"])
    metrics = evaluate_probabilities(
        df["is_phishing"].to_numpy(), probabilities, threshold
    )
    print("\nFROZEN-MODEL CROSS-DATASET RESULTS")
    print(json.dumps(metrics, indent=2))
    df["phishing_score"] = probabilities
    df["predicted_is_phishing"] = (probabilities >= threshold).astype(int)
    output_dir = args.output_dir or args.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, true_label, predicted_label in (
        ("false_positives", 0, 1),
        ("false_negatives", 1, 0),
    ):
        mask = df["is_phishing"].eq(true_label) & df["predicted_is_phishing"].eq(
            predicted_label
        )
        df.loc[mask].to_csv(output_dir / f"{name}.csv", index=False)
    report = {
        "evaluation_role": (
            "final_test_requested" if args.final_test else "development_stress_test"
        ),
        "independent_provenance_verified": False,
        "provenance_caution": (
            "File fingerprints and domain exclusion do not establish independent "
            "collection or remove all template leakage. Verify provenance separately."
        ),
        "dataset": str(args.input.resolve()),
        "dataset_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            (args.artifacts_dir / "model_metadata.json").read_bytes()
        ).hexdigest(),
        "label_mapping": {str(key): value for key, value in label_mapping.items()},
        "missing_scheme_policy": args.missing_scheme,
        "scheme_assumed_rows_before_cleaning": assumed_rows,
        "cleaning_report": cleaning_report,
        "conflicting_label_exclusion_enabled": args.exclude_conflicting_urls,
        **conflicts,
        "evaluated_class_counts": {
            str(label): int(count)
            for label, count in df["is_phishing"].value_counts().items()
        },
        "majority_class_accuracy": float(
            df["is_phishing"].value_counts(normalize=True).max()
        ),
        "rows_evaluated": len(df),
        "model_path": str(model_path.resolve()),
        "threshold": threshold,
        "exact_overlap_excluded": before - len(shared),
        "shared_domain_rows_before_optional_exclusion": shared_count,
        "shared_domains_excluded": args.exclude_shared_domains,
        "reference_data_dir": str(reference_dir.resolve()),
        "metrics": metrics,
    }
    (output_dir / "external_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Saved report and error examples to: {output_dir}")


if __name__ == "__main__":
    main()
