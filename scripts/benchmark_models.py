"""Compare URL models without touching the active API model or final test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from phishguard.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from phishguard.data import PROJECT_LABEL_MAPPING, clean_url_data, combine_sources
from phishguard.domains import get_registered_domain
from phishguard.experiments import URL_VARIANTS, run_benchmark, run_source_tuning
from phishguard.split import validate_training_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["benchmark", "tune"], default="benchmark")
    parser.add_argument(
        "--development-input",
        type=Path,
        action="append",
        default=[],
        help="Additional DEVELOPMENT CSV (url + label, 0=legitimate, 1=phishing).",
    )
    parser.add_argument("--development-label-column", default="label")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR / "mixed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Pilot cap: total rows in benchmark, per source in tune; 0 uses all.",
    )
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--maximum-fpr", type=float, default=0.01)
    parser.add_argument(
        "--variants", nargs="+", choices=URL_VARIANTS, default=list(URL_VARIANTS)
    )
    args = parser.parse_args()
    if args.max_rows < 0:
        parser.error("--max-rows must be nonnegative")
    if args.trees < 1 or not 0 <= args.maximum_fpr < 1:
        parser.error("Require positive --trees and 0 <= --maximum-fpr < 1")
    if args.mode == "benchmark" and args.folds < 2:
        parser.error("--folds must be at least two")
    splits = {
        name: pd.read_csv(args.data_dir / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    validate_training_splits(splits)
    hashes = {
        name: hashlib.sha256((args.data_dir / f"{name}.csv").read_bytes()).hexdigest()
        for name in splits
    }
    manifest = args.data_dir / "split_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["split_sha256"] != hashes:
        raise ValueError("Split files differ from their manifest")
    # Test is read ONLY for isolation/fingerprint checks, never fitted or scored.
    development = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    total_rows = len(development)
    if args.mode == "benchmark" and args.development_input:
        parser.error("--development-input is for source-tuning mode only")
    source_provenance = []
    if args.mode == "tune":
        source_provenance = (
            json.loads(manifest.read_text()).get("sources", [])
            if manifest.exists()
            else []
        )
        known_domains = set(
            pd.concat(splits.values())["url"].map(get_registered_domain)
        )
        frames = [development]
        for index, input_path in enumerate(args.development_input):
            frame, cleaning = clean_url_data(
                pd.read_csv(input_path),
                url_column="url",
                label_column=args.development_label_column,
                label_mapping=PROJECT_LABEL_MAPPING,
            )
            seen = frame["url"].map(get_registered_domain).isin(known_domains)
            frame = frame.loc[~seen].copy()
            if frame.empty or set(frame["is_phishing"].unique()) != {0, 1}:
                raise ValueError(
                    f"{input_path}: no two-class development source remains "
                    "after cleaning and known-domain exclusion"
                )
            name = f"external{index}:{input_path.parent.name}/{input_path.stem}"
            frame["source"] = name
            frames.append(frame)
            source_provenance.append(
                {
                    "source": name,
                    "path": str(input_path.resolve()),
                    "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "cleaning": cleaning,
                    "old_known_domain_rows_excluded": int(seen.sum()),
                }
            )
        development, cross_source_cleaning = combine_sources(frames)
        total_rows = len(development)
        if args.max_rows:
            development = pd.concat(
                [
                    frame.sample(
                        n=min(args.max_rows, len(frame)), random_state=args.seed
                    )
                    for _, frame in development.groupby("source", sort=True)
                ],
                ignore_index=True,
            )
    elif args.max_rows and len(development) > args.max_rows:
        development = development.sample(n=args.max_rows, random_state=args.seed)
    provenance = {
        "data_dir": str(args.data_dir.resolve()),
        "split_sha256": hashes,
        "development_splits": ["train", "validation"],
        "available_development_rows": total_rows,
        "pilot": len(development) < total_rows,
        "sampling": (
            "uniform label-blind row sample; domains disjoint during evaluation"
        ),
        "held_out_test_rows_not_scored": len(splits["test"]),
        "development_sources": source_provenance,
    }
    output = args.output_dir or ARTIFACTS_DIR / "experiments" / (
        f"url_{args.mode}_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    print(f"Development rows: {len(development):,} / {total_rows:,}", flush=True)
    print("No final-test scoring or active-model replacement.", flush=True)
    if args.mode == "tune":
        provenance["sampling"] = (
            "uniform label-blind rows per source; domain isolation preserved"
        )
        provenance["cross_source_cleaning"] = cross_source_cleaning
        report = run_source_tuning(
            development,
            output,
            pd.concat(splits.values(), ignore_index=True),
            seed=args.seed,
            maximum_fpr=args.maximum_fpr,
            trees=args.trees,
            provenance=provenance,
        )
        print(f"Saved source tuning: {output / 'tuning.json'}")
        print(f"Selected XGBoost configuration: {report['selected_xgboost']}")
        return
    report = run_benchmark(
        development,
        output,
        tuple(args.variants),
        args.folds,
        args.seed,
        args.maximum_fpr,
        args.trees,
        provenance,
    )
    print(f"Saved benchmark: {output / 'benchmark.json'}")
    print(f"Highest CV average precision: {report['highest_cv_average_precision']}")


if __name__ == "__main__":
    main()
