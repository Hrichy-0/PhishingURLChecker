"""Create domain-disjoint train, validation, and test CSV files."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from phishguard.config import PROCESSED_DATA_DIR
from phishguard.data import PROJECT_LABEL_MAPPING, clean_url_data, combine_sources
from phishguard.domains import get_registered_domain
from phishguard.split import split_by_domain


def print_split_summary(
    name: str,
    split_data: pd.DataFrame,
    total_rows: int,
) -> None:
    """Print size, domain count, and class balance for one split."""
    row_percentage = len(split_data) / total_rows * 100
    domain_count = split_data["url"].map(get_registered_domain).nunique()
    label_counts = split_data["is_phishing"].value_counts().sort_index()
    label_percentages = (
        split_data["is_phishing"].value_counts(normalize=True).sort_index().mul(100)
    )

    print(f"\n{name.upper()}")
    print(f"Rows: {len(split_data):,} ({row_percentage:.2f}%)")
    print(f"Registered domains: {domain_count:,}")

    for label, count in label_counts.items():
        class_name = "phishing" if label == 1 else "legitimate"
        print(
            f"{label} ({class_name}): {count:,} ({label_percentages.loc[label]:.2f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[PROCESSED_DATA_DIR / "clean_urls.csv"],
        help="One or more standardized CSVs containing url and is_phishing.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frames = []
    sources = []
    for index, input_path in enumerate(args.input):
        raw_data = pd.read_csv(input_path)
        frame, report = clean_url_data(
            raw_data,
            url_column="url",
            label_column="is_phishing",
            label_mapping=PROJECT_LABEL_MAPPING,
        )
        frame["source"] = f"{index}:{input_path.stem}"
        frames.append(frame)
        sources.append(
            {
                "source": f"{index}:{input_path.stem}",
                "path": str(input_path.resolve()),
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "cleaning": report,
            }
        )

    clean_data, combined_report = combine_sources(frames)
    print(f"Cross-source cleaning: {combined_report}")

    train_data, validation_data, test_data = split_by_domain(
        clean_data, random_state=args.seed
    )

    assert len(train_data) + len(validation_data) + len(test_data) == len(clean_data)

    splits = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
    }
    for name, split_data in splits.items():
        print_split_summary(name, split_data, len(clean_data))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{name}.csv"
        split_data.to_csv(output_path, index=False)
        print(f"Saved {name}: {output_path}")

    manifest = {
        "seed": args.seed,
        "grouping": "registered domain (including private public suffixes)",
        "sources": sources,
        **combined_report,
        "split_sha256": {
            name: hashlib.sha256(
                (args.output_dir / f"{name}.csv").read_bytes()
            ).hexdigest()
            for name in splits
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
