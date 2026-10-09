"""Clean a supported phishing URL dataset into the project's standard schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phishguard.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from phishguard.data import (
    PHIUSIIL_LABEL_MAPPING,
    PROJECT_LABEL_MAPPING,
    TEXT_LABEL_MAPPING,
    clean_url_data,
)

LABEL_MAPPINGS = {
    "phiusiil": PHIUSIIL_LABEL_MAPPING,
    "project": PROJECT_LABEL_MAPPING,
    "text": TEXT_LABEL_MAPPING,
}


def find_raw_csv() -> Path:
    """Return the only CSV in the raw data directory."""
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DATA_DIR}. Run scripts/download_data.py first."
        )

    if len(csv_files) > 1:
        raise RuntimeError(
            f"Multiple raw CSV files were found. Select one with --input: {csv_files}"
        )

    return csv_files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a phishing URL CSV for model training."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Source CSV. Defaults to the only CSV in data/raw.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "clean_urls.csv",
        help="Destination CSV (default: data/processed/clean_urls.csv).",
    )
    parser.add_argument(
        "--url-column",
        help="Column containing URLs. If omitted, URL then url are checked.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing the target label (default: label).",
    )
    parser.add_argument(
        "--label-convention",
        choices=sorted(LABEL_MAPPINGS),
        default="phiusiil",
        help=(
            "phiusiil: 0=phishing/1=legitimate; "
            "project: 0=legitimate/1=phishing; "
            "text: legitimate/phishing"
        ),
    )
    parser.add_argument("--phishing-label", help="Custom source value for phishing.")
    parser.add_argument(
        "--legitimate-label", help="Custom source value for legitimate."
    )
    return parser.parse_args()


def resolve_url_column(data: pd.DataFrame, requested: str | None) -> str:
    """Resolve the URL column without silently choosing an unrelated field."""
    if requested:
        if requested not in data.columns:
            raise ValueError(f"URL column not found: {requested}")
        return requested

    for candidate in ("URL", "url"):
        if candidate in data.columns:
            return candidate

    raise ValueError(
        "Could not find a URL column named URL or url. Specify it with --url-column."
    )


def main() -> None:
    args = parse_args()
    raw_data_path = args.input or find_raw_csv()

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {raw_data_path}")

    print(f"Reading raw data from: {raw_data_path}")
    raw_data = pd.read_csv(raw_data_path, low_memory=False)
    url_column = resolve_url_column(raw_data, args.url_column)
    label_mapping = LABEL_MAPPINGS[args.label_convention]
    if args.phishing_label is not None or args.legitimate_label is not None:
        if args.phishing_label is None or args.legitimate_label is None:
            raise ValueError("Specify both --phishing-label and --legitimate-label")
        if args.phishing_label.strip().lower() == args.legitimate_label.strip().lower():
            raise ValueError("Phishing and legitimate source labels must be different")
        label_mapping = {args.phishing_label: 1, args.legitimate_label: 0}

    print(f"URL column: {url_column}")
    print(f"Label column: {args.label_column}")
    print(f"Label convention: {args.label_convention}")
    print(f"Source-to-project label mapping: {label_mapping}")

    clean_data, report = clean_url_data(
        raw_data,
        url_column=url_column,
        label_column=args.label_column,
        label_mapping=label_mapping,
    )
    if clean_data.empty:
        raise ValueError("No valid labeled HTTP(S) URLs remain after cleaning")

    print("\nCLEANING REPORT")
    for rule, count in report.items():
        print(f"{rule}: {count:,}")

    print("\nCLEANED LABEL DISTRIBUTION")
    label_counts = clean_data["is_phishing"].value_counts().sort_index()
    label_percentages = (
        clean_data["is_phishing"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    for label, count in label_counts.items():
        class_name = "phishing" if label == 1 else "legitimate"
        percentage = label_percentages.loc[label]
        print(f"{label} ({class_name}): {count:,} rows ({percentage:.2f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_data.to_csv(args.output, index=False)
    print(f"\nSaved cleaned data to: {args.output.resolve()}")

    saved_data = pd.read_csv(args.output)
    assert list(saved_data.columns) == ["url", "is_phishing"]
    assert len(saved_data) == report["rows_after"]
    assert not saved_data["url"].duplicated().any()
    assert set(saved_data["is_phishing"].unique()) <= {0, 1}
    print("Saved file verification passed.")


if __name__ == "__main__":
    main()
