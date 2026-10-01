"""Inspect an external phishing URL dataset before evaluation."""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/external/balanced_urls.csv")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place balanced_urls.csv inside data/external/ first."
        )

    df = pd.read_csv(DATASET_PATH)

    print("=" * 60)
    print("EXTERNAL DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nFIRST 5 ROWS")
    print(df.head())

    print("\nMISSING VALUES")
    print(df.isna().sum())

    print("\nDUPLICATE ROWS")
    print(df.duplicated().sum())

    if "url" in df.columns:
        print("\nDUPLICATE URLS")
        print(df["url"].duplicated().sum())

        print("\nUNIQUE URLS")
        print(df["url"].nunique())

    if "label" in df.columns:
        print("\nLABEL DISTRIBUTION")
        print(df["label"].value_counts(dropna=False))

        print("\nLABEL DISTRIBUTION (%)")
        print(
            (df["label"].value_counts(normalize=True, dropna=False) * 100)
            .round(2)
        )

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()