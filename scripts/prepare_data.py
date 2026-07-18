from pathlib import Path

import pandas as pd

from phishguard.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from phishguard.data import clean_url_data


def find_raw_csv() -> Path:
    """Return the only CSV in the raw data directory."""
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DATA_DIR}. Run scripts/download_data.py first."
        )

    if len(csv_files) > 1:
        raise RuntimeError(f"Expected one raw CSV, but found: {csv_files}")

    return csv_files[0]


def main() -> None:
    raw_data_path = find_raw_csv()

    print(f"Reading raw data from: {raw_data_path}")
    raw_data = pd.read_csv(raw_data_path, low_memory=False)

    clean_data, report = clean_url_data(raw_data)

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

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "clean_urls.csv"

    clean_data.to_csv(output_path, index=False)

    print(f"\nSaved cleaned data to: {output_path}")

    saved_data = pd.read_csv(output_path)

    assert list(saved_data.columns) == ["url", "is_phishing"]
    assert len(saved_data) == report["rows_after"]
    assert not saved_data["url"].duplicated().any()
    assert set(saved_data["is_phishing"].unique()) == {0, 1}

    print("Saved file verification passed.")


if __name__ == "__main__":
    main()
