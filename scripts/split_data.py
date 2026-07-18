"""Create domain-disjoint train, validation, and test CSV files."""

import pandas as pd

from phishguard.config import PROCESSED_DATA_DIR
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
    input_path = PROCESSED_DATA_DIR / "clean_urls.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run scripts/prepare_data.py first."
        )

    print(f"Reading cleaned data from: {input_path}")
    clean_data = pd.read_csv(input_path)
    train_data, validation_data, test_data = split_by_domain(clean_data)

    assert len(train_data) + len(validation_data) + len(test_data) == len(clean_data)

    splits = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
    }
    for name, split_data in splits.items():
        print_split_summary(name, split_data, len(clean_data))
        output_path = PROCESSED_DATA_DIR / f"{name}.csv"
        split_data.to_csv(output_path, index=False)
        print(f"Saved {name}: {output_path}")


if __name__ == "__main__":
    main()
