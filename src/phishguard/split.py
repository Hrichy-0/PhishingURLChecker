from math import isclose

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from phishguard.domains import get_registered_domain

REQUIRED_COLUMNS = {"url", "is_phishing"}


def split_by_domain(
    data: pd.DataFrame,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split URL data while keeping registered domains together."""

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Cannot split an empty dataset")

    total_size = train_size + validation_size + test_size

    if not isclose(total_size, 1.0):
        raise ValueError("train_size, validation_size, and test_size must sum to 1.0")

    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("All split sizes must be greater than zero")

    working_data = data.copy()

    working_data["_domain_group"] = working_data["url"].map(get_registered_domain)

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )

    train_indices, remaining_indices = next(
        first_split.split(
            working_data,
            groups=working_data["_domain_group"],
        )
    )

    train_data = working_data.iloc[train_indices].copy()
    remaining_data = working_data.iloc[remaining_indices].copy()

    relative_validation_size = validation_size / (validation_size + test_size)

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_validation_size,
        random_state=random_state + 1,
    )

    validation_indices, test_indices = next(
        second_split.split(
            remaining_data,
            groups=remaining_data["_domain_group"],
        )
    )

    validation_data = remaining_data.iloc[validation_indices].copy()
    test_data = remaining_data.iloc[test_indices].copy()

    train_domains = set(train_data["_domain_group"])
    validation_domains = set(validation_data["_domain_group"])
    test_domains = set(test_data["_domain_group"])

    assert train_domains.isdisjoint(validation_domains)
    assert train_domains.isdisjoint(test_domains)
    assert validation_domains.isdisjoint(test_domains)

    def finalize(split_data: pd.DataFrame) -> pd.DataFrame:
        return split_data.drop(columns="_domain_group").reset_index(drop=True)

    return (
        finalize(train_data),
        finalize(validation_data),
        finalize(test_data),
    )
