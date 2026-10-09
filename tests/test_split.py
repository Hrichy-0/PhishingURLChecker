import pandas as pd
import pytest

from phishguard.domains import get_registered_domain
from phishguard.split import split_by_domain, validate_training_splits


def make_test_data() -> pd.DataFrame:
    rows = []

    for domain_number in range(40):
        domain = f"site{domain_number}.com"
        label = domain_number % 2

        rows.append(
            {
                "url": f"https://{domain}/",
                "is_phishing": label,
            }
        )

        rows.append(
            {
                "url": f"https://login.{domain}/account",
                "is_phishing": label,
            }
        )

    return pd.DataFrame(rows)


def test_split_by_domain_prevents_domain_overlap() -> None:
    data = make_test_data()

    train_data, validation_data, test_data = split_by_domain(data)

    assert len(train_data) + len(validation_data) + len(test_data) == len(data)

    train_domains = set(train_data["url"].map(get_registered_domain))
    validation_domains = set(validation_data["url"].map(get_registered_domain))
    test_domains = set(test_data["url"].map(get_registered_domain))

    assert train_domains.isdisjoint(validation_domains)
    assert train_domains.isdisjoint(test_domains)
    assert validation_domains.isdisjoint(test_domains)


def test_split_by_domain_is_reproducible() -> None:
    data = make_test_data()

    first_result = split_by_domain(data, random_state=42)
    second_result = split_by_domain(data, random_state=42)

    for first_split, second_split in zip(
        first_result,
        second_result,
        strict=True,
    ):
        pd.testing.assert_frame_equal(first_split, second_split)


def test_training_validation_rejects_domain_leakage() -> None:
    train = pd.DataFrame(
        {
            "url": ["https://same.example.com/", "https://other.test/"],
            "is_phishing": [0, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "url": ["https://same.example.com/about", "https://third.test/"],
            "is_phishing": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="Domain leakage"):
        validate_training_splits({"train": train, "validation": validation})


def test_training_validation_requires_both_labels() -> None:
    train, validation, test = split_by_domain(make_test_data())
    validation["is_phishing"] = 0
    with pytest.raises(ValueError, match="both binary labels"):
        validate_training_splits(
            {"train": train, "validation": validation, "test": test}
        )


def test_split_sizes_must_sum_to_one() -> None:
    data = make_test_data()

    with pytest.raises(ValueError, match="sum to 1.0"):
        split_by_domain(
            data,
            train_size=0.7,
            validation_size=0.1,
            test_size=0.1,
        )
