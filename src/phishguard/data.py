import pandas as pd

REQUIRED_COLUMNS = {"URL", "label"}
EXPECTED_LABELS = {0, 1}


def clean_url_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the raw PhiUSIIL data for URL-only model training."""

    missing_columns = REQUIRED_COLUMNS - set(raw_data.columns)

    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")

    data = raw_data.loc[:, ["URL", "label"]].copy()

    report = {
        "rows_before": len(data),
        "missing_urls_removed": int(data["URL"].isna().sum()),
        "missing_labels_removed": int(data["label"].isna().sum()),
    }

    data = data.dropna(subset=["URL", "label"])

    data["URL"] = data["URL"].astype("string").str.strip()

    blank_url_mask = data["URL"].eq("")
    report["blank_urls_removed"] = int(blank_url_mask.sum())
    data = data.loc[~blank_url_mask].copy()

    actual_labels = set(data["label"].unique())
    unexpected_labels = actual_labels - EXPECTED_LABELS

    if unexpected_labels:
        raise ValueError(f"Unexpected labels were found: {sorted(unexpected_labels)}")

    label_counts_per_url = data.groupby("URL")["label"].nunique()
    conflicting_urls = label_counts_per_url[label_counts_per_url > 1]

    if not conflicting_urls.empty:
        raise ValueError(f"Found {len(conflicting_urls)} URLs with conflicting labels")

    duplicate_url_mask = data.duplicated(subset=["URL"], keep="first")
    report["duplicate_urls_removed"] = int(duplicate_url_mask.sum())

    data = data.loc[~duplicate_url_mask].copy()

    # PhiUSIIL uses 0 = phishing and 1 = legitimate.
    # Our project uses 1 = phishing and 0 = legitimate.
    data["is_phishing"] = (1 - data["label"]).astype("int8")

    data = (
        data.rename(columns={"URL": "url"}).drop(columns="label").reset_index(drop=True)
    )

    report["rows_after"] = len(data)

    assert data["url"].notna().all()
    assert not data["url"].duplicated().any()
    assert set(data["is_phishing"].unique()) <= {0, 1}

    return data, report
