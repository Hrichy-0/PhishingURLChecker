import pandas as pd

from phishguard.config import RAW_DATA_DIR

csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DATA_DIR} Run scripts/download_data.py first."
    )

if len(csv_files) > 1:
    raise RuntimeError(
        f"Multiple CSV files found in {RAW_DATA_DIR}. "
        "Please ensure only one CSV file is present."
    )

data_path = csv_files[0]

print(f"Reading: {data_path}")
# reading the dataset
data = pd.read_csv(data_path, low_memory=False)

# inspecting dimensions and columns..Also rows
print("\nDATASET SHAPE")
print(data.shape)

print("\nCOLUMN NAMES")
print(data.columns.tolist())

print("\nFIRST FIVE ROWS")
print(data.head())

print("\nCOLUMN INFORMATION")
data.info()

# inspect data types and missing values
print("\nMISSING VALUES BY COLUMN")
missing_values = data.isna().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

print("\nLABEL COUNTS")
label_counts = data["label"].value_counts().sort_index()
print(label_counts)

print("\nLABEL PERCENTAGES")
label_percentages = (
    data["label"].value_counts(normalize=True).sort_index().mul(100).round(2)
)
print(label_percentages)

expected_labels = {0, 1}
actual_labels = set(data["label"].unique())
unexpected_labels = actual_labels - expected_labels

print("\nUNEXPECTED LABELS")
print(unexpected_labels)

# check for blank URLs
blank_urls = data["URL"].str.strip().eq("").sum()

print("\nBLANK URLS")
print(blank_urls)

# check duplicate rows and URLs
duplicate_rows = data.duplicated().sum()
duplicate_urls = data.duplicated(subset=["URL"]).sum()
unique_urls = data["URL"].nunique()

print("\nDUPLICATE INFORMATION")
print(f"Completely duplicate rows: {duplicate_rows}")
print(f"Duplicate URLs beyond the first: {duplicate_urls}")
print(f"Unique URLs: {unique_urls}")

# checking conflicting labels
labels_per_url = data.groupby("URL")["label"].nunique()
conflicting_urls = labels_per_url[labels_per_url > 1]

print("\nURLS WITH CONFLICTING LABELS")
print(f"Number of conflicting URLs: {len(conflicting_urls)}")

# check repeated domains
unique_domains = data["Domain"].nunique()
repeated_domains = data.duplicated(subset=["Domain"]).sum()

print("\nDOMAIN INFORMATION")
print(f"Unique domains: {unique_domains}")
print(f"Repeated domains beyond the first: {repeated_domains}")

# inspect samples from both classes
print("\nPHISHING URL EXAMPLES")
print(
    data.loc[data["label"] == 0, ["URL", "Domain", "label"]]
    .head(5)
    .to_string(index=False)
)

print("\nLEGITIMATE URL EXAMPLES")
print(
    data.loc[data["label"] == 1, ["URL", "Domain", "label"]]
    .head(5)
    .to_string(index=False)
)
