"""Download the selected Kaggle dataset into data/raw.

Kaggle may require credentials. See https://github.com/Kaggle/kagglehub for
authentication options.
"""

from pathlib import Path
from shutil import copy2

import kagglehub

from phishguard.config import KAGGLE_DATASET, RAW_DATA_DIR


def main() -> None:
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = sorted(cache_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in Kaggle download: {cache_dir}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for source in csv_files:
        destination = RAW_DATA_DIR / source.name
        copy2(source, destination)
        print(f"Copied {source.name} -> {destination}")


if __name__ == "__main__":
    main()
