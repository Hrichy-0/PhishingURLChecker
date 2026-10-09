"""Download version-pinned URL data, never repository code or listed websites."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from phishguard.config import PROJECT_ROOT

SOURCES = {
    "ealvaradob": {
        "repo": "ealvaradob/phishing-dataset",
        "revision": "55caf9af01ae498d68ff44886ff69a4b165eb54b",
        "files": ["urls.json"],
        "url_column": "text",
        "label_column": "label",
        "license": "apache-2.0 (publisher card; sources include Kaggle compilation)",
    },
    "phishnchips": {
        "repo": "AreLit/PhishNChips",
        "revision": "89afcc39610084298c4679159cb2e27d9ffffa46",
        "files": ["core_emails.csv"],
        "url_column": "url_raw",
        "label_column": "phish_label",
        "license": "other (mixed third-party sources; see SOURCE_LICENSES.md)",
    },
}


def extract_url_columns(
    payload: bytes, filename: str, url_column: str, label_column: str
) -> pd.DataFrame:
    """Read data-only formats and select URL/label fields, not other features."""
    stream = io.BytesIO(payload)
    if filename.endswith(".csv"):
        frame = pd.read_csv(stream, usecols=[url_column, label_column])
    elif filename.endswith(".json"):
        data = json.loads(payload)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of URL records")
        frame = pd.DataFrame(data).loc[:, [url_column, label_column]]
    else:
        raise ValueError("Only CSV and JSON data files are supported")
    return frame.rename(columns={url_column: "url", label_column: "label"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(SOURCES), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/external/huggingface",
    )
    args = parser.parse_args()
    spec = SOURCES[args.dataset]
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, entries = [], []
    for filename in spec["files"]:
        url = (
            f"https://huggingface.co/datasets/{spec['repo']}/"
            f"resolve/{spec['revision']}/{filename}"
        )
        print(f"Downloading pinned data file: {filename}", flush=True)
        with urlopen(url, timeout=60) as response:
            payload = response.read(100 * 1024 * 1024 + 1)
        if len(payload) > 100 * 1024 * 1024:
            raise ValueError("File exceeds the 100 MiB download limit")
        frame = extract_url_columns(
            payload, filename, spec["url_column"], spec["label_column"]
        )
        frames.append(frame)
        entries.append(
            {
                "filename": filename,
                "download_url": url,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": len(frame),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    destination = output_dir / "urls.csv"
    combined.to_csv(destination, index=False)
    manifest = {
        **spec,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "files": entries,
        "export_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "rows": len(combined),
        "label_counts": {
            str(key): int(count)
            for key, count in combined["label"].value_counts(dropna=False).items()
        },
        "selection": "All rows from the listed files; URL and label columns only",
        "protocol_policy": "Preserved unchanged; no scheme inference",
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved {len(combined):,} URL rows to {destination}")
    print(f"Source label counts: {manifest['label_counts']}")


if __name__ == "__main__":
    main()
