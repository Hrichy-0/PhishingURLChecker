"""Extract pinned PhreshPhish URL columns with HTTP ranges, without reading HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import fsspec
import pandas as pd
import pyarrow.parquet as pq

from phishguard.config import PROJECT_ROOT

REPO = "phreshphish/phreshphish"
REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"
COLUMNS = ["url", "label", "date"]


def extract_shard(filename: str, output_dir: Path) -> dict:
    """Use column projection and bounded range caching; preserve source values."""
    if not re.fullmatch(r"data/(train|test)-\d+\.parquet", filename):
        raise ValueError(f"Unexpected shard path: {filename}")
    path = output_dir / "shards" / (Path(filename).stem + ".csv")
    record_path = path.with_suffix(".json")
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if (
            record.get("revision") == REVISION
            and record.get("columns") == COLUMNS
            and record.get("export_sha256")
            == hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            return record
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}"
    filesystem = fsspec.filesystem("http", client_kwargs={"trust_env": True})
    with filesystem.open(
        url, "rb", block_size=256 * 1024, cache_type="bytes"
    ) as stream:
        parquet = pq.ParquetFile(stream)
        frame = parquet.read(columns=COLUMNS, use_threads=False).to_pandas()
        remote_size = stream.size
    temporary = path.with_suffix(".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    record = {
        "revision": REVISION,
        "filename": filename,
        "download_url": url,
        "columns": COLUMNS,
        "remote_file_bytes": remote_size,
        "rows": len(frame),
        "export_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/external/huggingface/phreshphish",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shards").mkdir(exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}"
    with urlopen(api, timeout=60) as response:
        listing = json.load(response)
    filenames = sorted(
        entry["rfilename"]
        for entry in listing["siblings"]
        if re.fullmatch(r"data/(train|test)-\d+\.parquet", entry["rfilename"])
    )
    if not filenames or listing["sha"] != REVISION:
        raise ValueError("Pinned dataset listing is missing or changed")
    print(f"Reading URL/label/date from {len(filenames)} pinned shards...", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(extract_shard, filename, args.output_dir): filename
            for filename in filenames
        }
        for future in as_completed(pending):
            records.append(future.result())
            print(f"Extracted {len(records)}/{len(filenames)} shards", flush=True)
    exports = {}
    for split in ("train", "test"):
        selected = [name for name in filenames if name.startswith(f"data/{split}-")]
        frame = pd.concat(
            [
                pd.read_csv(args.output_dir / "shards" / (Path(name).stem + ".csv"))
                for name in selected
            ],
            ignore_index=True,
        )
        path = args.output_dir / f"{split}.csv"
        frame.to_csv(path, index=False)
        exports[split] = {
            "rows": len(frame),
            "label_counts": frame["label"].value_counts().to_dict(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"Saved {split}: {exports[split]}", flush=True)
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "license": "CC BY 4.0; anti-phishing research only (publisher card)",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "columns": COLUMNS,
        "policy": "Original publisher train/test partitions and protocols preserved",
        "shards": sorted(records, key=lambda item: item["filename"]),
        "exports": exports,
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
