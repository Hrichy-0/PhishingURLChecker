"""Download every original pinned PhreshPhish Parquet file, including its HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq

from phishguard.config import PROJECT_ROOT

REPO = "phreshphish/phreshphish"
REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(entry: dict, destination: Path, progress) -> dict:
    filename = entry["path"]
    if not re.fullmatch(r"data/(train|test)-\d+\.parquet", filename):
        raise ValueError(f"Unexpected dataset path: {filename}")
    size = entry["size"]
    expected_hash = entry.get("lfs", {}).get("oid")
    if not expected_hash or len(expected_hash) != 64:
        raise ValueError(f"Missing upstream SHA-256: {filename}")
    path = destination / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        if sha256_file(path) == expected_hash:
            progress(filename, size)
            return {"filename": filename, "bytes": size, "sha256": expected_hash}
        raise ValueError(f"Existing file failed checksum: {path}")
    temporary = path.with_suffix(".parquet.part")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}"
    for attempt in range(8):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > size:
            raise ValueError(f"Partial file exceeds expected size: {temporary}")
        if offset == size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if offset and response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Wrong resumed range: {filename}")
                elif offset and response.status == 200:
                    offset = 0
                elif offset:
                    raise ValueError(f"Unsupported resume status: {response.status}")
                with temporary.open("ab" if offset else "wb") as stream:
                    written = offset
                    progress(filename, written)
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        written += len(chunk)
                        if written > size:
                            raise ValueError(
                                f"Response exceeded expected size: {filename}"
                            )
                        progress(filename, written)
            if temporary.stat().st_size == size:
                break
        except (OSError, TimeoutError) as error:
            if attempt == 7:
                raise
            print(f"Retrying {filename}: {type(error).__name__}", flush=True)
            time.sleep(min(2**attempt, 20))
    if temporary.stat().st_size != size or sha256_file(temporary) != expected_hash:
        raise ValueError(
            f"Downloaded file failed size/checksum verification: {filename}"
        )
    temporary.replace(path)
    return {"filename": filename, "bytes": size, "sha256": expected_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/raw/phreshphish"
    )
    parser.add_argument(
        "--url-export-dir",
        type=Path,
        default=PROJECT_ROOT / "data/external/huggingface/phreshphish_full",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.url_export_dir.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data"
    with urlopen(api, timeout=60) as response:
        entries = [entry for entry in json.load(response) if entry["type"] == "file"]
        if response.headers.get("Link"):
            raise ValueError(
                "Dataset listing is paginated; cannot claim a full download"
            )
    if not entries or any(
        not re.fullmatch(r"data/(train|test)-\d+\.parquet", entry["path"])
        for entry in entries
    ):
        raise ValueError("Dataset file listing is missing or unexpected")
    entries.sort(key=lambda entry: entry["path"])
    total = sum(entry["size"] for entry in entries)
    free = shutil.disk_usage(args.output_dir).free
    remaining = sum(
        max(
            0,
            entry["size"]
            - min(
                entry["size"],
                (
                    (args.output_dir / entry["path"]).stat().st_size
                    if (args.output_dir / entry["path"]).exists()
                    else 0
                ),
            ),
        )
        for entry in entries
    )
    if free < remaining + 5 * 1024**3:
        raise ValueError("Insufficient disk space for the complete dataset and exports")
    print(
        f"Full download: {len(entries)} original files, {total / 1e9:.2f} GB.",
        flush=True,
    )
    print(f"Local destination: {args.output_dir.resolve()}", flush=True)
    lock = threading.Lock()
    counts = {}
    last_report = [0.0]
    status_path = args.output_dir / "download_status.json"

    def progress(filename: str, count: int) -> None:
        with lock:
            counts[filename] = count
            now = time.monotonic()
            if now - last_report[0] < 15:
                return
            last_report[0] = now
            downloaded = sum(counts.values())
            status = {
                "status": "downloading",
                "bytes_downloaded": downloaded,
                "total_bytes": total,
                "percent": 100 * downloaded / total,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            print(
                f"Downloaded {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB "
                f"({status['percent']:.1f}%)",
                flush=True,
            )

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_file, entry, args.output_dir, progress)
            for entry in entries
        ]
        for future in as_completed(futures):
            records.append(future.result())
            print(f"Verified {len(records)}/{len(entries)} original files.", flush=True)
    exports = {}
    for split in ("train", "test"):
        selected = [e for e in entries if e["path"].startswith(f"data/{split}-")]
        frames = [
            pq.read_table(
                args.output_dir / e["path"], columns=["url", "label", "date"]
            ).to_pandas()
            for e in selected
        ]
        frame = pd.concat(frames, ignore_index=True)
        path = args.url_export_dir / f"{split}.csv"
        frame.to_csv(path, index=False)
        exports[split] = {
            "rows": len(frame),
            "sha256": sha256_file(path),
            "label_counts": frame["label"].value_counts().to_dict(),
        }
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "complete_dataset": True,
        "total_bytes": total,
        "local_full_dataset": str(args.output_dir.resolve()),
        "license": "CC BY 4.0; anti-phishing research (publisher card)",
        "files": sorted(records, key=lambda record: record["filename"]),
        "exports": exports,
        "training_policy": (
            "URL-only; URL/label/date exported from complete local files"
        ),
    }
    for directory in (args.output_dir, args.url_export_dir):
        (directory / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "percent": 100,
                "total_bytes": total,
                "files": len(records),
            },
            indent=2,
        )
    )
    print(
        "Complete dataset downloaded and every file verified against upstream SHA-256.",
        flush=True,
    )
    print(f"URL-only training exports: {exports}", flush=True)


if __name__ == "__main__":
    main()
