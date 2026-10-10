"""Verify pulled models and result files without requiring any datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "artifacts/transfer_manifest.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid manifest path")
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or digest(path) != entry["sha256"]
        ):
            failures.append(entry["path"])
    if failures:
        raise SystemExit("Missing or changed artifacts: " + ", ".join(failures))
    print(
        f"Verified {len(manifest['files'])} model/result files; dataset not required."
    )


if __name__ == "__main__":
    main()
