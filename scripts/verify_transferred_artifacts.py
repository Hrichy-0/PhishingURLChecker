"""Verify pulled models and result files without requiring any datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def ignored_by_git(root: Path, paths: list[str]) -> set[str]:
    """Paths deliberately kept out of the repository, e.g. raw-URL error files."""
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
    )
    return set(result.stdout.split())


def verify(root: Path) -> tuple[int, list[str], list[str]]:
    manifest = json.loads((root / "artifacts/transfer_manifest.json").read_text())
    missing = [
        entry["path"]
        for entry in manifest["files"]
        if not (root / entry["path"]).is_file()
    ]
    local_only = ignored_by_git(root, missing) if missing else set()
    failures = []
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid manifest path")
        if entry["path"] in local_only:
            continue
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or digest(path) != entry["sha256"]
        ):
            failures.append(entry["path"])
    return len(manifest["files"]), sorted(local_only), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    total, local_only, failures = verify(root)
    if failures:
        raise SystemExit("Missing or changed artifacts: " + ", ".join(failures))
    note = f"; {len(local_only)} local-only files not present" if local_only else ""
    print(f"Verified {total - len(local_only)} model/result files{note}.")


if __name__ == "__main__":
    main()
