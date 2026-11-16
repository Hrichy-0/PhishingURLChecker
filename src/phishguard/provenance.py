"""File checksums that prove inputs are the recorded ones and outputs were not touched.

Checksums establish a local provenance chain, not protection against a malicious
editor. ``native_html.fingerprint`` and ``evaluate_frozen_url.sha`` keep their own
copies: one is part of the HTML feature-cache key and the other is frozen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(paths, *, missing_ok: bool = False) -> dict[str, str]:
    """Checksum each file before a run; record the result as provenance."""
    return {
        str(path): sha256_file(path)
        for path in map(Path, paths)
        if not missing_ok or path.exists()
    }


def assert_unchanged(before: dict[str, str]) -> None:
    """Fail the run if any protected file changed or disappeared since ``snapshot``."""
    changed = [
        path
        for path, digest in before.items()
        if not Path(path).exists() or sha256_file(path) != digest
    ]
    if changed:
        raise RuntimeError(f"Protected artifacts changed during the run: {changed}")


def verify_split_files(directory: Path, manifest: dict, names) -> dict[str, str]:
    """Check ``<name>_rows.parquet`` files against the manifest's ``split_sha256``."""
    hashes = {}
    for name in names:
        hashes[name] = sha256_file(Path(directory) / f"{name}_rows.parquet")
        if hashes[name] != manifest["split_sha256"][name]:
            raise ValueError(f"Split checksum mismatch: {name}")
    return hashes
