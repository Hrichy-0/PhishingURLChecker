"""Manifest verification tolerates only git-ignored files that are absent."""

import hashlib
import json
import subprocess

from scripts.verify_transferred_artifacts import verify


def test_missing_ignored_files_are_skipped_but_others_fail(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("artifacts/**/*.csv\n")
    (tmp_path / "artifacts").mkdir()
    model = tmp_path / "artifacts" / "model.ubj"
    model.write_bytes(b"model")
    entries = [
        {
            "path": "artifacts/model.ubj",
            "bytes": 5,
            "sha256": hashlib.sha256(b"model").hexdigest(),
        },
        {"path": "artifacts/errors.csv", "bytes": 1, "sha256": "0" * 64},
    ]
    manifest = tmp_path / "artifacts" / "transfer_manifest.json"
    manifest.write_text(json.dumps({"files": entries}))

    assert verify(tmp_path) == (2, ["artifacts/errors.csv"], [])

    model.unlink()
    assert verify(tmp_path)[2] == ["artifacts/model.ubj"]
