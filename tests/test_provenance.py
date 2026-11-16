"""Protected-artifact guards and split-manifest checks."""

import hashlib

import pytest

from phishguard.provenance import (
    assert_unchanged,
    sha256_file,
    snapshot,
    verify_split_files,
)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"phishguard")
    assert sha256_file(path) == hashlib.sha256(b"phishguard").hexdigest()


def test_guard_passes_when_untouched_and_names_changed_files(tmp_path):
    kept, edited, removed = (tmp_path / name for name in ("kept", "edited", "gone"))
    for path in (kept, edited, removed):
        path.write_text("original")
    before = snapshot([kept, edited, removed])
    assert_unchanged(before)
    edited.write_text("changed")
    removed.unlink()
    with pytest.raises(RuntimeError) as error:
        assert_unchanged(before)
    assert str(edited) in str(error.value) and str(removed) in str(error.value)
    assert str(kept) not in str(error.value)


def test_snapshot_requires_files_unless_missing_ok(tmp_path):
    missing = tmp_path / "missing"
    assert snapshot([missing], missing_ok=True) == {}
    with pytest.raises(FileNotFoundError):
        snapshot([missing])


def test_split_files_must_match_manifest(tmp_path):
    for name in ("train", "validation"):
        (tmp_path / f"{name}_rows.parquet").write_bytes(name.encode())
    manifest = {
        "split_sha256": {
            name: sha256_file(tmp_path / f"{name}_rows.parquet")
            for name in ("train", "validation")
        }
    }
    hashes = verify_split_files(tmp_path, manifest, ("train", "validation"))
    assert hashes == manifest["split_sha256"]
    (tmp_path / "validation_rows.parquet").write_bytes(b"edited")
    with pytest.raises(ValueError, match="mismatch: validation"):
        verify_split_files(tmp_path, manifest, ("train", "validation"))
