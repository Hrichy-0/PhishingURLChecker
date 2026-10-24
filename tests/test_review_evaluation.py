"""Safe bundle input and training-feature reconstruction safeguards."""

import hashlib
import json
import socket
import zipfile

import pytest

from phishguard.native_html import HYBRID_FEATURE_NAMES
from scripts.evaluate_review_bundle import matrix_from_snapshots, verified_bundle


def bundle_fixture(tmp_path):
    payload = b"<title>Example</title><form><input type='password'></form>"
    digest = hashlib.sha256(payload).hexdigest()
    source = {"filename": "data/train-000.parquet", "sha256": "f" * 64}
    request = {
        "dataset_revision": "fixture",
        "sources": [source],
        "records": [
            {
                "request_id": "one",
                "url": "https://example.test/login",
                "expected_label": 1,
                "expected_date": "2025-01-01",
                "expected_html_sha256": digest,
            }
        ],
    }
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(request))
    report = {
        "status": "complete",
        "missing_request_ids": [],
        "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_revision": "fixture",
        "snapshot_byte_limit": 1000000,
        "requested_records": 1,
        "sources_checked": [source],
        "matches": [
            {
                "request_id": "one",
                "full_html_sha256": digest,
                "snapshot_file": f"snapshots/{digest}.txt",
                "date": "2025-01-01",
                "raw_label": "phish",
                "source": source["filename"],
                "saved_bytes": len(payload),
                "original_html_bytes": len(payload),
                "snapshot_sha256": digest,
                "truncated": False,
            }
        ],
    }
    members = {
        "bundle\\requests.json": json.dumps(request).encode(),
        "bundle\\review_manifest.json": json.dumps(report).encode(),
        f"bundle\\snapshots\\{digest}.txt": payload,
    }
    return path, members


def write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_reads_windows_bundle_without_network_or_extraction(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network in replay")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    request, members = bundle_fixture(tmp_path)
    path = tmp_path / "bundle.zip"
    write_zip(path, members)
    _, _, records, snapshots = verified_bundle(path, request)
    matrix = matrix_from_snapshots(records, snapshots)
    assert matrix.shape == (1, 38)
    assert matrix[0, HYBRID_FEATURE_NAMES.index("password_field_count")] == 1
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("case", ["path", "payload", "label"])
def test_rejects_unsafe_paths_modified_payload_and_wrong_labels(tmp_path, case):
    request, members = bundle_fixture(tmp_path)
    if case == "path":
        members["../escape.txt"] = b"bad"
    elif case == "payload":
        key = next(k for k in members if k.endswith(".txt"))
        members[key] = b"modified"
    else:
        report = json.loads(members["bundle\\review_manifest.json"])
        report["matches"][0]["raw_label"] = "benign"
        members["bundle\\review_manifest.json"] = json.dumps(report).encode()
    path = tmp_path / "bundle.zip"
    write_zip(path, members)
    with pytest.raises(ValueError):
        verified_bundle(path, request)


def test_reconstruction_restores_original_truncation_flag():
    request = {"url": "https://example.test/", "expected_html_sha256": "digest"}
    values = matrix_from_snapshots(
        [(request, {"truncated": True})], {"digest": "<title>Bounded prefix</title>"}
    )
    assert values[0, HYBRID_FEATURE_NAMES.index("html_truncated")] == 1
