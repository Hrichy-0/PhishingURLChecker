"""Offline extraction preserves provenance and never renders untrusted HTML."""

import hashlib
import json
import socket
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.export_review_bundle import MAX_SNAPSHOT_BYTES, extract_bundle, fingerprint


def fixture(tmp_path, html="<script>throw new Error('must not execute')</script>"):
    dataset = tmp_path / "raw"
    source = dataset / "data/train-000.parquet"
    source.parent.mkdir(parents=True)
    records = [
        {
            "url": f"https://site{i}.test/",
            "label": label,
            "date": "2025-01-01",
            "html": html,
        }
        for i, label in enumerate(("benign", "phish"))
    ]
    pq.write_table(pa.Table.from_pylist(records), source, row_group_size=1)
    digest = hashlib.sha256(html.encode()).hexdigest()
    request = {
        "dataset_revision": "test-revision",
        "sources": [
            {"filename": "data/train-000.parquet", "sha256": fingerprint(source)}
        ],
        "records": [
            {
                "request_id": str(i),
                "url": r["url"],
                "expected_label": i,
                "expected_date": r["date"],
                "expected_html_sha256": digest,
            }
            for i, r in enumerate(records)
        ],
    }
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(request))
    return dataset, path, source


def test_exports_only_plain_text_and_deduplicates_without_network(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network permitted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    dataset, request, source = fixture(tmp_path)
    before = fingerprint(source)
    output = tmp_path / "bundle"
    report = extract_bundle(request, dataset, output)
    assert report["status"] == "complete"
    assert len(report["matches"]) == 2
    assert len(list((output / "snapshots").glob("*.txt"))) == 1
    assert not list(output.rglob("*.html"))
    assert report["matches"][0]["source_row_index"] == 0
    assert report["matches"][1]["source_row_index"] == 1
    assert fingerprint(source) == before
    with pytest.raises(ValueError, match="Output exists"):
        extract_bundle(request, dataset, output)


def test_bounds_utf8_and_records_truncation(tmp_path):
    dataset, request, _ = fixture(tmp_path, html="é" * (MAX_SNAPSHOT_BYTES + 1))
    output = tmp_path / "bundle"
    report = extract_bundle(request, dataset, output)
    record = report["matches"][0]
    assert record["truncated"] is True
    assert record["saved_bytes"] <= MAX_SNAPSHOT_BYTES
    snapshot = output / record["snapshot_file"]
    snapshot.read_text(encoding="utf-8")
    assert fingerprint(snapshot) == record["snapshot_sha256"]


def test_rejects_wrong_source_identity_and_unsafe_paths(tmp_path):
    dataset, path, _ = fixture(tmp_path)
    request = json.loads(path.read_text())
    request["sources"][0]["filename"] = "../outside.parquet"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(path, dataset, tmp_path / "unsafe")
    request["sources"][0] = {"filename": "data/train-000.parquet", "sha256": "0" * 64}
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="checksum mismatch"):
        extract_bundle(path, dataset, tmp_path / "wrong-source")


def test_marks_wrong_snapshot_as_incomplete_instead_of_substituting(tmp_path):
    dataset, path, _ = fixture(tmp_path)
    request = json.loads(path.read_text())
    request["records"][0]["expected_html_sha256"] = "0" * 64
    path.write_text(json.dumps(request))
    report = extract_bundle(path, dataset, tmp_path / "partial")
    assert report["status"] == "incomplete_missing_records"
    assert report["missing_request_ids"] == ["0"]


def test_memory_failure_checkpoints_and_resume_finishes(tmp_path, monkeypatch):
    dataset, request, _ = fixture(tmp_path)
    original = pq.ParquetFile

    class FailingReader:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def iter_batches(self, **kwargs):
            assert kwargs["batch_size"] == 1
            assert kwargs["use_threads"] is False
            if kwargs["row_groups"] == [1]:
                raise MemoryError("simulated Arrow allocation failure")
            return self.reader.iter_batches(**kwargs)

    monkeypatch.setattr(pq, "ParquetFile", FailingReader)
    output = tmp_path / "bundle"
    with pytest.raises(MemoryError):
        extract_bundle(request, dataset, output)
    checkpoint = json.loads((output / "review_manifest.json").read_text())
    assert len(checkpoint["matches"]) == 1
    assert checkpoint["completed_sources"] == []
    assert checkpoint["status"] == "memory_error_resumable"
    monkeypatch.setattr(pq, "ParquetFile", original)
    result = extract_bundle(request, dataset, output, resume=True)
    assert result["status"] == "complete"
    assert len(result["matches"]) == 2
    assert len({m["request_id"] for m in result["matches"]}) == 2


def test_resume_migrates_original_manifest_and_skips_completed_shards(
    tmp_path, monkeypatch
):
    dataset, request, first_source = fixture(tmp_path)
    rows = pq.read_table(first_source).to_pylist()
    second_source = dataset / "data/train-001.parquet"
    pq.write_table(pa.Table.from_pylist(rows[:1]), first_source)
    pq.write_table(pa.Table.from_pylist(rows[1:]), second_source)
    inputs = json.loads(request.read_text())
    inputs["sources"] = [
        {"filename": f"data/{path.name}", "sha256": fingerprint(path)}
        for path in (first_source, second_source)
    ]
    request.write_text(json.dumps(inputs))
    output = tmp_path / "bundle"
    extract_bundle(request, dataset, output)
    path = output / "review_manifest.json"
    legacy = json.loads(path.read_text())
    legacy.pop("format_version")
    legacy.pop("completed_sources")
    legacy["status"] = "running"
    legacy["matches"] = legacy["matches"][:1]
    legacy["sources_checked"] = legacy["sources_checked"][:1]
    path.write_text(json.dumps(legacy))
    original = pq.ParquetFile

    def only_remaining(stream, **kwargs):
        assert Path(stream.name).name == second_source.name
        return original(stream, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", only_remaining)
    result = extract_bundle(request, dataset, output, resume=True)
    assert result["status"] == "complete"
    assert len(result["completed_sources"]) == 2
    assert len(result["matches"]) == 2


def test_resume_rejects_damaged_snapshots_and_changed_requests(tmp_path):
    dataset, request, _ = fixture(tmp_path)
    output = tmp_path / "bundle"
    result = extract_bundle(request, dataset, output)
    snapshot = output / result["matches"][0]["snapshot_file"]
    snapshot.write_text("corrupted")
    with pytest.raises(ValueError, match="snapshot failed"):
        extract_bundle(request, dataset, output, resume=True)
    request.write_text(request.read_text() + " ")
    with pytest.raises(ValueError, match="request checksum differs"):
        extract_bundle(request, dataset, output, resume=True)
