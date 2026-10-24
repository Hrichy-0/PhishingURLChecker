"""Prepare audit requests or extract bounded, inert HTML text from LOCAL shards.

Standalone on Windows: requires pandas/pyarrow already used for training.
No networking, browsing, HTML rendering, JavaScript execution or model fitting.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MAX_SNAPSHOT_BYTES = 1_000_000


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_requests(experiment: Path, cv: Path, destination: Path):
    import pandas as pd

    if destination.exists():
        raise ValueError("Request file exists; choose a new filename")
    manifest = json.loads((experiment / "split_manifest.json").read_text())
    frames = []
    for split in ("train", "validation"):
        path = experiment / f"{split}_rows.parquet"
        if fingerprint(path) != manifest["split_sha256"][split]:
            raise ValueError(f"Changed feature records: {split}")
        frame = pd.read_parquet(
            path,
            columns=[
                "url",
                "domain",
                "html_sha256",
                "is_phishing",
                "date",
            ],
        )
        frame["original_split"] = split
        frame["split_row_index"] = range(len(frame))
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    assignments = pd.read_parquet(cv / "role_assignments.parquet")
    cv_report = json.loads((cv / "cv_results.json").read_text())
    if cv_report["provenance"]["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("CV and feature records differ")
    if len(frame) != len(assignments) or assignments.row_index.to_list() != list(
        range(len(frame))
    ):
        raise ValueError("CV row mapping differs")
    if (
        fingerprint(cv / "role_assignments.parquet")
        != cv_report["output_sha256"]["role_assignments.parquet"]
    ):
        raise ValueError("CV assignment checksum mismatch")
    frame["component_id"] = assignments.component_id
    hashes = (
        frame.loc[frame.html_sha256.ne("")].groupby("html_sha256").is_phishing.nunique()
    )
    mixed = set(hashes.loc[hashes.gt(1)].index)
    selected = {
        index: ["mixed_label_exact_content"]
        for index in frame.index[frame.html_sha256.isin(mixed)]
    }
    largest = int(assignments.component_id.value_counts().index[0])
    group = frame.loc[frame.component_id.eq(largest)].copy()
    group["sampling_key"] = group.url.map(
        lambda url: hashlib.sha256(url.encode()).hexdigest()
    )
    # Diagnostic sampling, deliberately label-stratified; NOT an accuracy cohort.
    for label in (0, 1):
        sample = (
            group.loc[group.is_phishing.eq(label)]
            .sort_values("sampling_key")
            .drop_duplicates("domain")
            .head(6)
        )
        for index in sample.index:
            selected.setdefault(index, []).append("largest_component_representative")
    records = []
    for index in sorted(selected):
        row = frame.loc[index]
        records.append(
            {
                "request_id": f"record-{index:06d}",
                "development_row_index": index,
                "url": row.url,
                "expected_label": int(row.is_phishing),
                "expected_html_sha256": row.html_sha256,
                "expected_date": row.date,
                "domain": row.domain,
                "component_id": int(row.component_id),
                "original_split": row.original_split,
                "split_row_index": int(row.split_row_index),
                "reasons": selected[index],
            }
        )
    request = {
        "purpose": "Content/label audit only, not training or final evaluation",
        "dataset_revision": manifest["provenance"]["dataset_revision"],
        "split_manifest_sha256": fingerprint(experiment / "split_manifest.json"),
        "cv_roles_sha256": fingerprint(cv / "role_assignments.parquet"),
        "mixed_label_content_hashes": len(mixed),
        "largest_component_id": largest,
        "sampling": (
            "All mixed-label hash rows plus up to 6 distinct domains per class "
            "from largest component; deterministic URL-SHA ordering"
        ),
        "sources": [
            {"filename": shard["source"], "sha256": shard["key"]["source_sha256"]}
            for shard in manifest["provenance"]["shards"]
        ],
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(request, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    snapshots = len({r["expected_html_sha256"] for r in records})
    print(
        f"Prepared {len(records)} record requests; {snapshots} distinct snapshots",
        flush=True,
    )
    return request


def extract_bundle(request_path: Path, dataset: Path, output: Path, *, resume=False):
    if output.exists() and not resume:
        raise ValueError("Output exists; use a new directory, never overwrite a review")
    if resume and not (output / "review_manifest.json").is_file():
        raise ValueError("Resume requires the existing review_manifest.json")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    records = request["records"]
    if not records or len(records) > 200:
        raise ValueError("Require 1-200 requested records")
    if len({r["request_id"] for r in records}) != len(records):
        raise ValueError("Duplicate request IDs")
    wanted = {}
    for row in records:
        if not re.fullmatch(r"[a-f0-9]{64}", row["expected_html_sha256"]):
            raise ValueError("Invalid requested content hash")
        wanted.setdefault(row["url"], []).append(row)
    dataset = dataset.resolve()
    sources = []
    for source in request["sources"]:
        if not re.fullmatch(r"data/(train|test)-\d+\.parquet", source["filename"]):
            raise ValueError("Unsafe or unexpected source path")
        path = (dataset / source["filename"]).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError("Source must stay inside dataset directory")
        if not path.is_file():
            raise FileNotFoundError(f"Missing original shard: {path}")
        sources.append((source, path))
    report_path = output / "review_manifest.json"
    report = {
        "format_version": 2,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "request_sha256": fingerprint(request_path),
        "dataset_revision": request["dataset_revision"],
        "snapshot_byte_limit": MAX_SNAPSHOT_BYTES,
        "safety": "Inert UTF-8 .txt only; never rendered, fetched or executed",
        "requested_records": len(records),
        "sources_checked": [],
        "completed_sources": [],
        "matches": [],
        "candidate_versions_not_matching": 0,
    }
    if resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["request_sha256"] != fingerprint(request_path):
            raise ValueError(
                "Resume request checksum differs; keep the original requests"
            )
        if (
            json.loads((output / "requests.json").read_text(encoding="utf-8"))
            != request
        ):
            raise ValueError("Saved request contents differ")
        if report.get("snapshot_byte_limit") != MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot size policy differs")
        if report.get("format_version", 1) == 1:
            # V1 saved sources_checked only after an entire shard succeeded.
            report["completed_sources"] = list(report["sources_checked"])
            report["format_version"] = 2
        expected = {row["request_id"]: row for row in records}
        seen = set()
        source_hashes = {source["filename"]: source["sha256"] for source, _ in sources}
        for source in report["completed_sources"]:
            if source_hashes.get(source["filename"]) != source["sha256"]:
                raise ValueError("Completed source identity differs")
        for match in report["matches"]:
            target = expected.get(match["request_id"])
            digest = target["expected_html_sha256"] if target else None
            if (
                target is None
                or match["request_id"] in seen
                or match["full_html_sha256"] != digest
                or match["date"] != target["expected_date"]
                or {"benign": 0, "phish": 1}.get(
                    str(match["raw_label"]).strip().lower()
                )
                != target["expected_label"]
                or match["source"] not in source_hashes
                or match["snapshot_file"] != f"snapshots/{digest}.txt"
            ):
                raise ValueError("Saved match identity differs from request")
            snapshot = output / match["snapshot_file"]
            if (
                not snapshot.is_file()
                or snapshot.stat().st_size != match["saved_bytes"]
                or snapshot.stat().st_size > MAX_SNAPSHOT_BYTES
                or fingerprint(snapshot) != match["snapshot_sha256"]
            ):
                raise ValueError("Saved snapshot failed resume integrity checks")
            seen.add(match["request_id"])
        report["resumed_at"] = datetime.now(UTC).isoformat()
        report["resume_note"] = "Completed-source hashes reused; snapshots reverified"
    else:
        output.mkdir(parents=True)
        (output / "snapshots").mkdir()
        (output / "requests.json").write_text(
            json.dumps(request, indent=2) + "\n", encoding="utf-8"
        )

    def save():
        temporary = output / "review_manifest.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    matched = {match["request_id"] for match in report["matches"]}
    completed = {source["filename"] for source in report["completed_sources"]}
    print(
        f"Starting with {len(matched)}/{len(records)} verified records; "
        f"{len(completed)} completed shards",
        flush=True,
    )
    report["status"] = "running"
    save()

    def process_batch(batch, source, offset):
        # Keep HTML as an Arrow UTF-8 buffer: no full Python string + encode copy.
        for local_index in range(batch.num_rows):
            url = str(batch.column("url")[local_index].as_py() or "").strip()
            targets = [t for t in wanted.get(url, []) if t["request_id"] not in matched]
            if not targets:
                continue
            scalar = batch.column("html")[local_index]
            if not scalar.is_valid:
                report["candidate_versions_not_matching"] += 1
                continue
            payload = scalar.as_buffer()
            digest = hashlib.sha256(payload).hexdigest()
            raw_label = batch.column("label")[local_index].as_py()
            label = {"benign": 0, "phish": 1}.get(str(raw_label or "").strip().lower())
            date = str(batch.column("date")[local_index].as_py() or "")
            for target in targets:
                if (
                    digest != target["expected_html_sha256"]
                    or label != target["expected_label"]
                    or date != target["expected_date"]
                ):
                    report["candidate_versions_not_matching"] += 1
                    continue
                bounded = (
                    payload.slice(0, min(len(payload), MAX_SNAPSHOT_BYTES))
                    .to_pybytes()
                    .decode("utf-8", errors="ignore")
                    .encode("utf-8")
                )
                snapshot = output / "snapshots" / f"{digest}.txt"
                expected_hash = hashlib.sha256(bounded).hexdigest()
                if snapshot.exists():
                    if fingerprint(snapshot) != expected_hash:
                        raise ValueError("Existing snapshot has unexpected contents")
                else:
                    with snapshot.open("xb") as destination:
                        destination.write(bounded)
                report["matches"].append(
                    {
                        "request_id": target["request_id"],
                        "source": source["filename"],
                        "source_row_index": offset + local_index,
                        "raw_label": raw_label,
                        "date": date,
                        "full_html_sha256": digest,
                        "snapshot_file": f"snapshots/{digest}.txt",
                        "snapshot_sha256": expected_hash,
                        "original_html_bytes": len(payload),
                        "saved_bytes": len(bounded),
                        "truncated": len(bounded) < len(payload),
                    }
                )
                matched.add(target["request_id"])
                save()  # Preserve progress even if the next Arrow allocation fails.

    for number, (source, path) in enumerate(sources, 1):
        if len(matched) == len(records):
            break
        if source["filename"] in completed:
            continue
        gc.collect()
        pa.default_memory_pool().release_unused()
        if fingerprint(path) != source["sha256"]:
            report["status"] = "source_checksum_failed"
            save()
            raise ValueError(f"Original shard checksum mismatch: {path.name}")
        if source not in report["sources_checked"]:
            report["sources_checked"].append(source)
        try:
            stream = path.open("rb")
            parquet = pq.ParquetFile(stream, pre_buffer=False, buffer_size=0)
            if not {"url", "label", "date", "html"}.issubset(
                parquet.schema_arrow.names
            ):
                raise ValueError(f"Not a raw URL/HTML shard: {path.name}")
            offset = 0
            for row_group in range(parquet.num_row_groups):
                # Read only cheap URL columns first; HTML only for matching groups.
                urls = (
                    parquet.read_row_group(
                        row_group, columns=["url"], use_threads=False
                    )
                    .column("url")
                    .to_pylist()
                )
                remaining_urls = {
                    url
                    for url, targets in wanted.items()
                    if any(t["request_id"] not in matched for t in targets)
                }
                if any(str(url or "").strip() in remaining_urls for url in urls):
                    batch_offset = 0
                    for batch in parquet.iter_batches(
                        batch_size=1,
                        row_groups=[row_group],
                        columns=["url", "label", "date", "html"],
                        use_threads=False,
                    ):
                        process_batch(batch, source, offset + batch_offset)
                        batch_offset += batch.num_rows
                        del batch
                offset += parquet.metadata.row_group(row_group).num_rows
            report["completed_sources"].append(source)
            completed.add(source["filename"])
        except MemoryError:
            report["status"] = "memory_error_resumable"
            report["failed_source"] = source["filename"]
            save()
            raise
        finally:
            if "parquet" in locals():
                parquet.close()
                del parquet
            if "stream" in locals():
                stream.close()
                del stream
        print(
            f"Checked {number}/{len(sources)} shards; "
            f"matched {len(matched)}/{len(records)} records",
            flush=True,
        )
        save()
        if len(matched) == len(records):
            break
    report["missing_request_ids"] = [
        r["request_id"] for r in records if r["request_id"] not in matched
    ]
    report["status"] = (
        "complete"
        if not report["missing_request_ids"]
        else "incomplete_missing_records"
    )
    save()
    print(f"{report['status']}: {output / 'review_manifest.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--cv", type=Path)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify saved progress and resume an interrupted export",
    )
    args = parser.parse_args()
    if args.prepare:
        if args.experiment is None or args.cv is None:
            parser.error("Prepare mode requires --experiment and --cv")
        prepare_requests(args.experiment, args.cv, args.requests)
    else:
        if args.dataset_dir is None or args.output_dir is None:
            parser.error("Extraction requires --dataset-dir and --output-dir")
        report = extract_bundle(
            args.requests, args.dataset_dir, args.output_dir, resume=args.resume
        )
        if report["status"] != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
