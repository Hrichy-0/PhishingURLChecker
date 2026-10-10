"""Batch extraction and leakage checks for local, saved URL/HTML pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from phishguard.data import combine_sources, is_valid_http_url
from phishguard.domains import get_registered_domain
from phishguard.html_features import (
    CONTENT_FEATURE_NAMES,
    MAX_HTML_BYTES,
    extract_content_features,
)
from phishguard.lexical import extract_url_features
from phishguard.native import TRAINING_FEATURES

HTML_FEATURE_NAMES = CONTENT_FEATURE_NAMES + (
    "html_available",
    "html_truncated",
    "html_parse_error",
)
HYBRID_FEATURE_NAMES = TRAINING_FEATURES + HTML_FEATURE_NAMES
VARIANT_FEATURES = {
    "url_only": TRAINING_FEATURES,
    "html_features": HTML_FEATURE_NAMES,
    "url_html": HYBRID_FEATURE_NAMES,
}


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_features(
    url: str, html: str | None, max_html_bytes: int = MAX_HTML_BYTES
) -> dict[str, float]:
    """Retain unknowns as NaN, and parse at most a bounded UTF-8 prefix."""
    if not is_valid_http_url(url):
        raise ValueError("An absolute HTTP(S) URL is required")
    if not 1 <= max_html_bytes <= MAX_HTML_BYTES:
        raise ValueError(f"HTML byte limit must be between 1 and {MAX_HTML_BYTES}")
    values = dict.fromkeys(CONTENT_FEATURE_NAMES, float("nan"))
    values.update(html_available=0.0, html_truncated=0.0, html_parse_error=0.0)
    if not isinstance(html, str) or not html.strip():
        return values
    # Limit encoding work as well as parsing work. A Unicode character uses >=1 byte.
    prefix = html[:max_html_bytes].encode("utf-8", errors="replace")
    bounded = prefix[:max_html_bytes].decode("utf-8", errors="ignore")
    values["html_available"] = 1.0
    values["html_truncated"] = float(len(bounded) < len(html))
    try:
        if not bounded.strip():
            raise ValueError("No usable content within the parse limit")
        values.update(extract_content_features(url, bounded))
    except (ValueError, AssertionError):
        values["html_parse_error"] = 1.0
    return values


def snapshot_vector(
    url: str,
    html: str | None,
    feature_names: tuple[str, ...] = HYBRID_FEATURE_NAMES,
    max_html_bytes: int = MAX_HTML_BYTES,
) -> np.ndarray:
    """Use exactly the training feature order for later native-model inference."""
    if tuple(feature_names) not in VARIANT_FEATURES.values():
        raise ValueError("Unrecognized model feature schema")
    values = {
        **extract_url_features(url),
        **snapshot_features(url, html, max_html_bytes),
    }
    return np.asarray([[values[name] for name in feature_names]], dtype=np.float32)


def _extract_shard(
    path: Path, batch_rows: int, max_html_bytes: int
) -> tuple[pd.DataFrame, dict]:
    rows = []
    counters: Counter = Counter()
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        columns = ["url", "label", "date", "html"]
        if not set(columns).issubset(parquet.schema_arrow.names):
            raise ValueError(f"Missing URL/label/date/HTML columns in {path}")
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            for record in batch.to_pylist():
                counters["rows_read"] += 1
                url = str(record["url"] or "").strip()
                label = str(record["label"] or "").strip().lower()
                if label not in {"benign", "phish"}:
                    raise ValueError(f"Unknown label in {path.name}: {label!r}")
                if not is_valid_http_url(url):
                    counters["invalid_url_rows"] += 1
                    continue
                html = record["html"]
                html_values = snapshot_features(url, html, max_html_bytes)
                label_value = int(label == "phish")
                for flag in ("html_available", "html_truncated", "html_parse_error"):
                    counters[f"{flag}_class_{label_value}"] += int(html_values[flag])
                lexical = extract_url_features(url)
                rows.append(
                    {
                        "url": url,
                        "domain": get_registered_domain(url),
                        "is_phishing": label_value,
                        "date": str(record["date"] or ""),
                        # Hash actual content; publisher annotations are not features.
                        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()
                        if isinstance(html, str) and html.strip()
                        else "",
                        **{name: float(lexical[name]) for name in TRAINING_FEATURES},
                        **html_values,
                    }
                )
    counters["rows_retained"] = len(rows)
    return pd.DataFrame(rows), dict(counters)


def prepare_snapshot_cache(
    dataset_dir: Path,
    cache_dir: Path,
    *,
    batch_rows: int = 16,
    max_html_bytes: int = MAX_HTML_BYTES,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Resume verified per-shard feature caches; raw HTML is not copied into them."""
    if not 1 <= batch_rows <= 256 or not 1 <= max_html_bytes <= MAX_HTML_BYTES:
        raise ValueError("Invalid batch size or HTML byte limit")
    dataset_dir = dataset_dir.resolve()
    manifest = json.loads((dataset_dir / "download_manifest.json").read_text())
    if not manifest.get("complete_dataset"):
        raise ValueError("Finish the full dataset download before HTML training")
    cache_dir.mkdir(parents=True, exist_ok=True)
    extractor_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("html_features.py").read_bytes()
        + Path(__file__).with_name("lexical.py").read_bytes()
        + Path(__file__).with_name("domains.py").read_bytes()
    ).hexdigest()
    frames: dict[str, list] = {"train": [], "test": []}
    reports = []
    for index, entry in enumerate(manifest["files"], 1):
        relative = Path(entry["filename"])
        source = (dataset_dir / relative).resolve()
        if not source.is_relative_to(dataset_dir) or relative.suffix != ".parquet":
            raise ValueError("Manifest file must be a Parquet inside dataset directory")
        split = relative.name.split("-", 1)[0]
        if split not in frames:
            raise ValueError(f"Unexpected dataset split: {split}")
        if source.stat().st_size != entry["bytes"]:
            raise ValueError(f"Source size mismatch: {relative}")
        actual_hash = fingerprint(source)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Source checksum mismatch: {relative}")
        cache_path = cache_dir / relative.name
        info_path = cache_path.with_suffix(".json")
        key = {
            "source_sha256": actual_hash,
            "extractor_sha256": extractor_hash,
            "max_html_bytes": max_html_bytes,
            "feature_order": list(HYBRID_FEATURE_NAMES),
        }
        try:
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
        except json.JSONDecodeError:
            info = {}
        if (
            info.get("key") == key
            and cache_path.exists()
            and fingerprint(cache_path) == info.get("cache_sha256")
        ):
            with cache_path.open("rb") as stream:
                frame = pq.read_table(stream).to_pandas()
            action = "Reused"
        else:
            frame, cleaning = _extract_shard(source, batch_rows, max_html_bytes)
            temporary = cache_path.with_suffix(".parquet.tmp")
            with temporary.open("wb") as stream:
                pq.write_table(
                    pa.Table.from_pandas(frame, preserve_index=False), stream
                )
            temporary.replace(cache_path)
            info = {
                "key": key,
                "cleaning": cleaning,
                "cache_sha256": fingerprint(cache_path),
            }
            info_temporary = info_path.with_suffix(".json.tmp")
            info_temporary.write_text(json.dumps(info, indent=2) + "\n")
            info_temporary.replace(info_path)
            action = "Extracted"
        if not frame.empty:
            frames[split].append(frame)
        reports.append({"source": entry["filename"], **info})
        print(
            f"{action} {index}/{len(manifest['files'])}: {relative.name}, "
            f"{len(frame):,} rows",
            flush=True,
        )
    result, cleaning_report = {}, {}
    for split, parts in frames.items():
        if not parts:
            raise ValueError(f"No usable rows in publisher {split} split")
        result[split], cleaning_report[split] = combine_sources(parts)
        if set(result[split]["is_phishing"].unique()) != {0, 1}:
            raise ValueError(f"Both classes required in publisher {split}")
    return result, {
        "dataset_revision": manifest["revision"],
        "dataset_manifest_sha256": fingerprint(dataset_dir / "download_manifest.json"),
        "max_html_bytes": max_html_bytes,
        "shards": reports,
        "deduplication": cleaning_report,
        "unused_annotation_columns": ["target", "lang", "lang_score", "sha256"],
        "date_role": "audit metadata, not a predictive feature",
    }


def content_partition(data: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Keep domains AND identical HTML together using connected components."""
    parents = {domain: domain for domain in data["domain"].unique()}

    def root(domain):
        while parents[domain] != domain:
            parents[domain] = parents[parents[domain]]
            domain = parents[domain]
        return domain

    owners = {}
    for domain, digest in data[["domain", "html_sha256"]].itertuples(
        index=False, name=None
    ):
        if digest:
            previous = owners.setdefault(digest, domain)
            first, second = sorted((root(domain), root(previous)))
            parents[second] = first
    groups = {domain: root(domain) for domain in parents}
    buckets = {
        group: int.from_bytes(
            hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big"
        )
        / 2**64
        for group in set(groups.values())
    }
    bucket = data["domain"].map(groups).map(buckets)
    masks = {
        "train": bucket < 0.8,
        "validation": (bucket >= 0.8) & (bucket < 0.9),
        "test": bucket >= 0.9,
    }
    splits = {
        name: data.loc[mask].reset_index(drop=True) for name, mask in masks.items()
    }
    for name, frame in splits.items():
        if set(frame["is_phishing"].unique()) != {0, 1}:
            raise ValueError(f"Both classes required in {name}; inspect HTML groups")
    for name, first in splits.items():
        for other, second in splits.items():
            if name != other:
                if not set(first["domain"]).isdisjoint(second["domain"]):
                    raise ValueError("Domain leakage")
                hashes = set(first["html_sha256"]) - {""}
                if not hashes.isdisjoint(set(second["html_sha256"]) - {""}):
                    raise ValueError("Exact HTML leakage")
    return splits


def purge_publisher_test(
    test: pd.DataFrame, development: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    domains = test["domain"].isin(set(development["domain"]))
    hashes = test["html_sha256"].isin(set(development["html_sha256"]) - {""})
    frame = test.loc[~domains & ~hashes].reset_index(drop=True)
    if set(frame["is_phishing"].unique()) != {0, 1}:
        raise ValueError("No two-class publisher test after domain/HTML exclusions")
    return frame, {
        "rows_before": len(test),
        "domain_overlap_rows": int(domains.sum()),
        "exact_html_overlap_rows": int(hashes.sum()),
        "excluded_rows_union": int((domains | hashes).sum()),
        "rows_evaluated": len(frame),
    }
