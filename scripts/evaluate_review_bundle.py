"""Offline URL versus URL+HTML diagnostic replay; NOT an external accuracy test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import xgboost as xgb

from phishguard.native import TRAINING_FEATURES, evaluate_probabilities
from phishguard.native_html import HYBRID_FEATURE_NAMES, fingerprint, snapshot_vector

SCHEMAS = {"url_only": TRAINING_FEATURES, "url_html": HYBRID_FEATURE_NAMES}


def verified_bundle(bundle: Path, requests: Path):
    """Read bounded members directly; never extract, render or execute HTML."""
    expected = json.loads(requests.read_text(encoding="utf-8"))
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        if len(infos) > 205 or sum(i.file_size for i in infos) > 32_000_000:
            raise ValueError("Unexpected archive count or expanded size")
        names = {}
        for info in infos:
            raw = info.filename.replace("\\", "/")
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or ":" in raw:
                raise ValueError("Unsafe archive path")
            if info.is_dir():
                continue
            key = str(path)
            if key in names or info.flag_bits & 1:
                raise ValueError("Duplicate or encrypted archive member")
            names[key] = info
        manifests = [n for n in names if n.endswith("/review_manifest.json")]
        if len(manifests) != 1:
            raise ValueError("Require one review manifest")
        prefix = str(PurePosixPath(manifests[0]).parent) + "/"

        def read(name, limit):
            info = names.get(prefix + name)
            if info is None or info.file_size > limit:
                raise ValueError("Missing or oversized archive member")
            with archive.open(info) as stream:
                payload = stream.read(limit + 1)
            if len(payload) > limit:
                raise ValueError("Expanded member exceeds limit")
            return payload

        report = json.loads(read("review_manifest.json", 2_000_000))
        if json.loads(read("requests.json", 2_000_000)) != expected:
            raise ValueError("Bundle requests differ from original")
        if (
            report["status"] != "complete"
            or report.get("missing_request_ids")
            or report["request_sha256"] != fingerprint(requests)
            or report["dataset_revision"] != expected["dataset_revision"]
            or report["snapshot_byte_limit"] != 1_000_000
        ):
            raise ValueError("Incomplete bundle or request identity mismatch")
        targets = {r["request_id"]: r for r in expected["records"]}
        if len(targets) != len(expected["records"]) or report[
            "requested_records"
        ] != len(targets):
            raise ValueError("Duplicate request IDs or wrong record count")
        sources = {s["filename"]: s["sha256"] for s in expected["sources"]}
        checked = {s["filename"]: s["sha256"] for s in report["sources_checked"]}
        if any(sources.get(name) != digest for name, digest in checked.items()):
            raise ValueError("Reported shard checksum differs from request")
        records, seen, snapshots = [], set(), {}
        for match in report["matches"]:
            target = targets.get(match["request_id"])
            if target is None or match["request_id"] in seen:
                raise ValueError("Unexpected or duplicate returned record")
            digest = target["expected_html_sha256"]
            if (
                not re.fullmatch(r"[a-f0-9]{64}", digest)
                or match["full_html_sha256"] != digest
                or match["snapshot_file"] != f"snapshots/{digest}.txt"
                or match["date"] != target["expected_date"]
                or {"benign": 0, "phish": 1}.get(
                    str(match["raw_label"]).strip().lower()
                )
                != target["expected_label"]
                or match["source"] not in checked
            ):
                raise ValueError("Returned record identity differs")
            payload = read(match["snapshot_file"], 1_000_000)
            if (
                len(payload) != match["saved_bytes"]
                or hashlib.sha256(payload).hexdigest() != match["snapshot_sha256"]
                or bool(match["truncated"])
                != (len(payload) < match["original_html_bytes"])
                or (
                    not match["truncated"]
                    and hashlib.sha256(payload).hexdigest() != digest
                )
            ):
                raise ValueError("Snapshot size or checksum mismatch")
            html = payload.decode("utf-8", errors="strict")
            snapshots[digest] = html
            records.append((target, match))
            seen.add(match["request_id"])
        if seen != set(targets):
            raise ValueError("Missing requested records")
        required = {prefix + "requests.json", prefix + "review_manifest.json"}
        required |= {prefix + f"snapshots/{digest}.txt" for digest in snapshots}
        if set(names) != required:
            raise ValueError("Unexpected archive files")
    return expected, report, records, snapshots


def matrix_from_snapshots(records, snapshots):
    rows = []
    for request, match in records:
        values = snapshot_vector(
            request["url"], snapshots[request["expected_html_sha256"]]
        )[0]
        # A bounded exported file cannot reveal that the ORIGINAL page was longer.
        # Restore this known provenance flag; do not silently convert it to zero.
        values[HYBRID_FEATURE_NAMES.index("html_truncated")] = float(match["truncated"])
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def predict(path, names, values, rounds, expected_hash):
    if fingerprint(path) != expected_hash:
        raise ValueError("Model checksum mismatch")
    model = xgb.Booster()
    model.load_model(bytearray(path.read_bytes()))
    model.set_param({"device": "cpu", "nthread": 4})
    if model.feature_names != list(names) or model.num_features() != len(names):
        raise ValueError("Model feature schema differs")
    if not 0 < rounds <= model.num_boosted_rounds():
        raise ValueError("Invalid prediction tree range")
    return model.predict(
        xgb.DMatrix(values, feature_names=list(names), nthread=4),
        iteration_range=(0, rounds),
    )


def evaluate(
    bundle: Path, requests: Path, experiment: Path, cv_root: Path, output: Path
):
    if output.exists():
        raise ValueError("Choose a new report directory")
    expected, transfer, records, snapshots = verified_bundle(bundle, requests)
    matrix = matrix_from_snapshots(records, snapshots)
    split_manifest_path = experiment / "split_manifest.json"
    if fingerprint(split_manifest_path) != expected["split_manifest_sha256"]:
        raise ValueError("Experiment manifest differs from request")
    split_manifest = json.loads(split_manifest_path.read_text())
    frames = {}
    for split in ("train", "validation"):
        path = experiment / f"{split}_rows.parquet"
        if fingerprint(path) != split_manifest["split_sha256"][split]:
            raise ValueError("Feature-record checksum differs")
        frames[split] = pd.read_parquet(path)
    cached_rows, labels = [], []
    for request, _ in records:
        row = frames[request["original_split"]].iloc[request["split_row_index"]]
        if (
            row.url != request["url"]
            or row.html_sha256 != request["expected_html_sha256"]
            or row.is_phishing != request["expected_label"]
            or row.date != request["expected_date"]
        ):
            raise ValueError("Request does not map to original feature record")
        cached_rows.append(
            row.loc[list(HYBRID_FEATURE_NAMES)].to_numpy(dtype=np.float32)
        )
        labels.append(request["expected_label"])
    np.testing.assert_array_equal(matrix, np.asarray(cached_rows))
    labels = np.asarray(labels)
    details = pd.DataFrame(
        {
            "request_id": [r[0]["request_id"] for r in records],
            "development_row_index": [r[0]["development_row_index"] for r in records],
            "dataset_label": labels,
            "content_sha256": [r[0]["expected_html_sha256"] for r in records],
            "group_reason": ["|".join(r[0]["reasons"]) for r in records],
        }
    )
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Diagnostic replay on deliberately selected development records",
        "not_external_accuracy": True,
        "no_network_or_rendering": True,
        "bundle_sha256": fingerprint(bundle),
        "request_sha256": fingerprint(requests),
        "records": len(records),
        "unique_snapshots": len(snapshots),
        "dataset_class_counts": dict(Counter(map(str, labels))),
        "truncated_records": sum(bool(m["truncated"]) for _, m in records),
        "all_38_features_reproduce_cached_float32_values_exactly": True,
        "truncation_flag_restored_from_transfer_provenance": True,
        "raw_shard_hashes": "Windows-reported hashes match; not rehashed on Mac",
        "models": {},
        "label_groups": [],
        "limitations": [
            "Selected anomaly sample, not representative or independent",
            "Dataset labels are not independently adjudicated truth",
            "Original frozen-model results reuse train/validation records",
            "OOF models exclude the record component, but sample selection is post-hoc",
            "One truncated raw document cannot be fully rehashed from its saved prefix",
            "No extension integration, new training, cutoff changes or live-page test",
        ],
    }
    protected = {}
    for variant, names in SCHEMAS.items():
        columns = [HYBRID_FEATURE_NAMES.index(name) for name in names]
        values = matrix[:, columns]
        directory = experiment / variant
        metadata_path = directory / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["features"] != list(names):
            raise ValueError("Saved metadata feature schema differs")
        model_path = directory / "xgboost_model.ubj"
        protected.update({str(p): fingerprint(p) for p in (metadata_path, model_path)})
        frozen = predict(
            model_path,
            names,
            values,
            metadata["best_iteration"] + 1,
            metadata["model_sha256"],
        )
        details[f"{variant}_frozen_score"] = frozen
        cutoff = metadata["threshold"]
        details[f"{variant}_frozen_flag"] = frozen >= cutoff
        cv_dir = cv_root / (
            "talentud_url_grouped_cv_20261016"
            if variant == "url_only"
            else "talentud_combined_grouped_cv_20261016"
        )
        cv = json.loads((cv_dir / "cv_results.json").read_text())
        if (
            cv["features"] != list(names)
            or cv["provenance"]["split_sha256"] != split_manifest["split_sha256"]
        ):
            raise ValueError("CV feature/data provenance differs")
        oof_path = cv_dir / "out_of_fold.parquet"
        roles_path = cv_dir / "role_assignments.parquet"
        for path in (oof_path, roles_path):
            if fingerprint(path) != cv["output_sha256"][path.name]:
                raise ValueError("CV output checksum differs")
        if fingerprint(roles_path) != expected["cv_roles_sha256"]:
            raise ValueError("CV roles differ from sampling request")
        oof = (
            pd.read_parquet(oof_path)
            .set_index("row_index")
            .loc[details.development_row_index]
        )
        np.testing.assert_array_equal(oof.is_phishing.to_numpy(), labels)
        held_scores = np.empty(len(records), dtype=np.float32)
        policies = list(cv["folds"][0]["cutoffs"])
        cutoffs = {policy: np.empty(len(records)) for policy in policies}
        for record in cv["folds"]:
            selected = oof.fold.to_numpy() == record["fold"]
            if not selected.any():
                continue
            path = cv_dir / f"fold_{record['fold']}.ubj"
            held_scores[selected] = predict(
                path, names, values[selected], record["trees"], record["model_sha256"]
            )
            for policy in policies:
                cutoffs[policy][selected] = record["cutoffs"][policy]
        np.testing.assert_array_equal(held_scores, oof.score.to_numpy(dtype=np.float32))
        details[f"{variant}_held_score"] = held_scores
        details[f"{variant}_held_fold"] = oof.fold.to_numpy()
        summaries = {}
        for policy in policies:
            flags = held_scores >= cutoffs[policy]
            details[f"{variant}_{policy}_flag"] = flags
            # Metrics on binary flags permit a different inner-selected cutoff per fold.
            metric = evaluate_probabilities(labels, flags.astype(float), 0.5)
            summaries[policy] = {
                key: value
                for key, value in metric.items()
                if key not in {"roc_auc", "average_precision", "threshold"}
            }
        result["models"][variant] = {
            "features": len(names),
            "frozen_model_sha256": metadata["model_sha256"],
            "frozen_cutoff": cutoff,
            "frozen_diagnostic_metrics_against_dataset_labels": evaluate_probabilities(
                labels, frozen, cutoff
            ),
            "held_fold_raw_feature_scores_match_saved_oof_exactly": True,
            "held_fold_diagnostic_metrics_against_dataset_labels": summaries,
        }
    for digest in sorted(snapshots):
        subset = details.loc[details.content_sha256.eq(digest)]
        result["label_groups"].append(
            {
                "html_sha256": digest,
                "records": len(subset),
                "dataset_labels": sorted(map(int, subset.dataset_label.unique())),
                "url_held_scores_distinct": int(subset.url_only_held_score.nunique()),
                "combined_held_scores_distinct": int(
                    subset.url_html_held_score.nunique()
                ),
            }
        )
    result["held_f1_policy_decision_disagreements"] = int(
        (
            details.url_only_inner_f1_recall90_flag
            != details.url_html_inner_f1_recall90_flag
        ).sum()
    )
    result["saved_model_and_metadata_unchanged"] = all(
        fingerprint(Path(p)) == h for p, h in protected.items()
    )
    if not result["saved_model_and_metadata_unchanged"]:
        raise RuntimeError("Saved artifacts changed")
    output.mkdir(parents=True)
    details.to_parquet(output / "record_predictions.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"label_groups", "limitations"}
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--requests", type=Path, default=Path("data/processed/review_requests.json")
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/experiments/phreshphish_html_20261010"),
    )
    parser.add_argument("--cv-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.bundle, args.requests, args.experiment, args.cv_root, args.output)


if __name__ == "__main__":
    main()
