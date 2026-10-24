"""Prepare a blinded external URL pilot, then score ONLY after label review.

Never visits candidate URLs, retrains, tunes a cutoff, or updates the extension.
All source URLs, review evidence and predictions stay in the ignored data folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from phishguard.config import MAIN_URL_MODEL_DIR, PROJECT_ROOT
from phishguard.domains import domain_extractor
from phishguard.native import TRAINING_FEATURES, evaluate_probabilities, url_matrix
from phishguard.prediction import NativeURLService

FEED = (
    "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
)
TRANCO_ID = "https://tranco-list.eu/top-1m-id"
TARGET = 100  # A reviewable pilot, NOT a low-FPR certification sample.
SEED = "frozen-url-pilot-v1"
REVIEW_FIELDS = (
    "status",
    "review_label",
    "reviewer",
    "reviewed_at",
    "evidence",
    "notes",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(UTC).isoformat()


@lru_cache(maxsize=200_000)
def hostname_group(hostname):
    extracted = domain_extractor(hostname)
    return (extracted.top_domain_under_public_suffix or hostname).lower()


def registered(url, *, historical=False):
    """Never guess a scheme for benchmark URLs; bare historical hosts exclude only."""
    if not isinstance(url, str) or not url or any(c.isspace() for c in url):
        raise ValueError("Require a nonblank URL without whitespace")
    if historical and "://" not in url:
        url = "https://" + url  # Conservative exclusion, NEVER inference input.
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Require an explicit HTTP(S) URL")
    if not historical and (parsed.username or parsed.password):
        raise ValueError("Credentials in a benchmark URL require separate handling")
    _ = parsed.port  # Reject malformed ports rather than silently accepting them.
    return hostname_group(parsed.hostname)


def fetch(url, destination, limit=20_000_000):
    """Download only explicitly selected HTTPS dataset endpoints, not targets."""
    if urlsplit(url).scheme != "https":
        raise ValueError("Dataset downloads require HTTPS")
    request = Request(url, headers={"User-Agent": "PhishingURLChecker-research/0.1"})
    with urlopen(request, timeout=45) as response:
        payload = response.read(limit + 1)
        headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified")}
        resolved = response.url
    if len(payload) > limit:
        raise ValueError("Dataset download exceeds size bound")
    destination.write_bytes(payload)
    return {
        "url": url,
        "resolved_url": resolved,
        "retrieved_at": now(),
        "bytes": len(payload),
        "sha256": sha(destination),
        "headers": headers,
    }


def history_domains(root, output):
    """Inventory every local URL-bearing CSV/Parquet, including earlier tests.

    Exact source files are hashed. No claim is made about unavailable Windows
    files, unextracted archives or campaign-level independence.
    """
    paths = sorted(
        {
            p
            for base in (root / "data", root / "artifacts")
            for suffix in ("*.csv", "*.parquet")
            for p in base.rglob(suffix)
            if output.resolve() not in p.resolve().parents
        }
    )
    domains, inventory = set(), []
    for path in paths:
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
        else:
            names = list(pd.read_csv(path, nrows=0).columns)
        columns = [c for c in names if c.lower() == "url"]
        entry = {"path": str(path.relative_to(root)), "sha256": sha(path)}
        if len(columns) != 1:
            entry["status"] = "no_unique_url_column"
            inventory.append(entry)
            continue
        column = columns[0]
        batches = (
            (
                b.to_pandas()
                for b in parquet.iter_batches(
                    columns=[column], batch_size=50_000, use_threads=False
                )
            )
            if path.suffix == ".parquet"
            else pd.read_csv(path, usecols=[column], dtype=str, chunksize=50_000)
        )
        rows = invalid = 0
        for batch in batches:
            rows += len(batch)
            for url in batch[column].drop_duplicates():
                try:
                    domains.add(registered(url, historical=True))
                except ValueError:
                    invalid += 1
        entry.update(status="included", rows=rows, invalid_unique_values=invalid)
        inventory.append(entry)
    if not domains:
        raise ValueError("No historical domains found; refuse an empty overlap audit")
    return domains, inventory


def select_candidates(feed, ranking, known):
    """Pre-score, deterministic, at most one URL per private-PSL domain."""
    phish, counts = {}, Counter()
    for url in feed.splitlines():
        url = url.strip()
        counts["feed_rows"] += 1
        try:
            group = registered(url)
        except ValueError:
            counts["feed_invalid_or_credentials"] += 1
            continue
        if group in known:
            counts["feed_known_domain"] += 1
            continue
        if group in phish:
            counts["feed_repeated_domain"] += 1
        # Choose a URL by seeded hash, never by a model score.
        if group not in phish or order(url) < order(phish[group]):
            phish[group] = url
    rows = [
        {
            "url": url,
            "domain": group,
            "source": "OpenPhish",
            "source_label": "1",
            "rank": "",
        }
        for group, url in sorted(phish.items(), key=lambda item: order(item[0]))[
            :TARGET
        ]
    ]
    # Popularity supplies CANDIDATES only. Do not invent https://domain URLs or
    # label ranked sites legitimate without independent evidence.
    seen = known | set(phish)
    for lower, upper, quota in ((1, 1000, 34), (1001, 10000, 33), (10001, 100000, 33)):
        candidates = []
        for rank, host in ranking:
            if lower <= rank <= upper:
                group = hostname_group(host)
                if group not in seen:
                    candidates.append((group, rank))
        candidates.sort(key=lambda item: order(item[0]))
        added = 0
        for group, rank in candidates:
            if group in seen:
                continue
            seen.add(group)
            rows.append(
                {
                    "url": "",
                    "domain": group,
                    "source": "Tranco",
                    "source_label": "unknown",
                    "rank": str(rank),
                }
            )
            added += 1
            if added == quota:
                break
        counts[f"tranco_selected_rank_{lower}_{upper}"] = added
    for row in rows:
        row["record_id"] = order(row["source"] + "|" + row["domain"])
    counts["phishing_candidates"] = sum(r["source"] == "OpenPhish" for r in rows)
    counts["unlabeled_ranked_candidates"] = sum(r["source"] == "Tranco" for r in rows)
    return rows, dict(counts)


def order(value):
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def prepare(root, model_dir, output):
    if output.exists():
        raise ValueError(
            "Choose a new output directory; never overwrite a frozen pilot"
        )
    service = NativeURLService(model_dir)
    output.mkdir(parents=True)
    code = [
        Path(__file__).resolve(),
        root / "src/phishguard/lexical.py",
        root / "src/phishguard/native.py",
        root / "src/phishguard/domains.py",
        root / "src/phishguard/prediction.py",
    ]
    lock = {
        "created_at": now(),
        "purpose": "Blinded fresh-source URL-only PILOT",
        "model_dir": str(model_dir.resolve()),
        "model_sha256": service.model_sha256,
        "metadata_sha256": sha(model_dir / "model_metadata.json"),
        "threshold": service.metadata["threshold"],
        "rounds": service.rounds,
        "features": list(TRAINING_FEATURES),
        "candidate_target_per_source": TARGET,
        "selection_seed": SEED,
        "minimum_reviewed_domains_per_class": 30,
        "bootstrap_repetitions": 2000,
        "bootstrap_seed": 42,
        "software_sha256": {str(p.relative_to(root)): sha(p) for p in code},
        "versions": {
            name: __import__(name).__version__
            for name in ("numpy", "pandas", "xgboost", "tldextract", "pyarrow")
        },
        "rules": [
            "No training, threshold search, model promotion or live target visits",
            "Exclude every registered domain found in local URL CSV/Parquet history",
            "Reject malformed benchmark URLs; never infer a missing URL scheme",
            "One record per private-PSL domain; ambiguous labels excluded and counted",
            "Review all selected records before scoring; no score-driven replacements",
            "Tranco popularity is NOT a legitimate label; evidence is required",
            "OpenPhish is a provider label, not independently adjudicated ground truth",
            "Snapshot time is NOT first-seen time or proof of temporal independence",
        ],
    }
    write_json(output / "protocol.json", lock)  # Freeze BEFORE acquiring candidates.
    domains, inventory = history_domains(root, output)
    (output / "excluded_domains.txt").write_text("\n".join(sorted(domains)) + "\n")
    write_json(output / "history_inventory.json", inventory)
    sources = output / "sources"
    sources.mkdir()
    source_info = {"openphish": fetch(FEED, sources / "openphish.txt")}
    source_info["tranco_id"] = fetch(TRANCO_ID, sources / "tranco_id.txt", 100)
    list_id = (sources / "tranco_id.txt").read_text().strip()
    if not list_id.isalnum() or not 3 <= len(list_id) <= 12:
        raise ValueError("Unexpected Tranco permanent list ID")
    source_info["tranco"] = fetch(
        f"https://tranco-list.eu/download/{list_id}/1000000",
        sources / "tranco.csv",
        50_000_000,
    )
    ranking = pd.read_csv(
        sources / "tranco.csv", header=None, names=["rank", "domain"], nrows=100000
    )
    if not np.array_equal(ranking["rank"], np.arange(1, len(ranking) + 1)):
        raise ValueError("Unexpected Tranco CSV rank order")
    rows, counts = select_candidates(
        (sources / "openphish.txt").read_text(),
        list(ranking.itertuples(index=False, name=None)),
        domains,
    )
    if not rows:
        raise ValueError("No candidates survived the overlap audit")
    candidates = pd.DataFrame(rows)
    candidates.to_parquet(output / "candidates.parquet", index=False)
    review = candidates.copy()
    for field in REVIEW_FIELDS:
        review[field] = "pending" if field == "status" else ""
    review.to_csv(output / "review.csv", index=False)
    manifest = {
        "created_at": now(),
        "status": "awaiting_blinded_label_review",
        "predictions_computed": False,
        "model_sha256": service.model_sha256,
        "threshold": lock["threshold"],
        "known_local_domains": len(domains),
        "history_files_included": sum(i["status"] == "included" for i in inventory),
        "sources": source_info,
        "tranco_list_id": list_id,
        "selection": counts,
        "frozen_sha256": {
            name: sha(output / name)
            for name in (
                "protocol.json",
                "history_inventory.json",
                "excluded_domains.txt",
                "candidates.parquet",
            )
        },
    }
    write_json(output / "preparation.json", manifest)
    return manifest


def verified_review(candidates, review, known):
    if (
        set(review.columns) != set(candidates.columns) | set(REVIEW_FIELDS)
        or review.record_id.duplicated().any()
        or set(review.record_id) != set(candidates.record_id)
    ):
        raise ValueError("Review must contain every original candidate exactly once")
    original = candidates.set_index("record_id")
    rows = []
    counts = Counter()
    seen = set()
    for row in review.to_dict("records"):
        candidate = original.loc[row["record_id"]]
        for field in candidates.columns.difference(["url", "record_id"]):
            if row[field] != candidate[field]:
                raise ValueError("Immutable candidate identity changed")
        if candidate.source == "OpenPhish" and row["url"] != candidate.url:
            raise ValueError("Do not replace a selected phishing URL")
        status = row["status"]
        if status not in {"accepted", "rejected", "uncertain"}:
            raise ValueError("Label review incomplete; no predictions computed")
        if not all(
            row[field].strip() for field in ("reviewer", "reviewed_at", "evidence")
        ):
            raise ValueError("Every decision requires reviewer, date and evidence")
        if datetime.fromisoformat(row["reviewed_at"]).tzinfo is None:
            raise ValueError("Review timestamp must include a timezone")
        counts[status] += 1
        if status != "accepted":
            continue
        if row["review_label"] not in {"0", "1"}:
            raise ValueError("Accepted labels must be 0=legitimate or 1=phishing")
        group = registered(row["url"])
        if group != row["domain"] or group in known or group in seen:
            raise ValueError("Changed, previously seen or repeated registered domain")
        seen.add(group)
        row["is_phishing"] = int(row["review_label"])
        rows.append(row)
    return pd.DataFrame(rows), dict(counts)


def uncertainty(labels, scores, cutoff, repetitions, seed):
    """Stratified domain bootstrap (exactly one accepted URL per domain)."""
    labels = np.asarray(labels)
    predicted = np.asarray(scores) >= cutoff
    pools = [predicted[labels == label] for label in (0, 1)]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("Both classes required for uncertainty")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        fp, tp = [int(rng.choice(p, len(p), replace=True).sum()) for p in pools]
        tn, fn = len(pools[0]) - fp, len(pools[1]) - tp
        values.append(
            (
                tp / len(pools[1]),
                fp / len(pools[0]),
                (tp + tn) / len(labels),
                tp / max(1, tp + fp),
                2 * tp / max(1, 2 * tp + fp + fn),
            )
        )
    intervals = {
        name: np.quantile(np.asarray(values)[:, i], [0.025, 0.975]).tolist()
        for i, name in enumerate(
            ("recall", "false_positive_rate", "accuracy", "precision", "f1")
        )
    }

    # The bootstrap degenerates at zero errors; also report Wilson intervals.
    def wilson(successes, n):
        z, p = 1.959963984540054, successes / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        width /= 1 + z * z / n
        return [float(max(0, center - width)), float(min(1, center + width))]

    return {
        "conditional_stratified_domain_bootstrap_95": intervals,
        "wilson_95": {
            "recall": wilson(int(pools[1].sum()), len(pools[1])),
            "false_positive_rate": wilson(int(pools[0].sum()), len(pools[0])),
        },
        "limitations": "Conditional on this reviewed, source-selected pilot; not "
        "training, campaign, temporal or labeling uncertainty. "
        "Precision depends on the artificial sample prevalence.",
    }


def score(root, directory):
    import xgboost as xgb

    result_path = directory / "results.json"
    if result_path.exists() or (directory / "predictions.parquet").exists():
        raise ValueError("Already scored; do not replace this holdout result")
    manifest = json.loads((directory / "preparation.json").read_text())
    for name, digest in manifest["frozen_sha256"].items():
        if sha(directory / name) != digest:
            raise ValueError("Frozen benchmark input changed")
    lock = json.loads((directory / "protocol.json").read_text())
    for name, digest in lock["software_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Frozen inference/evaluation code changed")
    if any(__import__(n).__version__ != v for n, v in lock["versions"].items()):
        raise ValueError("Frozen dependency version changed")
    candidates = pd.read_parquet(directory / "candidates.parquet")
    review_path = directory / "review.csv"
    reviewed_sha = sha(review_path)
    reviewed = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    known = set((directory / "excluded_domains.txt").read_text().splitlines())
    accepted, decisions = verified_review(candidates, reviewed, known)
    if accepted.empty or any(
        (accepted.is_phishing == label).sum()
        < lock["minimum_reviewed_domains_per_class"]
        for label in (0, 1)
    ):
        raise ValueError("Need at least 30 reviewed domains per class; no scoring")
    model_dir = Path(lock["model_dir"])
    if sha(model_dir / "model_metadata.json") != lock["metadata_sha256"]:
        raise ValueError("Frozen metadata changed")
    service = NativeURLService(model_dir)
    if (
        service.model_sha256 != lock["model_sha256"]
        or service.rounds != lock["rounds"]
        or service.metadata["threshold"] != lock["threshold"]
        or list(TRAINING_FEATURES) != lock["features"]
    ):
        raise ValueError("Frozen model, cutoff or feature schema changed")
    matrix = url_matrix(accepted.url)
    scores = service.model.predict(
        xgb.DMatrix(matrix, feature_names=list(TRAINING_FEATURES), nthread=4),
        iteration_range=(0, service.rounds),
    )
    cutoff = lock["threshold"]
    result = {
        "created_at": now(),
        "purpose": "Source-specific reviewed external PILOT",
        "not_real_world_certification": True,
        "model_sha256": service.model_sha256,
        "preparation_sha256": sha(directory / "preparation.json"),
        "review_sha256": reviewed_sha,
        "accepted_domains": len(accepted),
        "review_decisions": decisions,
        "source_by_label": accepted.groupby(["source", "is_phishing"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        "metrics": evaluate_probabilities(accepted.is_phishing, scores, cutoff),
        "uncertainty": uncertainty(
            accepted.is_phishing,
            scores,
            cutoff,
            lock["bootstrap_repetitions"],
            lock["bootstrap_seed"],
        ),
        "known_domain_overlap": 0,
        "retrains_or_cutoff_searches": 0,
        "limitations": [
            "Only locally available history can be excluded",
            "Domain separation does not prove campaign independence",
            "Feed and ranking provenance can be predictive shortcuts",
            "Collection date does not establish first-seen date",
            "Human review and threat-feed labels can be wrong",
        ],
    }
    accepted["score"] = scores
    accepted["predicted_is_phishing"] = scores >= cutoff
    accepted.to_parquet(directory / "predictions.parquet", index=False)
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "score"])
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(PROJECT_ROOT, MAIN_URL_MODEL_DIR, args.directory)
    else:
        result = score(PROJECT_ROOT, args.directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
