"""Compare URL-only and URL+HTML models on the SAME offline snapshot rows."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from phishguard.config import ARTIFACTS_DIR
from phishguard.content import MAX_HTML_BYTES
from phishguard.data import clean_url_data
from phishguard.experiments import run_benchmark


def load_snapshots(
    path: Path,
    url_column: str,
    html_column: str,
    label_column: str,
    phishing_label: str,
    legitimate_label: str,
) -> tuple[pd.DataFrame, dict]:
    """Require explicit label definitions; retain first snapshot per cleaned URL."""
    if len({url_column, html_column, label_column}) != 3:
        raise ValueError("URL, HTML, and label columns must be different")
    if phishing_label.strip().lower() == legitimate_label.strip().lower():
        raise ValueError("Phishing and legitimate labels must differ")
    raw = pd.read_csv(path, usecols=[url_column, html_column, label_column])
    raw = raw.rename(
        columns={url_column: "url", html_column: "html", label_column: "label"}
    )
    raw["url"] = raw["url"].astype("string").str.strip()
    cleaned, url_report = clean_url_data(
        raw,
        url_column="url",
        label_column="label",
        label_mapping={phishing_label: 1, legitimate_label: 0},
    )
    # Fail on contradictory labels above, before removing any HTML rows.
    missing = raw["html"].isna() | raw["html"].astype("string").str.strip().eq("")
    too_large = (
        raw["html"]
        .fillna("")
        .map(lambda value: len(str(value).encode("utf-8")) > MAX_HTML_BYTES)
    )
    valid_html = raw.loc[~missing & ~too_large, ["url", "html"]].drop_duplicates("url")
    frames = cleaned.merge(valid_html, on="url", how="inner", validate="one_to_one")
    if frames.empty or set(frames["is_phishing"].unique()) != {0, 1}:
        raise ValueError("No two-class URL+HTML dataset remains after cleaning")
    return frames, {
        "url_cleaning": url_report,
        "missing_or_blank_html_rows": int(missing.sum()),
        "oversize_html_rows": int(too_large.sum()),
        "rows_with_usable_snapshots": len(frames),
        "duplicate_snapshot_policy": "first usable snapshot per exact URL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--html-column", default="html")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--phishing-label", required=True)
    parser.add_argument("--legitimate-label", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--maximum-fpr", type=float, default=0.01)
    args = parser.parse_args()
    if args.max_rows < 0:
        parser.error("--max-rows must be nonnegative")
    frames, cleaning = load_snapshots(
        args.input,
        args.url_column,
        args.html_column,
        args.label_column,
        args.phishing_label,
        args.legitimate_label,
    )
    usable_rows = len(frames)
    if args.max_rows and usable_rows > args.max_rows:
        frames = frames.sample(n=args.max_rows, random_state=args.seed)
    output = args.output_dir or ARTIFACTS_DIR / "experiments" / (
        "hybrid_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    run_benchmark(
        frames,
        output,
        variants=("xgboost_no_https", "char_logistic", "hybrid_xgboost"),
        folds=args.folds,
        seed=args.seed,
        maximum_fpr=args.maximum_fpr,
        trees=args.trees,
        provenance={
            "input": str(args.input.resolve()),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "label_mapping": {args.phishing_label: 1, args.legitimate_label: 0},
            "cleaning": cleaning,
            "pilot": len(frames) < usable_rows,
            "content_limit_bytes": MAX_HTML_BYTES,
            "caution": (
                "Domain-grouped exploratory CV only; temporal and near-duplicate "
                "HTML decontamination still required for real-world claims. "
                "Use a development subset, never a publisher final test split."
            ),
        },
    )
    print(f"Saved optional offline hybrid experiment: {output}")


if __name__ == "__main__":
    main()
