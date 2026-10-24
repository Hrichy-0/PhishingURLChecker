"""Run a transferred native URL/HTML model locally on CPU, including on a Mac."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xgboost as xgb

from phishguard.native_html import HTML_FEATURE_NAMES, fingerprint, snapshot_vector


def score_snapshot(model: xgb.Booster, values, names, metadata: dict) -> float:
    """Use only the saved best_iteration trees, as the API service does."""
    rounds = int(metadata.get("best_iteration", model.num_boosted_rounds() - 1)) + 1
    if not 0 < rounds <= model.num_boosted_rounds():
        raise ValueError("Model metadata has an invalid prediction tree range")
    data = xgb.DMatrix(values, feature_names=list(names))
    return float(model.predict(data, iteration_range=(0, rounds))[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--html-file", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.model_dir / "model_metadata.json").read_text())
    names = tuple(metadata["features"])
    if set(names) & set(HTML_FEATURE_NAMES) and args.html_file is None:
        parser.error("The HTML-derived/combined model requires --html-file")
    path = args.model_dir / "xgboost_model.ubj"
    if metadata.get("model_sha256") and fingerprint(path) != metadata["model_sha256"]:
        parser.error("Model file does not match its metadata checksum")
    html = (
        args.html_file.read_text(encoding="utf-8", errors="replace")
        if args.html_file
        else None
    )
    if args.html_file and not html.strip():
        parser.error("HTML file must contain a nonempty saved page")
    model = xgb.Booster()
    model.load_model(bytearray(path.read_bytes()))
    model.set_param({"device": "cpu", "nthread": 4})
    if model.num_features() != len(names):
        parser.error("Feature count differs from model metadata")
    values = snapshot_vector(
        args.url, html, names, metadata.get("max_html_bytes", 1_000_000)
    )
    score = score_snapshot(model, values, names, metadata)
    print(
        json.dumps(
            {
                "url": args.url,
                "model_type": metadata["model_type"],
                "phishing_probability": score,
                "threshold": metadata["threshold"],
                "is_phishing": score >= metadata["threshold"],
                "prediction_device": "cpu",
                "html_truncated": bool(values[0, names.index("html_truncated")])
                if "html_truncated" in names
                else None,
                "note": "Experimental advisory estimate; not a guarantee of safety",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
