"""Run a transferred native URL/HTML model locally on CPU, including on a Mac."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phishguard.native_html import HTML_FEATURE_NAMES, snapshot_vector
from phishguard.native_package import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--html-file", type=Path)
    args = parser.parse_args()
    try:
        model, metadata = load(args.model_dir)
    except ValueError as error:
        parser.error(str(error))
    names = model.features
    if set(names) & set(HTML_FEATURE_NAMES) and args.html_file is None:
        parser.error("The HTML-derived/combined model requires --html-file")
    html = (
        args.html_file.read_text(encoding="utf-8", errors="replace")
        if args.html_file
        else None
    )
    if args.html_file and not html.strip():
        parser.error("HTML file must contain a nonempty saved page")
    values = snapshot_vector(
        args.url, html, names, metadata.get("max_html_bytes", 1_000_000)
    )
    score = float(model.scores(values)[0])
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
