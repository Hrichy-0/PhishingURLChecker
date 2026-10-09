"""Inspect saved HTML safely; these signals are not a trained hybrid verdict."""

import argparse
import json
from pathlib import Path

from phishguard.content import MAX_HTML_BYTES, extract_content_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--html-file", type=Path, required=True)
    args = parser.parse_args()
    if args.html_file.stat().st_size > MAX_HTML_BYTES:
        parser.error("HTML file exceeds the 1 MB safety limit")
    html = args.html_file.read_text(encoding="utf-8")
    print(json.dumps(extract_content_features(args.url, html), indent=2))
    print("Offline signals only. No website was visited; no phishing verdict was made.")


if __name__ == "__main__":
    main()
