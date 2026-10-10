"""Show saved XGBoost feature importance percentages without loading a model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def render_report(metadata: dict) -> str:
    """Render normalized gain shares; never infer unavailable feature weights."""
    entries = metadata.get("feature_importances")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Metadata has no saved feature_importances list")
    weights = {}
    for entry in entries:
        name = entry["feature"]
        value = float(entry["gain_importance"])
        if not isinstance(name, str) or not name or name in weights:
            raise ValueError("Feature names must be unique, nonempty strings")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Gain importances must be finite and nonnegative")
        weights[name] = value
    features = metadata.get("features")
    if features is not None and set(features) != set(weights):
        raise ValueError("Saved feature list and importance entries do not match")
    total = math.fsum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("A positive finite total gain importance is required")
    lines = [
        "# Saved model inspection",
        "",
        f"Model: {metadata.get('model_type', 'unspecified')}",
        f"Version: {metadata.get('model_version', 'unspecified')}",
        f"Created: {metadata.get('created_at', 'unspecified')}",
        f"Threshold: {metadata.get('threshold', 'unspecified')}",
        "",
        "These are saved global gain importance shares, normalized to 100%.",
        "They are not per-URL contributions, detection rates, or percentiles.",
        "Correlated features can share importance; gain does not show direction.",
        "This reads metadata only and does not verify the model binary matches it.",
        "",
        "| Rank | URL feature | Gain importance share |",
        "|---:|---|---:|",
    ]
    for rank, (name, value) in enumerate(
        sorted(weights.items(), key=lambda item: item[1], reverse=True), start=1
    ):
        lines.append(f"| {rank} | {name} | {100 * value / total:.2f}% |")
    rows = metadata.get("dataset_rows")
    if rows:
        lines.extend(["", f"Recorded split sizes: {json.dumps(rows, sort_keys=True)}"])
    metrics = metadata.get("test_metrics", {})
    if metrics:
        lines.extend(["", "## Recorded test metrics", ""])
        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "false_positive_rate",
        ):
            if name in metrics:
                lines.append(f"- {name}: {100 * float(metrics[name]):.2f}%")
        lines.extend(["", "Recorded metrics have not been rerun by this command."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional Markdown report path")
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        report = render_report(metadata)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot inspect metadata: {error}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
