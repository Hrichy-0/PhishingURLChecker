"""Compare research candidates locally, without changing the API or visiting URLs."""

import argparse
import json
from pathlib import Path

import joblib

from phishguard.config import ARTIFACTS_DIR
from phishguard.data import is_valid_http_url
from phishguard.experiments import URL_VARIANTS


def compare_url(
    url: str,
    experiment_dir: Path,
    variants: tuple[str, ...] = URL_VARIANTS,
) -> dict:
    url = url.strip()
    if len(url) > 2048 or not is_valid_http_url(url):
        raise ValueError("Supply an absolute HTTP(S) URL of at most 2048 characters")
    results = {}
    if not variants or set(variants) - set(URL_VARIANTS):
        raise ValueError("Select known URL model variants")
    for variant in variants:
        path = experiment_dir / variant
        if not (path / "phishguard_model.joblib").exists():
            raise FileNotFoundError(
                f"Missing {variant} package; run benchmark_models.py"
            )
        # Load only locally trained/trusted artifacts, never downloaded joblib.
        model = joblib.load(path / "phishguard_model.joblib")
        metadata = json.loads((path / "model_metadata.json").read_text())
        score = float(model.predict_proba([url])[0, 1])
        threshold = float(metadata["threshold"])
        results[variant] = {
            "uncalibrated_score": score,
            "threshold": threshold,
            "above_alert_threshold": score >= threshold,
        }
    return {
        "url": url,
        "experimental": True,
        "results": results,
        "disclaimer": (
            "No site was visited. Scores are uncalibrated; below threshold does "
            "not mean safe. Candidates have unresolved cross-dataset errors."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=ARTIFACTS_DIR / "experiments" / "url_benchmark_full_v1",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=URL_VARIANTS, default=list(URL_VARIANTS)
    )
    args = parser.parse_args()
    print(
        json.dumps(
            compare_url(args.url, args.experiment_dir, tuple(args.variants)), indent=2
        )
    )


if __name__ == "__main__":
    main()
