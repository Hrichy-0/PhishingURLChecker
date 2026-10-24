"""Measure local URL-only service latency; not browser latency or accuracy."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

from phishguard.config import MAIN_URL_MODEL_DIR
from phishguard.prediction import NativeURLService

# Reserved domains/IPs only. These strings are never visited or given labels.
CASES = (
    "https://example.test/",
    "https://example.test/de/",
    "https://shop.example.test/products/item?id=123&lang=en",
    "http://192.0.2.1:8080/login/verify-account",
    "https://example.test/" + "article/" * 30,
    "https://example.test/search?q=%E2%82%AC&next=%2Faccount",
)


def benchmark(model_dir: Path, repeats: int = 600, warmup: int = 30) -> dict:
    if repeats < 1 or warmup < 0:
        raise ValueError("Require repeats >= 1 and warmup >= 0")
    start = time.perf_counter_ns()
    service = NativeURLService(model_dir)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    for index in range(warmup):
        service.predict(CASES[index % len(CASES)])
    elapsed = []
    for index in range(repeats):
        start = time.perf_counter_ns()
        service.predict(CASES[index % len(CASES)])
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Local Python service: URL features, CPU inference, response building",
        "not_measured": [
            "accuracy",
            "browser inference",
            "HTTP/popup latency",
            "memory",
            "energy",
        ],
        "workload": "Synthetic reserved URLs; no labels, fetching or website visits",
        "runtime": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "inference_device": "cpu",
            "threads": 4,
        },
        "model_sha256": service.model_sha256,
        "feature_count": len(service.metadata["features"]),
        "threshold": service.metadata["threshold"],
        "model_file_bytes": (model_dir / "xgboost_model.ubj").stat().st_size,
        "model_load_ms_after_imports": load_ms,
        "warmup_calls": warmup,
        "timed_calls": repeats,
        "latency_ms": {
            "median": float(np.median(elapsed)),
            "p95": float(np.percentile(elapsed, 95)),
            "p99": float(np.percentile(elapsed, 99)),
            "mean": float(np.mean(elapsed)),
        },
        "limitations": "One process on one device; not a cold application launch or CI",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MAIN_URL_MODEL_DIR)
    parser.add_argument("--repeats", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A benchmark must never overwrite an existing report or model artifact.
    if args.output.exists():
        parser.error("Output exists; choose a new report filename")
    report = benchmark(args.model_dir, args.repeats, args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
