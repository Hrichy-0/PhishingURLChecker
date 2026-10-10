# Pulling code, models, and results onto your existing Mac checkout

The code, active model pair, historical model artifacts, evaluation reports and
prediction outputs are now tracked in the repository. Use your existing checkout;
you do not need to clone it again. Raw datasets, feature caches, dataset split
files, Windows virtual environments and temporary tests are excluded.

```bash
cd /your/existing/PhishingURLChecker
git status
git pull --ff-only origin main
```

Replace the first path with your Mac checkout location. If Git reports local
changes, preserve/commit those changes before pulling. If it reports untracked
files under `artifacts/` that would be overwritten, preserve your old artifacts
directory first, then retry:

```bash
mv artifacts artifacts.mac-backup-20261011
git pull --ff-only origin main
```

Only use that backup command when Git reports that conflict and the backup path
does not already exist. Do not reset or delete your Mac work to force a pull.

## Existing environment or a new native Mac environment

Use your existing Mac `.venv` if it has Python 3.12. Otherwise create one with
Python 3.12 installed on the Mac. Do not copy the Windows `.venv`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-model-runtime.txt
.venv/bin/python -m pip install -e '.[data,dev]'
.venv/bin/python -m pip check
```

`requirements-model-runtime.txt` records versions used by these model artifacts.
Pip installs native Mac wheels, not Windows binaries. If XGBoost reports missing
`libomp.dylib`, its official Mac installation instructions require OpenMP:
`brew install libomp` with an existing Homebrew installation. This is a library
dependency, not a security-policy change. See
[XGBoost installation documentation](https://xgboost.readthedocs.io/en/stable/install.html).

## Inspect results and run the newly trained models

Read `docs/html-training-results-2026-10-11.md`, or the underlying
`artifacts/experiments/phreshphish_html_20261010/comparison.json`.

```bash
.venv/bin/python scripts/inspect_model_metadata.py --metadata artifacts/experiments/phreshphish_html_20261010/url_html/model_metadata.json
.venv/bin/python scripts/verify_transferred_artifacts.py
```

All three new native models run on Mac CPU; CUDA is not required for inference.
The new URL-only baseline can run with just a URL:

```bash
.venv/bin/python scripts/predict_native_snapshot.py --model-dir artifacts/experiments/phreshphish_html_20261010/url_only --url 'https://example.com/login'
```

The combined model needs a saved HTML file. This example uses a small synthetic
fixture, not a real phishing page or a demonstration of measured accuracy:

```bash
.venv/bin/python scripts/predict_native_snapshot.py --model-dir artifacts/experiments/phreshphish_html_20261010/url_html --url 'https://example.test/login' --html-file examples/hybrid/login_fixture.html
```

These commands do not fetch websites, execute HTML, or upload page contents.
The active FastAPI/extension model is still the old joblib URL-only model; native
experiments are separate and are not automatically activated.

Retraining/auditing raw input rows requires datasets to be downloaded separately.
The 36.58 GB PhreshPhish dataset has intentionally not been uploaded to GitHub.
Its pinned downloader remains in `scripts/download_phreshphish_full.py`.
The transfer manifest verifies all included model/result files without datasets.
