# PhishingURLChecker (PhishGuard ML)

PhishGuard is an end-to-end, URL-only phishing risk detector. It validates and
cleans labeled URL datasets, trains an XGBoost pipeline, serves predictions
through FastAPI, and connects to a Chrome Manifest V3 extension.

> **Project status:** learning/portfolio prototype. The full pipeline works.
> The active version uses broader training data and fixes the observed Apple
> deep-link false positive. Remaining false positives and source bias still
> require further evaluation.

The detector never visits a submitted URL. All 21 model features are computed from
the URL string, which keeps inference fast and avoids executing content from a
suspected site.

## Current measured performance

The active XGBoost model was trained on PhiUSIIL plus `balanced_urls.csv`.
Its 60,551-row test set is domain-disjoint from training and validation across
both sources. This is a mixed-source holdout, not a third-party external test.

| Metric | Result |
|---|---:|
| Accuracy | 93.19% |
| Phishing precision | 93.48% |
| Phishing recall | 92.74% |
| F1 | 93.11% |
| ROC-AUC | 98.33% |
| False-positive rate | 6.37% |
| False positives | 1,942 |
| False negatives | 2,182 |

These are offline dataset results, not a guarantee of future performance. See
[`docs/model-card.md`](docs/model-card.md) for limitations and
[`docs/xgboost-experiments.md`](docs/xgboost-experiments.md) for the same-split
logistic comparison (83.54% accuracy), the PhiUSIIL-only experiment (97.80%),
and the reproducible workflow.

Frozen-model evaluation on two additional Kaggle datasets found substantially
worse results, with high false-alarm rates and missing-protocol limitations.
The 93.19% holdout score must not be presented as universal accuracy. See
[`docs/external-evaluation.md`](docs/external-evaluation.md) for all assumptions,
metrics, class-imbalance baselines, and commands. The active model was unchanged.

Two Hugging Face evaluations likewise found poor results: 44.42% accuracy on
retained PhishNChips URLs and 71.50% on the retained complete-URL subset of
ealvaradob's URL dataset. Both excluded previously used domains and preserved
original protocols. See [`docs/huggingface-evaluation.md`](docs/huggingface-evaluation.md)
for the fixed-model procedure, high false-alarm rates, and selection limitations.

## New experimental alternatives

Domain-grouped cross-validation now compares XGBoost, XGBoost without the direct
HTTPS feature, and character TF-IDF + Logistic Regression. The character model
improves some external ranking metrics but still produces substantial false
alarms. **The active API/extension model has not been replaced.** An optional
offline URL+HTML feature extractor and same-row hybrid benchmark are implemented,
but no hybrid model has yet been trained on real website-content data.

See [`docs/model-comparison.md`](docs/model-comparison.md) for measured results,
the files to study, reproducible commands, and the remaining evaluation work.
Source-held-out XGBoost tuning is available through the same benchmark command
(`--mode tune`), with six explicit candidates rather than a large search framework.
The [file-purpose guide](docs/architecture.md#file-purpose-audit--2026-10-06)
distinguishes runtime files, current research commands, and optional/history files.

The [consolidated dataset audit](docs/dataset-audit-report.md) combines the
three read-only agent audits of training sources, uploaded URLs, and Hugging Face
data, including label conflicts, false-positive patterns, source bias, and next steps.

## Technology stack

- Python 3.11+
- pandas, scikit-learn, and XGBoost
- joblib model packaging
- FastAPI, Pydantic, and Uvicorn
- pytest and Ruff
- Chrome Extension Manifest V3 with plain HTML, CSS, and JavaScript

## Repository map

```text
api/                    FastAPI application
artifacts/              Generated model and metadata
data/raw/               Original Kaggle CSV
data/processed/         Clean and split CSVs
docs/                   Architecture and model documentation
extension/              Unpacked Chrome extension
scripts/                Reproducible data and training commands
src/phishguard/         Reusable data, feature, model, and prediction code
tests/                   Unit and API tests
```

## Set up

### Windows laptop (PowerShell / VS Code)

Install Git, Python 3.12, VS Code and Chrome. On a fresh laptop, clone the project
from PowerShell, then open that folder in VS Code:

```powershell
git clone https://github.com/Hrichy-0/PhishingURLChecker.git
cd PhishingURLChecker
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[data,dev]"
code .
```

If the clone already exists, use `git pull` inside it instead of cloning again;
preserve any local edits before pulling. In VS Code select
**Python: Select Interpreter → `.venv\Scripts\python.exe`**. These commands call
the environment directly, so PowerShell activation-policy changes are unnecessary.
Do not copy the Mac `.venv` to Windows; recreate it. See the official
[Python virtual-environment guide](https://docs.python.org/3/library/venv.html).

Verify without datasets or a trained model:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

XGBoost on Windows requires the Microsoft Visual C++ Redistributable runtime;
see the official [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html#python)
if a DLL import error occurs. GPU hardware is not required by this project's CPU workflow.

#### Move your existing data/model separately

GitHub includes code, tests, extension files and Markdown reports, **not** local
datasets, generated evaluation JSONs/CSVs or trained model binaries. Copy trusted
files privately from the Mac if you want the same experiments on the laptop:

- `data/` for the original and processed datasets (preserve file bytes).
- `artifacts/` for the saved models, metadata and detailed evaluation outputs.
- The uploaded `URL dataset.csv`, which currently lives outside the repository.

For API inference alone, the required pair is
`artifacts/phishguard_model.joblib` and `artifacts/model_metadata.json`.
Load only your own trusted joblib files, never arbitrary downloaded models.
The verified Mac environment uses NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1,
XGBoost 3.4.1 and joblib 1.6.0; match these versions when transferring its saved
Python model, rather than assuming any versions in the broad dependency ranges
are serialization-compatible:

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy==2.5.3" "pandas==2.3.3" "scikit-learn==1.9.1" "xgboost==3.4.1" "joblib==1.6.0"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

This Windows guide was checked against the project code and official installation
documentation, not executed on a Windows machine yet. Without the trusted model
pair, `/health` correctly reports degraded and predictions cannot run.

Saved experiment metadata contains absolute `/Users/...` Mac data paths. API
inference does not read those paths, but evaluation commands and old audit
references do. Keep the original metadata unchanged; transferred experiments need
an explicit Windows-local provenance/path migration before evaluation, or rebuild
datasets/model packages on Windows to generate local paths. Do not claim that
copying artifacts alone makes every historical evaluation command portable.

All commands below use `python` for brevity. In PowerShell without activation,
replace it with `.\.venv\Scripts\python.exe`; run multiline Bash examples as one
line (PowerShell does not use `\` for line continuation). Load `extension/` through
Chrome's **Load unpacked** after starting the API. Do not visit phishing examples.

### macOS / Linux shell

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,dev]'
```

On macOS, XGBoost may require `brew install libomp` before it can load.

## Rebuild the PhiUSIIL-only experiment

The raw dataset is the **PhiUSIIL Phishing URLs** Kaggle dataset. Its source
labels are `0 = phishing` and `1 = legitimate`; the cleaning step converts them
to the project convention `1 = phishing` and `0 = legitimate`.

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/split_data.py
python scripts/train_model.py --artifacts-dir artifacts/experiments/xgboost_phiusiil
```

To reproduce the active broader model, follow the mixed-source commands in
[`docs/xgboost-experiments.md`](docs/xgboost-experiments.md#reproduce-training).
`scripts/activate_model.py --artifacts-dir ...` activates an evaluated experiment
and archives the previous model. Restart the API after activation.

Generated data and artifacts are ignored by Git because they are reproducible
and potentially large.

### Prepare a different dataset

The cleaner can also standardize another CSV without changing the rest of the
training pipeline. The current external dataset has columns `url` and `label`,
with textual labels `legitimate` and `phishing`:

```bash
python scripts/prepare_data.py \
  --input data/external/balanced_urls.csv \
  --url-column url \
  --label-column label \
  --label-convention text \
  --output data/processed/sources/balanced.csv
```

This writes the same internal schema—`url` plus `is_phishing`—to
the selected output path. `split_data.py --input ... --output-dir ...` supports
one or several cleaned files, and `train_model.py --data-dir ...` accepts the
resulting splits. For a dataset using `0 = legitimate` and `1 = phishing`, use
`--label-convention project`. Always state the correct convention for a new
dataset because reversing numeric labels would silently train the wrong model.
For other two-class label names, specify both `--legitimate-label benign` and
`--phishing-label malicious`, along with the actual URL and label column names.

## Run and test the API

```bash
uvicorn api.main:app --reload
```

Open <http://127.0.0.1:8000/docs>, or send:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/login"}'
```

The current response field is named `phishing_probability`, but it should be
interpreted as a model score—not literal certainty that a site is phishing.
It also includes the selected threshold, a risk level, and human-readable
lexical signals.

## Load the Chrome extension

1. Keep the API running on `127.0.0.1:8000`.
2. Open `chrome://extensions`.
3. Enable **Developer mode**.
4. Choose **Load unpacked** and select the `extension/` directory.
5. Visit an HTTP or HTTPS page and click the PhishGuard toolbar action.

## Verify the project

```bash
ruff format --check src scripts tests api
ruff check src scripts tests api
pytest -q
```

## Important safety boundary

This model is an advisory layer, not a replacement for browser Safe Browsing,
endpoint protection, reputation feeds, or security analysts. A low score must
never be presented as proof that a website is safe.

## What this project teaches

- how to inspect and clean a labeled cybersecurity dataset;
- why label conventions and duplicate removal matter;
- why splitting by registered domain is safer than a random row split;
- how a scikit-learn pipeline prevents training/serving feature drift;
- why recall, precision, thresholds, and false positives matter more than
  accuracy alone in a security detector;
- how to package a model behind an API and call it from a Chrome extension;
- why strong offline metrics can still hide collection bias.

The complete chronological record is in
[`docs/project-journey.md`](docs/project-journey.md). The architecture is in
[`docs/architecture.md`](docs/architecture.md), and responsible-use details
are in [`docs/model-card.md`](docs/model-card.md).

## Roadmap

- [x] Inspect and clean the Kaggle dataset.
- [x] Make domain-disjoint train, validation, and test splits.
- [x] Train, evaluate, and package a lexical baseline model.
- [x] Expose predictions through FastAPI.
- [x] Connect a local Chrome Manifest V3 extension.
- [x] Add automated tests and linting.
- [x] Display the score as model risk rather than literal certainty.
- [ ] Add a separate real-world sanity set containing benign deep links.
- [x] Remove slash count, compare XGBoost, and retrain with broader data.
- [ ] Evaluate on an unused third source and assess score calibration.
- [ ] Add reputation/content signals before considering production use.

## Data and generated files

The raw Kaggle CSV, processed splits, and trained artifacts are intentionally
excluded from Git. Reproduce them with the commands above. This keeps the
repository small and avoids redistributing a third-party dataset.
