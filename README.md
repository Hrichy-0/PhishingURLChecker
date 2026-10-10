# PhishingURLChecker (PhishGuard ML)

PhishGuard is a phishing detection research project with a URL-only FastAPI
service and Chrome Manifest V3 extension. It also trains and evaluates separate
URL-only, HTML-derived, and combined URL+HTML XGBoost models locally.

> **Project status:** learning/research prototype. Native local training and
> saved-model evaluation are verified. The API/extension still uses the older
> URL-only joblib model; it requires working scikit-learn native dependencies.
> The current Windows policy blocks that serving dependency. See the local audit
> and Mac setup guide. False positives and source bias require further evaluation.

The API computes 21 features from the submitted URL string without visiting the
site. The experimental combined model adds 17 HTML-derived features from a saved
page snapshot. Its extractor parses HTML without executing JavaScript, fetching
resources, or following links. These experiments have not replaced the API model.

## Latest training results: URL + HTML

Full paired PhreshPhish training completed on **2026-10-11**, using the locally
downloaded **36.58 GB dataset (77 original Parquet files)**. Checksums and HTML
feature extraction took about **2 hours 40 minutes**; training and evaluation
from cached numeric features took **20.80 seconds** on the GTX 1660 Ti with CUDA.
The GPU accelerates fitting; parsing and file verification use CPU and disk.

All three variants used the same 397,884 training pages, 56,301 validation pages,
39,866 internal-test pages, and 84,746 retained publisher-test pages. Domains and
identical HTML were grouped within development and excluded from publisher test
when already present in development. Model selection and thresholds used validation.

| Model | Features | Publisher-test accuracy | Precision | Phishing recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| URL-only | 21 | 84.98% | 93.81% | 82.72% | 87.92% | 10.62% |
| HTML-derived | 17 | 84.43% | 86.22% | 90.97% | 88.53% | 28.31% |
| URL + HTML | 38 | 88.75% | 90.65% | 92.52% | 91.58% | 18.58% |

Adding HTML improved accuracy by **3.78 percentage points** and phishing recall
by **9.80 points**, while increasing false-positive rate by **7.96 points** at
the main thresholds. At separate thresholds targeting at most 1% validation
false positives, publisher-test recall was 61.35% for URL-only and 72.37% for
URL+HTML; measured publisher-test false-positive rates were 1.70% and 1.80%.

This comparison measures the effect of adding HTML on the same large dataset.
It does not isolate the effect of dataset size. Earlier experiments used different
test cohorts and cannot establish that comparison. Near-duplicate templates,
HTML truncation, and collection bias still require evaluation.

### Features and importance

HTML signals cover forms, password and hidden fields, form destinations, links,
scripts, iframes, images, title/brand/domain mismatch, page size, and three content
availability/parsing flags. Each document is capped at 1,000,000 UTF-8 bytes for
parsing; missing measurements remain missing rather than becoming zero counts.

The combined model's largest normalized gain importance shares were:

| Feature | Gain share |
|---|---:|
| Link count | 28.48% |
| URL path length | 10.20% |
| URL length | 5.31% |
| Query parameter count | 4.55% |
| Query length | 4.47% |

HTML-derived features account for **48.69%** of total normalized gain shares.
These percentages describe global split importance, not detection rates,
percentiles, causal effects, or the explanation of an individual prediction.

Saved models are under `artifacts/experiments/phreshphish_html_20261010/`, with
`url_only`, `html_features`, and `url_html` subdirectories. All three saved models
reproduced internal/publisher-test confusion matrices on CPU. See
[full results and limitations](docs/html-training-results-2026-10-11.md) and
[training/feature definitions](docs/native-html-training.md).

## Active API model performance

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

## Other experimental alternatives

Domain-grouped cross-validation now compares XGBoost, XGBoost without the direct
HTTPS feature, and character TF-IDF + Logistic Regression. The character model
improves some external ranking metrics but still produces substantial false
alarms. These URL benchmarks complement the completed PhreshPhish URL+HTML
comparison above. **The active API/extension model has not been replaced.**

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
- joblib packaging for the active API model; native UBJ models for local experiments
- PyArrow for batched Parquet dataset processing
- FastAPI, Pydantic, and Uvicorn
- pytest and Ruff
- Chrome Extension Manifest V3 with plain HTML, CSS, and JavaScript

## Repository map

```text
api/                    FastAPI application
artifacts/              Tracked models, metadata, evaluations, and research results
data/raw/               Local original datasets, including full PhreshPhish (ignored)
data/processed/         Local cleaned splits and numeric feature caches (ignored)
docs/                   Architecture and model documentation
extension/              Unpacked Chrome extension
scripts/                Reproducible data and training commands
src/phishguard/         Reusable data, feature, model, and prediction code
tests/                   Unit and API tests
```

## Set up

### Windows laptop (PowerShell / VS Code)

Use the existing checkout with Git, Python 3.12, VS Code and Chrome installed.
In the VS Code PowerShell terminal:

```powershell
cd "C:\Users\hrish\Documents\PhishingURLChecker"
git status
git pull --ff-only origin main
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-model-runtime.txt
.\.venv\Scripts\python.exe -m pip install -e ".[data,dev]"
code .
```

Preserve any local edits before pulling. Create `.venv` only if a suitable
environment does not already exist. In VS Code select
**Python: Select Interpreter → `.venv\Scripts\python.exe`**. These commands call
the environment directly, so PowerShell activation-policy changes are unnecessary.
Do not copy the Mac `.venv` to Windows; recreate it. See the official
[Python virtual-environment guide](https://docs.python.org/3/library/venv.html).

Run the project checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

XGBoost on Windows requires the Microsoft Visual C++ Redistributable runtime;
see the official [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html#python)
if a DLL import error occurs. GPU hardware is not required by this project's CPU workflow.

#### Models and local datasets

GitHub now includes code, tests, extension files, trained models, metadata,
evaluation JSONs/CSVs, and research reports. A pull brings these files into the
existing checkout. Raw datasets, dataset splits, feature caches, and virtual
environments remain local. Retraining requires the datasets separately.

For API inference alone, the required pair is
`artifacts/phishguard_model.joblib` and `artifacts/model_metadata.json`.
Load only your own trusted joblib files, never arbitrary downloaded models.
`requirements-model-runtime.txt` pins the model dependency versions, including
NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, XGBoost 3.4.1 and joblib 1.6.0.
Use these pins when loading the saved Python model:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model-runtime.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Native XGBoost training and saved-model evaluation have run on this Windows
laptop. The current Windows Application Control policy blocks a scikit-learn
native serving dependency, so the full API/extension workflow has not been
verified here. Without a loadable model pair, `/health` reports degraded.

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

### Pull onto your existing Mac checkout

```bash
cd /your/existing/PhishingURLChecker
git status
git pull --ff-only origin main
.venv/bin/python -m pip install -r requirements-model-runtime.txt
.venv/bin/python -m pip install -e '.[data,dev]'
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_transferred_artifacts.py
```

Use the existing Python 3.12 environment, or create a native Mac environment with
`python3.12 -m venv .venv` first. Do not copy the Windows environment. See the
[Mac pull guide](docs/mac-pull-and-models.md) for preserving local artifact
conflicts and resolving native library dependencies. The new models run on CPU;
CUDA is not required for Mac inference.

### Run a saved experimental model

In a Mac terminal, the URL-only model accepts a URL:

```bash
.venv/bin/python scripts/predict_native_snapshot.py --model-dir artifacts/experiments/phreshphish_html_20261010/url_only --url 'https://example.com/login'
```

The combined model requires a saved HTML file. This example uses a synthetic
fixture to demonstrate the command, not measured phishing detection accuracy:

```bash
.venv/bin/python scripts/predict_native_snapshot.py --model-dir artifacts/experiments/phreshphish_html_20261010/url_html --url 'https://example.test/login' --html-file examples/hybrid/login_fixture.html
```

On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.
These commands use the matching saved-model feature schema and do not fetch sites.

### Retrain locally on the full downloaded PhreshPhish dataset

In a VS Code PowerShell terminal, with the full dataset already downloaded:

```powershell
.\.venv\Scripts\python.exe -u scripts/train_native_html_model.py --dataset-dir data/raw/phreshphish --cache-dir data/processed/html_features_v1 --output-dir artifacts/experiments/phreshphish_html_rerun --trees 800 --threads 4 --device cuda
```

Choose a new output directory for each run. Use `--device cpu` for CPU training;
`--extract-only` performs extraction without fitting. This reads the full local
download in batches and reuses verified caches where available. It writes
separate experiment models; it does not activate them in the API. See the
[training guide](docs/native-html-training.md) for the complete procedure.

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

Raw datasets, processed splits, and feature caches are ignored by Git. Selected
model artifacts and research results are tracked for transfer and review.

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

Latest validation: 36 focused data/native-training tests passed, and Ruff checks
passed. All three new models reproduced internal/publisher-test results on CPU.
Full API/scikit-learn tests remain blocked on this Windows machine by its native
dependency policy; see [local audit](docs/local-check-2026-10-09.md).

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
- [x] Download and process the full PhreshPhish dataset locally.
- [x] Train and compare URL-only, HTML-derived, and URL+HTML models on matching rows.
- [x] Verify saved native models on CPU and publish models/results for Mac pulls.
- [ ] Integrate HTML inference into the API/extension after further evaluation.
- [ ] Evaluate near-duplicate templates, temporal splits, and HTML size/truncation effects.
- [ ] Add reputation signals and assess calibration before considering production use.

## Data and generated files

The repository includes selected trained models, metadata, feature-importance
reports, evaluation outputs, and the artifact checksum manifest. Pulling the
latest changes transfers those files without retraining.

Raw datasets (including the 36.58 GB PhreshPhish download), dataset-containing
split files, numeric caches, virtual environments, and temporary test files are
excluded. Download datasets separately when reproducing training. The full
PhreshPhish downloader is `scripts/download_phreshphish_full.py`.
