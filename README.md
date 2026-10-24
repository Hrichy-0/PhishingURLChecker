# PhishingURLChecker (PhishGuard ML)

PhishGuard is a research prototype for phishing detection that runs on the
user's own device. It is part of the TalentUD project **Evaluating On-Device
Phishing Detection Models on Real-World Web Pages**, which compares a URL-only
model with a URL + static-webpage-content model.

- **Models:** two XGBoost models trained on the PhreshPhish dataset. The URL-only
  model uses 21 features of the URL string; the URL+HTML model adds 17 features
  of the page's HTML.
- **Local API:** a FastAPI service on `127.0.0.1` that scores a URL, or a URL plus
  the HTML the browser already loaded. It never fetches websites.
- **Chrome extension:** a Manifest V3 popup with "Check this URL" (URL-only) and
  "Check URL + page content (experimental)" (URL+HTML).
- **Real-website testing:** a Playwright script that drives the real popup on a
  fixed list of websites.

Inference runs locally through the Python service; running the models inside the
browser itself is planned but not implemented yet. The full summary is in the
[project overview](docs/project-overview.md).

The separate [Word project documentation](PhishingURLChecker_Project_Documentation.docx)
records the implementation history, model comparisons, evaluation limitations,
current review pilot and Mac/Windows operating commands, updated 19 October 2026.

## Progress so far

### 1. Cross-validation

Five-fold grouped cross-validation on 454,185 development records. Records that
share a domain or identical page content stay in the same fold (154,074 groups).
A separate grouped inner validation set chose early stopping and the threshold;
the test sets were not used.

| Model (F1 threshold policy) | Accuracy | Precision | Recall | False-positive rate |
|---|---:|---:|---:|---:|
| URL-only | 90.93% | 90.09% | 89.95% | 8.26% |
| URL + HTML | 92.59% | 95.01% | 88.35% | 3.87% |

- **Paired bootstrap** (whole groups resampled): adding HTML lowered the
  false-positive rate by 4.4 percentage points (95% CI 3.8–5.0). The recall
  differences were not statistically conclusive.
- **Instability:** the URL-only false-positive rate varied from 4.25% to 16.71%
  across folds, driven by one large linked group of domains.

### 2. Leakage audit

- **Integrity:** all 578,797 records match their original checksums, and the URL
  features re-extract exactly.
- **Overlap:** no URL, domain or identical page content appears in more than one
  split or fold.
- **Label conflicts:** 66 records share identical content but have opposite
  labels. They are mostly generic pages ("store unavailable", server default,
  geo-blocked), so they were flagged rather than relabelled.

### 3. Real-website test

The Playwright runner opened 197 real legitimate websites (banks, government,
universities, news, shopping, email and software), excluding every domain seen in
the project's data and each verified as an official site. All 195 scored popups
matched the API's answer (median 76 ms).

| Legitimate sites | n | URL+HTML false alarms | URL-only false alarms |
|---|---:|---:|---:|
| Loaded normally (HTTP 200) | 152 | 29.6% | 27.0% |
| Blocked the automated browser (403/429/500) | 23 | 95.7% | 21.7% |

On new real sites both models falsely flag about 27–30% of normal pages, far
above the 4–8% measured in cross-validation. The URL+HTML model also flags almost
every bot-check or error page. Details: [RPA report](docs/archive/rpa-real-world-testing.md)
and [`results.json`](artifacts/experiments/rpa_url_html_20261024/results.json).

### Next steps

1. Close the gap between development and real-world false alarms: decide how
   blocked/error pages are treated, then test threshold or training changes on
   separate legitimate pages.
2. Finish independent label review for the fresh OpenPhish/Tranco benchmark and
   score it once.
3. Move inference into the browser and measure latency, memory and privacy.

These are research results, not production approval. A low score never means a
site is safe.

## Run the main local model

Start the main model from the repository root without an environment override:

```bash
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

It loads `artifacts/experiments/phreshphish_html_20261010/url_only/`, verifies the
model checksum and feature order, runs on CPU, and retains its saved cutoff
`0.4901971220970154`. `/health` and `/predict` identify the model, checksum and
21-feature count. The popup continues to label the result experimental. Restart
any existing API process and reload the unpacked extension after changing code.
On Windows use `.\.venv\Scripts\python.exe` instead of `.venv/bin/python`.

All 21 features are supplied correctly; the learned trees use 20, with no split
using `has_explicit_port`. Zero split usage is not a broken input. Feature
definitions are unchanged; existing suffix/counting limitations still apply.

To explicitly run the preserved older model on macOS:

```bash
PHISHGUARD_MODEL=legacy .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Leave `PHISHGUARD_EXPERIMENTAL_MODEL_DIR` unset for normal use; it remains a
research override and cannot be combined with `PHISHGUARD_MODEL=legacy`.
A missing/corrupt main model reports unavailable, never silently loads the old
model.

## Preliminary training comparison: URL + HTML

These results are preserved research evidence, not approval for deployment.
URL+content comparison remains within the submitted research scope. The URL-only
package from this experiment remains the main local model for now.

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
[full results and limitations](docs/archive/html-training-results-2026-10-11.md) and
[training/feature definitions](docs/archive/native-html-training.md).

## Legacy model performance (preserved for rollback)

The previous XGBoost model was trained on PhiUSIIL plus `balanced_urls.csv`.
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
[`docs/archive/model-card.md`](docs/archive/model-card.md) for limitations and
[`docs/archive/xgboost-experiments.md`](docs/archive/xgboost-experiments.md) for the same-split
logistic comparison (83.54% accuracy), the PhiUSIIL-only experiment (97.80%),
and the reproducible workflow.

Frozen-model evaluation on two additional Kaggle datasets found substantially
worse results, with high false-alarm rates and missing-protocol limitations.
The 93.19% holdout score must not be presented as universal accuracy. See
[`docs/archive/external-evaluation.md`](docs/archive/external-evaluation.md) for all assumptions,
metrics, class-imbalance baselines, and commands. The active model was unchanged.

Two Hugging Face evaluations likewise found poor results: 44.42% accuracy on
retained PhishNChips URLs and 71.50% on the retained complete-URL subset of
ealvaradob's URL dataset. Both excluded previously used domains and preserved
original protocols. See [`docs/archive/huggingface-evaluation.md`](docs/archive/huggingface-evaluation.md)
for the fixed-model procedure, high false-alarm rates, and selection limitations.

## Other experimental alternatives

Domain-grouped cross-validation now compares XGBoost, XGBoost without the direct
HTTPS feature, and character TF-IDF + Logistic Regression. The character model
improves some external ranking metrics but still produces substantial false
alarms. These URL benchmarks complement the completed PhreshPhish URL+HTML
comparison above. **The main API now selects the native URL-only model;
the older joblib pair and all research artifacts remain unchanged.**

See [`docs/archive/model-comparison.md`](docs/archive/model-comparison.md) for measured results,
the files to study, reproducible commands, and the remaining evaluation work.
Source-held-out XGBoost tuning is available through the same benchmark command
(`--mode tune`), with six explicit candidates rather than a large search framework.
The [file-purpose guide](docs/archive/architecture.md#file-purpose-audit--2026-10-06)
distinguishes runtime files, current research commands, and optional/history files.

The [consolidated dataset audit](docs/archive/dataset-audit-report.md) covers the
training sources, an uploaded URL dataset and the Hugging Face data, including
label conflicts, false-positive patterns, source bias and next steps.

## Technology stack

- Python 3.11+
- pandas, scikit-learn, and XGBoost
- native UBJ for the main URL model; joblib for the preserved legacy baseline
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
docs/                   Project overview and detailed reports (docs/archive/)
examples/               Inert HTML fixtures and the RPA website lists
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
evaluation JSONs and research reports. Result files that contain raw URLs stay local. A pull brings these files into the
existing checkout. Raw datasets, dataset splits, feature caches, and virtual
environments remain local. Retraining requires the datasets separately.

For default API inference, the required pair is `xgboost_model.ubj` and
`model_metadata.json` under
`artifacts/experiments/phreshphish_html_20261010/url_only/`. The older root-level
joblib/metadata pair is used only with `PHISHGUARD_MODEL=legacy`.
Load only your own trusted artifacts, never arbitrary downloaded joblib models.
`requirements-model-runtime.txt` pins the model dependency versions, including
NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, XGBoost 3.4.1 and joblib 1.6.0.
Use these pins when setting up the saved-model runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model-runtime.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Native XGBoost training and saved-model evaluation ran on the Windows training
laptop. Its policy previously blocked a scikit-learn native dependency in the
legacy serving path. The new default does not deserialize that legacy pipeline,
but full Windows API/extension behavior still needs verification on that machine;
do not disable security policy to make it work. Without a loadable selected model,
`/health` reports degraded.

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
[Mac pull guide](docs/archive/mac-pull-and-models.md) for preserving local artifact
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
[training guide](docs/archive/native-html-training.md) for the complete procedure.

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

To reproduce the older mixed-source model, follow the commands in
[`docs/archive/xgboost-experiments.md`](docs/archive/xgboost-experiments.md#reproduce-training).
`scripts/activate_model.py --artifacts-dir ...` replaces the legacy joblib pair
and archives the previous pair; it does not change the default native selection.
Development candidates still fail its activation guard. Restart the API after
changing the selected model or artifacts.

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

Latest validation (25 October 2026, macOS): 147 tests pass and Ruff checks pass.
On Windows, the scikit-learn legacy path was previously blocked by a native
dependency policy; see the [local audit](docs/archive/local-check-2026-10-09.md).

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
[`docs/archive/project-journey.md`](docs/archive/project-journey.md). The architecture is in
[`docs/archive/architecture.md`](docs/archive/architecture.md), and responsible-use details
are in [`docs/archive/model-card.md`](docs/archive/model-card.md).

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
- [x] Add an experimental URL+HTML endpoint and popup button for RPA testing.
- [x] Run grouped cross-validation and a leakage audit for both models.
- [x] Run the RPA test on 197 real legitimate websites.
- [ ] Reduce real-world false alarms before deciding on HTML integration.
- [ ] Score the fresh OpenPhish/Tranco benchmark after independent label review.
- [ ] Run the models inside the browser (ONNX) and measure latency and memory.
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
