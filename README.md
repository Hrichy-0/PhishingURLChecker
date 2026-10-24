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

On new real sites both saved models falsely flag about 27–30% of normal pages,
against 10.6% (URL-only) and 18.6% (URL+HTML) on their earlier PhreshPhish test. The URL+HTML model also flags almost
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

## Models served by the API

`POST /predict` uses the URL-only model in
`artifacts/experiments/phreshphish_html_20261010/url_only/` (cutoff
`0.4901971220970154`). `POST /predict_html` uses the URL+HTML model in
`.../url_html/` (cutoff `0.2209`). On loading, the API verifies each model's
checksum, feature order and tree range, and runs on CPU. `/health` reports both.
Restart the API and reload the unpacked extension after changing code.

All 21 features are supplied correctly; the learned trees use 20, with no split
using `has_explicit_port`. Zero split usage is not a broken input. Feature
definitions are unchanged; existing suffix/counting limitations still apply.

To run the preserved older model instead:

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

Requires Python 3.11+ (3.12 used in development), Git and Chrome. From the
repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-model-runtime.txt
.venv/bin/python -m pip install -e '.[data,dev]'
```

On Windows use `py -3.12 -m venv .venv` and `.\.venv\Scripts\python.exe` in place
of `.venv/bin/python`. `requirements-model-runtime.txt` pins the versions the saved
models were built with (XGBoost 3.4.1, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1).
If XGBoost reports a missing `libomp.dylib` on macOS, run `brew install libomp`;
on Windows it needs the Microsoft Visual C++ Redistributable.

The trained models and results are in the repository; raw datasets are not.
Check that the model files arrived intact:

```bash
.venv/bin/python scripts/verify_transferred_artifacts.py
```

## Run the API

```bash
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs>, or send:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/login"}'
```

`phishing_probability` is a model score, not a calibrated certainty. The response
also includes the threshold, a risk level and simple URL-based reasons.

## Load the Chrome extension

1. Keep the API running on `127.0.0.1:8000`.
2. Open `chrome://extensions` and enable **Developer mode**.
3. Choose **Load unpacked** and select the `extension/` directory.
4. Open an HTTP or HTTPS page and click the PhishGuard toolbar action.

## Try the saved models directly

Score a URL with the URL-only model, or a URL plus a saved HTML file with the
URL+HTML model. The example HTML is an inert fixture; neither command fetches sites.

```bash
.venv/bin/python scripts/predict_native_snapshot.py --model-dir artifacts/experiments/phreshphish_html_20261010/url_only --url 'https://example.com/login'
.venv/bin/python scripts/predict_native_snapshot.py --model-dir artifacts/experiments/phreshphish_html_20261010/url_html --url 'https://example.test/login' --html-file examples/hybrid/login_fixture.html
```

## Reproduce the main experiment

Download the [PhreshPhish dataset](https://huggingface.co/datasets/phreshphish/phreshphish)
(36.6 GB) with `scripts/download_phreshphish_full.py`, then train and compare the
URL-only, HTML-only and URL+HTML models on identical splits:

```bash
.venv/bin/python -u scripts/train_native_html_model.py --dataset-dir data/raw/phreshphish --cache-dir data/processed/html_features_v1 --output-dir artifacts/experiments/NEW_RUN --trees 800 --threads 4 --device cpu
```

Use a new output directory for each run; `--device cuda` uses an NVIDIA GPU.
Grouped cross-validation uses `scripts/cross_validate_native.py`, and the
real-website test is described in the
[RPA report](docs/archive/rpa-real-world-testing.md). Training details are in the
[HTML training guide](docs/archive/native-html-training.md).

## Verify the project

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

Latest validation (25 October 2026, macOS): 147 tests pass and Ruff checks pass.

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
- [x] Verify saved native models on CPU and publish models and results.
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
reports, evaluation outputs and the artifact checksum manifest, so the models run
without retraining. Result files that contain raw URLs are kept out of the repository.

Raw datasets (including the 36.58 GB PhreshPhish download), dataset-containing
split files, numeric caches, virtual environments, and temporary test files are
excluded. Download datasets separately when reproducing training. The full
PhreshPhish downloader is `scripts/download_phreshphish_full.py`.
