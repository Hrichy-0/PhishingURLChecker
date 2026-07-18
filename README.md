# PhishingURLChecker (PhishGuard ML)

PhishGuard is an end-to-end, URL-only phishing risk detector. It cleans and
splits the PhiUSIIL Kaggle dataset, trains an explainable logistic-regression
pipeline, serves predictions through FastAPI, and connects to a Chrome
Manifest V3 extension.

> **Project status:** learning/portfolio prototype. The full pipeline works,
> but real-world testing exposed dataset bias on ordinary deep links such as
> `https://www.apple.com/de/`. The next model iteration will address this
> before the extension is treated as a dependable security control.

The detector never visits a submitted URL. All 22 features are computed from
the URL string, which keeps inference fast and avoids executing content from a
suspected site.

## Current measured performance

The final test set is domain-disjoint from training and validation data.

| Metric | Result |
|---|---:|
| Accuracy | 94.41% |
| Phishing precision | 95.76% |
| Phishing recall | 90.85% |
| F1 | 93.24% |
| ROC-AUC | 98.49% |
| False positives | 398 |
| False negatives | 905 |

These are offline dataset results, not a guarantee of future performance. See
[`docs/model-card.md`](docs/model-card.md) for limitations.

## Technology

- Python 3.11+
- pandas and scikit-learn
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

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,dev]'
```

## Rebuild everything

The raw dataset is the **PhiUSIIL Phishing URLs** Kaggle dataset. Its source
labels are `0 = phishing` and `1 = legitimate`; the cleaning step converts them
to the project convention `1 = phishing` and `0 = legitimate`.

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/split_data.py
python scripts/train_model.py
```

Generated data and artifacts are ignored by Git because they are reproducible
and potentially large.

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
- [ ] Replace probability language with calibrated model-risk language.
- [ ] Add a separate real-world sanity set containing benign deep links.
- [ ] Remove or redesign shortcut-prone path features and retrain.
- [ ] Add reputation/content signals before considering production use.

## Data and generated files

The raw Kaggle CSV, processed splits, and trained artifacts are intentionally
excluded from Git. Reproduce them with the commands above. This keeps the
repository small and avoids redistributing a third-party dataset.
