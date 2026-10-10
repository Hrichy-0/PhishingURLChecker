# Local verification and larger-training preparation — 9 October 2026

## Subsequent update on the same date

The Mac data and artifacts have now been transferred. Active model and metadata
SHA-256 values match the previously recorded Mac hashes, and its feature
percentages were shown to the user. Dependencies were subsequently installed with
authorization. Ruff lint/format and dependency consistency checks pass.

Windows Application Control blocks scikit-learn's `_openmp_helpers` native
extension, so the full existing suite and API smoke test remain blocked. Native
XGBoost CPU fitting works normally. A separate million-URL native experiment was
launched without changing Windows security settings; see
[the native training workflow](native-local-training.md) and its local run log.
The earlier findings below describe the checkout before this transfer/install.

## Outcome

Runtime operation and training could not be verified. This Windows checkout has
no `data/` or `artifacts/` directories. The current model files are on the user's
Mac. Its `.venv` contains pip only, without the project dependencies. Dependency
installation failed inside the sandbox; the requested installation outside the
sandbox was declined. No training or model activation ran.

The request to see the old feature percentages before training is pending the
Mac metadata or the user's decision to proceed without that comparison.

## Checks actually executed

- All 38 existing Python source, command, API and test files parsed successfully
  with Python's AST parser.
- `node --check extension/popup.js` passed.
- `extension/manifest.json` parsed and declares Manifest V3.
- `python -m pytest -q` could not run: pytest is absent.
- Ruff lint and format checks could not run: Ruff is absent.
- No live API prediction, browser extension interaction or model evaluation ran.
- Alternate local Python installations do not have the complete dependencies
  needed for the test suite or XGBoost training.

Syntax checks do not establish runtime correctness. The API's `/health` currently
checks that the two artifact files exist; it does not check that a model can load
or predict. API unit tests use a fake prediction service, so passing those tests
alone would not verify a transferred or newly trained model.

## Current model: recorded, not verified on this laptop

The existing model card and experiment record describe:

- `URLFeatureExtractor` followed by `XGBClassifier`.
- 21 lexical URL features; `slash_count` excluded.
- Maximum depth 7, 800 boosting rounds, learning rate 0.05, four CPU threads.
- Threshold 0.4711279273033142, chosen on validation data.
- 574,716 cleaned rows from PhiUSIIL and `balanced_urls.csv`.
- Train 457,544; validation 56,621; test 60,551, grouped by registered domain.
- Recorded test accuracy 93.19%, phishing precision 93.48%, recall 92.74%,
  F1 93.11%, ROC-AUC 98.33%, and false-positive rate 6.37%.

The 21 features are URL/hostname/path/query length; dot, hyphen, underscore and
digit counts; digit ratio; subdomain and query parameter counts; percent-encoding
and special-character counts; IP hostname, HTTPS, @ symbol and explicit-port
indicators; suspicious-word count; character entropy; longest-token length; and
TLD length. Knowing this list does not reveal learned weights.

Historical external evaluations are substantially worse: PhishNChips 44.42%
accuracy / 88.22% false-positive rate, and the retained complete-URL ealvaradob
subset 71.50% accuracy / 34.36% false-positive rate. These are recorded results
from the Mac, not new local measurements.

### Feature importance percentages

Exact old-model percentages are unavailable without its saved metadata or binary.
`train_model.py` already writes `feature_importances` with `gain_importance`
values to `model_metadata.json`. The new standard-library inspection command can
read that file without installing ML packages or copying the larger model binary:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_model_metadata.py --metadata artifacts/model_metadata.json --output artifacts/reports/current-model.md
```

The percentage is `100 * feature_gain_importance / sum(feature_gain_importances)`.
This is a global relative importance share, not a percentile rank, detection
percentage, causal effect or contribution to a particular URL. Gain alone does
not say whether a feature increases or decreases the phishing score. For a
specific URL, native XGBoost contributions can be computed after loading the
trusted model, using its selected early-stopping tree range.

## Larger dataset candidates

These counts and licenses come from publisher pages retrieved on 9 October 2026;
the files have not been downloaded or audited locally.

| Dataset | Published raw rows | Fit for this experiment | Caveats |
|---|---:|---|---|
| [Flower Labs fed-phishing-urls](https://huggingface.co/datasets/flwrlabs/fed-phishing-urls) | 1,132,306 | Large, compact URL/label Parquet; card lists Apache-2.0 | Mixes full URLs and addresses without schemes; upstream ealvaradob was already used for project development; publisher label description includes phishing/malicious |
| [PhreshPhish](https://huggingface.co/datasets/phreshphish/phreshphish) | 666,315 | Newer phishing/benign URL and HTML pairs, with dates; CC BY 4.0 | Full dataset is 36.6 GB; use only URL/label/date columns for this URL-only experiment; some records lack schemes |
| [Grambeddings](https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/) | 800,000 | Balanced 400,000 phishing / 400,000 legitimate; collected 2019–2021 | Access requires a [request form](https://docs.google.com/forms/d/e/1FAIpQLScuwosnH9JIJYKUeTeR26xxiG6bWXhxYoHjoNSketW5MrCUxQ/viewform); academic-use terms; labels differ from the project convention |

Flower is the candidate for a million-row experiment; its usable row count cannot
be promised before cleaning. PhreshPhish offers clearer recent collection context
and a useful independent evaluation candidate. Increasing row count alone does
not resolve the source/protocol biases documented in the existing audits.

## Conditions for executing local training

1. Restore the current metadata for the requested before-training percentages,
   or receive the user's answer allowing that comparison to be skipped.
2. Install the project's runtime dependencies and check tools into `.venv` once
   installation is authorized. Parquet candidates also require a reader such as
   PyArrow, which is not currently in `pyproject.toml` or installed here.
3. Pin the selected dataset revision and save file hashes, original labels and
   original protocols. Audit class counts, invalid rows, conflicts and duplicates.
4. Do not silently manufacture HTTP/HTTPS for addresses without a scheme. A
   separate protocol-invariant experiment could explicitly normalize them, but
   must normalize serving features identically and disclose the assumption.
5. Keep registered domains entirely within a single split. Select model and
   threshold on training/validation only; keep an independent source or temporal
   test frozen. The earlier development datasets are not untouched final tests.
6. Save a new experiment under `artifacts/experiments/`, including metrics,
   feature percentages, provenance and runtime versions. Verify saved-pipeline
   predictions and API behavior before considering activation.

The existing `prepare_data.py`, `split_data.py` and `train_model.py` commands can
train a cleaned, complete-URL CSV. The current downloader supports CSV/JSON from
two older pinned sources; it does not yet support these Parquet candidates.
No dataset-specific download or preprocessing path has been executed here.
