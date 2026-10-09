# XGBoost experiments — 3 October 2026

## What changed and why

Version 2 uses `XGBClassifier` with 21 lexical URL features. The old
`slash_count` shortcut is omitted. The full transformer and classifier are
saved together, so preparation at inference matches training. The URL feature
definitions are shared across datasets; no page-content fields, brand
whitelists, or dataset row identifiers enter the model.

XGBoost learns successive decision trees. A tree can combine conditions such as
hostname length, IP-host detection, and suspicious words, rather than combining
all features in one linear formula as logistic regression does.

The implementation follows the official [parameter-tuning guide](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
and [scikit-learn estimator guide](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html):

- CPU histogram tree construction (`tree_method="hist"`), four training threads;
- learning rate 0.05 and at most 800 boosting rounds;
- row and feature sampling of 0.8;
- L2 regularization 5.0 and L1 regularization 0.1;
- a fixed search of depths 3, 5, and 7 with minimum child weights 10, 5, and 5;
- early stopping after 40 rounds without improvement in validation logloss;
- select the XGBoost candidate using validation F1;
- choose the threshold on validation data, targeting at least 90% recall;
- evaluate test data after selection finishes.

Unweighted XGBoost is used here. The available datasets are reasonably balanced;
blindly reweighting the classes can distort probability interpretation. The
threshold controls the recall tradeoff. Scores remain uncalibrated model scores.

All mixed-data candidates reached the 800-round limit. That means this run was
bounded by its training budget, rather than stopped early. More rounds are a
possible future validation experiment, not a proven improvement.

## Experiment 1: change the algorithm, keep PhiUSIIL training data

The original 188,753 / 23,300 / 23,317 domain-disjoint splits were preserved.
The selected XGBoost candidate had depth 7 and best iteration 634 (one-based).

| PhiUSIIL test metric | Original 22-feature logistic | New 21-feature XGBoost |
|---|---:|---:|
| Accuracy | 94.41% | 97.80% |
| Phishing precision | 95.76% | 99.02% |
| Phishing recall | 90.85% | 95.75% |
| F1 | 93.24% | 97.36% |
| False positives | 398 | 94 |
| False negatives | 905 | 420 |

This comparison changes both the algorithm and the feature list. A separate
21-feature logistic comparison on the same split achieved 95.46% accuracy and
94.70% F1; its full metrics are saved in the experiment metadata.

However, this XGBoost model still generalized poorly to `balanced_urls.csv`.
After invalid URLs, exact overlaps, and every domain used in any original split
were excluded, 293,638 external rows remained:

| Cross-dataset metric | Original logistic | PhiUSIIL-only XGBoost |
|---|---:|---:|
| Accuracy | 54.81% | 54.68% |
| Precision | 54.69% | 54.61% |
| Recall | 99.94% | 99.97% |
| F1 | 70.69% | 70.64% |
| ROC-AUC | 62.07% | 71.60% |
| Legitimate URLs incorrectly flagged | 99.32% | 99.65% |

These results confirm that XGBoost alone does not cure source-collection bias.
The external dataset was already examined during development, so even this
cross-dataset benchmark is not a blind, newly acquired external evaluation.

## Experiment 2: more representative training data

For the broader model, both available sources were cleaned and combined **before**
domain splitting. This places a domain entirely within one split even when its
URLs came from different sources.

- PhiUSIIL: 235,370 cleaned URLs.
- Balanced source: 339,356 cleaned URLs; 368 invalid rows and 276 remaining
  duplicate URLs were removed from its 340,000 raw rows.
- Cross-source exact duplicates removed: 10; conflicting labels found: 0.
- Combined usable rows: 574,716.

| Split | Rows |
|---|---:|
| Train | 457,544 |
| Validation | 56,621 |
| Test | 60,551 |

The selected candidate used depth 7, minimum child weight 5, and 800 trees.
The validation-selected threshold was 0.4711279273.

| Same mixed test set | Logistic, 21 features | XGBoost, 21 features |
|---|---:|---:|
| Accuracy | 83.54% | 93.19% |
| Precision | 81.22% | 93.48% |
| Recall | 86.91% | 92.74% |
| F1 | 83.97% | 93.11% |
| ROC-AUC | 91.72% | 98.33% |
| False-positive rate | 19.79% | 6.37% |
| False positives | 6,037 | 1,942 |
| False negatives | 3,932 | 2,182 |

The XGBoost test breakdown is 94.77% accuracy on the PhiUSIIL subset and 91.97%
on the balanced-source subset. The false-positive rate is 0.77% and 10.84%
respectively. Reporting the aggregate alone hides this source difference.

**This is a mixed-source, domain-disjoint holdout.** The balanced source is now
part of training and can no longer serve as an independent external test for
this model. A third source or a later verified time window is needed to assess
new-source generalization. Neither dataset's provenance or collection dates
have been independently verified in this experiment.

## Manual URL sanity checks

These are illustrative spot checks, not an independently sampled benchmark.
Apple, Google, GitHub, and OpenAI registered domains have examples in training.

| URL | Mixed XGBoost score / 100 | Result |
|---|---:|---|
| `https://www.apple.com/de/` | 5.63 | low |
| `https://www.google.com/maps/` | 3.70 | low |
| `https://github.com/features` | 15.67 | low |
| `https://openai.com/research/` | 11.59 | low |
| `https://example.com/` | 72.72 | high — known false positive |
| `http://192.0.2.10/login/verify-account` | 99.98 | high — synthetic warning pattern |

The documentation IP address and `.test` examples are synthetic strings, not
verified malicious websites. Their scores illustrate behavior only.

## Reproduce training

On macOS, install the OpenMP runtime if XGBoost cannot load:

```bash
brew install libomp
python -m pip install -e '.[data,dev]'
```

Create separate cleaned source files:

```bash
python scripts/prepare_data.py \
  --input data/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
  --url-column URL --label-convention phiusiil \
  --output data/processed/sources/phiusiil.csv

python scripts/prepare_data.py \
  --input data/external/balanced_urls.csv \
  --url-column url --label-convention text \
  --output data/processed/sources/balanced.csv

python scripts/split_data.py \
  --input data/processed/sources/phiusiil.csv data/processed/sources/balanced.csv \
  --output-dir data/processed/mixed

python scripts/train_model.py \
  --data-dir data/processed/mixed \
  --artifacts-dir artifacts/experiments/xgboost_mixed

python scripts/activate_model.py \
  --artifacts-dir artifacts/experiments/xgboost_mixed
```

Activation archives the previous API model. Restart Uvicorn and reload the
extension after activation. The native `.ubj` export is also saved in the
experiment directory; the API uses joblib because it includes the transformer.

## Use a different dataset

Specify its URL column, target column, and source label meaning explicitly:

```bash
python scripts/prepare_data.py \
  --input data/raw/another_dataset.csv \
  --url-column address --label-column status \
  --legitimate-label benign --phishing-label malicious \
  --output data/processed/sources/another.csv
```

Then split that cleaned CSV to a separate directory and pass it to
`train_model.py --data-dir`. Never re-label additional attack types as phishing
without a justified definition of the target task.

For an unused third dataset, evaluate a frozen model and optionally remove
shared domains as well as exact URL overlaps:

```bash
python scripts/evaluate_external.py \
  --input data/external/third_dataset.csv \
  --url-column address --label-column status \
  --legitimate-label benign --phishing-label malicious \
  --artifacts-dir artifacts/experiments/xgboost_mixed \
  --exclude-shared-domains
```

Do not adjust hyperparameters or the threshold against that external result
while continuing to describe it as an untouched final test.
