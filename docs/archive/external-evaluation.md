# Frozen-model external evaluation — 2026-10-03

## Outcome

The deployed mixed-source XGBoost model does not demonstrate reliable
cross-dataset performance on the two additional datasets tested here. Results
are substantially worse than its 93.19% mixed-source holdout accuracy.
Missing URL protocols also prevent these tests from being definitive estimates
of performance on complete browser URLs. Do not choose the better protocol
assumption after seeing the results and advertise it as the model's accuracy.

No training, activation, threshold selection, or feature-transformer changes
were performed. The operating threshold stayed at 0.4711279273033142.
SHA-256 checks confirmed that the active model and metadata were unchanged:

- Model: `89ae67ee2911e4072a699afb21307015f3f1635e98acfbab579f1ac2740188c7`
- Metadata: `7dd13d9ec074a4db7889f486dc0bab8a81399dc5388a1603a835d9abcf185c1e`

## Sources and preparation

Downloaded the published Kaggle archives on 2026-10-03:

1. [Phishing URLs Dataset with Extracted Features](https://www.kaggle.com/datasets/victusadi/phishing-urls-dataset-with-extracted-features)
   by Victus Adi. File: `phishing_features.csv`, 160,064 rows. Use `url` and
   `label`, mapping 0 to legitimate and 1 to phishing. Recompute features from
   URLs using the saved pipeline; do not use the publisher's feature columns.
2. [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
   by Tarun Tiwari. File: `phishing_site_urls.csv`, 549,346 rows. Use `URL` and
   `Label`, mapping `good` to legitimate and `bad` to phishing.

These are publisher-provided labels, not independently verified current website
verdicts. Source provenance, shared upstream feeds, collection dates, and label
quality are not fully audited. Different dataset names do not prove independent
collection. Raw data and archives remain ignored by Git and are not republished.
No listed URL was opened, resolved, or fetched.

The first file has 159,244 phishing and only 820 legitimate rows. Every legitimate
URL lacks a protocol. In the second file, 549,217 URL strings lack a protocol
at the start. Embedded `https://` text in a query is not the URL's protocol.

Two label-independent assumptions were tested separately: add `http://` or
`https://` only where the initial protocol is missing. Existing protocols stay
unchanged; unsupported protocols and malformed URLs are rejected. This is a
sensitivity test, not evidence of the original protocol. The evaluator's default
remains rejection rather than guessing a protocol.

All exact URLs and registered domains found in the active model's training,
validation, or original test splits are excluded. Reference split hashes must
match training metadata. There is no resampling to make scores look better.

The second dataset contains one URL with contradictory labels across three
records. All three records are excluded, not relabeled. After excluding conflicts,
1,198 invalid rows and 41,690 duplicate rows are removed. For the first dataset,
seven invalid rows and no duplicates are removed.

## Results

False-positive rate is the fraction of legitimate URLs incorrectly flagged.
Recall is the fraction of phishing URLs detected.

| Dataset | Missing-protocol assumption | Unseen-domain rows | Accuracy | Precision | Recall | F1 | False-positive rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Extracted Features | HTTP | 128,511 | 81.64% | 99.55% | 81.93% | 89.89% | 94.95% |
| Extracted Features | HTTPS | 128,511 | 81.64% | 99.55% | 81.93% | 89.89% | 95.15% |
| Phishing Site URLs | HTTP | 258,250 | 51.17% | 34.04% | 93.70% | 49.93% | 63.77% |
| Phishing Site URLs | HTTPS | 258,250 | 67.51% | 41.13% | 57.97% | 48.12% | 29.14% |

The retained class counts are identical between protocol assumptions:

- Extracted Features: 128,016 phishing / 495 legitimate. Predicting phishing
  for every URL would achieve 99.61% accuracy. The model's high precision reflects
  this extreme class imbalance; 470 or 471 of 495 legitimate URLs are flagged.
  This is unsuitable as a representative balanced benign/phishing benchmark.
- Phishing Site URLs: 67,123 phishing / 191,127 legitimate. Predicting legitimate
  for every URL would achieve 74.01% accuracy. Both evaluated model accuracies
  are below that baseline, although the model detects some phishing URLs.

The HTTP assumption detects more phishing but produces many more false alarms.
The HTTPS assumption misses 42.03% of phishing. Protocol sensitivity, source
distribution shift, and potential labeling differences all require investigation;
these measurements alone do not isolate their causal contributions.

## Reproduce without retraining

From the project root, with the existing virtual environment activated:

```bash
python scripts/evaluate_external.py \
  --input data/external/phishing_features.csv \
  --label-convention project \
  --missing-scheme http \
  --exclude-shared-domains \
  --output-dir artifacts/evaluations/kaggle_lexical_http

python scripts/evaluate_external.py \
  --input data/external/phishing_site_urls.csv \
  --url-column URL --label-column Label \
  --phishing-label bad --legitimate-label good \
  --missing-scheme http \
  --exclude-conflicting-urls --exclude-shared-domains \
  --output-dir artifacts/evaluations/kaggle_sites_http
```

Repeat with `--missing-scheme https` and distinct output directories ending in
`_https` to obtain both sensitivity results. Do not run `train_model.py`.

Each output directory contains `external_evaluation.json`, `false_positives.csv`,
and `false_negatives.csv`. JSON records the input/model fingerprints, explicit
label map and protocol assumption, class counts, exclusions, fixed threshold,
majority baseline, and full metrics. Never open suspected URLs from error files.

## Next evaluation

Keep this model frozen. Obtain a dataset with original complete HTTP(S) URLs,
documented labels and collection times, and substantial legitimate and phishing
coverage. Run the default `--missing-scheme reject` policy and exclude previously
used domains. If we later improve features, models, or thresholds in response to
these results, these datasets become development evidence; a fresh untouched
dataset must supply the final evaluation.
