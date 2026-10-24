# Hugging Face frozen-model evaluation — 2026-10-03

## Outcome

The active mixed-source XGBoost model was evaluated on two Hugging Face datasets
without training, threshold tuning, or changes to inference features. Accuracy
was 44.42% on the retained PhishNChips URLs and 71.50% on the retained complete
URL subset of the larger phishing dataset. Both results are below their respective
majority-class accuracy baselines and show substantial false alarms.

These results do not establish reliable generalization. The model remains an
educational prototype, not a dependable standalone phishing blocker.

## Sources

| Hugging Face repository | Selected file | Pinned revision |
|---|---|---|
| [AreLit/PhishNChips](https://huggingface.co/datasets/AreLit/PhishNChips) | `core_emails.csv` | `89afcc39610084298c4679159cb2e27d9ffffa46` |
| [ealvaradob/phishing-dataset](https://huggingface.co/datasets/ealvaradob/phishing-dataset) | `urls.json` only | `55caf9af01ae498d68ff44886ff69a4b165eb54b` |

PhishNChips is an email-agent benchmark. This experiment uses only `url_raw` and
`phish_label`, not email content, prompts, model results, or publisher features.
Consequently this is a URL-only stress test, not a reproduction of the benchmark's
LLM email-agent evaluations. Its card describes malicious indicators from
PhishTank, OpenPhish, and GitHub collections and legitimate seeds from Tranco and
other sources. Labels are publisher-provided, not independently verified website
verdicts. Some legitimate seeds and email text are derived/synthetic; label quality
and sample construction limit interpretation.

The ealvaradob card documents `1 = phishing`, `0 = benign` and identifies a Kaggle
URL compilation as its URL source. Only the URL-specific file is selected; the
mixed email/SMS/HTML files and repository loading script are not used. Hosting on
Hugging Face does not prove independent collection or independence from other
public benchmarks. No remote Python code or dataset-listed URL was executed,
opened, crawled, or resolved.

The PhishNChips card lists mixed source-specific rights; the ealvaradob card lists
Apache-2.0. Raw records are kept locally, ignored by Git, and not republished.

An additional Parquet candidate, `pirocheto/phishing-url`, was inspected but not
evaluated because installing a Parquet reader was not approved. No dependency was
installed and the project dependency list was left unchanged.

## Fixed evaluation policy

- Model: the existing `artifacts/phishguard_model.joblib` pipeline.
- Threshold: **0.4711279273033142**, unchanged from validation selection.
- Preserve original protocols; reject incomplete or invalid HTTP(S) URLs.
- Normalize labels to the project convention, `0 = legitimate`, `1 = phishing`.
- Deduplicate exact URL strings; exclude all contradictory-label records if any.
- Exclude exact URLs and registered domains found in the model's training,
  validation, or original test splits; verify reference split fingerprints.
- No resampling, fine-tuning, threshold search, or changes to serving features.

SHA-256 checks confirmed the active model and metadata are unchanged:

- Model: `89ae67ee2911e4072a699afb21307015f3f1635e98acfbab579f1ac2740188c7`
- Metadata: `7dd13d9ec074a4db7889f486dc0bab8a81399dc5388a1603a835d9abcf185c1e`

## Cleaning and overlap accounting

| Stage | PhishNChips core | ealvaradob URL file |
|---|---:|---:|
| Downloaded rows | 2,000 | 835,697 |
| Missing/invalid URLs removed | 0 | 608,193 |
| Duplicate URL rows removed | 303 | 14,295 |
| Exact previously used URLs excluded | 50 | 2,201 |
| Remaining rows on previously used domains excluded | 652 | 45,385 |
| Final evaluated URLs | 995 | 165,623 |
| Legitimate evaluated URLs | 501 | 19,338 |
| Phishing evaluated URLs | 494 | 146,285 |

No conflicting labels were found in either selected file. The larger file
contains 607,938 strings without an initial protocol. Unlike the previous Kaggle
sensitivity tests, none were assigned an assumed protocol. Rejecting incomplete
URLs changes the class distribution substantially: the original file contains
444,933 benign and 390,764 phishing rows, whereas the evaluated subset is 88.32%
phishing. Results therefore apply only to the retained complete-URL subset, not
the full 835,697-row dataset. Exclusion can introduce selection bias.

## Results

| Metric | PhishNChips | ealvaradob complete-URL subset |
|---|---:|---:|
| Accuracy | 44.42% | 71.50% |
| Majority-class accuracy baseline | 50.35% | 88.32% |
| Phishing precision | 46.42% | 94.09% |
| Phishing recall | 77.53% | 72.27% |
| F1 | 58.07% | 81.75% |
| ROC-AUC | 68.80% | 76.36% |
| False-positive rate | 88.22% | 34.36% |
| True negatives | 59 | 12,694 |
| False positives | 442 | 6,644 |
| False negatives | 111 | 40,561 |
| True positives | 383 | 105,724 |

False-positive rate means the proportion of legitimate URLs incorrectly flagged.
For PhishNChips, 442 of 501 legitimate URLs were flagged. High precision in the
larger test does not negate its 34.36% false-positive rate: phishing prevalence
is much higher there. Accuracy below a majority baseline is not proof that the
model has no ranking signal, but does show poor classification at the frozen
operating threshold for these populations.

Different source collection, label quality, URL shape distributions, and model
shortcuts may contribute. These measurements do not isolate a causal explanation.
The prior 93.19% original holdout accuracy must not be advertised as universal
accuracy or as an estimate for these external populations.

## Reproduce

Use the existing project environment. Downloads require network access; evaluation
is offline. No Hugging Face SDK, Parquet reader, or new package is needed.

```bash
python scripts/download_hf_urls.py --dataset phishnchips
python scripts/download_hf_urls.py --dataset ealvaradob

python scripts/evaluate_external.py \
  --input data/external/huggingface/phishnchips/urls.csv \
  --label-convention project \
  --exclude-shared-domains --exclude-conflicting-urls \
  --output-dir artifacts/evaluations/hf_phishnchips

python scripts/evaluate_external.py \
  --input data/external/huggingface/ealvaradob/urls.csv \
  --label-convention project \
  --exclude-shared-domains --exclude-conflicting-urls \
  --output-dir artifacts/evaluations/hf_ealvaradob
```

Each download directory includes `download_manifest.json` with the immutable
source revision, raw-file and exported-file fingerprints, label counts, and
field selection. Each evaluation directory includes `external_evaluation.json`,
`false_positives.csv`, and `false_negatives.csv`. Do not open URLs in error files.

If future model improvements use these findings, treat these datasets as
development evidence. Reserve a new, untouched, well-documented dataset for the
next final evaluation; do not tune on these results and claim an independent test.
