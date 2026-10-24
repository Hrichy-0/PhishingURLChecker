# PhishGuard model card

## Intended use

PhishGuard estimates phishing risk from an HTTP or HTTPS URL string. It is
designed for education, portfolio demonstration, and an additional browser
warning signal. It does not retrieve or execute content from the URL.

## Data

### Main local model: native PhreshPhish URL-only

The default API now selects
`artifacts/experiments/phreshphish_html_20261010/url_only/`, retaining its original
research metadata (`activation_eligible: false`). It is a local advisory default,
not an independently validated production release. Its SHA-256 is
`9d9ae6a4808b0ac002c0062e0b1ddc83edd198dc52bfdbad5689aeab36559b47`.

The extractor, metadata and booster agree on all 21 features in order. All are
supplied as float32 values; 20 are used in tree splits and `has_explicit_port`
is unused. Inference uses 480 trees on CPU and the saved threshold
`0.4901971220970154`. The native lexical definitions are unchanged, including
known naive subdomain-count semantics. The original joblib artifacts are untouched.

| Cohort | Rows | Accuracy | Recall | FPR |
|---|---:|---:|---:|---:|
| Internal test | 39,866 | 90.40% | 83.69% | 5.57% |
| Purged publisher test | 84,746 | 84.98% | 82.72% | 10.62% |

The publisher test shares collection lineage and an overlapping boundary date
with development; it is not independent proof of future browsing performance.
The 21-input/API parity check is functional verification, not accuracy evidence.
The feature Parquets have been recovered and verified. Five-fold grouped CV on
454,185 train+validation records is complete for the fixed URL-only configuration,
with inner-validation stopping/cutoffs and component-clustered conditional
intervals. The inner-F1 policy yielded 89.95% recall and 8.26% FPR; at a 1%
inner-validation FPR budget, outer recall was 71.98% and FPR 1.06%. This evaluates
five research fold models, not a replacement of this saved model. The original
saved cutoff is only a descriptive reference in CV because it was previously
selected on validation now inside development. No learning curve has run yet.
See the [current TalentUD evaluation plan](talentud-evaluation-plan.md).
HTML models remain experimental research candidates within the submitted scope
and are not validated by the URL-only CV. Since 23 October 2026 the saved URL+HTML
model is reachable only through the experimental `/predict_html` endpoint and the
popup's content button used for RPA testing
([rpa-real-world-testing.md](rpa-real-world-testing.md)); the default popup check
remains URL-only.

### Historical joblib model (remaining sections describe this baseline)

The historical XGBoost model uses PhiUSIIL and the local `balanced_urls.csv` source.
After cleaning and cross-source deduplication, 574,716 URLs remain. Registered
domains are kept entirely within one split across both sources. The original
PhiUSIIL-only model and splits were preserved for comparison.

| Split | Rows |
|---|---:|
| Train | 457,544 |
| Validation | 56,621 |
| Test | 60,551 |

## Model

The saved scikit-learn pipeline contains:

1. A custom transformer selecting 21 lexical URL features (slash count omitted).
2. An XGBoost binary classifier, depth 7, learning rate 0.05, up to 800 trees,
   row/column subsampling, and L1/L2 regularization.

Three configurations are compared by validation F1. Validation logloss controls
early stopping. This run reached the maximum of 800 rounds. XGBoost uses tree
splits, so this pipeline does not include standard scaling.

The operating threshold is chosen on validation data by maximizing F1 while
meeting a minimum 90% phishing-recall target. The test set is not involved in
training or threshold selection.

## Test results

- Accuracy: 0.9319
- Precision: 0.9348
- Recall: 0.9274
- F1: 0.9311
- ROC-AUC: 0.9833
- False-positive rate: 0.0637
- True negatives: 28,566
- False positives: 1,942
- False negatives: 2,182
- True positives: 27,861

The logistic baseline trained and evaluated on these same splits achieved
0.8354 accuracy and 0.8397 F1. The original 0.9441 accuracy was measured on a
different PhiUSIIL-only task and is not directly comparable with the mixed
holdout score. See [the experiment record](xgboost-experiments.md) for both.

## Known limitations

- Dataset sources can encode collection bias and may not represent future
  phishing campaigns.
- Both original sources are part of training. Two additional Kaggle datasets
  have since been tested without retraining, with poor performance and important
  missing-protocol caveats. See [external evaluation](external-evaluation.md).
  There is still no temporal holdout. The balanced-source subset has a 10.84%
  false-positive rate even though the aggregate rate is 6.37%.
- Frozen-model tests on two Hugging Face sources also found poor classification:
  44.42% accuracy / 88.22% false-positive rate on retained PhishNChips URLs and
  71.50% / 34.36% on a complete-URL subset of ealvaradob's URL compilation.
  See [the Hugging Face evaluation](huggingface-evaluation.md) for class balance,
  cleaning, source-label limitations, and the unchanged threshold.
- The exact origin, labeling process, and collection dates of the local
  balanced source have not been independently verified.
- URLs and labels age. Domains can change ownership or behavior.
- The detector cannot see page content, redirects, certificates, DNS, domain
  age, JavaScript behavior, or brand impersonation visible only after loading.
- Attackers can deliberately choose benign-looking URLs.
- The validation-selected threshold represents one security tradeoff and may
  need adjustment for a real organization.
- False negatives and false positives remain possible. The UI must not call a
  low-risk result "safe."
- The score is not calibrated for real-world use. A displayed value near 100%
  is a model output, not a claim of 100% certainty.

## Important validation lesson

An early version achieved over 99% offline accuracy but incorrectly classified
common benign root URLs because browsers add a trailing `/` that much of the
training data omitted. Canonicalizing equivalent root URLs reduced the metric
to a more realistic level and fixed that training-serving skew. This regression
is covered by an automated feature test.

A later sanity check found a second, more important shortcut: the official
Apple Germany URL `https://www.apple.com/de/` received an extremely high
phishing score. Contribution inspection showed that `slash_count` dominated
the logistic-regression decision. In this dataset, legitimate examples are
disproportionately root URLs while phishing examples more often contain paths.
The model therefore learned a data-collection pattern that does not generalize
to normal deep links.

This was a confirmed false positive, not evidence that Apple's page is unsafe.
Domain-disjoint splitting prevents domain leakage, but it cannot remove every
source-collection bias. Version 2 drops slash count and learns from both sources;
Apple's Germany URL now scores 5.63/100. Its domain was in training, so this
spot check does not establish unseen-domain generalization. `example.com` still
receives a high score and is a documented false positive. Further source and
temporal evaluation remains necessary.

## Responsible presentation

- Describe the numeric output as a **model risk score**, not certainty.
- Never label a low score as "safe."
- Do not fix individual false positives with a brand whitelist.
- Do not tune the threshold against the same sanity examples used to report
  generalization.
- Keep browser/reputation protections enabled; this model is only one signal.
