# PhreshPhish URL + HTML experiment — 2026-10-11

All three variants use the same 397,884 training pages, 56,301 validation pages,
39,866 internal-test pages and 84,746 purged publisher-test pages. Models and
thresholds were selected on validation only. None has replaced the active API model.

## Same-row publisher test

| Model | Features | Trees | Accuracy | Precision | Phishing recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| url_only | 21 | 480 | 84.98% | 93.81% | 82.72% | 87.92% | 10.62% |
| html_features | 17 | 36 | 84.43% | 86.22% | 90.97% | 88.53% | 28.31% |
| url_html | 38 | 64 | 88.75% | 90.65% | 92.52% | 91.58% | 18.58% |

Combined accuracy increased by 3.78 percentage points;
phishing recall increased by 9.80 points, while
FPR increased by 7.96 points.
The default combined threshold detected 5,489 additional phishing pages but
flagged 2,289 additional benign pages. HTML-derived features alone performed
worse on accuracy and produced a 28.31% FPR. More inputs do not improve every metric.

## Thresholds selected for <=1% validation FPR

These are validation-constrained thresholds, not thresholds optimized on publisher test.
The measured publisher-test FPR can exceed 1%.

| Model | Publisher accuracy | Publisher recall | Publisher FPR |
| --- | ---: | ---: | ---: |
| url_only | 73.89% | 61.35% | 1.70% |
| html_features | 62.78% | 44.04% | 0.74% |
| url_html | 81.13% | 72.37% | 1.80% |

At this operating point, the combined model detected 6,167 more phishing pages
with 30 more false positives than URL-only. Its 1.80% publisher FPR is still
too high to assume suitability for automatic blocking in low-prevalence browsing.

## What the combined model learned

Gain importance is a normalized global split-quality measure, not a causal
effect, detection rate, direction, or explanation of one URL.

| Feature | Gain share |
| --- | ---: |
| link_count | 28.48% |
| path_length | 10.20% |
| url_length | 5.31% |
| query_parameter_count | 4.55% |
| query_length | 4.47% |
| external_link_ratio | 4.16% |
| character_entropy | 3.78% |
| uses_https | 3.57% |
| longest_token_length | 2.78% |
| html_length | 2.77% |

HTML-derived features account for 48.69% of normalized gain shares.
HTTPS accounts for 3.57%; the model has shifted toward page structure and URL
context. This does not prove that any single feature caused the improvement.

## Data scale, timing, and limitations

All 77 original 36.58 GB Parquet files were checked and processed. After URL
cleaning, the development pool contains 494,051 paired pages. The full publisher
test has 166,505 usable pages; domain and exact-HTML exclusions removed 81,759.
Extraction/checksum verification took 2.66 hours;
the cached-feature CUDA fitting/evaluation pipeline took 20.80 seconds.
480, 36 and 64 trees were selected for URL-only, HTML-derived and combined models
respectively. Shorter models reflect validation early stopping/model selection,
not a skipped dataset. Each candidate was allowed up to 800 rounds.

The training pool is smaller than the previous mixed-source URL-only experiment:
the older sources have no matching HTML, so they were excluded for a fair
paired comparison. Earlier 80.95% accuracy used a different 78,989-row test
cohort and cannot be directly substituted for the 84.98% same-row baseline.
The experiment measures the benefit of adding HTML on this large pool. It does
not isolate the causal effect of dataset size; that needs a controlled learning curve.

Training dates span 2024-07-02 to 2025-09-08; publisher test dates span
2025-09-08 to 2025-12-16. They share a boundary date, so strict temporal
separation has not been demonstrated. Internal splits are not temporal.

The 1 MB parse cap truncated 10.44% of benign and 1.70% of phishing training
pages. This class imbalance can create a size/missingness shortcut; evaluate
ablations removing size/truncation features and different parse caps before
claiming robust page understanding. No parsed rows reported an HTML parse error.
Near-duplicate templates are not removed; only identical HTML is grouped/purged.
Publisher-test results have now been inspected and should not guide tuning.
The evaluation phishing prevalence is much higher than ordinary browsing.

## Verification and transfer

The saved model checksums, feature schemas, tree counts and split checksums were
verified. All three publisher/internal test confusion matrices reproduced on
CPU, supporting inference on a Mac without CUDA. Small validation-boundary
differences occur from GPU/CPU floating-point rounding; see verification_report.json.

Models, metadata and prediction outputs are tracked in Git. Raw datasets,
feature caches and dataset-containing split Parquets are intentionally excluded.
Dataset: [PhreshPhish](https://huggingface.co/datasets/phreshphish/phreshphish),
pinned revision eabec4b7a66324b79cc8a0ad856d1731dc26fe1a, CC BY 4.0, anti-phishing research.
Paper: [Dalton et al., 2025](https://arxiv.org/abs/2507.10854).

## Professor evaluation follow-up

The main **local advisory** API/extension model now selects `url_only` by default,
not the old joblib model. Its original `activation_eligible: false` metadata is
preserved: this selection is not production approval. All trained model and
metadata files remain byte-for-byte unchanged. The old joblib model is still
available with `PHISHGUARD_MODEL=legacy` for rollback/comparison.

### Completed on this Mac

- All 128 transferred model/result checksums verify.
- Extractor order, metadata order and booster schema match on 21 lexical inputs;
  all are supplied as float32 values. The selected booster has 480 trees.
- Twenty features appear in learned splits; `has_explicit_port` has no split
  usage. This is a learned outcome, not a missing/broken input. No feature
  definition was changed under saved weights.
- Six benign/reserved synthetic URL-string cases give identical direct-booster
  and in-process API scores/decisions at saved cutoff `0.4901971220970154`.
  Apple `/de/` scores 0.1312856972, below the cutoff. These cases are functional
  checks, not a labeled accuracy study. No websites were visited.
- A Chrome-extension-origin preflight succeeds, and `/health` identifies the
  main model checksum plus 21-feature count. Actual Chrome popup behavior has
  not been reverified by automation.
- Validation, internal-test and publisher-test confusion matrices reproduce
  from the existing saved URL-only prediction CSVs, restoring float32 scores
  before comparison with the original float32 cutoff. This is an output audit,
  **not fresh inference on the raw training/test inputs**.

Machine-readable results: `artifacts/reports/main-url-model-verification.json`.

### Professor's checklist: current status

Scope clarification: the applicant retains the original submitted URL versus
URL+static-content comparison. The earlier proposed URL-only restriction is
withdrawn. The [TalentUD evaluation plan](talentud-evaluation-plan.md) records
current work: URL-only grouped CV is complete; HTML/browser comparison remains
pending. Existing HTML hashes are leakage-control metadata, not predictive
inputs to the URL-only model. The extension still uses URL-only inference.

| Requested check | Available evidence | Still needed |
|---|---|---|
| Cross-validation | Earlier mixed-source URL experiments have grouped CV; these are not CV results for the new PhreshPhish model | New grouped CV on the current development records |
| Leakage | Transferred verification reports record domain/exact-HTML isolation and split hashes | Recheck the actual feature records locally; near-template and source/time leakage remain unproven |
| Statistically meaningful estimates | Saved confusion counts and cohort sizes are consistent | Domain/HTML-component clustered confidence intervals and fold variability; row-independent intervals would ignore clustering |
| Real-life extension test | Local API contract checks and manual spot checks | A fixed, independently labeled website list, automated UI/response checks, and an explicit safe browsing protocol |

Validation, test and publisher cohorts span different class prevalences and
source conventions. Train/validation dates overlap internal test, and publisher
test shares the development boundary date. Do not describe this as a strict
temporal evaluation or an untouched independent final test. Lower-FPR cutoffs
were validation-selected, but do not guarantee the target on new sites.

### Input blocker: no training-feature Parquets on this Mac

The models, metadata and prediction outputs were transferred; the following
numeric-feature/record Parquets were not. Copy them privately from the Windows
experiment directory into the same relative location here:

```text
artifacts/experiments/phreshphish_html_20261010/train_rows.parquet
artifacts/experiments/phreshphish_html_20261010/validation_rows.parquet
artifacts/experiments/phreshphish_html_20261010/test_rows.parquet
artifacts/experiments/phreshphish_html_20261010/publisher_test_rows.parquet
```

The first two enable development learning curves/CV. Test/publisher records add
domain/HTML group identities needed for clustered uncertainty analysis. These
files contain URL/record metadata and extracted numeric features, not raw HTML;
they remain ignored by Git. Compare their bytes with the saved split manifest
before trusting them. Saved score/label CSVs alone cannot train new models or
support a domain-clustered analysis. Downloading 36.58 GB is not required if these
feature files are available. No learning-curve/CV fit has run on this Mac yet.

### Predeclared experiment to run once inputs arrive

1. Verify split-manifest and Parquet fingerprints, schema/label conventions,
   feature order, URL validity, duplicates/conflicts, domain and identical-HTML
   overlap. Preserve all original records/artifacts; report exclusions explicitly.
2. Use only development train+validation records. Keep domains and identical-HTML
   connected components together in five outer folds. Within each outer training
   portion, reserve a separate grouped inner-validation set for early stopping
   and cutoff selection; never use outer labels to choose either.
3. Use nested, label-blind subsets of 10%, 25%, 50% and 100% of available **inner
   fit groups**. Report actual row/group counts: 100% here is not necessarily the
   old model's 397,884-row fit. Fix the 21-feature definitions and model/search
   budget across sizes, use CPU on Mac, and retain seed/model provenance.
4. Measure held-fold recall, FPR, precision, F1, AP and ROC-AUC at the default
   validation-selection policy and at validation false-alarm budgets of 1%/2%.
   Also report training metrics so underfit/overfit patterns can be assessed.
   Do not promise those FPR budgets on held folds. Existing hyperparameters were
   previously selected on this development collection, so this remains a
   development experiment, not a newly blinded final performance estimate.
5. Report fold distributions and 95% component-cluster bootstrap intervals,
   clearly separating uncertainty conditional on fitted models from variability
   across retraining/splits. Do not use ordinary per-row intervals as if pages
   from the same domain/template were independent.
6. Freeze the selected protocol before final testing on genuinely fresh,
   temporally documented and source-reviewed data. Existing inspected publisher
   results must not guide another cutoff and then be called untouched final data.
7. Only then assess a fixed real-website extension workflow. Use public benign
   pages first; do not automatically open dataset-listed phishing URLs in the
   user's normal browser. HTML checking remains a separate research candidate
   requiring explicit snapshot/privacy and browser-parity safeguards.

Cross-validation and leakage checks may lower or raise estimates; they do not
automatically improve the detector. Statistical uncertainty is not label-quality
validation. More rows are useful only if held-group performance improves at an
acceptable false-alarm cost.
