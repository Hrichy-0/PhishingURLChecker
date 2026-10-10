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
