# TalentUD: on-device phishing evaluation

Research title: **Evaluating On-Device Phishing Detection Models on Real-World Web Pages**.
The submitted research scope is retained at the applicant's request. The earlier
proposal to narrow the topic to URL-only is withdrawn; no amendment is requested.

See the [consolidated measured follow-up](professor-evaluation-followup.md)
for paired URL/URL+content CV, the cutoff-composition audit, paired uncertainty,
actual benign Chrome popup tests and the remaining reliability gates.

## Scope and research question

Can lightweight on-device models combine URL and static webpage-content signals
to detect previously unseen phishing pages while limiting false positives,
protecting privacy and providing useful explanations?

Retain the submitted plan's comparison of a URL baseline and a URL+webpage-content
candidate, planned ONNX browser inference, real-world evaluation and explainability.
The submitted 22-feature logistic baseline and its 94.4% result are historical,
not the current XGBoost model or an apples-to-apples comparison.

- Current operational URL baseline: native XGBoost, 21 lexical URL features.
- Retain the original logistic baseline from the proposal for a fair same-cohort
  comparison; do not compare its old accuracy directly with newer cohorts.
- URL+static-content modeling remains in research scope. Existing HTML-only and
  combined experiments provide preliminary evidence, not production validation.
  The original CV below is URL-only. A same-role URL+content CV run is now complete
  and documented in the consolidated follow-up; neither validates browser inference.
- Keep all earlier artifacts/tests for reproducibility. No model is activated
  merely because the research scope includes it.
- Current extension calls a loopback Python API. This is on-device inference via
  a local service, **not yet inference inside Chrome**. Browser-only packaging,
  extractor parity and resource testing remain engineering milestones. Do not
  claim successful ONNX export or browser integration before verifying compatibility.
  Measure extraction/score parity, local-only network behavior and resource costs
  for both URL and selected static-content signals. Fully dynamic/interactive
  behavior remains outside the submitted scope.

## Professor's advice: evaluation before real-site automation

Transfer audit (2026-10-16): all four replacement/current files are readable and
match their original manifest hashes. The initially invalid validation copy is
resolved. Across 578,797 records, all 21 float32 URL features reproduce exactly,
registered domains recompute consistently, and row/label counts match the manifest.
No invalid/duplicate URLs or cross-split URL/domain/nonempty exact-content hash
overlap was found. This does not prove near-template, source or temporal independence.
See `artifacts/reports/copied-records-audit-2026-10-16.json`. Development CV on
454,185 train+validation records is now complete; results are below.

### 1. Recover records and audit leakage

Copy privately from the Windows experiment directory, preserving relative paths:

```text
artifacts/experiments/phreshphish_html_20261010/train_rows.parquet
artifacts/experiments/phreshphish_html_20261010/validation_rows.parquet
artifacts/experiments/phreshphish_html_20261010/test_rows.parquet
artifacts/experiments/phreshphish_html_20261010/publisher_test_rows.parquet
```

Only train+validation enter new CV (expected 454,185 records before audit).
Verify SHA-256 against `split_manifest.json`, labels, URL validity, feature order,
duplicate/conflicting URLs, domain identities and exact-content overlap. Do not
silently drop records or replace the missing inputs with old mixed-source data.
Test/publisher records are for isolation checks and later grouped uncertainty,
not for selecting hyperparameters or cutoffs. Existing publisher scores are
already inspected; they cannot become a fresh untouched final test.

Dropping HTML models does not require ignoring existing `html_sha256` metadata.
It can still identify duplicate pages for leakage control without reading HTML
or including content in prediction. Preserve same-domain and same nonempty hash
records in connected components. Missing hashes must not join unrelated pages.
Report component sizes, class counts and exclusions. Exact hashes do not prove
near-template, source or temporal independence.

### 2. New grouped CV on current development records

1. Concatenate only audited train+validation. Keep all 21 definitions fixed.
2. Assign connected components to five stratified outer folds, with a fixed seed.
   Check both classes and zero component/domain/hash overlap in every split.
   If a giant component makes five folds infeasible, report it and revise the
   protocol explicitly rather than splitting that component to improve results.
3. Within each outer training fold reserve grouped inner-validation records.
   Only inner-fit records train trees; only inner validation selects stopping
   rounds and thresholds. Outer labels are used for evaluation, never tuning.
   Start with the current hyperparameters and a declared fixed round budget.
4. Evaluate the same outer records for each model. Report confusion counts,
   precision, recall, F1, FPR, ROC-AUC and average precision. Evaluate the saved
   0.490197 cutoff as a descriptive reference and an inner-validation-only
   threshold policy at 1%/2% FPR budgets; neither budget is guaranteed on new data.
5. Save out-of-fold scores, component IDs, fold IDs, inner-selected cutoffs and
   provenance locally. Report pooled counts and fold distributions, plus 95%
   component-cluster bootstrap intervals. Clarify that bootstrap intervals
   conditional on fitted models do not capture all retraining variability.
6. This is development CV: current hyperparameters were already selected using
   this collection. It is not a new unbiased final generalization claim.

Existing `scripts/benchmark_models.py` targets older CSV cohorts and domain-only
grouping; it must not be run unchanged and described as CV of these records.
The record-aware command is `scripts/cross_validate_native.py`. It fits either
the URL or URL+content XGBoost candidate via `--variant`, rechecks file hashes and exact split isolation, and loads
holdout URL/domain/hash metadata only for isolation checks, not their labels or
feature values. It refuses an existing output directory and never activates a model.

### Completed URL-only grouped CV: 2026-10-16

Run command (use a new output name to repeat; existing runs are not overwritten):

```bash
.venv/bin/python scripts/cross_validate_native.py \
  --output artifacts/experiments/talentud_url_grouped_cv_20261016 \
  --trees 800 --threads 4 --bootstrap-draws 1000
```

Five outer folds covered every one of the 454,185 development records exactly once
(247,616 legitimate, 206,569 phishing). The 173,444 domains collapse to 154,074
domain/exact-content connected components. The largest has 32,629 rows; median
component size is 1, p95 is 5 and p99 is 20. No component/domain/nonempty exact hash
crosses fit/inner/outer roles. Outer folds contain 90,835-90,841 records each.

The fixed depth-7 configuration uses the current model's regularization and
subsampling, CPU/four threads, seed 42 + fold, up to 800 trees and patience 40.
Only inner validation controls early stopping and new cutoffs. Best tree counts
were 404, 800, 800, 295 and 577. Two folds reached the fixed round budget; we did
not enlarge it after inspecting outer results. Each saved model was sliced to
its selected tree count. No HTML features/models were fitted.

#### Pooled out-of-fold results

| Cutoff policy | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Saved 0.490197 cutoff: descriptive reference only | 91.50% | 93.49% | 87.40% | 90.34% | 5.08% |
| Inner-validation F1, with >=90% inner recall | 90.93% | 90.09% | 89.95% | 90.02% | 8.26% |
| Inner-validation FPR <=1% | 86.68% | 98.26% | 71.98% | 83.09% | 1.06% |
| Inner-validation FPR <=2% | 89.74% | 96.92% | 79.97% | 87.64% | 2.12% |

The reference cutoff was previously chosen using the old validation records,
which now participate in development CV. It is not a clean nested-selection
estimate. The other policies select one cutoff per fold on that fold's inner
validation only; their pooled results do not represent a single deployable cutoff.
These are five newly fitted fold models, not a re-evaluation of the unchanged
480-tree extension model. Do not compare their 5.08% reference FPR with the old
10.62% publisher FPR and claim the deployed model improved: the cohorts differ.

Mean fold ROC-AUC is 96.92% (sample SD 1.27 percentage points); mean fold average
precision is 96.87% (SD 1.17 points). Pooled ranking summaries are in JSON but are
descriptive because scores come from different fitted models. No ranking CIs are
claimed; the bootstrap intervals below concern confusion-derived metrics only.

#### Fold variability at the inner F1/recall policy

| Fold | Trees | Inner-selected cutoff | Outer recall | Outer FPR |
|---|---:|---:|---:|---:|
| 1 | 404 | 0.384258 | 85.49% | 7.22% |
| 2 | 800 | 0.528205 | 94.21% | 5.28% |
| 3 | 800 | 0.521683 | 83.33% | 4.25% |
| 4 | 295 | 0.407602 | 91.40% | 7.82% |
| 5 | 577 | 0.130803 | 95.33% | 16.71% |

The most important finding is threshold instability, not a new best accuracy.
Fold 5's F1 policy selects a low cutoff and produces many more false alarms.
Its grouped inner-validation subset has 116,963 rows versus 58,098-75,218 in
other folds; selecting 20% of components is not selecting 20% of rows. Large
components and class/composition differences warrant development investigation.
This association is not proof of the individual cause of each false positive.
The 1% FPR policy has outer fold FPRs 0.87%-1.17%, but recall varies 56.38%-84.57%.

#### Group-clustered uncertainty

All 1,000 bootstrap replicates were valid. Whole components were resampled within
outer folds; reported 95% percentile intervals are conditional on these fitted
models and chosen cutoffs. They do not account for full refitting uncertainty,
overlapping training-set dependencies, dataset bias or label errors, and are not
confidence guarantees for future browsing populations.

| Policy | Recall 95% interval | FPR 95% interval |
|---|---:|---:|
| Saved cutoff reference | 83.59%-91.57% | 4.45%-5.71% |
| Inner F1 / recall | 86.92%-93.22% | 7.27%-9.14% |
| Inner FPR <=1% | 64.52%-79.83% | 0.93%-1.20% |
| Inner FPR <=2% | 74.08%-86.16% | 1.87%-2.35% |

#### Files, verification and interpretation

- `artifacts/experiments/talentud_url_grouped_cv_20261016/cv_results.json`:
  full protocol, input/code/model hashes, role sizes/class counts, fold train/
  validation/outer metrics, cutoffs, pooled counts, fold SDs and bootstrap intervals.
- `out_of_fold.parquet`: one prediction per development row, original split,
  row index, component/fold IDs and each fold's cutoffs. No URL strings copied.
- `role_assignments.parquet`: each row's fit/inner/outer role per fold.
- `fold_1.ubj` through `fold_5.ubj`: research fold models, not deployed packages.
- `scripts/cross_validate_native.py`: one reproducible command; no framework added.
- `tests/test_native_cv.py`: grouping transitivity/empty hashes, role isolation,
  tied thresholds, cluster bootstrap and tiny real-training smoke tests.

After the run, all five models were independently reloaded on CPU: all 454,185
OOF scores reproduced exactly, roles/labels matched, and every outer and pooled
confusion matrix reproduced. Main native and legacy model/metadata hashes are
unchanged. Full suite: 104 tests passed, two dependency deprecation warnings;
Ruff lint/format and diff checks passed. Parquet outputs remain ignored by Git.

This completes the new development-CV step of the professor's advice, not an
independent final validation. No test/publisher scoring, threshold promotion,
extension update, HTML training or live phishing browsing occurred. Next: audit
the unstable F1 cutoffs and component composition as development work, agree on
the false-alarm/recall tradeoff, then freeze a selection protocol for genuinely
fresh source/time-reviewed real-world data. Do not tune against that final cohort.

Method references: [scikit-learn grouped stratification](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
and [XGBoost training/early stopping](https://xgboost.readthedocs.io/en/stable/python/python_intro.html).

Optional after CV: a fixed 10/25/50/100% learning curve using nested, label-blind
samples of inner-fit components, leaving inner validation and outer folds fixed.

The URL and URL+static-content XGBoost comparison now uses the same leakage-controlled
outer/inner partitions, equal declared budgets and no outer-test tuning. Existing feature
Parquets can support numeric comparisons, but browser extractor parity still needs
representative saved HTML fixtures/raw snapshots; aggregate columns cannot
reconstruct page content. Additional content-model CV was subsequently authorized
by the user's request to work through all evaluation recommendations; no saved
deployment package was replaced.

### 3. On-device engineering measurements

First record the current local-service CPU baseline:

```bash
.venv/bin/python scripts/benchmark_on_device.py \
  --output artifacts/reports/on-device-local-service-2026-10-16.json
```

This measures model loading after imports and warmed single-URL service latency
on synthetic reserved URL strings. It does not visit websites, measure accuracy,
include HTTP/Chrome latency, or establish real-browser performance. Repeat in new
processes on both laptops with new filenames; record power mode/load and device
details before comparing. A report filename is not a frozen final evaluation.

First completed Mac ARM64 CPU run: 30 warm-up calls, 600 timed calls, median
0.225 ms, p95 0.292 ms, model load after imports 73.25 ms, model file 2,407,390
bytes. The report above records runtime versions, checksum and exact values.
This is one synthetic microbenchmark, not a real-site or browser performance claim.

Then implement browser-only inference as a separately tested candidate. Require:

- Shared golden URL cases: Python vs browser parsing and all 21 float32 features,
  including root slashes, Unicode, percent escapes, ports, IPs and invalid inputs.
- Score tolerance and exact threshold-decision checks, especially near cutoff.
- Model/feature-version hashes, no remote runtime/model/CDN requests, offline test,
  minimal permissions and no persistent URL logging by default.
- Cold start, warmed median/p95 latency, model/extension bytes, peak browser
  memory, errors and repeatability. Separate feature, inference and popup time.

Changing Python feature definitions to match browser parsing silently is not
acceptable: document changes and retrain/version if needed. Current handwritten
reasons are lexical observations, not causal or SHAP explanations.

### 4. Freeze protocol, then real-world testing

Agree on false-alarm targets and a genuinely new, time-stamped, source-reviewed
cohort before selection. Include diverse legitimate regional/login/long-path and
hosted pages, plus independently verified phishing URL records. Store label
evidence, collection/review dates, redirects and original/final URLs; popularity,
HTTPS or a model prediction alone cannot establish a ground-truth label.

Separate (a) detection metrics from labeled URL strings and (b) safe extension UI
integration/latency tests. URL-only inference does not need to open phishing sites.
Begin UI automation on public benign pages and inert local fixtures in an isolated
profile. Do not visit live phishing URLs in the normal user profile or enter
credentials, submit forms, download or execute content. Any actual hostile-site
browsing needs a separately approved isolated lab protocol. Keep browser protection
enabled and report blocked/unreachable pages rather than removing them silently.

Freeze model/checksum/cutoff and record expected API vs popup decisions; log only
consented research data. Report errors and exclusions as well as FPR/recall and
grouped uncertainty. No auto-deployment or cutoff adjustment based on final results.
