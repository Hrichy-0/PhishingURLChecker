# XGBoost and character TF-IDF comparison

For the newer source-held-out tuning work, see the final section below.

## Outcome

The optional character TF-IDF + Logistic Regression candidate improves recall
at a low-false-alarm operating point on domain-grouped development folds and
has stronger ranking metrics on our two external development datasets. It
still has unacceptable external false positives. **No candidate was activated.**
The existing API and Chrome extension continue using the original active model.

No new dependencies were installed, and no dataset-listed website was visited.

## What the new models learn

- **XGBoost:** the existing 21 numeric URL features, using fixed parameters:
  up to 400 trees, depth 5, learning rate 0.05, and inner-validation early stopping.
- **XGBoost without `uses_https`:** the same configuration with that direct
  feature removed. Other numeric features can still encode protocol indirectly
  (for example URL length). This is not a fully protocol-invariant model.
- **Character TF-IDF + Logistic Regression:** remove the URL protocol and
  fragment, normalize the optional root slash, then learn 3–5-character patterns.
  TF-IDF vocabulary and document frequencies are fit on training rows only.
  The representation is lowercased, sparse, capped at 100,000 patterns,
  and uses `min_df=2`, sublinear TF, and L2-normalized vectors. The classifier
  uses `C=1`, balanced class weights, and the liblinear solver.

For a learning example, `login` produces character patterns such as `log`,
`ogi`, `gin`, `logi`, and `login`. TF-IDF weights patterns using their frequency
in each training URL and across the training corpus. Logistic Regression learns
which weighted patterns are associated with the label. A learned association
is not proof that a word or URL is malicious.

Primary references: [TF-IDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

## Evaluation design

The full benchmark uses **514,165 development URLs** from the existing train
and validation splits (274,334 legitimate, 239,831 phishing; 321,575 registered
domain groups). The old 60,551-row test split is checked for isolation and
fingerprinted but is **not scored or used for fitting** in this experiment.

Five outer `StratifiedGroupKFold` folds group registered domains, including
private suffixes. All candidates receive exactly the same role assignments.
Within each outer-training fold, a domain-disjoint 20% inner validation split
chooses early stopping and an alert threshold. Parameters are fixed, not
optimized using the outer evaluation rows. No TF-IDF vocabulary is learned
from inner validation or outer evaluation rows.

The threshold maximizes inner-validation recall subject to empirical FPR <=1%.
That budget is **not guaranteed outside the inner validation population**.
A no-alert threshold above 1 is permitted if no positive can meet the budget.
We also retain outer-fold results at threshold 0.5 for transparency.

The highest mean outer-fold average precision is recorded, but the software
does not automatically deploy a winner. Average precision is a non-interpolated
precision–recall summary, not the same as trapezoidal area under the PR curve.
Fold standard deviations describe variation; they are not confidence intervals
or proof of statistical significance.

[Grouped CV reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

## Full benchmark results

Five-fold means at the inner-validation-selected low-FPR operating point:

| Candidate | Accuracy | Phishing recall | FPR | Average precision |
|---|---:|---:|---:|---:|
| XGBoost | 87.45% | 74.28% | 1.034% | 97.85% |
| XGBoost without HTTPS feature | 81.14% | 60.73% | 1.013% | 96.38% |
| Character TF-IDF + Logistic Regression | 89.88% | 79.45% | 0.997% | 97.54% |

At a fixed 0.5 threshold, mean accuracy/recall/FPR are respectively:

- XGBoost: 92.79% / 92.40% / 6.88%.
- XGBoost without HTTPS feature: 90.10% / 87.52% / 7.64%.
- Character model: 92.42% / 88.35% / 4.02%.

This explains why a lower accuracy can accompany fewer false alarms: operating
thresholds trade recall against false positives. It is not a regression caused
solely by model quality. The previously reported 93.19% active-model holdout
result uses a different training/evaluation setup and threshold; it is not a
directly comparable CV result.

Removing HTTPS reduces in-source ranking quality, but helps ranking on one
external dataset. This does not establish a universal benefit or isolate all
causes of the dataset shift.

## External development checks

After CV, each package is trained on the same 411,036 fit URLs, retaining
103,129 separate validation URLs. Its threshold is selected on those validation
rows, not on external results. There is no subsequent refit on validation rows.
Existing complete-URL cleaning and exclusion of domains from **all three old
splits** produce exactly the same external rows for every candidate.

| External set | Candidate | Accuracy | Recall | FPR | ROC-AUC |
|---|---|---:|---:|---:|---:|
| PhishNChips, 995 URLs | XGBoost | 70.35% | 41.09% | 0.80% | 66.74% |
| PhishNChips | XGBoost without HTTPS feature | 66.93% | 34.82% | 1.40% | 61.48% |
| PhishNChips | Character model | 71.26% | 90.89% | **48.10%** | 86.18% |
| ealvaradob, 165,623 URLs | XGBoost | 47.15% | 41.86% | 12.83% | 73.59% |
| ealvaradob | XGBoost without HTTPS feature | 47.47% | 41.40% | 6.57% | 79.99% |
| ealvaradob | Character model | 81.32% | 79.73% | **6.70%** | 94.95% |

Thresholds are 0.8673874139785767 (XGBoost), 0.845186173915863
(XGBoost without HTTPS feature), and 0.7461666756597595 (character model).
Do not lower or raise them using these external results and then describe the
same datasets as untouched final tests.

The character candidate improves ealvaradob ROC-AUC and FPR relative to the
original active model (76.36% ROC-AUC and 34.36% FPR). That comparison includes
changes in representation, training procedure, and threshold; it does not
attribute the improvement to one change alone. Its 81.32% accuracy still falls
below the 88.32% majority baseline in that phishing-heavy retained subset.

PhishNChips is an adapted email benchmark, and ealvaradob loses many incomplete
URL records under our strict protocol policy. Their source labels are not
independently adjudicated here. See [external limitations](huggingface-evaluation.md).
These results show useful ranking improvements, **not dependable protection**.
Model scores remain uncalibrated.

## Files to read and run in VS Code

1. `src/phishguard/text_features.py`: stateless normalization.
2. `src/phishguard/modeling.py`: TF-IDF and Logistic Regression pipeline.
3. `src/phishguard/experiments.py`: domain folds, inner validation, threshold,
   fold metrics, and separate model packages.
4. `scripts/benchmark_models.py`: reproducible experiment entry point.
5. `tests/test_experiments.py`: invariants and model serialization tests.

```bash
source .venv/bin/activate
python -m pytest -q
ruff check .
ruff format --check .

# Small learning/pipeline check. Outputs never overwrite a previous run.
python scripts/benchmark_models.py --max-rows 50000

# Full development comparison; leaves the old test and active model alone.
python scripts/benchmark_models.py --max-rows 0 --folds 5 --trees 400

# Locally compare the already-created full-run candidates, not the active API.
python scripts/compare_url_models.py --url 'https://www.apple.com/de/'
```

The run created during this session is
`artifacts/experiments/url_benchmark_full_v1/benchmark.json`.
Each candidate subdirectory has `phishguard_model.joblib`,
`model_metadata.json`, and `out_of_fold_predictions.csv`; external reports and
error examples are in its `external_phishnchips/` and `external_ealvaradob/`.
The smaller first run is `artifacts/experiments/url_benchmark_pilot_v1/`.
Artifacts are ignored by Git. Load only trusted, locally generated joblib files.

To repeat an external development check, substitute each candidate name:

```bash
python scripts/evaluate_external.py \
  --input data/external/huggingface/phishnchips/urls.csv \
  --label-convention project --exclude-shared-domains --exclude-conflicting-urls \
  --artifacts-dir artifacts/experiments/url_benchmark_full_v1/char_logistic \
  --output-dir artifacts/experiments/url_benchmark_full_v1/char_logistic/external_phishnchips
```

## Optional hybrid path: implemented but not trained on real content yet

`src/phishguard/content.py` combines 20 numeric URL features with 14 HTML
aggregates for a separate XGBoost candidate. Saved HTML is parsed as text with
the standard library, never rendered and never visited. Scripts, forms, images,
and links are not executed or fetched. Input values are not retained as features.

Signals include password-field presence, form/link domain relationships,
iframe/script counts, and a small title-brand/domain mismatch reference. None
is an automatic verdict: legitimate SSO and regional brand sites can trigger
them. HTML size and structure can also create dataset shortcuts.

Try the inert fixture without opening it in a browser:

```bash
python scripts/inspect_html.py \
  --url 'https://example.test/login' \
  --html-file examples/hybrid/login_fixture.html
```

The output is feature values, **not a hybrid phishing prediction**.
There is no real-data-trained hybrid artifact or hybrid API/extension mode yet.

For a small, appropriately licensed development CSV containing `url`, `html`,
and publisher labels `phish`/`benign`:

```bash
python scripts/benchmark_hybrid.py \
  --input data/external/html_development.csv \
  --phishing-label phish --legitimate-label benign --max-rows 10000
```

This compares the two URL representations and the HTML-enhanced XGBoost
on identical rows/folds. It rejects conflicting labels, missing snapshots,
and snapshots over 1 MB, and records cleaning exclusions. The first usable
snapshot per exact URL is retained. HTML limits and missing protocols can bias
the retained population, so class distributions must be checked.

Next candidate: [PhreshPhish](https://huggingface.co/datasets/phreshphish/phreshphish),
which provides URL–HTML pairs and dates. Use a small version-pinned development
subset, preserving the publisher's final test. Before deployment, add temporal
testing and near-duplicate HTML/template decontamination, calibration analysis,
and a fresh final external evaluation. Domain grouping alone does not rule out
phishing-kit/template leakage. These steps have not yet been completed.

No arbitrary server-side fetching or automatic live-phishing browser testing
is introduced. Any future extension content mode should be opt-in, collect
only necessary nonsensitive aggregates, and not upload raw private page HTML.

## Source-held-out XGBoost tuning — 2026-10-06

This uses the existing `benchmark_models.py --mode tune` entry point. No new
Python files or search-framework dependencies were added. Shared source merging
and packaging were consolidated into the existing modules; old commands remain
available for reproducibility.

### What changes are searched?

Six explicit XGBoost hypotheses, with the character model as a comparison:

1. Baseline: existing numeric features, depth 5, unweighted.
2. Source balanced: equal total weight for each observed source/class pair,
   computed using fit rows only.
3. Regularized: depth 3, min child weight 15, gamma 0.5, L1 1, L2 15,
   row/column subsampling 0.7, and source/class balancing.
4. Protocol invariant: source balanced, with a constant scheme before numeric
   extraction and the direct HTTPS feature removed. This also eliminates
   HTTP/HTTPS differences in URL length and entropy, unlike the old ablation.
5. Protocol invariant + regularized.
6. Reduced shape: protocol invariant + regularized, additionally excluding URL,
   hostname, path, and query lengths. Other remaining features can still convey
   URL shape; this is an ablation, not a claim to remove all shortcuts.

The search is deliberately small. Trees are capped at 400 and stop using inner
validation log loss. [XGBoost tuning guidance](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html).
Equalized source/class weights do not create representative population
probabilities. Output scores remain uncalibrated.
The character comparison retains its existing balanced classifier class weights
in addition to the source/class sample weights. Thus its effective class weighting
differs from XGBoost; this is a model-family comparison, not an isolated causal
experiment about weighting alone. Early stopping uses pooled, unweighted inner
validation log loss; candidate selection uses macro held-source average precision.

### Which data can be used?

PhiUSIIL and the balanced source come from the existing train/validation splits.
Previously tested PhishNChips and ealvaradob URL records are now explicitly
**development data**, not independent final tests. Their strict complete-URL
selection and source-label quality limitations still apply.

Additional development inputs require `url` and numeric `label` columns with
0=legitimate and 1=phishing (or specify `--development-label-column is_phishing`
for standardized files). Use `prepare_data.py` to adapt different source labels
first. Protocols are not inferred. New sources lose domains already present in
all three old splits, preserving the old historical holdout and avoiding easy
source-overlap artifacts. Cross-source contradictory labels are all excluded;
exact duplicate URLs retain their first consistent source.

Each tuning fold holds out one whole source, purges its registered domains from
the other sources, then makes an inner domain-disjoint fit/validation split.
Every hypothesis receives the same fold roles. The inner validation selects a
threshold that satisfies the empirical FPR budget separately in each two-class
validation source. Held-source labels never choose that fold's threshold or
early stopping. They DO choose the overall candidate, so the search scores are
selection-biased development evidence, not an unbiased final accuracy estimate.

The selected XGBoost maximizes mean held-source average precision, breaking ties
by lower worst-source FPR. This does not guarantee a good deployment operating
point: recall and false alarms must be reviewed separately. The character model
is a comparison, not a candidate eligible for automatic replacement.

### Run the lean workflow

```bash
source .venv/bin/activate
python -m pytest -q
ruff check .
ruff format --check .

python scripts/benchmark_models.py --mode tune \
  --development-input data/external/huggingface/phishnchips/urls.csv \
  --development-input data/external/huggingface/ealvaradob/urls.csv \
  --max-rows 50000
```

In tuning mode the cap is **per source**, sampled uniformly without stratifying
on labels. Zero uses all available development rows. The five-fold benchmark's
cap remains a total-row cap. `--folds`/`--variants` apply to benchmark mode, not
source tuning; source tuning uses the explicit six hypotheses and one fold per
source. Small sources stay small; weighting is not a substitute for collecting
better independently labeled data.

Each run has `tuning.json`, two saved model packages (`xgboost/`, `char_logistic/`),
and a shared `reference/` containing fitted, validation, and historical known-domain
exclusion records. The reference's `test.csv` is **not a new scored final test**.
Hashes pin references and inputs. Development candidates cannot be activated by
`activate_model.py` without a separate approval/evaluation step.

### Fresh final evaluation is still required

Do not reuse the four tuning sources and call them fresh. On a new independently
collected URL dataset, after verifying label meaning and provenance:

```bash
python scripts/evaluate_external.py \
  --input data/external/new_independent_urls.csv \
  --label-convention project --final-test \
  --artifacts-dir artifacts/experiments/source_tuning_full_v1/xgboost \
  --output-dir artifacts/experiments/source_tuning_full_v1/xgboost/final_evaluation
```

`--final-test` rejects known-used dataset fingerprints, disallows assumed
protocols, and always excludes known domains using the model's saved references.
The model and threshold stay frozen. A different file hash does **not** prove
independent collection, temporal isolation, or absence of template leakage.
The report labels provenance as not independently verified by this program.
No fresh final dataset has yet been supplied or evaluated, and the active
API/extension model remains unchanged. This is intentionally not a deployment.

### Completed full-source search

The pilot used 150,995 URLs (up to 50,000 per source). The subsequent full run
used **680,783 development URLs**: 209,084 PhiUSIIL, 305,081 balanced-source,
995 retained PhishNChips, and 165,623 retained ealvaradob URLs. It completed all
six XGBoost hypotheses and the character comparison on identical source-fold
roles. No additional source-level contradictory labels or exact duplicates
were found after the documented cleaning/domain exclusions.

| Hypothesis | Mean held-source AP | Mean phishing recall | Worst-source FPR |
|---|---:|---:|---:|
| XGBoost baseline | 79.07% | 31.27% | 10.13% |
| Source balanced | 81.33% | 33.09% | 10.00% |
| Regularized | 81.42% | 30.97% | 8.44% |
| **Protocol invariant** | **83.70%** | **37.16%** | **4.36%** |
| Protocol + regularized | 81.23% | 33.81% | 3.75% |
| Reduced shape | 76.14% | 22.40% | 2.51% |
| Character comparison | 91.03% | 42.69% | 14.57% |

AP means average precision, a ranking measure, **not accuracy**. Means give
each source equal weight regardless of size. Worst-source FPR is the largest
legitimate-URL false-alarm rate among the four held sources. The empirical 1%
inner-validation target is not a guarantee on the held source. Very low recall
shows that this operating point misses many phishing examples.

Protocol invariance was selected by the predeclared XGBoost selection criterion.
Its per-source recall/FPR were:

- PhiUSIIL: 37.75% / 2.87%.
- Balanced source: 45.63% / 1.80%.
- PhishNChips: 25.71% / 0.40%.
- ealvaradob: 39.57% / 4.36%.

The changes help but **do not establish dependable cross-dataset protection**.
The lower-FPR reduced-shape hypothesis sacrifices even more recall. The
character model has stronger ranking but its 14.57% worst-source FPR is also
unsuitable for an automatic blocker. Selection uses these development scores;
a fresh independently collected final test remains necessary.

Full report: `artifacts/experiments/source_tuning_full_v1/tuning.json`.
Pilot report: `artifacts/experiments/source_tuning_v1/tuning.json`.
Only selected XGBoost and character packages are saved per run. Neither is active.

To reproduce the full search (outputs get a new timestamped directory):

```bash
python scripts/benchmark_models.py --mode tune \
  --development-input data/external/huggingface/phishnchips/urls.csv \
  --development-input data/external/huggingface/ealvaradob/urls.csv \
  --max-rows 0 --trees 400

python scripts/compare_url_models.py --url 'https://www.apple.com/de/' \
  --experiment-dir artifacts/experiments/source_tuning_full_v1 \
  --variants xgboost char_logistic
```

The Apple example checks the command/inference path, not independent accuracy:
its domain is already known in development. Do not visit error-example URLs.

## Deployed XGBoost: validation-only threshold comparison

This analysis keeps the deployed weights and threshold unchanged. It uses the
56,621 rows in the saved mixed validation split (30,092 legitimate and 26,529
phishing), checks split fingerprints and train/validation domain isolation, and
reproduces the saved validation metrics. Test labels and test scores are not
loaded. The recent uploaded URL dataset is not used to select thresholds.

| Operating point | Threshold | Validation FPR | Phishing recall | False positives | Missed phishing |
|---|---:|---:|---:|---:|---:|
| Current active setting | 0.471128 | 6.46% | 94.13% | 1,945 | 1,558 |
| Pooled validation 2% budget | 0.785329 | 2.00% | 84.55% | 601 | 4,099 |
| Pooled validation 1% budget | 0.881261 | 1.00% | 77.90% | 300 | 5,864 |

These are empirical development budgets, not guarantees on new websites. This
same validation set already informed early stopping and model selection, so its
results are not independent performance estimates. Increasing the cutoff reduces
false alarms but also misses more attacks; it does not improve the model's score
ranking. The 2% cutoff is a candidate for further evaluation, not an approved
deployment change.

The aggregate budget can hide source differences. At the 2% cutoff, PhiUSIIL
validation FPR is 0.36%, while the balanced source is 3.33%. At the 1% cutoff,
these are 0.16% and 1.68%. Do not substitute these figures for the previously
reported 11.78% FPR on the uploaded external dataset.

Reproduce with the existing command (no new script or retraining):

```bash
.venv/bin/python scripts/evaluate_external.py --validation-thresholds \
  --output-dir artifacts/evaluations/validation_thresholds_2026_10_06
```

The output `validation_thresholds.json` records fixed-cutoff comparisons,
per-source metrics, model/validation fingerprints and limitations. No alternative
model package is created. Before any activation, evaluate a validation-chosen
cutoff frozen on external data; do not pick another cutoff from external results
and describe those same results as an untouched final test.

## Fixed candidate cutoff: external stress tests — 2026-10-08

The validation-selected 2% cutoff **0.7853294610977173** was frozen before
evaluating external labels. The same deployed XGBoost weights were scored at
that cutoff and at the active cutoff **0.4711279273033142**. No training,
external threshold search, activation, URL visit, or new dependency was involved.
Model and active metadata fingerprints were verified unchanged after evaluation.

| External cohort | Rows | Current FPR | Candidate FPR | Current phishing recall | Candidate phishing recall |
|---|---:|---:|---:|---:|---:|
| Uploaded `URL dataset.csv` | 211,269 | 11.78% | 7.53% | 91.50% | 69.30% |
| PhishNChips extracted URLs | 995 | 88.22% | 16.97% | 77.53% | 62.55% |
| ealvaradob complete-URL subset | 165,623 | 34.36% | 18.90% | 72.27% | 55.47% |

The uploaded cohort exactly reproduces the 2026-10-06 common model comparison:
567 invalid URLs, 9,265 exact known URLs, and 229,075 additional shared-domain
rows were excluded. Both the active model references and previously used research
references were excluded to retain that same cohort. The current-threshold
confusion matrix was reproduced exactly. At the candidate cutoff, 6,518 false
alarms are avoided but 12,884 additional phishing URLs are missed. Accuracy drops
from 89.12% to 86.11%. On its HTTPS-only subset, recall drops from 67.41% to
49.24%; its protocol/label shortcut continues to limit interpretation.

The Hugging Face cohorts use the active model's three saved reference splits for
URL/domain exclusion, matching their earlier frozen-model evaluations. Neither
assumes missing protocols. PhishNChips is an email-derived URL stress test;
ealvaradob excludes 608,193 invalid/incomplete URLs and 14,295 duplicates before
overlap exclusion, so its results do not describe the full original population.
These sources previously informed broader development work. None of these tests
is an untouched independent final evaluation, and labels/provenance are not
independently verified. See `docs/huggingface-evaluation.md` for source limitations.

**Decision: do not activate this cutoff based on these results.** It lowers
false alarms but does not deliver the empirical validation 2% budget externally,
and recall deteriorates substantially. The extension remains unchanged. Further
work should audit false positives and improve the training data/representation,
then reserve fresh data for final evaluation rather than repeatedly searching
cutoffs on these external cohorts.

The single machine-readable output is
`artifacts/evaluations/fixed_threshold_external_2026_10_08/external_evaluation.json`.
It records both operating points, confusion matrices, protocol subsets, paired
changes, cleaning/exclusion counts, the threshold-selection report fingerprint,
and all input/reference/model fingerprints. No new source file was added.

## False-positive pattern audit — 2026-10-08

This post-evaluation diagnostic compares publisher-labeled legitimate URLs that
were flagged with legitimate URLs that were accepted, using the identical
external cohorts and frozen thresholds above. It reproduces their false-positive
counts, verifies source/reference/model fingerprints, and uses no network visits.
It does not verify publisher labels or establish causal explanations. Slices
overlap, URL counts are not independent domain counts, and no significance tests
or independent confirmations were performed.

### Uploaded dataset: hostname shape is the strongest observed pattern

| Legitimate URL slice | Legitimate rows | False positives | Slice FPR |
|---|---:|---:|---:|
| Hostname at least 30 characters | 13,008 | 8,890 | 68.34% |
| Hostname shorter than 30 characters | 140,217 | 9,164 | 6.54% |
| Blogspot private suffix | 7,033 | 6,067 | 86.26% |
| At least two naively counted subdomains | 24,877 | 14,551 | 58.49% |
| Any predefined account/login/payment-style token | 368 | 121 | 32.88% |

Blogspot accounts for 33.60% of the 18,054 false positives, while predefined
suspicious tokens account for just 0.67%. Consequently deleting the word feature
alone would not address the bulk of this source's errors. Long URL text and paths
are not uniformly harmful: URLs of at least 100 characters have 13.75% FPR here,
only modestly above the 11.78% overall rate. Nonroot paths have 11.24% FPR versus
13.24% for root paths.

### Subdomain feature definition needs revision in a new trained model

`features.py` currently defines subdomain count as
`max(number_of_hostname_parts - 2, 0)`. This treats `www.example.co.uk` as having
two subdomains instead of one. It also does not distinguish tenant domains under
private suffixes such as Blogspot from subdomains of that tenant, and gives an IP
address a DNS-style label count. The offline Public Suffix List extractor used
for domain grouping provides a more appropriate reference definition.

In the uploaded legitimate cohort, the naive count exceeds the PSL-aware count
for 13,142 URLs; 6,556 of these are false positives. However, genuinely two-or-more
PSL-aware subdomains still have 65.85% FPR (8,195 of 12,445), so overcounting is
not the whole explanation. The `co.uk` slice itself has only 4.32% FPR here.
Do not claim that correcting this definition would remove all of these errors.

**Do not change this feature in the active serving pipeline alone.** The saved
classifier learned the old values. A corrected feature requires a versioned
training experiment and corresponding saved transformer, followed by evaluation.
The active feature code and model were not changed during this audit.

### Source-specific conventions and other datasets

- In active-model PhiUSIIL training rows, all 107,778 legitimate URLs use HTTPS,
  start with `www.`, and have a root path. The other training source is different:
  legitimate rows are 71.99% `www.`, 2.90% root path, and 33.30% HTTP. Its phishing
  rows are 97.67% HTTP. Such source-specific patterns can encourage shortcuts;
  these distributions do not isolate a causal mechanism.
- PhishNChips legitimate examples are almost all HTTPS root URLs without `www.`:
  498 of 501 lack `www.`, and 500 have root paths. The overall FPR is 88.22%.
  Long paths and suspicious keywords cannot explain most errors there. The three
  `www.` examples are too few to estimate a reliable protective effect.
- In the ealvaradob retained subset, legitimate HTTP URLs have 49.87% FPR versus
  12.98% for HTTPS; HTTP accounts for 84.12% of false positives. Long hostname
  association persists within HTTPS: at least 30 characters gives 42.73% FPR
  versus 12.12% otherwise. Blogspot FPR is 92.20% (130 of 141); sample construction
  and label quality still limit interpretation.

### Model contribution check and next experiment

Native XGBoost TreeSHAP was computed for a seeded sample of up to 100 false
positives and 100 accepted legitimate URLs per source. Contributions were summed
and transformed back to probabilities to verify that they reproduce predictions,
using the same early-stopping tree range. Positive contributions increase the
phishing log-odds, not a verified probability of real-world harm.

Hostname length had the largest mean positive contribution on the uploaded false
positive sample; subdomain count led the PhishNChips sample, and protocol led the
ealvaradob sample. These sampled explanations are consistent with observed source
patterns, not proof of causation or independent confirmation. See the official
[XGBoost prediction documentation](https://xgboost.readthedocs.io/en/stable/prediction.html)
for native contribution outputs and early-stopping behavior.

The next proposed experiment is a versioned PSL-aware feature definition plus
better independently labeled legitimate coverage of hosted blogs, long/nested
hostnames, non-`www.` roots, and HTTP, alongside realistic HTTPS phishing. Keep
domains and sources isolated; compare with the unchanged deployed baseline at a
validation-chosen false-positive budget. Do not whitelist Blogspot or other shared
hosting providers, and do not retune on these audited external cohorts then call
them untouched final tests.

Full diagnostic output:
`artifacts/evaluations/false_positive_patterns_2026_10_08/false_positive_patterns.json`.
It includes pattern support, counterpart FPR, false-positive coverage, per-source
and HTTPS-only slices, feature medians, keyword/domain counts, the semantic audit,
training-shape summaries and sampled model contributions. No new source file,
dependency, retraining or active threshold change was needed.
