# Professor evaluation follow-up: measured progress and open gates

The original TalentUD scope is retained.
This report separates completed checks from unresolved reliability questions.

## 1. Cross-validation: both XGBoost candidates completed

Both candidates used the same 454,185 train+validation records, 154,074 connected
domain/exact-content groups and identical fit/inner/outer row assignments in five
outer folds. Role assignments were compared exactly. Fixed depth-7 settings and
equal budgets (800 rounds, patience 40) were used; grouped inner validation alone
controlled stopping and thresholds. Test/publisher records were not scored.

| Candidate / inner-validation policy | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| URL-only / F1 with >=90% inner recall | 90.93% | 90.09% | 89.95% | 8.26% |
| URL+content / F1 with >=90% inner recall | 92.59% | 95.01% | 88.35% | 3.87% |
| URL-only / <=1% inner FPR | 86.68% | 98.26% | 71.98% | 1.06% |
| URL+content / <=1% inner FPR | 90.14% | 98.45% | 79.57% | 1.04% |
| URL-only / <=2% inner FPR | 89.74% | 96.92% | 79.97% | 2.12% |
| URL+content / <=2% inner FPR | 92.23% | 97.10% | 85.47% | 2.13% |

The combined candidate uses 21 URL + 17 cached static-content features. Its best
round counts were 542, 798, 209, 535 and 496. All five combined models were
reloaded on CPU and all 454,185 OOF scores reproduced exactly. Mean fold ROC-AUC
was 98.60% (SD 0.45 percentage points), AP 98.45% (SD 0.53 points).

These are newly fitted research fold models, not new scores for either original
saved model. Existing hyperparameters were already selected on this collection;
this is development evidence, not unbiased final evaluation. The original
proposal's logistic baseline still needs a same-cohort comparison. No candidate
was promoted and the extension's model/cutoff remain unchanged.

## 2. Leakage and label/composition audit

All four source-file hashes match the original manifest. Cached URL features
reproduce exactly and registered domains recompute consistently. No exact URL,
domain or nonempty exact-content hash crosses the original splits or a CV role.
This is evidence against those specific leakage mechanisms, not proof of absence
of similar templates, source artifacts or temporal dependence. Date ranges overlap.

An additional development check found **9 exact-content hashes shared by both
labels, affecting 66 records**. Same HTML on different URLs is not automatically
a wrong label (a generic/error/login page can be reused), but this needs raw-content
and label-evidence review. The feature Parquets cannot reconstruct the raw HTML.
No records have been relabeled or deleted.

### Why fold 5's URL F1 cutoff moved

Its grouped inner-validation subset contains 116,963 rows. A single connected
component (ID 8) has 32,629 rows across 3,340 domains, including 31,626 phishing
labels. This is 27.90% of that inner subset. The large component stays together,
as required to prevent leakage, but strongly influences row-weighted threshold
selection. Its size comes from transitive domain/content connections; it is not
one single identical page or one single domain.

Diagnostic only: keeping the fitted model and scores fixed, removing this
component from cutoff selection moves the inner F1/recall cutoff from **0.130803
to 0.491091**. Removing each of the next four largest components individually
does not move the original cutoff. This isolates strong sensitivity to the large
group in this selection calculation; it is not an approved exclusion rule or
new deployment cutoff. No outer results were used to choose a replacement.

The original F1 policy has outer FPR 4.25%-16.71%. Component-balanced cutoff
selection, stronger inner grouping stratification or fixed FPR-budget policies
are reasonable *future development experiments*, not post-hoc fixes to present
as newly independent validation. Preserve this run and predeclare new policies.

## 3. Statistical uncertainty: paired comparison, not just two scores

Both models' predictions were paired on exactly the same held-fold records and
whole components resampled together within folds (1,000 replicates). Differences
below are URL+content minus URL-only, in **percentage points**.

| Policy / metric | Difference | Conditional 95% interval |
|---|---:|---:|
| F1 policy / recall | -1.61 | -10.59 to +4.98 |
| F1 policy / FPR | -4.38 | -5.03 to -3.76 |
| 1% FPR policy / recall | +7.59 | -7.81 to +18.06 |
| 1% FPR policy / FPR | -0.02 | -0.11 to +0.08 |
| 2% FPR policy / recall | +5.50 | -6.76 to +14.39 |

The low-FPR recall intervals include zero: **the observed recall gain is not
conclusive under this analysis**. The F1-policy FPR reduction is more consistent
conditional on these fitted models, but does not prove real-world superiority.
Intervals omit full retraining uncertainty, dependence induced by overlapping
training folds, source/temporal drift and label errors. No universal statistical
significance or future-population guarantee is claimed. Five folds are also not
five independent real-world datasets. Do not optimize a new policy on the final
cohort and subsequently call it untouched.

## 4. Actual Chrome UI automation: first benign pilot completed

The installed extension's source folder was verified as this repository's
`extension/`. It was reloaded without changing its permissions or model, and a
local-only API was started on `127.0.0.1:8000`. Native UI automation opened each
public page, opened the real popup, clicked Check and read its displayed result.

| Fixed page | Popup score | Popup result | API/display consistency |
|---|---:|---|---|
| https://www.apple.com/de/ | 13.1/100 | LOW RISK | Pass |
| https://www.python.org/ | 29.2/100 | LOW RISK | Pass |
| https://docs.python.org/3/ | 70.3/100 | HIGH RISK: false positive on official documentation | Pass |
| https://www.mozilla.org/en-US/ | 9.5/100 | LOW RISK | Pass |

All four actual API POSTs returned HTTP 200. Every popup showed 21 features,
checksum prefix `9d9ae6a4808b` and cutoff 0.4902, with the expected rounded score
and risk state. On a Chrome-internal non-HTTP page, the Check button was visibly
disabled. No forms were submitted, credentials entered, security interstitials
bypassed or live phishing URLs visited.

**A correct popup can display an incorrect model decision.** Python documentation
is a concrete false-positive case, not a UI failure. Offline TreeSHAP for this
prediction gives the largest upward margin contribution to path length (+1.946),
then longest token (+0.589) and digit ratio (+0.344). These are model associations
in log-odds units, not evidence that documentation or short paths are unsafe.
The generic popup reason does not explain this individual model decision.

This is a four-page engineering pilot in the existing browser profile, not an
isolated-profile or blinded accuracy study. All four domains already occur in
development; Apple Germany's exact URL also occurs in training. They cannot be
called unseen external cases. No population FPR/accuracy is estimated from these
four selected pages. Content inference and browser-local inference were not tested.
The API remains running locally for further manual checks.

## 5. Remaining gates before claiming the professor's advice is fully addressed

1. Review the suspicious content/label groups and similar-template/source/time
   dependence. Obtain the relevant original HTML and label evidence rather than
   trying to infer it from 17 aggregate numbers. Do not silently drop hard groups.
2. Freeze an evaluation policy and a genuinely fresh, domain-disjoint cohort with
   source/time provenance and reviewed labels for both classes. The current four
   pages form a UI regression list only, not that final benchmark.
3. Extend UI automation in an isolated test profile; separate benign site visits
   and inert positive fixtures from phishing-URL-string accuracy evaluation.
   Do not browse live hostile pages in the normal profile. Record redirects,
   errors, blocked/unreachable pages, permissions, model identity and parity.
4. Complete browser-local export/extractor parity and latency/memory/privacy
   measurements under the original plan. This is separate from the local API
   engineering smoke check and cannot be claimed complete yet.

The professor's reliability gate is **not yet fully passed**. We now have measured
limitations rather than merely a plan. Do not change
the extension's model or cutoff based on this development comparison.

## Reproduce and find the evidence

```bash
.venv/bin/python scripts/cross_validate_native.py --variant url_html \
  --output artifacts/experiments/NEW_combined_cv --trees 800 --threads 4
.venv/bin/python scripts/cross_validate_native.py --compare-runs \
  artifacts/experiments/talentud_url_grouped_cv_20261016 \
  artifacts/experiments/talentud_combined_grouped_cv_20261016 \
  --output artifacts/reports/NEW_paired_comparison.json
.venv/bin/pytest -q
```

- `artifacts/reports/professor_evaluation_followup_20261016.json`: consolidated
  findings, fixed UI cases, observed scores and completion/remaining-gate status.
- `artifacts/reports/talentud_paired_cv_20261016.json`: paired intervals/point differences.
- `artifacts/experiments/talentud_combined_grouped_cv_20261016/cv_results.json`:
  complete combined CV results, hashes, fold details and individual-model intervals.
- The corresponding `out_of_fold.parquet` and `role_assignments.parquet` remain
  local/ignored, as do the input datasets. No GitHub push occurred.
- Verification: 105 tests pass (two dependency deprecation warnings).

## Raw-content review handoff: original Windows shards are available

The applicant confirmed the original shards remain on Windows. The Mac prepared
78 requests: all 66 mixed-label exact-content rows plus six distinct-domain
representatives per class from the largest component. They map to 21 distinct
HTML snapshots. These are diagnostic samples, not final evaluation records.

`scripts/export_review_bundle.py` has two roles: prepare a deterministic request
list from verified development records/CV roles, or run standalone on Windows to
extract only matching snapshots from local original shards. It verifies source
SHA-256, URL/content-hash/label/date identity, records source row positions and
UTF-8 truncation, and saves deduplicated `.txt` snapshots capped at 1 MB each.
No browsing, fetching, rendering, execution, training or original-file edits occur.
It reports incomplete requests explicitly instead of substituting another version.

The private local handoff is `data/processed/review_request_package.zip` (script,
request list and README, about 14 KB). Extract its top-level
`review_request_package/` folder inside Windows `data/processed/`, then run from
the Windows project root:

```powershell
.\.venv\Scripts\python.exe data\processed\review_request_package\export_review_bundle.py --requests data\processed\review_request_package\review_requests.json --dataset-dir data\raw\phreshphish --output-dir data\processed\phreshphish_review_bundle
Compress-Archive -Path data\processed\phreshphish_review_bundle -DestinationPath data\processed\phreshphish_review_bundle.zip
```

Change only `--dataset-dir` if the raw dataset root lives elsewhere. It must
contain `data/train-000.parquet`, not numeric feature files. Hashing/locating
records may scan many original shards; output is capped at roughly 21 MB of
snapshot text, plus manifests. Transfer the resulting ZIP privately to the Mac;
do not upload request/snapshot data to GitHub or open snapshots as web pages.

The real raw-content audit remains pending the returned bundle. Four new tests
cover inert extraction, source integrity, row positions, deduplication, UTF-8
truncation, safe paths and explicit missing-snapshot reporting.

### Windows memory failure: resumable exporter update

The Windows run reported 67/78 matches after 68/77 shards, then raised MemoryError
inside Arrow HTML batch decoding. The exact allocation/RAM conditions are not
known. The exporter now uses serial single-record batches, no prefetch and direct
Arrow UTF-8 buffer hashing rather than full Python HTML/encoding copies. Pandas
is imported only for request preparation, not standalone extraction. See
[Arrow read controls](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html).

`--resume` migrates the original saved manifest, verifies request/snapshot identity,
skips completed shards, and checkpoints every new match through atomic manifest
replacement. New-format checked and completed sources are tracked separately, so
an interrupted source is not silently skipped. A repeated MemoryError records the
failed source and remains resumable; large individual Parquet pages can still
exceed available memory, so this is not a Windows success claim.

Private handoff: `data/processed/review_resume_fix.zip`, containing only the updated
script and instructions. Replace only the Windows script; preserve request JSON
and `phreshphish_review_bundle/`, then repeat the extraction command with `--resume`.
New tests simulate a mid-shard allocation failure, resume legacy partial progress
without rereading a completed shard, and reject changed requests/damaged snapshots.

## Returned bundle: URL versus URL+HTML replay (2026-10-19)

The Windows resume completed all 78 requests by shard 76. The returned ZIP was
read directly without extracting or rendering it. All request identities, labels,
dates, snapshot sizes/checksums and recorded shard hashes match the original
request. There are 21 distinct snapshots totaling 3,314,625 bytes (24 legitimate
and 54 phishing **dataset labels** across 78 records). Original raw shards were
not rehashed on the Mac; the Windows provenance is retained. One document is
truncated, so its full raw hash cannot be independently recomputed from the prefix.

The pure offline extractor reconstructed all **38 features exactly** as float32
against the original numeric feature records. The 21 URL inputs match too.
The original `html_truncated` flag for the bounded snapshot was restored from
verified transfer metadata; reading the shortened file alone would lose it.
Both original saved models and corresponding held-component CV fold models were
then scored on these reconstructed inputs. All 78 scores for each held-fold model
reproduce the existing OOF scores exactly. No retraining or threshold selection
was performed and no saved model/metadata changed.

### Original saved model replay: diagnostic, not independent accuracy

| Model | Features | Saved cutoff | Phishing-labeled records flagged | Legitimate-labeled records flagged |
|---|---:|---:|---:|---:|
| URL-only | 21 | 0.490197 | 51/54 | 3/24 |
| URL+HTML | 38 | 0.220903 | 54/54 | 18/24 |

These records come from the original train/validation splits, so these are not
unseen-model accuracy estimates. The saved models have different cutoffs and
training histories; differences cannot be interpreted as calibrated probability
changes or a causal effect of adding one particular feature.

### Held-component replay: inner-selected F1/recall policy

| Model | Phishing-labeled records flagged | Legitimate-labeled records flagged |
|---|---:|---:|
| URL-only | 52/54 | 4/24 |
| URL+HTML | 53/54 | 11/24 |

Each record was scored using the fold model that excluded its entire connected
component from fitting and cutoff selection. Compared with URL-only, the combined
model flags one additional phishing-labeled record and seven additional
legitimate-labeled records (eight decision disagreements). This anomaly sample
was selected after inspecting development data and intentionally overrepresents
problem cases. It is not representative of normal browsing, not independently
adjudicated ground truth and not evidence that combined models are universally
worse. The broader paired CV results above remain a separate development result.
Other fixed policies and full confusion counts are in the machine report.

### Plain-text content findings

All 66 mixed-label records map to nine exact-content hashes:

| Observed snapshot type | Hashes | Records | Dataset phishing / legitimate |
|---|---:|---:|---:|
| Shopify unavailable or nonexistent store pages | 5 | 46 | 32 / 14 |
| Apache default server page | 1 | 13 | 12 / 1 |
| Country-blocked access page | 1 | 3 | 2 / 1 |
| Connection-error/login application shell | 1 | 2 | 1 / 1 |
| Same astrology-report page content | 1 | 2 | 1 / 1 |

This supports a **capture-content/URL-label mismatch risk**, not automatic label
correction. A previously phishing URL may later return a generic unavailable or
blocked page. Identical default content on distinct URLs cannot by itself prove
which URL label is correct. The astrology case especially needs original URL and
label-evidence review rather than adjudication from page content alone.

The twelve large-component representatives include varied legitimate login/content
pages and phishing-labeled bank/cancellation content, a 404 page, a JavaScript app
shell, Google Drawings content and an empty plaintext-browser wrapper. This sample
does not show the whole 32,629-row component is one template. It confirms that
transitive grouping and capture availability need careful interpretation. No linked
scripts/resources were fetched; no JavaScript was executed.

Recommended next development step: define capture-usability categories and review
them without using model correctness to decide exclusions; report full-data and
predeclared usable-capture subsets separately. Preserve source URL labels and
explicitly mark unavailable/blocked/empty/JS-dependent captures. Do not relabel a
phishing URL as legitimate just because its captured page is unavailable. A fresh
source/time-reviewed holdout is still required before extension/model promotion.

Evidence:

- `artifacts/experiments/review_bundle_replay_20261019/summary.json`: verified
  counts, artifact hashes, feature parity, fixed/held-fold diagnostic results.
- `record_predictions.parquet` in that directory: request-level scores/flags and
  dataset labels, kept ignored by Git.
- `scripts/evaluate_review_bundle.py`: reproducible bounded offline ZIP verifier,
  extractor-parity and paired-model replay command.
- `tests/test_review_evaluation.py`: Windows ZIP names, path/payload/label rejection,
  no-network replay and preserved truncation metadata. Full suite: 117 tests pass.

```bash
.venv/bin/python scripts/evaluate_review_bundle.py \
  --bundle data/processed/phreshphish_review_bundle_completed.zip \
  --output artifacts/experiments/NEW_review_bundle_replay
```

At this point (19 October) the extension was still URL-only, and both saved
models and cutoffs were unchanged. This is offline testing of URL and HTML features, not
HTML integration into Chrome or a live-page accuracy experiment.

## URL-only external pilot preparation — 19 October 2026

HTML work is deferred while we examine the current URL-only model. Its checksum,
21-feature schema, saved 480-tree range and cutoff `0.4901971220970154` are frozen.
The extension is unchanged. The new unscored pilot excludes 952,770 registered
domains from 67 local URL-bearing CSV/Parquet files, including earlier evaluations.
It contains 100 OpenPhish feed candidates and 100 unlabeled Tranco candidates
(pinned list `467YX`), all pending independent evidence-based label review.
Popularity is not treated as a legitimate label, and retrieval dates are not
misrepresented as first-seen dates. No accuracy results have been computed.

`docs/archive/frozen-url-benchmark.md` explains the private review worksheet, frozen-input
checks, once-only scoring and uncertainty limitations. The software suite now has
134 passing tests. This prepares a fresh-source pilot; it does not complete label
adjudication or establish real-world reliability.

## RPA real-website testing with URL+HTML: tooling ready (23 October 2026)

At the user's direction, the real-life tests use the **URL+HTML model** on loaded
page content. This is a new experimental path. The earlier statement in this
document that the extension is URL-only described the state before this change.
The changes:

- A separate `POST /predict_html` endpoint.
- A "Check URL + page content (experimental)" popup button (adds only the
  `scripting` permission).
- A Playwright runner, `scripts/rpa_extension_check.py`. It opens the real popup
  with CDP `Extensions.triggerAction` in a fresh profile.

The URL-only model, `/predict` and both saved cutoffs are unchanged. Nothing was
retrained.

The smoke check (3 real pages and 3 simulated login fixtures) passed the
display/API match on 6 of 6. The URL+HTML model flagged mozilla.org's homepage,
which the URL-only model did not. This is not an evaluation. The full
pre-registered list of real legitimate sites and the results are still pending.
See [rpa-real-world-testing.md](rpa-real-world-testing.md).
