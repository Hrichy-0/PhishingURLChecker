# PhishingURLChecker — consolidated dataset audit

Date: 2026-10-08. Scope: active-model training sources, uploaded URL dataset,
and the two local Hugging Face URL exports. Three read-only agents audited these
areas; the coordinator checked artifact identities, key counts, and the two
representation-conflicting training pairs. This is one consolidated report, not
three additional code modules.

## Executive decision

Keep the active model and extension unchanged. The data has strong collection
shortcuts, a confirmed representation-level label-conflict gap, and substantial
external false positives. Domain isolation passes for the saved training splits,
but this does not establish real-world reliability or source/time independence.

The next justified work is a separately versioned data/feature experiment, not
another threshold increase or a hosting-platform whitelist. All findings below
are either observed facts or explicitly qualified hypotheses. “Legitimate” means
the dataset publisher's label; no website safety was independently verified.

## 1. Scope, methods, and evidence

The audits inspected local CSVs, cleaning/feature/splitting code, manifests,
saved model metadata, evaluation JSONs, and false-positive diagnostics. Checks
covered labels, syntax, missing values, exact and alternative-key duplicates,
URL/hostname/domain overlap, source-specific shapes, exclusions, clustered
errors, and frozen-threshold results. The uploaded and HF audits independently
recomputed predictions; the training audit did not rerun model fitting or inference.

No listed URL was visited, resolved, scraped, or executed. Primary HF dataset
cards were consulted for provenance, not website safety. No dependencies were
installed, data rewritten, files deleted, model retrained, or threshold deployed.
No new source-code file was added. The existing suite passes: **75 tests**, with
two dependency deprecation warnings. Tests do not validate publisher labels.

Key local evidence, linked relative to this report:

- [Active metadata](../../artifacts/model_metadata.json) and
  [mixed split manifest](../../data/processed/mixed/split_manifest.json).
- [Validation cutoff comparison](../../artifacts/evaluations/validation_thresholds_2026_10_06/validation_thresholds.json).
- [Same-cohort model comparison](../../artifacts/evaluations/url_dataset_2026_10_06/common_evaluation.json).
- [Frozen external cutoff comparison](../../artifacts/evaluations/fixed_threshold_external_2026_10_08/external_evaluation.json).
- [Full false-positive diagnostic](../../artifacts/evaluations/false_positive_patterns_2026_10_08/false_positive_patterns.json).
- [Cleaning](../../src/phishguard/data.py), [features](../../src/phishguard/features.py),
  [domain grouping](../../src/phishguard/domains.py), and [API](../../api/main.py).
- [HF manifests: PhishNChips](../../data/external/huggingface/phishnchips/download_manifest.json)
  and [ealvaradob](../../data/external/huggingface/ealvaradob/download_manifest.json).

Raw/derived data and generated artifacts remain local/ignored by Git; these
links can therefore be unavailable in a public repository checkout.

## 2. Training-data integrity: verified separation, not universal leakage clearance

All split fingerprints match both active metadata and the saved manifest.
Prepared source fingerprints match too. Each split has zero missing URL, label,
or source values and zero exact duplicate URLs.

| Split | Rows | Legitimate | Phishing | Registered domains | Hostnames |
|---|---:|---:|---:|---:|---:|
| Train | 457,544 | 244,242 | 213,302 | 285,844 | 330,622 |
| Validation | 56,621 | 30,092 | 26,529 | 35,731 | 41,984 |
| Test | 60,551 | 30,508 | 30,043 | 35,731 | 44,132 |
| Total | 574,716 | 304,842 | 269,874 | 357,306 | — |

Every split pair has **zero shared exact URLs, hostnames, or registered domains**
under the project's offline Public Suffix List (PSL), including private suffixes.
The supplementary normalized-key audit also found zero cross-split matches.
This does not test shared templates, campaign membership, redirects, collection
lineage, time leakage, or whether the bundled PSL covers every platform tenant.

### Label transformation and reconstruction

The audits re-applied cleaning to local raw sources and verified row-by-row URL
and internal-label agreement with prepared source CSVs.

| Source | Original labels / rows | Cleaning | Prepared rows |
|---|---|---|---:|
| PhiUSIIL | 235,795; source `1` legitimate: 134,850; source `0` phishing: 100,945 | 425 exact URL duplicates; no missing/blank/invalid removals | 235,370: 134,850 legitimate / 100,520 phishing |
| Balanced URLs | 340,000; textual legitimate/phishing: 170,000 each | 368 invalid URLs, 276 duplicates | 339,356: 170,000 legitimate / 169,356 phishing |

Prepared sources share 10 exact URLs with consistent labels; merging removes
these 10 duplicates and yields 574,716 rows. No exact contradictory source labels
were found. Internally, **0 means legitimate and 1 means phishing**. The reversed
PhiUSIIL source mapping must not be applied to textual or project-label datasets.
Transformation consistency is not proof that the original labels are correct.
The balanced source's original collection, labeling process and dates are not
verified by the local files; source-path and content fingerprints do not fill that gap.

### Confirmed cleaning gap: opposing labels for identical model inputs

Two root-URL groups have opposite labels, four rows total, all in training:

- HTTPS root on `www.education.gouv.fr`: the empty-root spelling is legitimate;
  the trailing-slash spelling is phishing. Both are PhiUSIIL records.
- HTTP root on `www.ipinfodb.com`: the empty-root spelling is phishing in
  PhiUSIIL; the trailing-slash spelling is legitimate in the balanced source.

The active extractor removes a root slash, so each pair produces **identical
feature vectors with opposing targets**. The coordinator independently verified
the equality. These are data inconsistencies relative to this model's representation,
not cross-split leakage or independently adjudicated website verdicts. Four rows
cannot explain the overall external error rate. Future preparation should detect
and quarantine such conflicts rather than guess which label is right.

A broader screening key found 977 extra rows across 833 multi-row groups: train
781, validation 105, test 91. That key lowercases scheme/netloc and removes root
slashes and fragments. It is **not a universally safe merge policy**: userinfo
can be case-sensitive and fragments can distinguish application states. The two
confirmed opposing-label pairs above require only root-slash equivalence.
There was no full census of all collisions in the 21-dimensional feature space.

289 hostnames and 977 registered domains contain both labels. This is not
automatically an error: different paths, compromised sites, and shared hosting
can legitimately contain both classes.

## 3. Training-source shortcuts and correlated observations

Training-only source/class shape statistics:

| Source / class | Rows | HTTPS | `www.` host | Root path | Hostname ≥30 characters |
|---|---:|---:|---:|---:|---:|
| PhiUSIIL legitimate | 107,778 | 100% | 100% | 100% | 2.69% |
| PhiUSIIL phishing | 78,222 | 47.24% | 42.93% | 71.52% | 23.83% |
| Balanced legitimate | 136,464 | 66.70% | 71.99% | 2.90% | 0.92% |
| Balanced phishing | 135,080 | 2.33% | 27.94% | 7.00% | 23.29% |

All 134,850 legitimate PhiUSIIL rows across all splits are HTTPS, `www.`, root
URLs without queries. Balanced training contains 131,933 HTTP phishing URLs
versus 45,444 HTTP legitimate URLs. Across training, long hostnames occur in
4,152/244,242 legitimate rows versus 50,106/213,302 phishing rows. Median path
length is 0/1 for PhiUSIIL legitimate/phishing and 32/27 for balanced legitimate/phishing.

**Hypothesis:** these collection/class conventions encourage shortcut learning.
Domain separation does not remove them. No causal ablation was performed.

Rows also cluster within hosts/domains despite exact deduplication:

| Split | Singleton hostnames | Singleton domains | Largest domain cluster |
|---|---:|---:|---:|
| Train | 273,371 | 222,650 | 1,556 rows |
| Validation | 34,984 | 27,922 | 559 rows |
| Test | 36,746 | 27,870 | 3,079 rows |

The largest train hostname has 1,196 rows; its ten largest hosts contribute
4,671 rows, and its ten largest domains contribute 8,733. The ten largest test
domains contribute 5,632 rows, or 9.30% of that test. Ordinary row-independent
confidence intervals could be misleading. Domain-clustered intervals and
per-domain reporting are proposed, not computed in this audit.

Validation informed three hyperparameter candidates, early stopping, and
F1/recall-constrained threshold selection. Best iteration 799 reaches the allowed
800 trees; this is not evidence of convergence or a need to add trees. Reused
validation results are development estimates, not independent confirmation.

## 4. Feature definitions: revise only in a separately trained version

The active `subdomain_count` is `max(nonempty hostname labels - 2, 0)`, not
PSL-aware. It conflates subdomains with multipart public suffixes, private
hosting suffixes, and IP address labels. A schematic `www.example.co.uk` yields
two instead of one. Domain grouping already uses an offline PSL-aware extractor.

Naive counts exceed PSL-aware counts in 79,361/457,544 training rows:
13,780 PhiUSIIL legitimate, 24,113 PhiUSIIL phishing, 15,762 balanced legitimate,
and 25,706 balanced phishing. In the uploaded legitimate evaluation cohort,
13,142 rows are overcounted, including 6,556 false positives. However, genuinely
two-or-more PSL-aware subdomains still have 65.85% FPR there. Overcounting is
not the whole explanation, and these counts do not predict the benefit of a fix.

`tld_length` describes the final hostname label, not the whole PSL suffix. This
is a semantic/documentation distinction, not automatically a bug. A future
experiment should distinguish public suffix, private tenant domain, and true
subdomain features explicitly. **Do not change definitions underneath the saved
classifier**; preserve the baseline transformer and retrain a versioned candidate.

## 5. Uploaded dataset audit

Local input: `URL dataset.csv` (a locally uploaded file), SHA-256
`accb2dfbfd3329a8b5cb1b85dcad90314a660c272be755cb45d0f6865014b466`.

It contains 450,176 rows with `url`/`type`: 345,738 legitimate, 104,438 phishing.
There are no nulls, blanks, leading/trailing URL whitespace, exact duplicates,
or exact contradictory labels. Cleaning removes 567 invalid/non-HTTP(S) URLs:
36 legitimate and 531 phishing. Scheme anomalies include 35 legitimate `httpss:`
values and FTP records; malformed syntax is not proof of a wrong phishing label.

### Protocol shortcut and filtered population

Of 449,609 cleaned rows, **all 345,702 legitimate URLs are HTTPS**. Phishing
contains 97,428 HTTP and 6,479 HTTPS URLs. The retained cohort repeats the
shortcut: 153,225 HTTPS legitimate URLs, 55,138 HTTP phishing, 2,906 HTTPS phishing.
A protocol-only rule scores 98.62% accuracy on that cohort, demonstrating dataset
bias, not a real-world detection rule. The upload has no verified origin, dates,
license, collection method, or label adjudication; why protocols look this way
cannot be established.

To reproduce the same-model comparison, the reference union covers both active
mixed splits and later source-tuning references. After syntax cleaning:

- 9,265 exact known URLs are excluded: 419 legitimate / 8,846 phishing.
- Another 229,075 shared-domain rows are excluded: 192,058 legitimate / 37,017 phishing.
- 211,269 remain: 153,225 legitimate / 58,044 phishing, only 47.0% of cleaned rows.
- Legitimate URLs span 84,700 hostnames / 81,246 private-aware registered domains;
  phishing spans 38,330 hostnames / 35,147 domains.

These asymmetric exclusions change prevalence. Scores describe this filtered
cohort, not the complete upload or representative browsing traffic.

### Major false-positive slices

Slices overlap; do not add their counts together. Rates use publisher-labeled
legitimate URLs at the active threshold.

| Slice | URLs | False positives | URL FPR | Unique domains | Domains with FP |
|---|---:|---:|---:|---:|---:|
| All legitimate | 153,225 | 18,054 | 11.78% | 81,246 | 10,568 |
| Hostname ≥30 characters | 13,008 | 8,890 | 68.34% | 9,044 | 5,776 |
| Blogspot tenant suffix | 7,033 | 6,067 | 86.26% | 5,874 | 5,018 |
| Long hostname, excluding Blogspot | 8,550 | 4,432 | 51.84% | 5,260 | 1,992 |
| At least two naive subdomains | 24,877 | 14,551 | 58.49% | 14,221 | 8,206 |
| At least two PSL-aware subdomains | 12,445 | 8,195 | 65.85% | 4,653 | 2,994 |
| Naive count exceeds PSL count | 13,142 | 6,556 | 49.89% | 10,003 | 5,334 |
| Host without `www.` | 361 | 259 | 71.75% | 348 | 255 |

Shorter hostnames have 6.54% FPR; excluding all Blogspot URLs gives 8.20% FPR.
The candidate cutoff still flags 4,445 Blogspot URLs (63.20%) and 6,722 long
hostnames (51.68%). Blogspot accounts for 33.60% of all current false positives,
but long-host errors persist without it. Almost all legitimate URLs start `www.`;
the non-`www.` slice has limited support and may not generalize.

Predefined login/payment-style tokens cover only 121 false positives (0.67%).
Explicit-port and `@` slices have just 12 and 17 legitimate examples. Long URLs
and queries are not universally harmful: URL length ≥100 has 13.75% FPR and
queries 12.64%; nonroot paths have 11.24% versus root paths 13.24%.

### Domain concentration and representation checks

The equal-domain mean of each domain's URL FPR is 11.98% overall, 63.21% for long
hosts, 84.63% for Blogspot, and 36.75% for long hosts excluding Blogspot. These
are not domain-classification accuracies. Error-heavy repeats affect some rates,
but do not explain away the findings. Top ten domains account for 7.60% of all
false positives; top ten Blogspot tenants account for only 2.19% of Blogspot errors.
13.01% of legitimate domains have at least one falsely flagged URL.

A fragment-preserving key lowercasing scheme/hostname and treating empty root
as `/` finds 288 extra equivalent rows in cleaned data (170 phishing / 118
legitimate), and 106 in the retained cohort (102 phishing / 4 legitimate), with
no opposing-label groups. It preserves userinfo, explicit ports, path/query case,
ordering, fragments, encodings, `www.`, and HTTP/HTTPS distinctions. These small
residual duplicates cannot explain the main false positives. Preliminary counts
282/101 were corrected to **288/106** after handling uppercase hosts consistently.

## 6. Hugging Face audit: provenance and population selection

| Stage | PhishNChips | ealvaradob |
|---|---:|---:|
| Original rows | 2,000 | 835,697 |
| Original legitimate / phishing | 1,000 / 1,000 | 444,933 / 390,764 |
| Invalid HTTP(S) URLs | 0 | 608,193 |
| Exact duplicates after validation | 303 | 14,295 |
| Clean deduplicated URLs | 1,697 | 213,209 |
| Exact active-reference exclusions | 50 | 2,201 |
| Additional shared-domain exclusions | 652 | 45,385 |
| Retained URLs | 995 | 165,623 |
| Retained legitimate / phishing | 501 / 494 | 19,338 / 146,285 |
| Retained hostnames / domains | 967 / 954 | 143,218 / 117,549 |

Both exports have `url`/`label`, no missing fields, project labels `{0,1}`, and
fingerprints matching pinned manifests. No exact contradictory labels were found.
Neither evaluation invents missing protocols.

PhishNChips' 303 removed duplicates are legitimate; clean classes are 697/1,000.
Reference exclusion removes another 196 legitimate and 456 phishing rows, all
50 exact exclusions phishing. ealvaradob has 607,938 strings without schemes:
392,898 legitimate / 215,040 phishing. Its total invalid removals are 392,907
legitimate / 215,296 phishing, so some invalidity has other causes. Its additional
domain exclusions remove 22,955 legitimate / 22,430 phishing; exact exclusions
remove 22 legitimate / 2,179 phishing.

Only 4.35% of ealvaradob's original legitimate rows survive, versus 37.44% of
phishing. Phishing prevalence rises from 46.76% to 88.32%. Results therefore
describe a selected complete-URL population, not the original full dataset.

### Task suitability, provenance, and rights

- [PhishNChips](https://huggingface.co/datasets/AreLit/PhishNChips) is an LLM
  email-agent benchmark. Our export uses only `url_raw`/`phish_label` from
  `core_emails.csv`, pinned at `89afcc39610084298c4679159cb2e27d9ffffa46`.
  Removing email context does not reproduce its intended task. Feed-derived
  phishing and seeded/generated benchmark context do not independently verify
  website labels. Its sources have mixed rights; the project's MIT/synthetic
  content terms do not imply every upstream indicator is unrestricted.
- [ealvaradob](https://huggingface.co/datasets/ealvaradob/phishing-dataset) exports
  `urls.json`, pinned at `55caf9af01ae498d68ff44886ff69a4b165eb54b`. Its card identifies
  a Kaggle URL compilation from other feeds/repositories. HF hosting does not
  establish independent collection. Apache-2.0 is publisher metadata, not an
  audit of upstream rights. Actual local duplicate counts must take precedence
  over a generic card description of “deduplicated” data.

Current primary cards supplement local pinned manifests and documentation;
pinned-card pages were unavailable in the agent's browsing check, so current
wording is not asserted to match the old revision. Per-row collection times,
source identities, and adjudication context were not retained in the URL-only
exports. No remote repository loading code was executed. Do not republish raw
data assuming these metadata labels alone settle permissions.

## 7. Hugging Face false positives and independence

### PhishNChips: broad root-domain failures, not long hosts

All 501 retained legitimate URLs are HTTPS; 500 have root paths, 498 lack `www.`,
none have hostnames ≥30 characters, and none have queries/fragments. The model
flags 442 different legitimate domains, one legitimate URL each. Thus repeated
pages, Blogspot, or long-host clusters cannot explain this source's errors.

Non-`www.` FPR is 442/498 (88.76%); `www.` FPR is 0/3, too small to establish
robustness. Root FPR is 441/500 (88.20%); the one nonroot URL is flagged. Phishing
contains 204 HTTP / 290 HTTPS, 162 roots, 113 long hosts, 12 `www.`, 30 queries,
and two fragments. These are benchmark-construction differences, not a browsing
sample. One legitimate error changes FPR by about 0.20 percentage points; 1%/2%
empirical budgets allow at most five/ten errors in this cohort.

### ealvaradob: protocol and source effects

Legitimate HTTP FPR is 5,589/11,208 (49.87%), versus HTTPS 1,055/8,130 (12.98%).
HTTP accounts for 84.12% of false positives. Long-host FPR is 406/619 (65.59%),
but long hosts account for only 6.11% of all false positives. Within HTTPS, long
hosts still have 42.73% FPR versus 12.12% for shorter hosts.

Root URLs have 2,227/3,879 FPR (57.41%) versus nonroots 4,417/15,459 (28.57%).
`www.` FPR is 29.40%, versus non-`www.` 41.78%. These correlated comparisons
do not isolate causes. Legitimate URLs span 13,761 hostnames / 13,329 domains;
5,201 domains have a false positive. Equal-domain mean URL FPR is 36.52%, above
the URL-weighted 34.36%, so a few large sites do not explain away the failures.

Example publisher-label domain clusters are `sensagent.com` (110/115 current
false positives; 74 candidate), `dictionary.com` (97/158; 0 candidate),
`definitions.net` (82/94; 15), `explained.today` (82/83; 81), and `academic.ru`
(73/425; 57). These are inert dataset groups, not verified-safe sites or whitelist
recommendations. Blogspot has 130/141 current false positives (92.20%).

### Equivalent records and cross-source dependence

A fragment-preserving HF key lowercases scheme/hostname, normalizes default
HTTP/HTTPS ports and empty root paths, and preserves userinfo/path/query/fragment.
Retained PhishNChips has zero extra equivalent rows; retained ealvaradob has
1,474 (1,469 phishing / five legitimate), with no opposing-label groups. Before
domain exclusion, clean ealvaradob has 1,542 extras. If fragments are additionally
dropped, these counts become 1,552 retained and 1,655 before exclusion. The latter
is only a near-duplicate screen: fragments can represent distinct SPA states.
These policies differ from the uploaded/training screens; do not sum the counts
or apply them as one automatic cleanup rule.

No tested equivalent-key matches remain against active references. Across the
two HF sources, before reference exclusion there are three exact common URLs,
62 shared domains and five fragment-dropped matches. Retained cohorts have zero
exact common strings but **11 shared domains and one fragment-preserving
equivalent URL** (same legitimate label; root slash difference). Do not treat
these datasets as completely independent. ealvaradob has three mixed-label hosts
and ten mixed-label domains; these are not necessarily labeling errors.

Both sources were subsequently used in source-tuning experiments and are now
development data. Historical active-model stress tests remain valid observations,
not untouched final-test estimates. Domain purging cannot prove absence of common
feeds, templates, campaigns, or collection lineage.

## 8. Threshold conclusions and model explanation limits

Both operating points were independently reproduced; no external labels were
used to select the candidate. Current cutoff: `0.4711279273033142`; validation-chosen
candidate: `0.7853294610977173`.

| Cohort | Rows | Current FPR → candidate FPR | Current recall → candidate recall |
|---|---:|---:|---:|
| Uploaded same-reference-union cohort | 211,269 | 11.78% → 7.53% | 91.50% → 69.30% |
| PhishNChips | 995 | 88.22% → 16.97% | 77.53% → 62.55% |
| ealvaradob complete-URL subset | 165,623 | 34.36% → 18.90% | 72.27% → 55.47% |

| Cohort | False alarms avoided | Additional phishing missed |
|---|---:|---:|
| Uploaded | 6,518 | 12,884 |
| PhishNChips | 357 | 74 |
| ealvaradob | 2,990 | 24,575 |

Uploaded accuracy falls from 89.12% to 86.11%, and HTTPS-only recall falls from
67.41% to 49.24%. ealvaradob accuracy falls from 71.50% to 58.47%, both below its
88.32% majority-class baseline. PhishNChips accuracy rises from 44.42% to 72.86%.
Accuracy/precision alone can obscure skewed class prevalence. The candidate
does not meet a 2% FPR on any external cohort and is not approved for deployment.

The prior native TreeSHAP diagnostic used seeded samples of 100 FP and 100
accepted legitimate URLs per source, except PhishNChips has only 59 accepted
legitimate URLs. Contribution sums were checked against model probabilities using
the matching tree range. On uploaded sampled FPs, hostname length has mean
contribution +2.788 margin units while naive subdomain count is −0.527. Hence the
high-subdomain-error association is not proof of a positive causal contribution.
PhishNChips sampled contributions emphasize subdomain/path features; ealvaradob
emphasizes protocol. This is a sampled model explanation, not a website verdict,
causal ablation, or fresh confirmation. See [XGBoost prediction documentation](https://xgboost.readthedocs.io/en/stable/prediction.html).

## 9. Syntax and API-compatibility limitations

The local cleaner checks parser syntax, HTTP(S), hostname presence, whitespace,
and port bounds. It does not establish DNS validity, browser interpretation, or
safety. Cleaned uploaded data still includes 165 single-label hosts (all phishing),
30 underscore-host records (29 phishing), and two literal-percent hostnames (both
phishing). Retained ealvaradob contains six userinfo URLs, two trailing-dot hosts,
487 underscore-host records, five empty-internal-label hosts, four non-ASCII
hosts, and 564 fragments. These are review flags, not automatic evidence of
incorrect labels or a universally invalid URL.

The API limits URL input to 2,048 characters; evaluation cleaning has no equivalent
length limit. The cleaned upload contains one valid URL of length 2,314, labeled
phishing; ealvaradob contains two valid over-limit URLs (maximum 3,992). PhishNChips
has none. Survival of these over-limit records in each final external cohort was
not verified, so these are input-population compatibility flags, not confirmed
differences in the reported retained cohorts. Browser/UI parity was not tested in
this dataset audit.

## 10. Prioritized next work — not implemented by this audit

1. **Data consistency:** add carefully defined representation-equivalence checks
   before future splitting; quarantine opposing-label groups and preserve original
   URL text. Resolve labels using independent evidence, not brand popularity.
2. **Versioned features:** trial PSL-aware subdomain/suffix/tenant features alongside
   the unchanged baseline. Retrain matching transformers; do not patch serving
   definitions under old artifacts. Use controlled ablations to measure benefit.
3. **Better labeled coverage:** independently source legitimate long/nested hosts,
   hosted tenants, non-`www.` roots, deep links and HTTP, plus realistic HTTPS
   phishing. Do not simply add these audited evaluation rows to training and still
   describe them as test data. Do not whitelist brands or hosting platforms.
4. **Honest evaluation:** source-held-out/domain-purged comparisons, validation-only
   fixed operating points, protocol/majority baselines, per-source/per-domain
   metrics and domain-clustered uncertainty. Keep transformations and cohorts explicit.
5. **Fresh final data:** record collection dates, source lineage, label evidence and
   license/context; reserve time/source-independent data before further selection.
   Original holdout and audited external sets are development evidence now.
6. **Compatibility:** inspect flagged URL formats and add conservative parser/API
   compatibility tests before claiming equivalent browser behavior.

Not established: current website safety/ownership, independent publisher labels,
temporal/source/template independence, statistical significance or confidence
intervals, the causal effect of any feature, or production browser reliability.

## Artifact identity and preservation

The model and metadata remained unchanged throughout this audit:

```text
Model SHA-256:
89ae67ee2911e4072a699afb21307015f3f1635e98acfbab579f1ac2740188c7
Metadata SHA-256:
7dd13d9ec074a4db7889f486dc0bab8a81399dc5388a1603a835d9abcf185c1e
```

The uploaded audit's revised canonical counts supersede its preliminary messages.
The broader normalized-key findings remain screening results with explicitly
different policies. Historical artifacts are preserved; no labels or old results
were silently rewritten. This report is the single human-readable deliverable.
