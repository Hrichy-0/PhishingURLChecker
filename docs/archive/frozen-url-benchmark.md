# Frozen URL-only external pilot

## What this step teaches

A holdout must be independent of model development **and label decisions must
not be influenced by predictions**. Downloading a different dataset is not
enough. We therefore freeze the model first, check historical domain overlap,
select candidates without scoring them, review their labels, and only then run
the locked evaluation once.

This is a small, source-specific pilot, not a claim of real-world reliability.
The Chrome extension, 21 lexical features and cutoff `0.4901971220970154` are
unchanged. No retraining or threshold search is part of this workflow.

### Preparation completed: 19 October 2026

- Scanned 81 local CSV/Parquet files; 67 contained URL columns. The other 14
  contain predictions, row assignments or scores without URL columns.
- Built an exclusion set of **952,770 previously observed local domains**.
- OpenPhish snapshot: 300 rows; removed 21 rows on known domains and collapsed
  23 repeated-domain rows, then deterministically selected 100 candidates.
- Tranco list `467YX`: selected 100 previously unseen domain candidates across
  the three predeclared rank bands (34 / 33 / 33).
- **200 candidates, all pending review. No predictions or accuracy results.**
- Full software suite: **134 tests pass**, with two existing dependency
  deprecation warnings. Ruff and Git whitespace checks pass.

The frozen model SHA-256 is
`9d9ae6a4808b0ac002c0062e0b1ddc83edd198dc52bfdbad5689aeab36559b47`.
The protocol SHA-256 is
`7bf75c5638a25e5d5529673e79fae68246d7120dd4f040b43b04f1433d353007`.
The source snapshot and candidate identities are retained privately. Do not
publish source URLs, review evidence or derived URL-level data to GitHub.

### Evaluation attempted on 20 October 2026

The frozen scoring command was run and exited with
`Label review incomplete; no predictions computed`. All 200 statuses remain
`pending`; reviewed labels, reviewers and evidence are blank. The 100 Tranco
candidates still need verified URL addresses. No prediction file or results JSON
was created, so accuracy, recall and FPR cannot yet be measured. Model and
metadata checksums still match the frozen protocol; neither weights nor cutoff
was changed. The 17 targeted benchmark-safeguard tests pass.

The next step is independent evidence-based review, not bypassing the safeguard
or treating source/ranking membership as verified truth. The dated attempt is
recorded in `artifacts/reports/frozen-url-pilot-attempt-20261020.json`.

### Verification options reviewed on 20 October 2026

The missing step is label adjudication, not a model repair. URL syntax,
reachability and HTTPS checks cannot establish whether a page is phishing.
Labels must refer to the exact URL and relevant observation date, and be decided
without seeing our model's output.

Recommended next work, **not yet implemented or performed**:

1. Match phishing candidates against a dated local copy of the
   [PhishTank verified feed](https://phishtank.net/developer_info.php). Preserve
   exact URL matches, phish IDs, verification times and provenance. Its community
   verification is useful corroboration, not infallible truth. A missing match
   means unresolved, not legitimate; different feeds can share upstream sources.
2. For Tranco candidates, obtain the exact page address from an official
   organization reference or authoritative directory and record dated ownership
   and page-purpose evidence. Use a fixed page-selection rule. Tranco ranks
   domains; it does not certify the safety of every URL on them. Do not invent
   HTTPS addresses or automatically label an entire domain benign.
3. Use existing reputation reports as supplementary evidence where appropriate.
   [VirusTotal](https://docs.virustotal.com/docs/how-it-works) aggregates scanner
   judgments, including false positives; a generic malicious verdict is not
   necessarily phishing. Submission results are shared, so do not submit private
   or token-bearing URLs. No candidate has been submitted by this work.
4. Keep uncertain cases unresolved or excluded with reasons. A second reviewer
   should examine conflicts. Retain all selected candidates and report acceptance
   coverage by source; do not silently replace difficult cases. A separately
   sourced, timestamped labeled dataset is an alternative study, but still needs
   provenance and historical overlap checks before evaluation.

Agree and record the evidence acceptance criteria before scoring. Any change to
the frozen protocol must be explicit rather than bypassing its safeguards.
Prefer local feed matching to sending candidate URLs to third parties; no live
suspected-site visits are needed for that matching step. No labels, cohort,
model, cutoff or extension were changed during this methods review, and no
predictions or new performance metrics were generated.

## Files and their purpose

Only one new script and one test file implement this workflow:

- `scripts/evaluate_frozen_url.py`: prepare an unscored review queue; later verify
  all frozen inputs and score the completed review offline.
- `tests/test_frozen_url.py`: overlap, label-review, input-integrity, no-network
  and uncertainty safeguards.

Private generated files are in
`data/processed/frozen_url_benchmark_20261019_v2/`, which Git ignores:

| File | Purpose |
|---|---|
| `protocol.json` | Model/metadata/code checksums, cutoff, feature order, versions and selection rules frozen before acquisition |
| `preparation.json` | Download timestamps, source checksums, pinned Tranco ID and candidate counts; no scores |
| `history_inventory.json` | Every scanned local CSV/Parquet, checksum and URL-column coverage |
| `excluded_domains.txt` | Domains found in those historical files, including earlier test sets |
| `sources/` | Unmodified downloaded feed/ranking snapshots for reproducibility |
| `candidates.parquet` | Immutable, unscored candidate identities |
| `review.csv` | The editable label-review worksheet |
| `results.json`, `predictions.parquet` | Created only after review passes; do not exist yet |

Historical exclusion covers locally available URL-bearing CSV/Parquet records,
not inaccessible Windows files or unextracted archives. Source archives already
have extracted CSV copies locally. Missing schemes in historical data are used
only for conservative domain exclusion, never as invented inference inputs.
Grouping uses the same offline private-PSL domain rules as development. Distinct
tenants or domains can still belong to the same phishing campaign.

## Sources and limitations

1. [OpenPhish Community Feed](https://openphish.com/phishing_feeds.html): candidate
   phishing URLs with provider-supplied labels. We select at most one URL per
   unseen registered domain using a seeded hash, not model scores. The snapshot
   has a retrieval timestamp, not per-URL first-seen evidence. Its labels are not
   automatically accepted as independent ground truth. Keep the feed and derived
   URL-level data private; consult the
   [source terms](https://openphish.com/terms.html) before sharing anything.
2. [Tranco](https://tranco-list.eu/): **unlabeled domain candidates**, selected
   across ranks 1–1,000, 1,001–10,000 and 10,001–100,000 from a pinned list ID.
   Popularity is not proof of legitimacy. We deliberately leave their URL fields
   blank: neither the scheme nor a homepage URL should be invented. The reviewer
   supplies a verified URL on the selected domain and independent evidence.

The target is 100 candidates from each source. Those choices make the pilot
reviewable, not representative of ordinary browsing. Feed-vs-ranking, URL path,
HTTPS and hosting differences can all create shortcuts. Report source-by-label
counts, HTTP/HTTPS and path composition when interpreting eventual results; do
not call this a deployment accuracy estimate.

The newer [PhiShark dataset](https://phishark2026dataset.com/) is another potential
source, but its metadata is included with restricted academic access requiring
approval and a signed agreement. No application has been submitted. Its website
and paper describe different release sizes/date windows, so an eventual granted
release must be pinned and audited rather than assumed interchangeable.

## Review `review.csv` before any predictions

Open the CSV **as text in VS Code**. Treat suspect URLs as inert text: do not
click them, load them in a browser, submit credentials or bypass security warnings.
Be careful opening untrusted text as a spreadsheet because cells may be
interpreted as formulas or clickable links.

Keep `record_id`, `domain`, `source`, `source_label` and `rank` unchanged. Keep
OpenPhish URLs unchanged. For a Tranco candidate, fill `url` only after verifying
the exact HTTP(S) address on its selected domain. Use a documented page-selection
rule before looking at predictions; do not choose only easy homepages. A service
domain without a reviewable public website can be rejected with a reason, not
silently replaced.

Complete these six columns for **every** candidate:

- `status`: `accepted`, `rejected` or `uncertain`.
- `review_label`: `0` legitimate or `1` phishing for accepted records; blank for
  unresolved/excluded records. Never set labels from our model's prediction.
- `reviewer`: who made the decision.
- `reviewed_at`: ISO timestamp with timezone, e.g. `2026-10-19T21:00:00+02:00`.
- `evidence`: the independent evidence/reference and its date. For legitimate
  URLs, an official organization/directory reference or authenticated ownership
  evidence is useful; popularity or absence from a blocklist alone is not proof.
  For suspected phishing, use independent reputable evidence or safely captured
  provenance; the same upstream feed mirrored elsewhere is not independent.
- `notes`: scope, uncertainty, inaccessible/stale content or exclusion reason.

A nonblank evidence field is a procedural gate, **not automated validation of
its truth**. Use `uncertain` when the evidence is insufficient; do not manufacture
certainty to reach a sample target. Count all exclusions. Source label conflicts
need explicit explanation, not automatic relabeling. A second reviewer is useful
for disputed cases. No source needs to be contacted and no suspicious site needs
to be visited to complete an uncertainty decision.

The script requires all decisions and at least 30 accepted domains per class.
That is a minimum pilot size, not enough to establish a very low false-positive
rate. Even 0 errors in 100 legitimate examples has a two-sided Wilson 95% upper
bound around 3.7%. Larger, more diverse cohorts will still be needed.

## Commands

The initial preparation has already been run. An incomplete first attempt in
`frozen_url_benchmark_20261019/` stopped at a download-size guard before candidate
selection; it is preserved, not a scored cohort. The completed pilot uses `_v2`
and handles Tranco's plain CSV download with a 50 MB bound.
For a **new** future cohort:

```bash
.venv/bin/python scripts/evaluate_frozen_url.py prepare \
  --directory data/processed/NEW_frozen_url_pilot
```

After review is complete, evaluate the existing pilot once:

```bash
.venv/bin/python scripts/evaluate_frozen_url.py score \
  --directory data/processed/frozen_url_benchmark_20261019_v2
```

Scoring verifies model, metadata, inference/evaluation code, dependency versions,
frozen candidates and the exclusion set. It uses the saved tree range and cutoff,
extracts all 21 features, and does not make network requests. Results include
confusion counts, recall, FPR, precision, F1, source composition and conditional
domain-bootstrap intervals. Wilson recall/FPR intervals remain informative when
the bootstrap degenerates at zero errors. Precision depends on this artificial
sample prevalence, not the real browsing phishing rate.

Do not edit the locked inference/evaluation code or dependencies before scoring;
such changes require a newly documented protocol/cohort. Checksums establish a
local provenance chain, not cryptographic notarization against a malicious editor.
After scoring, retain the result. Investigate errors separately, and use any
lessons for development with a **different** future holdout—not cutoff tuning on
this one. Nothing automatically activates a model or modifies Chrome.
