# Local URL + HTML training

Adding code does not change either saved URL-only model. A new training run creates
three separate native XGBoost artifacts: `url_only`, `html_features`, and `url_html`.
Retraining can change feature importance, tree splits, thresholds and predictions.
No candidate is automatically loaded by the current API or browser extension.

## Features and interpretation

The main combined model uses all 21 current URL features plus 17 HTML-derived
features: 14 content aggregates and three completeness flags (38 total).

| Group | Signals |
| --- | --- |
| Credential collection | Form count, password fields, hidden fields |
| Submission behavior | Fraction of HTTP forms targeting another registered domain; unsupported form destinations |
| Page resources | Link count, external link ratio, script count, external script ratio, iframe count, image count |
| Page identity and size | HTML length, title length, title brand/domain mismatch count |
| Completeness | HTML available, HTML truncated, HTML parse error |

These are learned signals, not rules declaring that login forms or external links
are malicious. The small brand reference list cannot cover every brand, reseller,
SSO service or legitimate third-party login. Input values are not retained. The
extractor does not execute JavaScript, load resources or open links. Relational
HTML signals use the page URL as context; `html_features` is not a raw-text-only
model. HTML length is the length of the parsed prefix, not the full document.

Absent content stays `NaN` rather than becoming misleading zero counts. Native
XGBoost handles those missing values. A maximum 1,000,000 UTF-8 bytes per document
are parsed, with a separate truncation flag. Pages remain in the paired comparison
even if content is unavailable; flags are reported separately for both classes.
Features beyond the prefix, including late forms, can be missed.

## Data and fair comparisons

`scripts/train_native_html_model.py` reads the already downloaded full local
PhreshPhish files. It verifies original checksums, then parses batches of 16 rows
and saves resumable per-shard numeric feature caches. It does not load 36.6 GB of
HTML into memory or redownload any data. Changing the extractor or parse limit
invalidates caches. Source checksums and cache checksums are checked on reuse.

The publisher training portion alone forms the development pool. URL-only,
HTML-derived, and combined variants use identical rows and identical splits.
Domains connected by identical HTML are kept together, using reproducible 80/10/10
group buckets. The publisher test removes domains and exact HTML hashes occurring
anywhere in development. Contradictory URL labels are excluded and recorded.
Missing protocols are rejected rather than guessed.

All variants receive the same depth-3/5/7 search, at most 800 trees per candidate,
early stopping on validation log loss, and model selection by validation F1 under
a 90% validation recall constraint. A second threshold is selected using only
validation to target <=1% FPR. Test FPR is measured and may exceed that target.
No test result selects the model or either threshold. Numeric model-input columns
never include the publisher label, target-brand annotation, hash, date or language
metadata. Dates are retained for auditing.

Exact HTML grouping does not remove near-duplicate templates. Internal splits are
not temporal; dates are reported without claiming chronological separation. This
publisher test has already been inspected in earlier URL experiments, so final
research claims should use a fresh independent evaluation. HTML size, language,
collection artifacts and missingness can also act as dataset shortcuts.

## Run in a VS Code PowerShell terminal

Full HTML extraction and paired CUDA training completed on 2026-10-11. See
[`html-training-results-2026-10-11.md`](html-training-results-2026-10-11.md) for
measured results.
Existing Python, PyArrow and XGBoost dependencies are sufficient. CPU is default;
`--device cuda` uses the GTX 1660 Ti. A CUDA request that falls back to CPU is an
error. No security-policy or PowerShell execution-policy change is required.

Validation on 2026-10-10: 36 focused tests passed, including a small CPU comparison,
cache recovery and overlap exclusion. All three variants also passed a six-round
synthetic CUDA train/save/reload check on the GTX 1660 Ti. Feature extraction was
checked on 15 valid real training snapshots (two required truncation, none had a
parse error). These checks are software validation, not real-data hybrid accuracy
results. The moved strict HTML parser matches its original implementation.

```powershell
.\.venv\Scripts\python.exe -u scripts/train_native_html_model.py --dataset-dir data/raw/phreshphish --cache-dir data/processed/html_features_v1 --output-dir artifacts/experiments/phreshphish_html_20261011_rerun --trees 800 --threads 4 --device cuda
```

To extract and cache features without fitting models, add `--extract-only`.
Extraction and original-file verification use CPU and disk; GPU accelerates
XGBoost fitting. Parsing the HTML may take much longer than the numeric fitting.
Choose a new output directory for each experiment; completed results are not
overwritten. Feature caches can be reused across experiments.

Outputs include per-variant UBJ models and metadata, gain importance shares,
validation/internal/publisher-test predictions, exact split-row files, checksum
provenance, completeness/exclusion counts, and `comparison.json`. The active
joblib model and completed URL-only experiment are hash checked for preservation.
The current API still accepts only a URL. Integrating HTML at inference requires
the same extraction/order/limits and an explicit content input path.

`snapshot_vector` in `phishguard.native_html` supplies that shared inference
representation. Cached output and experiment rows include URLs and content hashes
but do not include raw HTML or form input values.

## TalentUD research question

Does adding explainable HTML structure improve unseen-domain phishing recall at
a fixed false-positive budget? Compare all three paired variants, with confidence
intervals and future temporal/near-template evaluation. Report both improvements
and regressions, including extraction latency, resource use and incomplete-content
behavior. The current mixed-source URL-only metrics are useful background but are
not a fair substitute for the new same-row baseline.
