# Local native XGBoost training

## Current full-download run

The user requested that all original dataset files, including HTML, be downloaded,
while retaining URL-only training. The initial column-only training run was
stopped before it saved a model. The replacement local job is:

```powershell
.\scripts\run_full_dataset_training.ps1
```

It downloads all 77 original Parquet files (36.58 GB) into
`data/raw/phreshphish/data/`, preserves publisher partitions, verifies each file
against its upstream SHA-256, and supports resuming partial downloads. Only after
every file is verified does it export URL/label/date columns from the local full
files, clean the training portion and run the native CPU training. HTML remains
on disk and is not a training feature, as requested.

Full-run results go to
`artifacts/experiments/phreshphish_full_download_20261009/`. Its progress logs are
`data/raw/phreshphish/full-download.log` and
`data/raw/phreshphish/training-after-download.log`.
`data/raw/phreshphish/download_status.json` records transfer progress.

The shared lexical/data/native checks pass (28 tests). Ruff lint and formatting
also pass. Native model loading uses XGBoost's supported in-memory UBJ interface
with normal Python file I/O. The full scikit-learn API suite remains blocked by
the separately documented native-extension policy.

This Windows environment can load and train XGBoost through its native CPU API.
Its scikit-learn `_openmp_helpers` extension is blocked by Windows Application
Control. The native workflow uses supported XGBoost interfaces and does not alter
Windows security policy, unblock libraries, or request security exclusions.

## Data

The PhreshPhish URL, label and date columns were extracted from pinned revision
`eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`. The full website HTML is not used.
The [publisher dataset](https://huggingface.co/datasets/phreshphish/phreshphish)
has 498,255 training rows and 168,060 test rows. Its CC BY 4.0 terms restrict use
to anti-phishing research. Attribution: Dalton et al., *PhreshPhish: A Real-World,
High-Quality, Large-Scale Phishing Website Dataset and Benchmark* (2025),
https://arxiv.org/abs/2507.10854.

Local source files:

- `data/external/huggingface/phreshphish/train.csv`
- `data/external/huggingface/phreshphish/test.csv`
- `data/external/huggingface/phreshphish/download_manifest.json`

Cleaning retains 494,051 HTTP(S) training URLs: 272,531 benign and 221,520 phishing.
It rejects 4,204 invalid/incomplete records, including 4,189 benign records without
a protocol. Protocols are preserved, not guessed; this exclusion changes the
population represented by the experiment.

The new training pool combines PhiUSIIL, the previous balanced source, and the
PhreshPhish publisher training portion. After 168 cross-source duplicates are
removed, 1,068,609 unique URLs remain. No contradictory cross-source URL labels
were found. Historical mixed-source test rows are included in this new pool, so
the newly partitioned mixed holdout is not directly comparable to the old one.
The PhreshPhish publisher test portion is kept separate throughout selection.

## Execution

The project `.venv/Scripts/python.exe` executes `scripts/train_native_model.py`.
All feature extraction, fitting and evaluation run locally. No OpenAI API,
hosted model endpoint, GPU rental or cloud training job is used.

From PowerShell in the project root:

```powershell
.\scripts\run_local_training.ps1
```

The initial run uses four CPU threads and evaluates three configurations, with
depths 3, 5 and 7 and up to 800 rounds each. Validation logloss controls early
stopping; validation F1 selects the winner and threshold, subject to 90% recall.
The pure NumPy threshold calculation favors the higher threshold if F1 is tied.

URL features are shared with the original serving code through
`src/phishguard/lexical.py`. Their definitions were moved unchanged; the original
`URLFeatureExtractor` class remains in `phishguard.features` for compatibility.
Both pipelines use the same 21 features and omit slash count. No feature was
silently normalized or redefined.

Registered domains, including private suffixes, are assigned to deterministic
SHA-256 buckets for approximately 80% training, 10% validation and 10% internal
test. Domain isolation and two-class coverage are checked before fitting. This
partition method differs from the previous GroupShuffleSplit experiment.

The frozen model is evaluated on the publisher test after removing every domain
found anywhere in the mixed pool. The previous native XGBoost model is also
evaluated on the same retained rows using its original threshold. Neither result
selects the new model or threshold. Domain exclusion changes the test population;
it does not prove independent collection or eliminate shared-campaign leakage.

## Outputs and inspection

Prepared splits and feature arrays are under `data/processed/phreshphish_mixed_v1/`.
Experiment outputs are under `artifacts/experiments/phreshphish_mixed_20261009/`:

- `training.log`: live local progress and results.
- `xgboost_model.ubj`: selected native booster, containing only the chosen trees.
- `model_metadata.json`: parameters, validation/internal-test metrics, provenance
  and normalized gain feature importances.
- `phreshphish_external_evaluation.json`: filtered publisher-test results and
  the previous native model's same-row comparison.
- `feature-importance.md`: feature percentage report.
- `run_status.json`: successful completion marker and elapsed time.
- `requirements-local.txt`: installed package versions.

To watch the log in a separate PowerShell terminal:

```powershell
Get-Content .\artifacts\experiments\phreshphish_mixed_20261009\training.log -Tail 20 -Wait
```

To start another experiment after the first one has completed:

```powershell
.\scripts\run_local_training.ps1 -ExperimentName phreshphish_mixed_run2
```

Do not run simultaneous experiments against the same prepared-splits directory.

The active API model pair is not overwritten. Native UBJ is a separate artifact
format; the current API expects a scikit-learn joblib pipeline. Activating this
native model requires a deliberate serving integration and its own verification.
Training success does not resolve the scikit-learn DLL block or verify the full
existing API/extension test suite.
