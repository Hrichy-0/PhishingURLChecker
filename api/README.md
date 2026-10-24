# PhishGuard API

Start the service from the repository root with the saved native URL-only model:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Interactive documentation is available at <http://127.0.0.1:8000/docs>.

The default is `artifacts/experiments/phreshphish_html_20261010/url_only/`.
`GET /health` should identify `Native XGBoost: url_only`, `feature_count: 21`,
and checksum prefix `9d9ae6a4808b`. The model remains experimental/advisory;
the saved cutoff is 0.4901971220970154, not a calibrated safety probability.
Use the project's virtual-environment Python. Restart the API after changing
configuration. `PHISHGUARD_MODEL=legacy` explicitly selects the old joblib pair;
no research override may be set at the same time. A missing/bad model reports
degraded instead of silently falling back.

`POST /predict_html` (`{"url": ..., "html": ...}`) is a separate experimental
endpoint for the saved URL+HTML model (`url_html/`, 38 features, checksum prefix
`41f3e3c9a25b`, cutoff 0.22090303897857666). It scores HTML the browser already
loaded, never fetches or stores pages, parses at most 1 MB and reports
`html_truncated`. Its model failing returns 503 there only; `/predict` is
unaffected. See `docs/archive/rpa-real-world-testing.md`.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/login"}'
```
