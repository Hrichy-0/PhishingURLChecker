# PhishGuard API

Start the service from the repository root after training the model:

```bash
uvicorn api.main:app --reload
```

Interactive documentation is available at <http://127.0.0.1:8000/docs>.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/login"}'
```
