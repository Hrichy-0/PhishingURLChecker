# PhishGuard Chrome extension

1. Start the API with `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
   using the project's virtual-environment Python.
2. Open `chrome://extensions`.
3. Enable **Developer mode**.
4. Choose **Load unpacked** and select this `extension/` directory.
5. Open an HTTP or HTTPS page and click the PhishGuard toolbar action.

**Check this URL** sends only the active tab URL to `127.0.0.1`.
**Check URL + page content (experimental)** also sends the page's current HTML
(`document.documentElement.outerHTML`, read with the `scripting` permission on the
clicked tab only) to `127.0.0.1/predict_html`. Neither button makes the API visit
the site.

The main local model is the native URL-only model. After reloading this unpacked
extension, its model information should show `Native XGBoost: url_only`,
`21 features`, checksum prefix `9d9ae6a4808b` and cutoff `0.4902`.
The experimental/advisory warning remains; main local selection is not production
approval. The content button uses the research-only URL+HTML model (`url_html`,
38 features, prefix `41f3e3c9a25b`, cutoff `0.2209`); see
`docs/archive/rpa-real-world-testing.md`. A server already running with
the old configuration must be restarted before the popup uses the new default.
