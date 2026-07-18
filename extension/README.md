# PhishGuard Chrome extension

1. Start the API with `uvicorn api.main:app --reload`.
2. Open `chrome://extensions`.
3. Enable **Developer mode**.
4. Choose **Load unpacked** and select this `extension/` directory.
5. Open an HTTP or HTTPS page and click the PhishGuard toolbar action.

The extension sends only the active tab URL to `127.0.0.1`. It does not visit,
scrape, or execute content from the submitted site.
