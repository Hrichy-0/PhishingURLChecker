# RPA real-website testing with the URL+HTML model

Added 23 October 2026. The professor's next step is real-life testing: visit a
fixed list of real websites with UI automation and read what the Chrome extension
shows. These checks use the **experimental URL+HTML model** on the page the
browser has loaded. The URL-only `/predict` path, its model and its cutoff are
unchanged.

## What runs

| Part | Behaviour |
|---|---|
| API `POST /predict_html` | Takes `{url, html}`. Scores it with the saved model in `artifacts/experiments/phreshphish_html_20261010/url_html/` (38 features, SHA-256 `41f3e3c9a25b…`, cutoff `0.22090303897857666`, `best_iteration` 63). Checks the checksum, feature order and tree range when it loads. Parses at most the first 1 MB, the same limit as training, and flags truncation. Rejects request bodies over 5,000,000 characters. Never fetches the page and never stores the HTML. A missing model returns 503 from this endpoint only. `/health` reports `content_model_available`. |
| Extension | A second button, "Check URL + page content (experimental)". It reads `document.documentElement.outerHTML` from the active tab via `chrome.scripting` and sends it to `/predict_html` on `127.0.0.1`. The only new permission is `scripting`; `activeTab` limits access to the tab the user clicked on. |
| Runner `scripts/rpa_extension_check.py` | Uses Playwright and Chromium with a fresh temporary profile and `extension/` loaded unpacked. Visits each page, opens the **real action popup** with CDP `Extensions.triggerAction` and clicks the content button. Saves the popup's result (and, with `--screenshots`, a PNG of the popup), the API response for the same request (to check the display matches) and the URL-only score. It also scores the raw HTTP response body as a check on capture mode. |

Opening the popup with `chrome.action.openPopup()` was tried and rejected. It does
not grant `activeTab`, so the popup cannot see the tab. Simulated key presses
don't trigger extension shortcuts. A real OS-level keystroke needs macOS
Accessibility permission. `Extensions.triggerAction` (which needs the
`--enable-unsafe-extension-debugging` flag) acts like a user click.

## Safety

- The real list (`examples/rpa/sites.csv`) holds legitimate pages only. It is
  fixed before any scoring and contains no domains from development data. The
  loader rejects reserved test hosts in this list.
- Simulated phishing-like cases (`examples/rpa/simulated.csv`, 20 rows) use only
  `.example`, `.test`, `.invalid` or `192.0.2.0/24` hosts. Their pages are served
  from inert local fixtures. Every other request from those pages is aborted, so
  nothing leaves the machine. These cases test the HIGH-risk UI path, not
  accuracy.
- The runner never submits forms, enters credentials or bypasses interstitials.
  Real phishing URLs are never browsed.

## Run

```bash
python3.12 -m venv .venv-rpa                      # separate; frozen .venv untouched
.venv-rpa/bin/python -m pip install playwright httpx
.venv-rpa/bin/python -m playwright install chromium
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
.venv-rpa/bin/python scripts/rpa_extension_check.py \
  --sites examples/rpa/sites.csv --simulated examples/rpa/simulated.csv \
  --output artifacts/experiments/rpa_url_html_YYYYMMDD
```

The output directory must be new. `results.json` holds each record and a summary
with:

- display/API match count
- error types
- URL+HTML and URL-only flag rates on real sites, with Wilson 95% intervals,
  overall and per category
- the number of sites where the two models disagree
- redirects to another host
- the median score difference between rendered DOM and raw HTML
- popup latency (median and p95)
- the simulated-case flag rate per category

Popup screenshots are saved next to it only when `--screenshots` is passed.

## Smoke check, 23 October 2026 (engineering check, not an evaluation)

Three real public pages (python.org, mozilla.org, a Wikipedia article) and three
simulated login cases were run once:

- All 6 popups opened, and every displayed result matched the API's answer.
- No errors.
- Median popup latency was 72 ms (HTML capture plus inference).
- The URL+HTML model flagged mozilla.org's homepage HIGH (46.5/100 against
  cutoff 22.1). The URL-only model did not flag it.
- The rendered-DOM and raw-HTML scores differed by a median of only 0.0007.
- All three simulated login cases showed HIGH.

All three real domains may also appear in development data. Three selected
pages give no false-positive estimate. Separately, a minimal `<p>Hello</p>` page
on `example.com` scored 0.90 offline. Thin or unusual pages may be flagged, and
the full list is needed to measure this.

A second run covered all 20 simulated cases. All 20 displayed results matched
the API, with no errors. All 15 phishing-like fixtures showed HIGH. **All 5
benign control fixtures (a plain newsletter article on reserved hosts) also
showed HIGH.** The simulated cases therefore don't show that the model separates
these fixtures. Together with the mozilla.org result, this is an early warning
that the saved URL+HTML model and its 0.2209 cutoff may flag many small or
unusual legitimate pages. The real-site list must measure this before anyone
draws a conclusion.

## Full run, 24–25 October 2026

Run at commit `6e31c16` with the frozen 197-site list and 20 simulated cases.
Output: `artifacts/experiments/rpa_url_html_20261024/results.json` (no
screenshots kept).

- **Coverage:** 195 of 217 cases scored. Of the 22 failures, 12 were
  timeouts, 2 were network errors and 1 was a page too large for the API.
  The other 7 failed because the runner could not read the raw response body;
  that bug is now fixed, but those 7 sites were not re-run.
- **Display check:** all 195 popups matched the API's answer exactly.
- **Latency:** median 76 ms from click to result (p95 144 ms).

**False alarms on 175 scored real legitimate sites**, with Wilson 95%
intervals:

| Pages | n | URL+HTML flagged | URL-only flagged |
|---|---:|---:|---:|
| All scored | 175 | 67 (38.3%, 31.4–45.7) | 46 (26.3%, 20.3–33.3) |
| Normal response (HTTP 200) | 152 | 45 (29.6%, 22.9–37.3) | 41 (27.0%, 20.6–34.5) |
| Blocked or error response (HTTP 403/429/500) | 23 | 22 (95.7%, 79.0–99.2) | 5 (21.7%, 9.7–41.9) |

- **Blocked pages:** 23 sites returned HTTP 403, 429 or 500 to the automated
  browser, so the page scored was probably a bot-check or error page, not the
  real site. The URL+HTML model flagged 22 of these 23. This matches the
  earlier finding that generic unavailable or blocked pages in the training
  data are often labelled phishing.
- **Normal pages:** on pages that loaded normally, the two models' false-alarm
  rates are similar, about 27–30%. Development cross-validation measured
  4–8%, so there is a large gap on new real sites.
- **Disagreements:** the models disagreed on 41 sites. URL+HTML alone flagged
  31; URL-only alone flagged 10.
- **By page type (URL+HTML):** homepages 48/124, login pages 8/16, long query
  strings 8/8, deep links 2/19, docs 1/8. All five scored email sites were
  flagged.
- **Raw HTML vs rendered page:** the decision changed on 18 of 175 pages when
  the raw HTML was scored instead of the page after JavaScript had run. 19
  pages hit the 1 MB parse limit.
- **Simulated cases:** all 20 were flagged HIGH, including the 5 harmless
  controls, so they show the HIGH path works but not that the model separates
  phishing from harmless pages.

These are legitimate sites only, so the run measures false alarms, not
phishing detection. Sites were chosen before scoring and the cutoffs were not
changed.

## Limitations

- The extension sends the DOM after JavaScript has run. Training used crawler
  snapshots, so the runner also records the raw HTML score to measure this
  difference.
- The reasons shown in the popup are still URL-only rules. They do not explain
  the HTML features.
- One machine, one Chromium build (Chrome for Testing 156) and one run per list.
  Real sites change over time.
- The saved URL+HTML model has `activation_eligible: false`. This is a research
  test, not promotion.
