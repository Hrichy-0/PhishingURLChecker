"""Drive the real extension popup on listed pages and record its URL+HTML result.

Runs in the separate `.venv-rpa` (Playwright + httpx). Real sites are visited
read-only; simulated cases use reserved hosts served from local fixtures, so no
request for them leaves the machine. Never put phishing URLs in the real list.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import math
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_DIR = PROJECT_ROOT / "extension"
RESERVED_SUFFIXES = (".example", ".test", ".invalid")
RESERVED_NETWORK = ipaddress.ip_network("192.0.2.0/24")
DEBUG_PORT = 9333


def is_reserved_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host) in RESERVED_NETWORK
    except ValueError:
        return host.endswith(RESERVED_SUFFIXES)


def load_cases(path: Path, simulated: bool) -> list[dict]:
    """Validate a fixed list before any page is visited."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    seen, cases = set(), []
    for number, row in enumerate(rows, start=2):
        url = (row.get("url") or "").strip()
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"{path}:{number}: url must be absolute http(s)")
        if url in seen:
            raise ValueError(f"{path}:{number}: duplicate url")
        seen.add(url)
        if not (row.get("category") or "").strip():
            raise ValueError(f"{path}:{number}: category is required")
        if simulated != is_reserved_host(parts.hostname):
            kind = "simulated cases need" if simulated else "real sites must not use"
            raise ValueError(f"{path}:{number}: {kind} a reserved host")
        case = {"url": url, "category": row["category"].strip(), "simulated": simulated}
        if simulated:
            fixture = PROJECT_ROOT / row["fixture"]
            if not fixture.is_file():
                raise ValueError(f"{path}:{number}: fixture not found")
            case["fixture"] = row["fixture"]
        cases.append(case)
    return cases


def extension_id(path: Path = EXTENSION_DIR) -> str:
    """Chrome derives an unpacked extension's ID from its absolute path."""
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:32]
    return "".join("abcdefghijklmnop"[int(char, 16)] for char in digest)


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    centre = rate + z * z / (2 * total)
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    denominator = 1 + z * z / total
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("popup")]
    real = [r for r in scored if not r["simulated"]]
    simulated = [r for r in scored if r["simulated"]]

    def flagged(rows: list[dict], key: str = "popup") -> dict:
        count = sum(r[key]["is_phishing"] for r in rows)
        return {
            "flagged": count,
            "total": len(rows),
            "rate": count / len(rows) if rows else None,
            "wilson_95": wilson(count, len(rows)),
        }

    by_category: dict[str, list[dict]] = {}
    for record in real:
        by_category.setdefault(record["category"], []).append(record)
    latencies = [r["popup_ms"] for r in scored]
    raw_deltas = [
        abs(r["raw_html"]["phishing_probability"] - r["popup"]["phishing_probability"])
        for r in real
        if r.get("raw_html")
    ]
    errors: dict[str, int] = {}
    for record in records:
        if record.get("error"):
            errors[record["error_kind"]] = errors.get(record["error_kind"], 0) + 1
    return {
        "cases": len(records),
        "scored": len(scored),
        "errors": errors,
        "popup_api_parity": sum(r["parity"] for r in scored),
        "real_sites_url_html_flagged": flagged(real),
        "real_sites_url_only_flagged": flagged(
            [r for r in real if r.get("url_only")], "url_only"
        ),
        "real_sites_model_disagreements": sum(
            r["url_only"]["is_phishing"] != r["popup"]["is_phishing"]
            for r in real
            if r.get("url_only")
        ),
        "real_sites_by_category": {
            name: flagged(rows) for name, rows in sorted(by_category.items())
        },
        "redirected_to_other_host": sum(
            urlsplit(r["final_url"]).hostname != urlsplit(r["url"]).hostname
            for r in real
        ),
        "raw_vs_rendered_abs_score_difference_median": statistics.median(raw_deltas)
        if raw_deltas
        else None,
        "popup_ms_median": statistics.median(latencies) if latencies else None,
        "popup_ms_p95": statistics.quantiles(latencies, n=20)[-1]
        if len(latencies) >= 2
        else None,
        "simulated_flagged_by_category": {
            name: flagged([r for r in simulated if r["category"] == name])
            for name in sorted({r["category"] for r in simulated})
        },
    }


def check_case(
    playwright,
    context,
    cdp,
    client,
    api: str,
    case: dict,
    shots: Path | None,
    index: int,
) -> dict:
    record = {**case, "final_url": None, "http_status": None}
    page = context.new_page()
    debug_browser = None
    try:
        if case["simulated"]:
            body = (PROJECT_ROOT / case["fixture"]).read_text(encoding="utf-8")
            host = urlsplit(case["url"]).hostname

            def route(route, request, body=body, host=host):
                # Only the reserved host is answered locally; nothing else may load.
                if urlsplit(request.url).hostname == host:
                    route.fulfill(body=body, content_type="text/html")
                else:
                    route.abort()

            page.route("**/*", route)
        response = page.goto(case["url"], wait_until="load", timeout=30_000)
        page.wait_for_timeout(1_000)
        record["final_url"] = page.url
        record["http_status"] = response.status if response else None
        raw_html = None
        if response and not case["simulated"]:
            try:
                raw_html = response.text()
            except Exception:  # Body can be unavailable after redirects; optional.
                record["raw_html_error"] = "response body unavailable"

        tabs = cdp.send("Target.getTargets", {"filter": [{"type": "tab"}]})[
            "targetInfos"
        ]
        tab = next(t for t in tabs if t["url"] == page.url)
        cdp.send(
            "Extensions.triggerAction",
            {"id": extension_id(), "targetId": tab["targetId"]},
        )
        # The action popup is only visible to a CDP client attached after it opens.
        popup = None
        for _ in range(50):
            time.sleep(0.1)
            if debug_browser:
                debug_browser.close()
            debug_browser = playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{DEBUG_PORT}"
            )
            popup = next(
                (
                    p
                    for c in debug_browser.contexts
                    for p in c.pages
                    if p.url.endswith("/popup.html")
                ),
                None,
            )
            if popup:
                break
        if popup is None:
            raise RuntimeError("popup did not open")
        popup.wait_for_selector("#check-content:not([disabled])", timeout=5_000)
        start = time.perf_counter()
        with popup.expect_response("**/predict_html", timeout=30_000) as response_info:
            popup.click("#check-content")
        popup.wait_for_selector(
            "#result:not(.hidden), #error:not(.hidden)", timeout=30_000
        )
        record["popup_ms"] = (time.perf_counter() - start) * 1000
        if shots:
            popup.screenshot(path=str(shots / f"{index:03d}.png"))
        api_result = response_info.value.json()
        if not response_info.value.ok:
            raise RuntimeError(f"/predict_html {response_info.value.status}")
        shown = popup.locator("#probability").inner_text()
        badge = popup.locator("#risk-badge").inner_text()
        expected = (
            f"Model risk score: {api_result['phishing_probability'] * 100:.1f}/100"
        )
        record["popup"] = {
            key: api_result[key]
            for key in (
                "phishing_probability",
                "is_phishing",
                "risk_level",
                "threshold",
                "model_sha256",
                "html_truncated",
            )
        }
        record["popup_display"] = {
            "score": shown,
            "badge": badge,
            "model": popup.locator("#model-info").inner_text(),
        }
        record["parity"] = (
            shown == expected and badge.lower() == f"{api_result['risk_level']} risk"
        )
        popup.close()

        url_only = client.post(f"{api}/predict", json={"url": page.url})
        if url_only.is_success:
            record["url_only"] = {
                k: url_only.json()[k] for k in ("phishing_probability", "is_phishing")
            }
        if raw_html is not None:
            raw = client.post(
                f"{api}/predict_html", json={"url": page.url, "html": raw_html}
            )
            if raw.is_success:
                record["raw_html"] = {
                    k: raw.json()[k] for k in ("phishing_probability", "is_phishing")
                }
    except (
        Exception
    ) as error:  # Record and continue; one bad site must not stop the run.
        record["error"] = str(error).splitlines()[0][:300]
        record["error_kind"] = type(error).__name__
    finally:
        if debug_browser:
            debug_browser.close()
        page.close()
    return record


def run(
    cases: list[dict], output: Path, api: str, headed: bool, screenshots: bool = False
) -> dict:
    import httpx
    from playwright.sync_api import sync_playwright

    client = httpx.Client(timeout=30)
    health = client.get(f"{api}/health").json()
    if not health.get("content_model_available"):
        raise SystemExit(
            "The API's URL+HTML model is unavailable; start the API first."
        )
    output.mkdir(parents=True)
    shots = output / "screenshots" if screenshots else None
    if shots:
        shots.mkdir()
    records = []
    with sync_playwright() as playwright, tempfile.TemporaryDirectory() as profile:
        context = playwright.chromium.launch_persistent_context(
            profile,
            channel="chromium",
            headless=not headed,
            args=[
                f"--remote-debugging-port={DEBUG_PORT}",
                "--enable-unsafe-extension-debugging",
                f"--disable-extensions-except={EXTENSION_DIR}",
                f"--load-extension={EXTENSION_DIR}",
            ],
        )
        cdp = context.browser.new_browser_cdp_session()
        for index, case in enumerate(cases):
            record = check_case(
                playwright, context, cdp, client, api, case, shots, index
            )
            records.append(record)
            status = record.get("error") or record["popup"]["risk_level"]
            print(
                f"[{index + 1}/{len(cases)}] {case['category']}: {status}", flush=True
            )
        context.close()
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Real extension popup, URL+HTML content mode, fresh Chromium profile",
        "method": (
            "Playwright; popup opened with CDP Extensions.triggerAction "
            "(grants activeTab like a click)"
        ),
        "api_health": health,
        "summary": summarize(records),
        "records": records,
        "limitations": [
            "Rendered DOM after load, not the crawler snapshots used in training",
            "Simulated cases test the UI path, not real-world phishing accuracy",
            "One machine, one browser build, one run",
        ],
    }
    (output / "results.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, help="Real legitimate pages CSV")
    parser.add_argument(
        "--simulated", type=Path, help="Reserved-host fixture cases CSV"
    )
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, help="Check only the first N of each list")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--screenshots", action="store_true", help="Save a PNG of each popup"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; use a new run directory")
    cases = []
    if args.sites:
        cases += load_cases(args.sites, simulated=False)[: args.limit]
    if args.simulated:
        cases += load_cases(args.simulated, simulated=True)[: args.limit]
    if not cases:
        raise SystemExit("Pass --sites and/or --simulated")
    report = run(cases, args.output, args.api, args.headed, args.screenshots)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
