"""Pure offline HTML aggregates: no network, rendering, or script execution."""

from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urljoin

from phishguard.data import is_valid_http_url
from phishguard.domains import get_registered_domain

MAX_HTML_BYTES = 1_000_000
CONTENT_FEATURE_NAMES = (
    "html_length",
    "form_count",
    "password_field_count",
    "hidden_field_count",
    "external_form_ratio",
    "unsupported_form_ratio",
    "link_count",
    "external_link_ratio",
    "iframe_count",
    "script_count",
    "external_script_ratio",
    "image_count",
    "title_length",
    "title_brand_domain_mismatch_count",
)
# This is a small explanatory reference list, not a trusted-domain allowlist.
# It cannot cover all brands, subsidiaries, resellers, SSO, or localization.
BRAND_DOMAINS = {
    "apple": {"apple.com", "icloud.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "microsoftonline.com"},
    "google": {"google.com"},
    "paypal": {"paypal.com"},
    "amazon": {"amazon.com", "amazon.de", "amazon.co.uk"},
    "facebook": {"facebook.com", "meta.com"},
}


class _SnapshotParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.counts: Counter[str] = Counter()
        self.form_actions: list[str] = []
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.base_href: str | None = None
        self.title_parts: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.counts[tag] += 1
        values = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "input":
            kind = (values.get("type") or "").lower()
            if kind in {"password", "hidden"}:
                self.counts[f"{kind}_field"] += 1
            # Never retain input values, including password/hidden values.
        elif tag == "form":
            self.form_actions.append(values.get("action") or "")
        elif tag == "a" and values.get("href"):
            self.links.append(values["href"] or "")
        elif tag == "script" and values.get("src"):
            self.scripts.append(values["src"] or "")
        elif tag == "base" and self.base_href is None and values.get("href"):
            self.base_href = values["href"]

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


def _external_ratio(destinations: list[str], base: str, domain: str) -> float:
    external = 0
    valid = 0
    for value in destinations:
        try:
            destination = urljoin(base, value)
            if is_valid_http_url(destination):
                valid += 1
                external += get_registered_domain(destination) != domain
        except ValueError:
            continue
    return external / max(1, valid)


def extract_content_features(url: str, html: str) -> dict[str, float]:
    """Parse a bounded, untrusted snapshot as text; return only aggregate values.

    HTMLParser does not execute JavaScript or fetch images, scripts, or links.
    These are signals, NOT rules establishing that a page is phishing.
    """
    if not is_valid_http_url(url):
        raise ValueError("Content features require an absolute HTTP(S) URL")
    if not isinstance(html, str) or not html.strip():
        raise ValueError("A nonempty HTML snapshot is required")
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        raise ValueError(f"HTML exceeds the {MAX_HTML_BYTES:,}-byte snapshot limit")
    parser = _SnapshotParser()
    parser.feed(html)
    parser.close()
    domain = get_registered_domain(url)
    base = url
    if parser.base_href:
        try:
            candidate = urljoin(url, parser.base_href)
            if is_valid_http_url(candidate):
                base = candidate
        except ValueError:
            pass
    unsupported = 0
    # Empty form action submits to the document URL, not an external <base> URL.
    forms = []
    for action in parser.form_actions:
        try:
            destination = urljoin(base, action) if action else url
            forms.append(destination)
            unsupported += not is_valid_http_url(destination)
        except ValueError:
            unsupported += 1
    title = " ".join(parser.title_parts).strip()
    title_words = set(re.findall(r"[a-z]+", title.lower()))
    mismatch = sum(
        brand in title_words and domain not in allowed
        for brand, allowed in BRAND_DOMAINS.items()
    )
    values = {
        "html_length": len(html),
        "form_count": parser.counts["form"],
        "password_field_count": parser.counts["password_field"],
        "hidden_field_count": parser.counts["hidden_field"],
        "external_form_ratio": _external_ratio(forms, url, domain),
        "unsupported_form_ratio": unsupported / max(1, len(parser.form_actions)),
        "link_count": parser.counts["a"],
        "external_link_ratio": _external_ratio(parser.links, base, domain),
        "iframe_count": parser.counts["iframe"],
        "script_count": parser.counts["script"],
        "external_script_ratio": _external_ratio(parser.scripts, base, domain),
        "image_count": parser.counts["img"],
        "title_length": len(title),
        "title_brand_domain_mismatch_count": mismatch,
    }
    return {name: float(values[name]) for name in CONTENT_FEATURE_NAMES}
