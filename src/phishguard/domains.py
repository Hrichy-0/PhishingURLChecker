from urllib.parse import urlsplit

import tldextract

domain_extractor = tldextract.TLDExtract(
    cache_dir=None,
    suffix_list_urls=(),
    include_psl_private_domains=True,
)


def get_registered_domain(url: str) -> str:
    """Return the registered domain used to group related URLs."""

    parsed_url = urlsplit(url)
    hostname = parsed_url.hostname

    if not parsed_url.scheme or not hostname:
        raise ValueError(f"Invalid absolute URL: {url!r}")

    extracted = domain_extractor(hostname)
    registered_domain = extracted.top_domain_under_public_suffix

    # IP addresses and local-style hostnames may not have a public suffix.
    if not registered_domain:
        registered_domain = hostname

    return registered_domain.lower()
