"""Leakage-safe grouping: link domains that share exact HTML, and prove splits disjoint.

``native.domain_partition`` (frozen) and ``native_html.content_partition`` (part of
the HTML feature-cache key) keep their own copies; new code should use this module.
These controls do not catch near-duplicate templates across distinct domains.
"""

from __future__ import annotations

import pandas as pd


def connected_groups(frame: pd.DataFrame) -> list[int]:
    """Transitive domain/exact-content groups; content hashes are NOT features."""
    if frame[["domain", "html_sha256"]].isna().any().any():
        raise ValueError("Missing grouping metadata")
    parents = {domain: domain for domain in frame.domain.unique()}

    def root(domain):
        while parents[domain] != domain:
            parents[domain] = parents[parents[domain]]
            domain = parents[domain]
        return domain

    owners = {}
    for domain, digest in frame[["domain", "html_sha256"]].itertuples(index=False):
        if not domain:
            raise ValueError("Empty domain")
        if digest:  # Empty hashes must not join all unavailable pages.
            previous = owners.setdefault(digest, domain)
            first, second = sorted((root(domain), root(previous)))
            parents[second] = first
    roots = frame.domain.map({domain: root(domain) for domain in parents})
    return pd.factorize(roots, sort=True)[0]


def assert_disjoint(
    frames: dict[str, pd.DataFrame], columns=("domain", "html_sha256"), context=""
) -> None:
    """Raise if any two named frames share a non-empty value in any column."""
    names = list(frames)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            for column in columns:
                shared = (set(frames[name][column]) - {""}) & (
                    set(frames[other][column]) - {""}
                )
                if shared:
                    raise ValueError(
                        f"{context}{column} leakage between {name} and {other}"
                    )
