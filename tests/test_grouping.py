"""Shared leakage controls: exact-HTML components and pairwise disjointness."""

import pandas as pd
import pytest

from phishguard.grouping import assert_disjoint, connected_groups


def test_shared_html_links_domains_but_empty_hashes_do_not():
    frame = pd.DataFrame(
        {
            "domain": ["a.test", "b.test", "c.test", "d.test", "e.test"],
            "html_sha256": ["h1", "h1", "h2", "", ""],
        }
    )
    groups = connected_groups(frame)
    assert groups[0] == groups[1]
    assert len({groups[1], groups[2], groups[3], groups[4]}) == 4


@pytest.mark.parametrize("bad", [None, ""])
def test_missing_or_empty_domain_is_rejected(bad):
    frame = pd.DataFrame({"domain": ["a.test", bad], "html_sha256": ["h", "h"]})
    with pytest.raises(ValueError):
        connected_groups(frame)


def test_disjointness_names_column_and_splits_and_ignores_empty_values():
    train = pd.DataFrame({"domain": ["a.test"], "html_sha256": [""]})
    test = pd.DataFrame({"domain": ["b.test"], "html_sha256": [""]})
    assert_disjoint({"train": train, "test": test})
    test.loc[0, "html_sha256"] = train.loc[0, "html_sha256"] = "same"
    with pytest.raises(ValueError, match="Fold 1: html_sha256 leakage between train"):
        assert_disjoint({"train": train, "test": test}, context="Fold 1: ")
