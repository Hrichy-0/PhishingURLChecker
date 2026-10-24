"""RPA list validation and summaries; no browser or network is used."""

import pytest

from scripts.rpa_extension_check import (
    extension_id,
    is_reserved_host,
    load_cases,
    summarize,
    wilson,
)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_reserved_hosts_cover_test_domains_and_documentation_ips():
    assert is_reserved_host("login.example")
    assert is_reserved_host("192.0.2.10")
    assert not is_reserved_host("www.python.org")
    assert not is_reserved_host("8.8.8.8")


def test_real_list_rejects_duplicates_bad_schemes_and_reserved_hosts(tmp_path):
    ok = write(tmp_path / "ok.csv", "url,category\nhttps://www.python.org/,docs\n")
    assert load_cases(ok, simulated=False)[0]["simulated"] is False
    for body in (
        "url,category\nftp://x.org/,docs\n",
        "url,category\nhttps://a.org/,docs\nhttps://a.org/,docs\n",
        "url,category\nhttps://a.example/,docs\n",
        "url,category\nhttps://a.org/,\n",
    ):
        with pytest.raises(ValueError):
            load_cases(write(tmp_path / "bad.csv", body), simulated=False)


def test_simulated_cases_require_reserved_hosts_and_existing_fixtures(tmp_path):
    fixture = "examples/hybrid/login_fixture.html"
    good = write(
        tmp_path / "s.csv", f"url,category,fixture\nhttps://a.test/,x,{fixture}\n"
    )
    assert load_cases(good, simulated=True)[0]["fixture"] == fixture
    real = write(
        tmp_path / "r.csv", f"url,category,fixture\nhttps://a.org/,x,{fixture}\n"
    )
    with pytest.raises(ValueError, match="reserved"):
        load_cases(real, simulated=True)
    missing = write(
        tmp_path / "m.csv", "url,category,fixture\nhttps://a.test/,x,nope.html\n"
    )
    with pytest.raises(ValueError, match="fixture"):
        load_cases(missing, simulated=True)


def test_shipped_simulated_list_is_valid():
    from scripts.rpa_extension_check import PROJECT_ROOT

    cases = load_cases(PROJECT_ROOT / "examples/rpa/simulated.csv", simulated=True)
    assert len(cases) == 20


def test_extension_id_is_32_letters_a_to_p():
    value = extension_id()
    assert len(value) == 32 and set(value) <= set("abcdefghijklmnop")


def test_wilson_interval_bounds():
    assert wilson(0, 0) is None
    low, high = wilson(0, 10)
    assert low == pytest.approx(0, abs=1e-12) and 0.27 < high < 0.29
    low, high = wilson(5, 10)
    assert low < 0.5 < high


def test_summary_counts_flags_parity_disagreement_and_errors():
    def record(category, flagged, url_only, simulated=False):
        return {
            "url": "https://a.org/",
            "final_url": "https://a.org/",
            "category": category,
            "simulated": simulated,
            "popup_ms": 50.0,
            "parity": True,
            "popup": {"is_phishing": flagged, "phishing_probability": 0.4},
            "url_only": {"is_phishing": url_only, "phishing_probability": 0.1},
            "raw_html": {"is_phishing": flagged, "phishing_probability": 0.5},
        }

    records = [
        record("docs", True, False),
        record("docs", False, False),
        record("fixture", True, True, simulated=True),
        {
            "url": "https://b.org/",
            "category": "docs",
            "simulated": False,
            "error": "timeout",
            "error_kind": "TimeoutError",
        },
    ]
    summary = summarize(records)
    assert summary["scored"] == 3
    assert summary["errors"] == {"TimeoutError": 1}
    assert summary["popup_api_parity"] == 3
    assert summary["real_sites_url_html_flagged"]["flagged"] == 1
    assert summary["real_sites_url_html_flagged"]["total"] == 2
    assert summary["real_sites_model_disagreements"] == 1
    assert summary["raw_vs_rendered_abs_score_difference_median"] == pytest.approx(0.1)
    assert summary["simulated_flagged_by_category"]["fixture"]["flagged"] == 1
