"""Data-only import checks; tests never contact Hugging Face or dataset URLs."""

import pytest

from scripts.download_hf_urls import SOURCES, extract_url_columns


def test_csv_import_preserves_urls_and_labels_and_drops_other_fields():
    payload = b"url_raw,phish_label,email_content\nhttps://a.test/login,1,ignore\nhttp://b.test/,0,ignore\n"
    frame = extract_url_columns(payload, "core_emails.csv", "url_raw", "phish_label")
    assert frame.columns.tolist() == ["url", "label"]
    assert frame["url"].tolist() == ["https://a.test/login", "http://b.test/"]
    assert frame["label"].tolist() == [1, 0]


def test_unsupported_download_format_is_rejected():
    with pytest.raises(ValueError, match="Only CSV and JSON"):
        extract_url_columns(b"", "loader.py", "url", "label")


def test_hf_sources_are_pinned_to_immutable_revisions():
    for spec in SOURCES.values():
        assert len(spec["revision"]) == 40
        assert all(character in "0123456789abcdef" for character in spec["revision"])
        assert all(name.endswith((".csv", ".json")) for name in spec["files"])


def test_json_url_import_excludes_non_url_fields():
    payload = b'[{"text":"https://a.test/","label":1,"extra":"ignored"}]'
    frame = extract_url_columns(payload, "urls.json", "text", "label")
    assert frame.to_dict("records") == [{"url": "https://a.test/", "label": 1}]


def test_json_url_import_rejects_an_unexpected_container():
    with pytest.raises(ValueError, match="JSON array"):
        extract_url_columns(b"{}", "urls.json", "text", "label")
