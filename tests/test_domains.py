import pytest

from phishguard.domains import get_registered_domain


@pytest.mark.parametrize(
    ("url", "expected_domain"),
    [
        (
            "https://login.example.com/reset",
            "example.com",
        ),
        (
            "https://forums.bbc.co.uk/news",
            "bbc.co.uk",
        ),
        (
            "https://subdomain.project.github.io/page",
            "project.github.io",
        ),
        (
            "http://192.168.1.10/login",
            "192.168.1.10",
        ),
    ],
)
def test_get_registered_domain(
    url: str,
    expected_domain: str,
) -> None:
    assert get_registered_domain(url) == expected_domain


def test_get_registered_domain_rejects_relative_url() -> None:
    with pytest.raises(ValueError, match="Invalid absolute URL"):
        get_registered_domain("/login/reset")
