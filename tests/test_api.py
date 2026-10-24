import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import app, get_content_model_service, get_model_service


class FakeService:
    metadata = {"model_type": "Test URL model"}
    model_sha256 = "test-checksum"
    experimental = False

    def predict(self, url: str) -> dict[str, object]:
        return {
            "url": url,
            "is_phishing": True,
            "phishing_probability": 0.91,
            "risk_level": "high",
            "threshold": 0.55,
            "reasons": ["Test reason"],
            "model_version": "test",
        }


@pytest.fixture
def client():
    # A fresh Git clone excludes trained artifacts. API unit tests must not
    # require local model binaries or deserialize a real model.
    app.dependency_overrides[get_model_service] = lambda: FakeService()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("available", [False, True])
def test_health_reports_model_artifact(client, monkeypatch, available):
    def service():
        if not available:
            raise HTTPException(status_code=503, detail="Missing model")
        return FakeService()

    monkeypatch.setattr("api.main.get_model_service", service)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_available"] is available
    assert response.json()["status"] == ("ok" if available else "degraded")


def test_predict_validates_and_returns_prediction(client) -> None:
    response = client.post("/predict", json={"url": "https://example.test/login"})

    assert response.status_code == 200
    assert response.json()["phishing_probability"] == 0.91
    assert "guarantee" in response.json()["disclaimer"]


def test_predict_rejects_non_http_input(client) -> None:
    response = client.post("/predict", json={"url": "javascript:alert(1)"})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "url", ["https://example.test/a b", "https://example.test:99999"]
)
def test_predict_rejects_invalid_http_urls(client, url):
    assert client.post("/predict", json={"url": url}).status_code == 422


def test_native_override_with_wrong_schema_reports_degraded(tmp_path, monkeypatch):
    import json

    get_model_service.cache_clear()
    monkeypatch.delenv("PHISHGUARD_MODEL", raising=False)
    monkeypatch.setenv("PHISHGUARD_EXPERIMENTAL_MODEL_DIR", str(tmp_path))
    (tmp_path / "model_metadata.json").write_text(
        json.dumps({"features": ["html_length"]})
    )
    try:
        response = TestClient(app).get("/health")
        assert response.json()["status"] == "degraded"
        assert response.json()["model_available"] is False
        assert (
            TestClient(app)
            .post("/predict", json={"url": "https://example.test/"})
            .status_code
            == 503
        )
    finally:
        get_model_service.cache_clear()


def test_main_model_is_native_and_legacy_requires_explicit_selection(monkeypatch):
    from phishguard.config import MAIN_URL_MODEL_DIR

    monkeypatch.delenv("PHISHGUARD_MODEL", raising=False)
    monkeypatch.delenv("PHISHGUARD_EXPERIMENTAL_MODEL_DIR", raising=False)
    native, legacy = object(), object()
    calls = []

    def load_native(path):
        calls.append(path)
        return native

    monkeypatch.setattr("api.main.NativeURLService", load_native)
    monkeypatch.setattr("api.main.ModelService", lambda: legacy)
    get_model_service.cache_clear()
    try:
        assert get_model_service() is native
        assert calls == [MAIN_URL_MODEL_DIR]
        monkeypatch.setenv("PHISHGUARD_MODEL", "legacy")
        get_model_service.cache_clear()
        assert get_model_service() is legacy
    finally:
        get_model_service.cache_clear()


def test_missing_main_model_does_not_silently_fall_back(monkeypatch):
    monkeypatch.delenv("PHISHGUARD_MODEL", raising=False)
    monkeypatch.delenv("PHISHGUARD_EXPERIMENTAL_MODEL_DIR", raising=False)

    def missing(path):
        raise FileNotFoundError("Main model missing")

    def forbidden():
        raise AssertionError("No silent legacy fallback")

    monkeypatch.setattr("api.main.NativeURLService", missing)
    monkeypatch.setattr("api.main.ModelService", forbidden)
    get_model_service.cache_clear()
    try:
        assert TestClient(app).get("/health").json()["model_available"] is False
    finally:
        get_model_service.cache_clear()


class FakeContentService(FakeService):
    def predict(self, url: str, html: str) -> dict[str, object]:
        return {**super().predict(url), "html_truncated": len(html) > 10}


def test_predict_html_returns_content_prediction(client) -> None:
    app.dependency_overrides[get_content_model_service] = FakeContentService
    response = client.post(
        "/predict_html",
        json={"url": "https://example.test/login", "html": "<form></form>!"},
    )

    assert response.status_code == 200
    assert response.json()["html_truncated"] is True
    assert "page-content" in response.json()["disclaimer"]


def test_predict_html_rejects_oversized_or_missing_html(client) -> None:
    app.dependency_overrides[get_content_model_service] = FakeContentService
    url = "https://example.test/"

    assert client.post("/predict_html", json={"url": url}).status_code == 422
    oversized = {"url": url, "html": "a" * 5_000_001}
    assert client.post("/predict_html", json=oversized).status_code == 422


def test_missing_content_model_returns_503_without_touching_url_model(
    client, monkeypatch
):
    def missing(path):
        raise FileNotFoundError("Content model missing")

    monkeypatch.setattr("api.main.NativeURLHTMLService", missing)
    get_content_model_service.cache_clear()
    try:
        content = {"url": "https://example.test/", "html": "<p></p>"}
        assert client.post("/predict_html", json=content).status_code == 503
        assert client.get("/health").json()["content_model_available"] is False
        url_only = {"url": "https://example.test/"}
        assert client.post("/predict", json=url_only).status_code == 200
    finally:
        get_content_model_service.cache_clear()
