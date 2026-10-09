import pytest
from fastapi.testclient import TestClient

from api.main import app, get_model_service


class FakeService:
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
def test_health_reports_model_artifact(client, tmp_path, monkeypatch, available):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    if available:
        model_path.touch()
        metadata_path.touch()
    monkeypatch.setattr("api.main.MODEL_PATH", model_path)
    monkeypatch.setattr("api.main.MODEL_METADATA_PATH", metadata_path)
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
