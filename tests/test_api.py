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


def test_health_reports_model_artifact() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["model_available"] is True


def test_predict_validates_and_returns_prediction() -> None:
    app.dependency_overrides[get_model_service] = lambda: FakeService()
    try:
        client = TestClient(app)
        response = client.post("/predict", json={"url": "https://example.test/login"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["phishing_probability"] == 0.91
    assert "guarantee" in response.json()["disclaimer"]


def test_predict_rejects_non_http_input() -> None:
    response = TestClient(app).post("/predict", json={"url": "javascript:alert(1)"})

    assert response.status_code == 422
