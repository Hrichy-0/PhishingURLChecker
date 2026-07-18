from phishguard.prediction import ModelService, build_reasons


class FakeModel:
    def predict_proba(self, urls: list[str]) -> list[list[float]]:
        return [[0.15, 0.85] for _ in urls]


def test_build_reasons_detects_obvious_lexical_signals() -> None:
    reasons = build_reasons("http://user@192.168.1.5/login/verify")

    assert any("IP address" in reason for reason in reasons)
    assert any("@ symbol" in reason for reason in reasons)
    assert any("HTTPS" in reason for reason in reasons)


def test_model_service_applies_saved_threshold() -> None:
    service = object.__new__(ModelService)
    service.model = FakeModel()
    service.metadata = {"threshold": 0.55, "model_version": "test"}

    result = service.predict("https://example.test/login")

    assert result["is_phishing"] is True
    assert result["risk_level"] == "high"
    assert result["model_version"] == "test"
