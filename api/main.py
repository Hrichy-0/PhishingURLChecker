"""FastAPI application serving the packaged phishing model."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from phishguard.config import MODEL_METADATA_PATH, MODEL_PATH
from phishguard.prediction import ModelService

app = FastAPI(
    title="PhishGuard API",
    version="0.1.0",
    description="URL-only phishing risk estimates without visiting submitted sites.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    url: str = Field(min_length=4, max_length=2_048)

    @field_validator("url")
    @classmethod
    def validate_absolute_http_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("url must be an absolute HTTP or HTTPS URL")
        return value


class PredictionResponse(BaseModel):
    url: str
    is_phishing: bool
    phishing_probability: float
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    reasons: list[str]
    model_version: str
    disclaimer: str = (
        "This lexical model is an advisory signal and cannot guarantee "
        "that a URL is safe."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_available: bool


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    try:
        return ModelService()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available = MODEL_PATH.exists() and MODEL_METADATA_PATH.exists()
    return HealthResponse(
        status="ok" if available else "degraded",
        model_available=available,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    service: Annotated[ModelService, Depends(get_model_service)],
) -> PredictionResponse:
    return PredictionResponse(**service.predict(request.url))
