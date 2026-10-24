"""FastAPI application serving the packaged phishing model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from xgboost.core import XGBoostError

from phishguard.config import MAIN_URL_MODEL_DIR, URL_HTML_MODEL_DIR
from phishguard.data import is_valid_http_url
from phishguard.prediction import (
    ModelService,
    NativeURLHTMLService,
    NativeURLService,
)

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
        if not is_valid_http_url(value):
            raise ValueError("url must be an absolute HTTP or HTTPS URL")
        return value


class ContentPredictionRequest(PredictionRequest):
    # The extractor parses at most the model's 1 MB prefix and flags truncation.
    html: str = Field(max_length=5_000_000)


class PredictionResponse(BaseModel):
    url: str
    is_phishing: bool
    phishing_probability: float
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    reasons: list[str]
    model_version: str
    model_type: str = "URL model"
    model_sha256: str | None = None
    experimental: bool = False
    feature_count: int | None = None
    disclaimer: str = (
        "This lexical model is an advisory signal and cannot guarantee "
        "that a URL is safe."
    )


class ContentPredictionResponse(PredictionResponse):
    html_truncated: bool
    disclaimer: str = (
        "This experimental URL and page-content model is an advisory signal "
        "and cannot guarantee that a page is safe."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_available: bool
    model_type: str | None = None
    model_sha256: str | None = None
    experimental: bool = False
    feature_count: int | None = None
    content_model_available: bool = False
    content_model_sha256: str | None = None


@lru_cache(maxsize=1)
def get_model_service() -> ModelService | NativeURLService:
    try:
        experiment = os.environ.get("PHISHGUARD_EXPERIMENTAL_MODEL_DIR")
        selection = os.environ.get("PHISHGUARD_MODEL", "url_only")
        if selection not in {"url_only", "legacy"}:
            raise ValueError("PHISHGUARD_MODEL must be url_only or legacy")
        if selection == "legacy":
            if experiment:
                raise ValueError("Remove the experiment override before using legacy")
            return ModelService()
        return NativeURLService(Path(experiment) if experiment else MAIN_URL_MODEL_DIR)
    except (OSError, ValueError, KeyError, TypeError, XGBoostError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@lru_cache(maxsize=1)
def get_content_model_service() -> NativeURLHTMLService:
    try:
        return NativeURLHTMLService(URL_HTML_MODEL_DIR)
    except (OSError, ValueError, KeyError, TypeError, XGBoostError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


def content_model_identity() -> dict[str, object]:
    try:
        service = get_content_model_service()
    except HTTPException:
        return {"content_model_available": False}
    return {
        "content_model_available": True,
        "content_model_sha256": service.model_sha256,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        service = get_model_service()
    except HTTPException:
        return HealthResponse(
            status="degraded", model_available=False, **content_model_identity()
        )
    return HealthResponse(
        status="ok",
        model_available=True,
        model_type=service.metadata.get("model_type"),
        model_sha256=service.model_sha256,
        experimental=service.experimental,
        feature_count=len(service.metadata.get("features", [])) or None,
        **content_model_identity(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    service: Annotated[ModelService | NativeURLService, Depends(get_model_service)],
) -> PredictionResponse:
    return PredictionResponse(**service.predict(request.url))


@app.post("/predict_html", response_model=ContentPredictionResponse)
def predict_html(
    request: ContentPredictionRequest,
    service: Annotated[NativeURLHTMLService, Depends(get_content_model_service)],
) -> ContentPredictionResponse:
    """Score HTML the browser already loaded; the API never fetches the page."""
    return ContentPredictionResponse(**service.predict(request.url, request.html))
