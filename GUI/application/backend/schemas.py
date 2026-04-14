from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="News headline to classify")


class PredictResponse(BaseModel):
    is_clickbait: bool
    confidence: float = Field(..., ge=0.0, le=100.0)
    label: int = Field(..., ge=0, le=1)
    model: str
    device: str
    cached: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    model: str
    cache: dict[str, int]
