from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------


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
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    timestamp: str = Field(default_factory=_utc_now_iso)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    titles: list[str] = Field(..., min_length=1, max_length=100)
    model_key: str | None = Field(
        default=None,
        description="Model key to use for all predictions. Falls back to server default.",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use result caching for duplicate titles.",
    )


class BatchPredictItem(PredictResponse):
    title: str


class BatchPredictResponse(BaseModel):
    results: list[BatchPredictItem]
    total: int
    cached_count: int
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=_utc_now_iso)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="News headline to classify")
    model_keys: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of model keys to compare (2-10 models).",
    )


class CompareItem(BaseModel):
    model: str
    label: int
    confidence: float = Field(ge=0.0, le=100.0)
    is_clickbait: bool
    elapsed_ms: float


class CompareResponse(BaseModel):
    results: list[CompareItem]
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=_utc_now_iso)


class HealthResponse(BaseModel):
    status: str
    device: str
    model: str
    cache: dict[str, int]
    uptime_seconds: float
    requests_total: int
    requests_cached: int
    avg_latency_ms: float


class MetricsResponse(BaseModel):
    requests_total: int
    requests_success: int
    requests_failed: int
    requests_cached: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    uptime_seconds: float
    cache_hit_rate: float
    model_stats: dict[str, int]
