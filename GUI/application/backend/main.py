from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import load_settings
from .metrics import MetricsCollector, get_metrics
from .rate_limiter import get_or_create_ip_limiter
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictItem,
    CompareRequest,
    CompareResponse,
    CompareItem,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    _utc_now_iso,
)
from .service import ORCDPredictService

settings = load_settings()
predict_service = ORCDPredictService(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_result = await run_in_threadpool(predict_service.warm_up)
    print("[ORCD] Model warmup complete:", warmup_result)
    yield
    print("[ORCD] Shutting down ORCD service")


app = FastAPI(
    title="Realtime Clickbait Detection API",
    description="FastAPI backend for ORCD clickbait detection used by Chrome Extension",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    snapshot = await run_in_threadpool(predict_service.health_snapshot)
    metrics = get_metrics().snapshot()
    return HealthResponse(
        status="ok",
        device=snapshot["device"],
        model=snapshot["model"],
        cache=snapshot["cache"],
        uptime_seconds=predict_service.uptime_seconds(),
        requests_total=metrics["requests_total"],
        requests_cached=metrics["requests_cached"],
        avg_latency_ms=metrics["avg_latency_ms"],
    )


@app.get("/warmup")
async def warmup() -> dict[str, Any]:
    result = await run_in_threadpool(predict_service.warm_up)
    return result


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    metrics_data = get_metrics().snapshot()
    return MetricsResponse(**metrics_data)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    client_ip = _get_client_ip(request)
    limiter = get_or_create_ip_limiter()

    if not limiter.acquire(client_ip):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please slow down.",
                "retry_after_seconds": 5,
            },
        )

    started = time.perf_counter()
    metrics_collector = get_metrics()

    try:
        output = await run_in_threadpool(predict_service.predict, payload.title)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        metrics_collector.record(
            latency_ms=elapsed_ms,
            success=True,
            cached=output.get("cached", False),
            model_key=output.get("model", "unknown"),
        )
    except ValueError as exc:
        get_metrics().record(0, False, False, "predict")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        get_metrics().record(0, False, False, "predict")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        status = 503 if "API key" in message else 500
        get_metrics().record(0, False, False, "predict")
        raise HTTPException(status_code=status, detail=message) from exc
    except Exception as exc:  # noqa: BLE001
        get_metrics().record(0, False, False, "predict")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return PredictResponse(
        is_clickbait=output["is_clickbait"],
        confidence=output["confidence"],
        label=output["label"],
        model=output["model"],
        device=output["device"],
        cached=output["cached"],
        request_id=str(uuid.uuid4())[:8],
        elapsed_ms=elapsed_ms,
        timestamp=_utc_now_iso(),
    )


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    client_ip = _get_client_ip(request)
    limiter = get_or_create_ip_limiter()

    num_tokens = len(payload.titles)
    if not limiter.acquire(client_ip, cost=num_tokens):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Batch size {num_tokens} exceeds remaining token budget. Reduce batch size or wait.",
                "retry_after_seconds": 5,
            },
        )

    started = time.perf_counter()

    try:
        results = await run_in_threadpool(
            predict_service.predict_batch,
            payload.titles,
            payload.model_key,
            payload.use_cache,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        cached_count = sum(1 for r in results if r.get("cached", False))
        total = len(results)

        batch_items = [
            BatchPredictItem(
                title=r["title"],
                is_clickbait=r["is_clickbait"],
                confidence=r["confidence"],
                label=r["label"],
                model=r["model"],
                device=r["device"],
                cached=r["cached"],
                request_id=str(uuid.uuid4())[:8],
                elapsed_ms=r.get("elapsed_ms", 0.0),
                timestamp=_utc_now_iso(),
            )
            for r in results
        ]

        return BatchPredictResponse(
            results=batch_items,
            total=total,
            cached_count=cached_count,
            elapsed_ms=elapsed_ms,
            request_id=str(uuid.uuid4())[:8],
            timestamp=_utc_now_iso(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc


@app.post("/compare", response_model=CompareResponse)
async def compare(payload: CompareRequest, request: Request) -> CompareResponse:
    client_ip = _get_client_ip(request)
    limiter = get_or_create_ip_limiter()

    if not limiter.acquire(client_ip, cost=len(payload.model_keys)):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many comparison requests. Please wait.",
                "retry_after_seconds": 5,
            },
        )

    try:
        results = await run_in_threadpool(
            predict_service.predict_compare,
            payload.title,
            payload.model_keys,
        )

        compare_items = [
            CompareItem(
                model=r["model"],
                label=r["label"],
                confidence=r["confidence"],
                is_clickbait=r["is_clickbait"],
                elapsed_ms=r.get("elapsed_ms", 0.0),
            )
            for r in results
        ]

        return CompareResponse(
            results=compare_items,
            request_id=str(uuid.uuid4())[:8],
            timestamp=_utc_now_iso(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Comparison error: {exc}") from exc


@app.delete("/cache")
async def clear_cache() -> dict[str, Any]:
    await run_in_threadpool(predict_service.clear_cache)
    return {"status": "ok", "message": "Prediction cache cleared"}


@app.get("/models")
async def list_models() -> dict[str, Any]:
    """Return available models."""
    return {
        "models": predict_service._service.available_models(),
        "default": settings.model_key,
        "fallback": settings.fallback_local_model_key,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=False)
