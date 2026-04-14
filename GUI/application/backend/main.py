from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .config import load_settings
from .schemas import HealthResponse, PredictRequest, PredictResponse
from .service import ORCDPredictService

settings = load_settings()
predict_service = ORCDPredictService(settings=settings)

app = FastAPI(
    title="Realtime Clickbait Detection API",
    description="FastAPI backend for ORCD clickbait detection used by Chrome Extension",
    version="1.0.0",
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
    return HealthResponse(**snapshot)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    try:
        output = await run_in_threadpool(predict_service.predict, payload.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        status = 503 if "API key" in message else 500
        raise HTTPException(status_code=status, detail=message) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return PredictResponse(**output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=False)
