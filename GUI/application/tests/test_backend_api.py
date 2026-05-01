"""
Tests for the FastAPI backend endpoints.
Run with: cd GUI/application && PYTHONPATH="$(pwd):$(pwd)/tests" pytest tests/ -v
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestMetricsCollector:
    """Tests for the MetricsCollector."""

    def test_metrics_empty(self):
        from backend.metrics import MetricsCollector

        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["requests_total"] == 0
        assert snap["requests_success"] == 0
        assert snap["cache_hit_rate"] == 0.0

    def test_metrics_record_and_snapshot(self):
        from backend.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record(latency_ms=100.0, success=True, cached=False, model_key="test")
        mc.record(latency_ms=200.0, success=True, cached=True, model_key="test")
        mc.record(latency_ms=300.0, success=False, cached=False, model_key="test2")

        snap = mc.snapshot()
        assert snap["requests_total"] == 3
        assert snap["requests_success"] == 2
        assert snap["requests_failed"] == 1
        assert snap["requests_cached"] == 1
        assert snap["cache_hit_rate"] == pytest.approx(1 / 3, rel=0.01)
        assert snap["avg_latency_ms"] == 200.0
        assert snap["p50_latency_ms"] == 200.0
        assert snap["model_stats"]["test"] == 2
        assert snap["model_stats"]["test2"] == 1

    def test_percentile_single_value(self):
        from backend.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record(latency_ms=42.0, success=True, cached=False, model_key="x")
        snap = mc.snapshot()
        assert snap["p50_latency_ms"] == 42.0
        assert snap["p95_latency_ms"] == 42.0
        assert snap["p99_latency_ms"] == 42.0


class TestTokenBucketRateLimiter:
    """Tests for the TokenBucketRateLimiter."""

    def test_allows_under_limit(self):
        from backend.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(bucket_size=10, refill_rate=0, burst_size=10)
        assert limiter.acquire("client1") is True
        assert limiter.acquire("client1") is True
        assert limiter.acquire("client1") is True

    def test_blocks_over_limit(self):
        from backend.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(bucket_size=2, refill_rate=0, burst_size=2)
        assert limiter.acquire("client1") is True
        assert limiter.acquire("client1") is True
        assert limiter.acquire("client1") is False  # bucket exhausted

    def test_different_clients_separate_buckets(self):
        from backend.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(bucket_size=1, refill_rate=0, burst_size=1)
        assert limiter.acquire("client1") is True
        assert limiter.acquire("client1") is False
        assert limiter.acquire("client2") is True  # different client


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_response_has_extra_fields(self):
        """Verify the response schema includes new fields."""
        from backend.schemas import PredictResponse

        resp = PredictResponse(
            is_clickbait=True,
            confidence=85.5,
            label=1,
            model="test-model",
            device="cpu",
            cached=False,
        )
        assert hasattr(resp, "request_id")
        assert hasattr(resp, "elapsed_ms")
        assert hasattr(resp, "timestamp")
        assert resp.elapsed_ms == 0.0  # default


class TestBatchEndpoint:
    """Tests for the /batch-predict endpoint."""

    def test_batch_request_valid(self):
        from backend.schemas import BatchPredictRequest

        req = BatchPredictRequest(
            titles=["Title 1", "Title 2"],
            model_key="gemini-zero",
            use_cache=True,
        )
        assert len(req.titles) == 2
        assert req.model_key == "gemini-zero"

    def test_batch_request_max_100_titles(self):
        from backend.schemas import BatchPredictRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchPredictRequest(titles=["x"] * 101)

    def test_batch_response_structure(self):
        from backend.schemas import BatchPredictItem, BatchPredictResponse

        item = BatchPredictItem(
            title="Test",
            is_clickbait=True,
            confidence=75.0,
            label=1,
            model="test",
            device="cpu",
            cached=False,
        )
        resp = BatchPredictResponse(
            results=[item],
            total=1,
            cached_count=0,
        )
        assert len(resp.results) == 1
        assert resp.total == 1


class TestCompareEndpoint:
    """Tests for the /compare endpoint."""

    def test_compare_request_min_2_models(self):
        from backend.schemas import CompareRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompareRequest(title="Test", model_keys=["model1"])

    def test_compare_request_max_10_models(self):
        from backend.schemas import CompareRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompareRequest(title="Test", model_keys=[f"model{i}" for i in range(11)])

    def test_compare_response_structure(self):
        from backend.schemas import CompareItem, CompareResponse

        item = CompareItem(
            model="gemini-zero",
            label=1,
            confidence=88.0,
            is_clickbait=True,
            elapsed_ms=120.5,
        )
        resp = CompareResponse(results=[item])
        assert len(resp.results) == 1
        assert resp.results[0].model == "gemini-zero"


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_response_has_new_fields(self):
        from backend.schemas import HealthResponse

        resp = HealthResponse(
            status="ok",
            device="cuda",
            model="generate-and-predict",
            cache={"hits": 10, "misses": 5, "maxsize": 1024, "currsize": 3},
            uptime_seconds=120.5,
            requests_total=15,
            requests_cached=10,
            avg_latency_ms=80.0,
        )
        assert hasattr(resp, "uptime_seconds")
        assert hasattr(resp, "requests_total")
        assert hasattr(resp, "requests_cached")
        assert hasattr(resp, "avg_latency_ms")


class TestORCDPredictService:
    """Tests for ORCDPredictService batch/compare methods."""

    def test_normalize_title(self):
        from backend.service import ORCDPredictService

        assert ORCDPredictService._normalize_title(None) == ""
        assert ORCDPredictService._normalize_title("  Hello   World  ") == "Hello World"
        assert ORCDPredictService._normalize_title("") == ""

    def test_is_api_backend_model(self):
        from backend.service import ORCDPredictService

        assert ORCDPredictService._is_api_backend_model("orcd-gpt35") is True
        assert ORCDPredictService._is_api_backend_model("gpt4o-zero") is True
        assert ORCDPredictService._is_api_backend_model("gemini-zero") is True
        assert ORCDPredictService._is_api_backend_model("bart-mnli") is False
        assert ORCDPredictService._is_api_backend_model("generate-and-predict") is True
        assert ORCDPredictService._is_api_backend_model("") is False
        assert ORCDPredictService._is_api_backend_model(None) is False

    def test_extract_api_key(self):
        from backend.service import ORCDPredictService

        assert ORCDPredictService._extract_api_key("sk-abc123xyz") == "sk-abc123xyz"
        assert ORCDPredictService._extract_api_key("") == ""
        assert ORCDPredictService._extract_api_key("key: sk-abc123") == "sk-abc123"
        assert ORCDPredictService._extract_api_key("ORCD_API_KEY=sk-abc123") == "sk-abc123"

    def test_should_fallback_disabled(self):
        from backend.service import ORCDPredictService

        svc = ORCDPredictService.__new__(ORCDPredictService)
        svc._settings = MagicMock()
        svc._settings.enable_api_fallback = False

        result = svc._should_fallback_to_local("orcd-gpt35", "bart-mnli", RuntimeError("timeout"))
        assert result is False

    def test_should_fallback_timeout(self):
        from backend.service import ORCDPredictService

        svc = ORCDPredictService.__new__(ORCDPredictService)
        svc._settings = MagicMock()
        svc._settings.enable_api_fallback = True

        result = svc._should_fallback_to_local("orcd-gpt35", "bart-mnli", RuntimeError("connection timeout"))
        assert result is True

    def test_should_fallback_api_key_error(self):
        from backend.service import ORCDPredictService

        svc = ORCDPredictService.__new__(ORCDPredictService)
        svc._settings = MagicMock()
        svc._settings.enable_api_fallback = True

        result = svc._should_fallback_to_local("orcd-gpt35", "bart-mnli", RuntimeError("invalid API key"))
        assert result is True

    def test_should_fallback_same_model(self):
        from backend.service import ORCDPredictService

        svc = ORCDPredictService.__new__(ORCDPredictService)
        svc._settings = MagicMock()
        svc._settings.enable_api_fallback = True

        result = svc._should_fallback_to_local("bart-mnli", "bart-mnli", RuntimeError("timeout"))
        assert result is False  # Same model key

    def test_should_fallback_local_to_api(self):
        from backend.service import ORCDPredictService

        svc = ORCDPredictService.__new__(ORCDPredictService)
        svc._settings = MagicMock()
        svc._settings.enable_api_fallback = True

        result = svc._should_fallback_to_local("bart-mnli", "orcd-gpt35", RuntimeError("timeout"))
        assert result is False  # Should not fallback API -> local if primary is local
