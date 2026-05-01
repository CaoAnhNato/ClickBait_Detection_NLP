from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class RequestRecord:
    latency_ms: float
    success: bool
    cached: bool
    model_key: str
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[RequestRecord] = []
        self._start_time = time.time()
        self._model_counts: dict[str, int] = {}

    def record(self, latency_ms: float, success: bool, cached: bool, model_key: str) -> None:
        with self._lock:
            self._records.append(
                RequestRecord(
                    latency_ms=latency_ms,
                    success=success,
                    cached=cached,
                    model_key=model_key,
                )
            )
            self._model_counts[model_key] = self._model_counts.get(model_key, 0) + 1
            # Keep at most 10,000 records to prevent unbounded memory growth
            if len(self._records) > 10_000:
                self._records = self._records[-5_000:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if not self._records:
                return {
                    "requests_total": 0,
                    "requests_success": 0,
                    "requests_failed": 0,
                    "requests_cached": 0,
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "uptime_seconds": round(time.time() - self._start_time, 2),
                    "cache_hit_rate": 0.0,
                    "model_stats": dict(self._model_counts),
                }

            total = len(self._records)
            success = sum(1 for r in self._records if r.success)
            cached = sum(1 for r in self._records if r.cached)
            latencies = sorted(r.latency_ms for r in self._records)
            avg = sum(latencies) / total
            p50 = self._percentile(latencies, 50)
            p95 = self._percentile(latencies, 95)
            p99 = self._percentile(latencies, 99)
            cache_rate = cached / total if total > 0 else 0.0

            return {
                "requests_total": total,
                "requests_success": success,
                "requests_failed": total - success,
                "requests_cached": cached,
                "avg_latency_ms": round(avg, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "cache_hit_rate": round(cache_rate, 4),
                "model_stats": dict(self._model_counts),
            }

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: int) -> float:
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * percentile / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(f)]
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


# Global singleton
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics
