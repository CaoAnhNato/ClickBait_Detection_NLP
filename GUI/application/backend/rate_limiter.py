from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import Any


class TokenBucketRateLimiter:
    """Token bucket rate limiter - each client has a bucket refilled at refill_rate per second."""

    def __init__(
        self,
        bucket_size: int = 100,
        refill_rate: float = 10.0,
        burst_size: int | None = None,
    ) -> None:
        self._bucket_size = bucket_size
        self._refill_rate = refill_rate
        self._burst_size = burst_size if burst_size is not None else bucket_size
        self._buckets: dict[str, float] = defaultdict(lambda: float(self._burst_size))
        self._last_refill: dict[str, float] = defaultdict(time.time)
        self._lock = Lock()
        self._total_requests: dict[str, int] = defaultdict(int)
        self._total_rejected: dict[str, int] = defaultdict(int)

    def _refill(self, key: str) -> float:
        now = time.time()
        elapsed = now - self._last_refill[key]
        tokens_to_add = elapsed * self._refill_rate
        self._buckets[key] = min(self._burst_size, self._buckets[key] + tokens_to_add)
        self._last_refill[key] = now
        return self._buckets[key]

    def acquire(self, key: str, cost: int = 1) -> bool:
        with self._lock:
            current = self._refill(key)
            if current >= cost:
                self._buckets[key] -= cost
                self._total_requests[key] += 1
                return True
            self._total_rejected[key] += 1
            return False

    def stats(self, key: str | None = None) -> dict[str, Any]:
        with self._lock:
            if key is None:
                total_req = sum(self._total_requests.values())
                total_rej = sum(self._total_rejected.values())
                return {
                    "total_requests": total_req,
                    "total_rejected": total_rej,
                    "active_clients": len(self._buckets),
                }
            return {
                "requests": self._total_requests.get(key, 0),
                "rejected": self._total_rejected.get(key, 0),
                "available_tokens": round(self._buckets.get(key, 0.0), 2),
            }


# Per-IP rate limiter (keyed by client IP)
_per_ip_limiter: TokenBucketRateLimiter | None = None


def get_ip_limiter() -> TokenBucketRateLimiter:
    global _per_ip_limiter
    if _per_ip_limiter is None:
        _per_ip_limiter = TokenBucketRateLimiter(bucket_size=60, refill_rate=10.0, burst_size=30)
    return _per_ip_limiter


def get_or_create_ip_limiter(
    bucket_size: int = 60,
    refill_rate: float = 10.0,
    burst_size: int = 30,
) -> TokenBucketRateLimiter:
    global _per_ip_limiter
    if _per_ip_limiter is None:
        _per_ip_limiter = TokenBucketRateLimiter(
            bucket_size=bucket_size,
            refill_rate=refill_rate,
            burst_size=burst_size,
        )
    return _per_ip_limiter
