"""Distributed rate limiting, token-bucket throttling, and tenant quota governance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class TenantTier(StrEnum):
    """Supported tenant service tiers for rate limiting and quota governance."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class RateLimitAlgorithm(StrEnum):
    """Supported rate-limiting enforcement algorithms."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class QuotaPolicy:
    """Rate limit and daily quota policy for a tenant tier."""

    tier: TenantTier
    requests_per_second: float
    burst_capacity: int
    window_seconds: int
    max_requests_per_window: int
    daily_quota: int
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.HYBRID

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy configuration to a JSON-compatible dictionary."""
        return {
            "tier": self.tier.value,
            "requests_per_second": self.requests_per_second,
            "burst_capacity": self.burst_capacity,
            "window_seconds": self.window_seconds,
            "max_requests_per_window": self.max_requests_per_window,
            "daily_quota": self.daily_quota,
            "algorithm": self.algorithm.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QuotaPolicy:
        """Deserialize a QuotaPolicy from a dictionary."""
        return cls(
            tier=TenantTier(payload["tier"]),
            requests_per_second=float(payload["requests_per_second"]),
            burst_capacity=int(payload["burst_capacity"]),
            window_seconds=int(payload["window_seconds"]),
            max_requests_per_window=int(payload["max_requests_per_window"]),
            daily_quota=int(payload["daily_quota"]),
            algorithm=RateLimitAlgorithm(
                payload.get("algorithm", RateLimitAlgorithm.HYBRID.value)
            ),
        )


DEFAULT_TIER_POLICIES: dict[TenantTier, QuotaPolicy] = {
    TenantTier.FREE: QuotaPolicy(
        tier=TenantTier.FREE,
        requests_per_second=5.0,
        burst_capacity=15,
        window_seconds=60,
        max_requests_per_window=120,
        daily_quota=2000,
    ),
    TenantTier.PRO: QuotaPolicy(
        tier=TenantTier.PRO,
        requests_per_second=25.0,
        burst_capacity=60,
        window_seconds=60,
        max_requests_per_window=1000,
        daily_quota=50000,
    ),
    TenantTier.ENTERPRISE: QuotaPolicy(
        tier=TenantTier.ENTERPRISE,
        requests_per_second=100.0,
        burst_capacity=250,
        window_seconds=60,
        max_requests_per_window=5000,
        daily_quota=1000000,
    ),
}


@dataclass(slots=True)
class RateLimitDecision:
    """Outcome of evaluating a single request against tenant rate limits and quotas."""

    allowed: bool
    tenant_id: str
    tier: TenantTier
    limit: int
    remaining: int
    reset_at: float
    retry_after_seconds: float = 0.0
    reason: str = "ok"
    daily_used: int = 0
    daily_quota: int = 0
    blocked: bool = False
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_headers(self) -> dict[str, str]:
        """Build standard HTTP rate-limit response headers."""
        headers = {
            "X-RateLimit-Limit": str(max(0, self.limit)),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(round(self.reset_at))),
        }
        if not self.allowed:
            retry_after = max(1, int(round(self.retry_after_seconds)))
            headers["Retry-After"] = str(retry_after)
        return headers

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision details for diagnostics, API responses, and logs."""
        return {
            "allowed": self.allowed,
            "tenant_id": self.tenant_id,
            "tier": self.tier.value,
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "retry_after_seconds": round(self.retry_after_seconds, 3),
            "reason": self.reason,
            "daily_used": self.daily_used,
            "daily_quota": self.daily_quota,
            "blocked": self.blocked,
            "evaluated_at": self.evaluated_at,
        }
