"""Distributed rate limiting, token-bucket throttling, and tenant quota governance."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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


@dataclass(slots=True)
class TokenBucketState:
    """Mutable token-bucket state for a tenant."""

    tokens: float
    last_refill: float

    def refill(self, now: float, rate: float, capacity: int) -> None:
        """Refill tokens proportionally to elapsed seconds up to capacity."""
        if now <= self.last_refill:
            return
        elapsed = now - self.last_refill
        self.tokens = min(float(capacity), self.tokens + elapsed * rate)
        self.last_refill = now


@dataclass(slots=True)
class SlidingWindowState:
    """Mutable sliding-window request timestamp queue for a tenant."""

    timestamps: deque[float] = field(default_factory=deque)

    def prune(self, now: float, window_seconds: int) -> None:
        """Remove timestamps older than the active sliding window."""
        cutoff = now - window_seconds
        while self.timestamps and self.timestamps[0] <= cutoff:
            self.timestamps.popleft()


@dataclass(frozen=True, slots=True)
class AbuseGuardrailConfig:
    """Configuration for automated abuse detection and quarantine."""

    rejection_threshold: int = 20
    quarantine_seconds: float = 300.0
    violation_window_seconds: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize guardrail config to a JSON-compatible dictionary."""
        return {
            "rejection_threshold": self.rejection_threshold,
            "quarantine_seconds": self.quarantine_seconds,
            "violation_window_seconds": self.violation_window_seconds,
        }


class RateLimiterEngine:
    """Thread-safe multi-tenant rate limiter with token bucket, sliding window,
    daily quota governance, abuse quarantine, and snapshot state sync.
    """

    def __init__(
        self,
        *,
        default_tier: TenantTier = TenantTier.FREE,
        policies: dict[TenantTier, QuotaPolicy] | None = None,
        abuse_config: AbuseGuardrailConfig | None = None,
        enabled: bool = False,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.default_tier = default_tier
        self.policies: dict[TenantTier, QuotaPolicy] = dict(
            policies or DEFAULT_TIER_POLICIES
        )
        self.abuse_config = abuse_config or AbuseGuardrailConfig()
        self.enabled = enabled
        self._time_fn: Callable[[], float] = time_fn or time.time

        self._tenant_tiers: dict[str, TenantTier] = {}
        self._buckets: dict[str, TokenBucketState] = {}
        self._windows: dict[str, SlidingWindowState] = {}
        self._daily_counts: dict[str, tuple[str, int]] = {}
        self._violations: dict[str, deque[float]] = {}
        self._blocked_until: dict[str, tuple[float, str]] = {}

        self._requests_allowed_total: int = 0
        self._requests_rejected_total: int = 0

    def _now(self, now: float | None = None) -> float:
        return self._time_fn() if now is None else now

    @staticmethod
    def _day_key(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")

    def get_policy(self, tenant_id: str) -> QuotaPolicy:
        """Return the active QuotaPolicy for a tenant."""
        with self._lock:
            tier = self._tenant_tiers.get(tenant_id, self.default_tier)
            return self.policies[tier]

    def set_tenant_tier(self, tenant_id: str, tier: TenantTier) -> QuotaPolicy:
        """Assign or update a tenant's service tier and reset bucket to capacity."""
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            self._tenant_tiers[tenant_id] = tier
            policy = self.policies[tier]
            now = self._now()
            self._buckets[tenant_id] = TokenBucketState(
                tokens=float(policy.burst_capacity),
                last_refill=now,
            )
            return policy

    def block_tenant(
        self,
        tenant_id: str,
        duration_seconds: float | None = None,
        reason: str = "manual_block",
    ) -> float:
        """Quarantine a tenant until now + duration_seconds."""
        with self._lock:
            now = self._now()
            dur = (
                self.abuse_config.quarantine_seconds
                if duration_seconds is None
                else max(1.0, float(duration_seconds))
            )
            until = now + dur
            self._blocked_until[tenant_id] = (until, reason)
            return until

    def unblock_tenant(self, tenant_id: str) -> bool:
        """Remove a tenant's quarantine and clear recent violation history."""
        with self._lock:
            existed = tenant_id in self._blocked_until
            self._blocked_until.pop(tenant_id, None)
            self._violations.pop(tenant_id, None)
            return existed

    def _record_violation(self, tenant_id: str, now: float) -> bool:
        """Track a rejection and return True if it triggers auto-quarantine."""
        violations = self._violations.setdefault(tenant_id, deque())
        cutoff = now - self.abuse_config.violation_window_seconds
        while violations and violations[0] <= cutoff:
            violations.popleft()
        violations.append(now)
        if len(violations) >= self.abuse_config.rejection_threshold:
            until = now + self.abuse_config.quarantine_seconds
            self._blocked_until[tenant_id] = (until, "abuse_threshold_exceeded")
            return True
        return False

    def _daily_used(self, tenant_id: str, now: float) -> int:
        day = self._day_key(now)
        recorded = self._daily_counts.get(tenant_id)
        if recorded is None or recorded[0] != day:
            return 0
        return recorded[1]

    def check_and_consume(
        self,
        tenant_id: str,
        cost: int = 1,
        *,
        now: float | None = None,
    ) -> RateLimitDecision:
        """Evaluate rate limits and daily quota for a tenant request."""
        if cost < 1:
            raise ValueError("cost must be >= 1")
        tenant = tenant_id.strip() or "anonymous"

        with self._lock:
            current = self._now(now)
            policy = self.get_policy(tenant)
            day = self._day_key(current)
            daily_used = self._daily_used(tenant, current)

            # 1. Quarantine / abuse block check
            blocked_entry = self._blocked_until.get(tenant)
            if blocked_entry is not None:
                until, block_reason = blocked_entry
                if current < until:
                    self._requests_rejected_total += 1
                    return RateLimitDecision(
                        allowed=False,
                        tenant_id=tenant,
                        tier=policy.tier,
                        limit=policy.burst_capacity,
                        remaining=0,
                        reset_at=until,
                        retry_after_seconds=until - current,
                        reason=f"quarantined:{block_reason}",
                        daily_used=daily_used,
                        daily_quota=policy.daily_quota,
                        blocked=True,
                    )
                self._blocked_until.pop(tenant, None)

            # 2. Daily quota check
            if daily_used + cost > policy.daily_quota:
                self._requests_rejected_total += 1
                self._record_violation(tenant, current)
                next_midnight = (
                    datetime.fromtimestamp(current, tz=UTC)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .timestamp()
                    + 86400.0
                )
                return RateLimitDecision(
                    allowed=False,
                    tenant_id=tenant,
                    tier=policy.tier,
                    limit=policy.daily_quota,
                    remaining=max(0, policy.daily_quota - daily_used),
                    reset_at=next_midnight,
                    retry_after_seconds=max(1.0, next_midnight - current),
                    reason="daily_quota_exceeded",
                    daily_used=daily_used,
                    daily_quota=policy.daily_quota,
                    blocked=tenant in self._blocked_until,
                )

            # 3. Sliding window check
            window = self._windows.setdefault(tenant, SlidingWindowState())
            window.prune(current, policy.window_seconds)
            if (
                policy.algorithm
                in (
                    RateLimitAlgorithm.SLIDING_WINDOW,
                    RateLimitAlgorithm.HYBRID,
                )
                and len(window.timestamps) + cost > policy.max_requests_per_window
            ):
                self._requests_rejected_total += 1
                quarantined = self._record_violation(tenant, current)
                oldest = window.timestamps[0] if window.timestamps else current
                reset_at = oldest + policy.window_seconds
                return RateLimitDecision(
                    allowed=False,
                    tenant_id=tenant,
                    tier=policy.tier,
                    limit=policy.max_requests_per_window,
                    remaining=max(
                        0, policy.max_requests_per_window - len(window.timestamps)
                    ),
                    reset_at=reset_at,
                    retry_after_seconds=max(0.1, reset_at - current),
                    reason="sliding_window_exceeded",
                    daily_used=daily_used,
                    daily_quota=policy.daily_quota,
                    blocked=quarantined,
                )

            # 4. Token bucket check
            bucket = self._buckets.get(tenant)
            if bucket is None:
                bucket = TokenBucketState(
                    tokens=float(policy.burst_capacity),
                    last_refill=current,
                )
                self._buckets[tenant] = bucket
            else:
                bucket.refill(
                    current,
                    policy.requests_per_second,
                    policy.burst_capacity,
                )

            if policy.algorithm in (
                RateLimitAlgorithm.TOKEN_BUCKET,
                RateLimitAlgorithm.HYBRID,
            ):
                if bucket.tokens < float(cost):
                    self._requests_rejected_total += 1
                    quarantined = self._record_violation(tenant, current)
                    needed = float(cost) - bucket.tokens
                    wait_seconds = needed / max(0.001, policy.requests_per_second)
                    return RateLimitDecision(
                        allowed=False,
                        tenant_id=tenant,
                        tier=policy.tier,
                        limit=policy.burst_capacity,
                        remaining=int(bucket.tokens),
                        reset_at=current + wait_seconds,
                        retry_after_seconds=wait_seconds,
                        reason="token_bucket_exhausted",
                        daily_used=daily_used,
                        daily_quota=policy.daily_quota,
                        blocked=quarantined,
                    )
                bucket.tokens -= float(cost)

            # 5. Consume sliding window + daily quota
            for _ in range(cost):
                window.timestamps.append(current)
            new_daily = daily_used + cost
            self._daily_counts[tenant] = (day, new_daily)
            self._requests_allowed_total += 1

            if policy.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                limit_val = policy.max_requests_per_window
                remaining_val = max(0, limit_val - len(window.timestamps))
                reset_val = current + policy.window_seconds
            else:
                limit_val = policy.burst_capacity
                remaining_val = max(0, int(bucket.tokens))
                missing = max(0.0, float(policy.burst_capacity) - bucket.tokens)
                reset_val = current + (missing / max(0.001, policy.requests_per_second))

            return RateLimitDecision(
                allowed=True,
                tenant_id=tenant,
                tier=policy.tier,
                limit=limit_val,
                remaining=remaining_val,
                reset_at=reset_val,
                retry_after_seconds=0.0,
                reason="ok",
                daily_used=new_daily,
                daily_quota=policy.daily_quota,
                blocked=False,
            )

    def get_tenant_usage(
        self,
        tenant_id: str,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Return detailed rate-limit and quota usage snapshot for a tenant."""
        with self._lock:
            current = self._now(now)
            policy = self.get_policy(tenant_id)
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                tokens_available = float(policy.burst_capacity)
            else:
                bucket.refill(
                    current,
                    policy.requests_per_second,
                    policy.burst_capacity,
                )
                tokens_available = bucket.tokens

            window = self._windows.get(tenant_id)
            window_used = 0
            if window is not None:
                window.prune(current, policy.window_seconds)
                window_used = len(window.timestamps)

            daily_used = self._daily_used(tenant_id, current)
            blocked_entry = self._blocked_until.get(tenant_id)
            is_blocked = blocked_entry is not None and current < blocked_entry[0]
            return {
                "tenant_id": tenant_id,
                "tier": policy.tier.value,
                "algorithm": policy.algorithm.value,
                "tokens_available": round(tokens_available, 3),
                "burst_capacity": policy.burst_capacity,
                "requests_per_second": policy.requests_per_second,
                "window_used": window_used,
                "max_requests_per_window": policy.max_requests_per_window,
                "window_seconds": policy.window_seconds,
                "daily_used": daily_used,
                "daily_quota": policy.daily_quota,
                "daily_remaining": max(0, policy.daily_quota - daily_used),
                "blocked": is_blocked,
                "blocked_until": blocked_entry[0]
                if is_blocked and blocked_entry
                else None,
                "block_reason": blocked_entry[1]
                if is_blocked and blocked_entry
                else None,
            }

    def get_stats(self, *, now: float | None = None) -> dict[str, Any]:
        """Return aggregate rate-limiter telemetry and tenant summaries."""
        with self._lock:
            current = self._now(now)
            known_tenants = sorted(
                set(self._tenant_tiers)
                | set(self._buckets)
                | set(self._windows)
                | set(self._daily_counts)
                | set(self._blocked_until)
            )
            blocked_count = sum(
                1
                for tenant_id in known_tenants
                if tenant_id in self._blocked_until
                and current < self._blocked_until[tenant_id][0]
            )
            tenants = [
                self.get_tenant_usage(tenant_id, now=current)
                for tenant_id in known_tenants
            ]
            return {
                "enabled": self.enabled,
                "default_tier": self.default_tier.value,
                "requests_allowed_total": self._requests_allowed_total,
                "requests_rejected_total": self._requests_rejected_total,
                "active_tenants": len(known_tenants),
                "blocked_tenants": blocked_count,
                "abuse_config": self.abuse_config.to_dict(),
                "tenants": tenants,
            }

    def export_state(self) -> dict[str, Any]:
        """Export full rate-limiter state as a deterministic JSON-serializable dict."""
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "enabled": self.enabled,
                "default_tier": self.default_tier.value,
                "requests_allowed_total": self._requests_allowed_total,
                "requests_rejected_total": self._requests_rejected_total,
                "tenant_tiers": {k: v.value for k, v in self._tenant_tiers.items()},
                "buckets": {
                    k: {"tokens": v.tokens, "last_refill": v.last_refill}
                    for k, v in self._buckets.items()
                },
                "windows": {k: list(v.timestamps) for k, v in self._windows.items()},
                "daily_counts": {
                    k: {"day": v[0], "count": v[1]}
                    for k, v in self._daily_counts.items()
                },
                "blocked_until": {
                    k: {"until": v[0], "reason": v[1]}
                    for k, v in self._blocked_until.items()
                },
            }

    def import_state(
        self,
        state: dict[str, Any],
        *,
        merge: bool = False,
    ) -> None:
        """Import or merge distributed rate-limiter state from a snapshot dict."""
        with self._lock:
            if not merge:
                self._tenant_tiers.clear()
                self._buckets.clear()
                self._windows.clear()
                self._daily_counts.clear()
                self._blocked_until.clear()
                self._violations.clear()
                self._requests_allowed_total = int(
                    state.get("requests_allowed_total", 0)
                )
                self._requests_rejected_total = int(
                    state.get("requests_rejected_total", 0)
                )
            else:
                self._requests_allowed_total = max(
                    self._requests_allowed_total,
                    int(state.get("requests_allowed_total", 0)),
                )
                self._requests_rejected_total = max(
                    self._requests_rejected_total,
                    int(state.get("requests_rejected_total", 0)),
                )

            for tenant_id, tier_str in state.get("tenant_tiers", {}).items():
                self._tenant_tiers[str(tenant_id)] = TenantTier(tier_str)

            for tenant_id, bucket_data in state.get("buckets", {}).items():
                tid = str(tenant_id)
                incoming = TokenBucketState(
                    tokens=float(bucket_data["tokens"]),
                    last_refill=float(bucket_data["last_refill"]),
                )
                if merge and tid in self._buckets:
                    existing = self._buckets[tid]
                    self._buckets[tid] = TokenBucketState(
                        tokens=min(existing.tokens, incoming.tokens),
                        last_refill=max(existing.last_refill, incoming.last_refill),
                    )
                else:
                    self._buckets[tid] = incoming

            for tenant_id, stamps in state.get("windows", {}).items():
                tid = str(tenant_id)
                parsed = [float(ts) for ts in stamps]
                if merge and tid in self._windows:
                    combined = sorted([*self._windows[tid].timestamps, *parsed])
                    self._windows[tid] = SlidingWindowState(timestamps=deque(combined))
                else:
                    self._windows[tid] = SlidingWindowState(
                        timestamps=deque(sorted(parsed))
                    )

            for tenant_id, daily in state.get("daily_counts", {}).items():
                tid = str(tenant_id)
                day = str(daily["day"])
                count = int(daily["count"])
                if (
                    merge
                    and tid in self._daily_counts
                    and self._daily_counts[tid][0] == day
                ):
                    count = max(self._daily_counts[tid][1], count)
                self._daily_counts[tid] = (day, count)

            for tenant_id, blk in state.get("blocked_until", {}).items():
                tid = str(tenant_id)
                until = float(blk["until"])
                reason = str(blk.get("reason", "imported_block"))
                if merge and tid in self._blocked_until:
                    if until > self._blocked_until[tid][0]:
                        self._blocked_until[tid] = (until, reason)
                else:
                    self._blocked_until[tid] = (until, reason)

    def save_state(self, path: str | Path) -> Path:
        """Persist rate-limiter state snapshot to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path, *, merge: bool = False) -> bool:
        """Load rate-limiter state snapshot from a JSON file if it exists."""
        target = Path(path)
        if not target.exists():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.import_state(payload, merge=merge)
        return True


_global_rate_limiter = RateLimiterEngine()


def get_global_rate_limiter() -> RateLimiterEngine:
    """Return the process-wide RateLimiterEngine singleton."""
    return _global_rate_limiter


def set_global_rate_limiter(engine: RateLimiterEngine) -> None:
    """Replace the process-wide RateLimiterEngine singleton."""
    global _global_rate_limiter
    _global_rate_limiter = engine
