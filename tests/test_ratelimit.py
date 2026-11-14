"""Tests for distributed rate limiting, token-bucket throttling, and tenant quotas."""

from __future__ import annotations

from pathlib import Path

from music_recommender.ratelimit import (
    DEFAULT_TIER_POLICIES,
    AbuseGuardrailConfig,
    QuotaPolicy,
    RateLimitAlgorithm,
    RateLimitDecision,
    RateLimiterEngine,
    TenantTier,
)


def test_quota_policy_roundtrip_and_defaults() -> None:
    """Default tier policies should cover all TenantTier values and roundtrip JSON."""
    assert set(DEFAULT_TIER_POLICIES.keys()) == {
        TenantTier.FREE,
        TenantTier.PRO,
        TenantTier.ENTERPRISE,
    }
    free_policy = DEFAULT_TIER_POLICIES[TenantTier.FREE]
    pro_policy = DEFAULT_TIER_POLICIES[TenantTier.PRO]
    ent_policy = DEFAULT_TIER_POLICIES[TenantTier.ENTERPRISE]

    assert free_policy.requests_per_second < pro_policy.requests_per_second
    assert pro_policy.requests_per_second < ent_policy.requests_per_second
    assert free_policy.daily_quota < pro_policy.daily_quota < ent_policy.daily_quota

    payload = pro_policy.to_dict()
    restored = QuotaPolicy.from_dict(payload)
    assert restored == pro_policy
    assert restored.algorithm == RateLimitAlgorithm.HYBRID


def test_rate_limit_decision_headers_and_serialization() -> None:
    """RateLimitDecision should emit standard headers and Retry-After on rejection."""
    allowed_decision = RateLimitDecision(
        allowed=True,
        tenant_id="tenant-a",
        tier=TenantTier.PRO,
        limit=60,
        remaining=59,
        reset_at=1700000060.2,
        daily_used=1,
        daily_quota=50000,
    )
    allowed_headers = allowed_decision.to_headers()
    assert allowed_headers == {
        "X-RateLimit-Limit": "60",
        "X-RateLimit-Remaining": "59",
        "X-RateLimit-Reset": "1700000060",
    }
    assert "Retry-After" not in allowed_headers

    rejected_decision = RateLimitDecision(
        allowed=False,
        tenant_id="tenant-a",
        tier=TenantTier.FREE,
        limit=15,
        remaining=0,
        reset_at=1700000010.0,
        retry_after_seconds=0.4,
        reason="token_bucket_exhausted",
        daily_used=15,
        daily_quota=2000,
    )
    rejected_headers = rejected_decision.to_headers()
    assert rejected_headers["X-RateLimit-Remaining"] == "0"
    assert rejected_headers["Retry-After"] == "1"

    as_dict = rejected_decision.to_dict()
    assert as_dict["allowed"] is False
    assert as_dict["tier"] == "free"
    assert as_dict["reason"] == "token_bucket_exhausted"


def test_token_bucket_burst_and_refill() -> None:
    """Token bucket should allow burst up to capacity, reject, and refill over time."""
    now = [1000.0]
    policies = {
        TenantTier.FREE: QuotaPolicy(
            tier=TenantTier.FREE,
            requests_per_second=2.0,
            burst_capacity=3,
            window_seconds=60,
            max_requests_per_window=100,
            daily_quota=500,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
        TenantTier.PRO: DEFAULT_TIER_POLICIES[TenantTier.PRO],
        TenantTier.ENTERPRISE: DEFAULT_TIER_POLICIES[TenantTier.ENTERPRISE],
    }
    engine = RateLimiterEngine(policies=policies, time_fn=lambda: now[0])

    # Consume all 3 burst tokens
    for expected_remaining in (2, 1, 0):
        decision = engine.check_and_consume("t1")
        assert decision.allowed is True
        assert decision.remaining == expected_remaining

    # 4th request immediately fails
    rejected = engine.check_and_consume("t1")
    assert rejected.allowed is False
    assert rejected.reason == "token_bucket_exhausted"
    assert rejected.retry_after_seconds > 0

    # Advance 1.0s -> 2.0 tokens refilled
    now[0] += 1.0
    refilled_1 = engine.check_and_consume("t1")
    refilled_2 = engine.check_and_consume("t1")
    assert refilled_1.allowed is True
    assert refilled_2.allowed is True
    assert engine.check_and_consume("t1").allowed is False


def test_sliding_window_and_daily_quota_enforcement() -> None:
    """Sliding window and daily quota limits should both enforce and reset."""
    now = [2000.0]
    policies = {
        TenantTier.FREE: QuotaPolicy(
            tier=TenantTier.FREE,
            requests_per_second=100.0,
            burst_capacity=100,
            window_seconds=10,
            max_requests_per_window=2,
            daily_quota=3,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
        ),
        TenantTier.PRO: DEFAULT_TIER_POLICIES[TenantTier.PRO],
        TenantTier.ENTERPRISE: DEFAULT_TIER_POLICIES[TenantTier.ENTERPRISE],
    }
    engine = RateLimiterEngine(policies=policies, time_fn=lambda: now[0])

    assert engine.check_and_consume("t2").allowed is True
    assert engine.check_and_consume("t2").allowed is True
    win_reject = engine.check_and_consume("t2")
    assert win_reject.allowed is False
    assert win_reject.reason == "sliding_window_exceeded"

    # Advance past 10s window -> 3rd daily request succeeds, 4th hits quota
    now[0] += 11.0
    assert engine.check_and_consume("t2").allowed is True
    daily_reject = engine.check_and_consume("t2")
    assert daily_reject.allowed is False
    assert daily_reject.reason == "daily_quota_exceeded"

    # Advance to next UTC day -> daily quota resets
    now[0] += 86400.0
    assert engine.check_and_consume("t2").allowed is True


def test_abuse_quarantine_and_unblock() -> None:
    """Repeated rejections should quarantine a tenant until unblocked or expired."""
    now = [5000.0]
    policies = {
        TenantTier.FREE: QuotaPolicy(
            tier=TenantTier.FREE,
            requests_per_second=1.0,
            burst_capacity=1,
            window_seconds=60,
            max_requests_per_window=100,
            daily_quota=1000,
        ),
        TenantTier.PRO: DEFAULT_TIER_POLICIES[TenantTier.PRO],
        TenantTier.ENTERPRISE: DEFAULT_TIER_POLICIES[TenantTier.ENTERPRISE],
    }
    engine = RateLimiterEngine(
        policies=policies,
        abuse_config=AbuseGuardrailConfig(
            rejection_threshold=3,
            quarantine_seconds=120.0,
            violation_window_seconds=60.0,
        ),
        time_fn=lambda: now[0],
    )

    assert engine.check_and_consume("bad-tenant").allowed is True
    # 3 consecutive rejections trigger quarantine
    assert engine.check_and_consume("bad-tenant").allowed is False
    assert engine.check_and_consume("bad-tenant").allowed is False
    third = engine.check_and_consume("bad-tenant")
    assert third.allowed is False
    assert third.blocked is True

    # Subsequent request is blocked with quarantine reason
    quarantined = engine.check_and_consume("bad-tenant")
    assert quarantined.allowed is False
    assert quarantined.blocked is True
    assert quarantined.reason.startswith("quarantined:")

    # Admin unblock + tier upgrade restores access
    assert engine.unblock_tenant("bad-tenant") is True
    engine.set_tenant_tier("bad-tenant", TenantTier.PRO)
    usage = engine.get_tenant_usage("bad-tenant")
    assert usage["tier"] == "pro"
    assert usage["blocked"] is False
    assert engine.check_and_consume("bad-tenant").allowed is True


def test_distributed_state_export_import_and_persistence(tmp_path: Path) -> None:
    """RateLimiterEngine should export, merge, save, and load state snapshots."""
    now = [8000.0]
    node_a = RateLimiterEngine(enabled=True, time_fn=lambda: now[0])
    node_b = RateLimiterEngine(enabled=True, time_fn=lambda: now[0])

    node_a.set_tenant_tier("shared", TenantTier.PRO)
    assert node_a.check_and_consume("shared").allowed is True
    assert node_a.check_and_consume("shared").allowed is True

    snapshot_path = tmp_path / "ratelimit_state.json"
    node_a.save_state(snapshot_path)
    assert node_b.load_state(snapshot_path, merge=True) is True

    stats_b = node_b.get_stats()
    assert stats_b["requests_allowed_total"] == 2
    usage_b = node_b.get_tenant_usage("shared")
    assert usage_b["tier"] == "pro"
    assert usage_b["daily_used"] == 2
