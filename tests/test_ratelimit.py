"""Tests for distributed rate limiting, token-bucket throttling, and tenant quotas."""

from __future__ import annotations

from music_recommender.ratelimit import (
    DEFAULT_TIER_POLICIES,
    QuotaPolicy,
    RateLimitAlgorithm,
    RateLimitDecision,
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
