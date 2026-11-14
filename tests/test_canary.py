"""Tests for shadow deployment, canary traffic splitting, and A/B guardrails."""

from __future__ import annotations

import pytest

from music_recommender.canary import (
    ABGuardrailConfig,
    CanaryPolicy,
    DeploymentMode,
    RoutingDecision,
)


def test_canary_policy_and_guardrail_config_validation() -> None:
    """Policies and guardrail configs should validate bounds and roundtrip JSON."""
    policy = CanaryPolicy(
        canary_weight=0.25,
        shadow_enabled=True,
        shadow_sample_rate=0.5,
        champion_version="v1",
        candidate_version="v2",
        mode=DeploymentMode.CANARY,
    )
    restored_policy = CanaryPolicy.from_dict(policy.to_dict())
    assert restored_policy == policy

    with pytest.raises(ValueError, match="canary_weight"):
        CanaryPolicy(canary_weight=1.5)
    with pytest.raises(ValueError, match="shadow_sample_rate"):
        CanaryPolicy(shadow_sample_rate=-0.1)

    guardrail = ABGuardrailConfig(
        min_samples_per_arm=20,
        max_error_rate_delta=0.03,
        max_latency_p95_ratio=1.4,
        max_latency_ms=400.0,
        min_ctr_lift_z=-2.0,
    )
    restored_guardrail = ABGuardrailConfig.from_dict(guardrail.to_dict())
    assert restored_guardrail == guardrail

    with pytest.raises(ValueError, match="min_samples_per_arm"):
        ABGuardrailConfig(min_samples_per_arm=0)
    with pytest.raises(ValueError, match="max_latency_p95_ratio"):
        ABGuardrailConfig(max_latency_p95_ratio=0.9)


def test_routing_decision_serialization() -> None:
    """RoutingDecision should serialize arm, shadow_arm, and mode cleanly."""
    decision = RoutingDecision(
        entity_id="user_42",
        arm="canary",
        shadow_arm=None,
        bucket_score=0.1234567,
        mode=DeploymentMode.CANARY,
        champion_version="v1",
        candidate_version="v2",
    )
    payload = decision.to_dict()
    assert payload["entity_id"] == "user_42"
    assert payload["arm"] == "canary"
    assert payload["bucket_score"] == 0.123457
    assert payload["mode"] == "canary"
