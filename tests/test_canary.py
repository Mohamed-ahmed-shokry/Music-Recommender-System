"""Tests for shadow deployment, canary traffic splitting, and A/B guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.canary import (
    ABGuardrailConfig,
    CanaryPolicy,
    DeploymentMode,
    RoutingDecision,
    ShadowCanaryRouter,
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


def test_sticky_bucketing_and_shadow_comparisons() -> None:
    """Router should route deterministically and record shadow comparisons."""
    router = ShadowCanaryRouter(
        policy=CanaryPolicy(
            canary_weight=0.25,
            shadow_enabled=True,
            shadow_sample_rate=1.0,
            champion_version="v1",
            candidate_version="v2",
            mode=DeploymentMode.CANARY,
        )
    )

    first = router.route("listener_101")
    second = router.route("listener_101")
    assert first.arm == second.arm
    assert first.bucket_score == second.bucket_score

    canary_count = sum(
        1 for i in range(400) if router.route(f"user_{i}").arm == "canary"
    )
    assert 60 <= canary_count <= 140

    comp = router.record_shadow_comparison(
        entity_id="listener_101",
        champion_items=["a1", "a2", "a3"],
        candidate_items=["a2", "a3", "a4"],
        champion_latency_ms=12.0,
        candidate_latency_ms=15.0,
    )
    assert comp["jaccard_overlap"] == 0.5
    assert comp["latency_delta_ms"] == 3.0

    status = router.get_status()
    assert status["shadow_summary"]["sample_count"] == 1
    assert status["shadow_summary"]["mean_jaccard_overlap"] == 0.5


def test_guardrail_evaluation_and_auto_rollback() -> None:
    """Guardrails should detect insufficient data, promote_ready, and auto-rollback."""
    router = ShadowCanaryRouter(
        policy=CanaryPolicy(
            canary_weight=0.2,
            mode=DeploymentMode.CANARY,
            champion_version="v1",
            candidate_version="v2",
        ),
        guardrails=ABGuardrailConfig(
            min_samples_per_arm=20,
            max_error_rate_delta=0.05,
            max_latency_p95_ratio=1.5,
            max_latency_ms=200.0,
            min_ctr_lift_z=-1.96,
        ),
    )

    # Insufficient data
    insuff = router.evaluate_guardrails()
    assert insuff.status == "insufficient_data"
    assert insuff.should_rollback is False

    # Healthy / promote_ready when canary outperforms champion
    for _ in range(30):
        router.record_outcome(
            "champion", latency_ms=20.0, success=True, converted=False
        )
        router.record_outcome("canary", latency_ms=22.0, success=True, converted=True)
    promote_eval = router.evaluate_guardrails()
    assert promote_eval.status == "promote_ready"
    assert promote_eval.should_rollback is False
    assert promote_eval.z_score > 1.96

    # Trigger rollback on error-rate spike
    for _ in range(20):
        router.record_outcome(
            "canary", latency_ms=250.0, success=False, converted=False
        )
    rb_eval = router.evaluate_guardrails(auto_rollback=True)
    assert rb_eval.status == "rollback_triggered"
    assert rb_eval.should_rollback is True
    assert router.policy.mode == DeploymentMode.ROLLED_BACK
    assert router.policy.canary_weight == 0.0
    assert router.route("any_user").arm == "champion"


def test_canary_state_persistence(tmp_path: Path) -> None:
    """ShadowCanaryRouter should export, save, load, and restore state."""
    router_a = ShadowCanaryRouter()
    router_a.update_policy(
        canary_weight=0.3,
        shadow_enabled=True,
        shadow_sample_rate=0.5,
        candidate_version="v3",
    )
    router_a.record_outcome("champion", latency_ms=10.0, success=True, converted=True)
    router_a.record_outcome("canary", latency_ms=11.0, success=True, converted=False)

    snapshot = tmp_path / "canary_state.json"
    router_a.save_state(snapshot)

    router_b = ShadowCanaryRouter()
    assert router_b.load_state(snapshot) is True
    assert router_b.policy.canary_weight == 0.3
    assert router_b.policy.candidate_version == "v3"
    assert router_b.get_status()["arms"]["champion"]["requests"] == 1
    assert router_b.get_status()["arms"]["canary"]["requests"] == 1
