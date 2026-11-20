"""Tests for federated privacy, DP budgets, and consent ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.privacy import (
    ConsentRecord,
    ConsentScope,
    FederatedPrivacyManager,
    PrivacyBudgetConfig,
    PrivacyBudgetState,
)


def test_consent_and_privacy_budget_models() -> None:
    """ConsentRecord, PrivacyBudgetConfig, and PrivacyBudgetState should roundtrip."""
    rec = ConsentRecord(
        user_id="u1",
        granted_scopes={ConsentScope.PERSONALIZATION, ConsentScope.FEDERATED_UPDATES},
    )
    assert rec.has_scope(ConsentScope.FEDERATED_UPDATES) is True
    assert rec.has_scope("analytics") is False
    restored_rec = ConsentRecord.from_dict(rec.to_dict())
    assert restored_rec.granted_scopes == rec.granted_scopes

    with pytest.raises(ValueError, match="user_id"):
        ConsentRecord(user_id="  ")

    cfg = PrivacyBudgetConfig(
        max_epsilon=5.0, default_delta=1e-5, clip_norm=2.0, noise_multiplier=1.2
    )
    assert PrivacyBudgetConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="max_epsilon"):
        PrivacyBudgetConfig(max_epsilon=0.0)
    with pytest.raises(ValueError, match="clip_norm"):
        PrivacyBudgetConfig(clip_norm=-1.0)

    state = PrivacyBudgetState(
        user_id="u1",
        epsilon_spent=2.0,
        delta_spent=1e-5,
        queries_count=2,
        max_epsilon=5.0,
    )
    assert state.remaining_epsilon == pytest.approx(3.0)
    assert state.exhausted is False
    restored_state = PrivacyBudgetState.from_dict(state.to_dict())
    assert restored_state.epsilon_spent == 2.0
    assert restored_state.queries_count == 2


def test_dp_clipping_and_budget_enforcement() -> None:
    """Manager should L2-clip vectors, inject DP noise, and enforce max_epsilon."""
    mgr = FederatedPrivacyManager(
        config=PrivacyBudgetConfig(max_epsilon=1.5, clip_norm=1.0, noise_multiplier=0.0)
    )
    clipped = mgr.clip_and_add_dp_noise([3.0, 4.0], epsilon=1.0, seed=42)
    assert clipped == pytest.approx([0.6, 0.8])

    s1 = mgr.consume_privacy_budget("u1", epsilon=1.0)
    assert s1.remaining_epsilon == pytest.approx(0.5)
    assert mgr.can_spend("u1", 0.5) is True
    assert mgr.can_spend("u1", 0.6) is False

    with pytest.raises(ValueError, match="exhausted"):
        mgr.consume_privacy_budget("u1", epsilon=0.6)


def test_consent_gated_federated_aggregation() -> None:
    """Federated aggregation should accept consented users and reject unconsented."""
    mgr = FederatedPrivacyManager(
        config=PrivacyBudgetConfig(max_epsilon=1.0, clip_norm=2.0, noise_multiplier=0.1)
    )
    mgr.set_consent("u_ok", [ConsentScope.FEDERATED_UPDATES])
    mgr.set_consent("u_analytics", [ConsentScope.ANALYTICS])

    res = mgr.aggregate_federated_updates(
        [
            ("u_ok", [1.0, 0.0]),
            ("u_analytics", [0.0, 1.0]),
            ("u_none", [0.5, 0.5]),
        ],
        epsilon_per_user=0.6,
        seed=7,
    )
    assert res["accepted_users"] == ["u_ok"]
    assert res["rejected_users"] == {
        "u_analytics": "missing_federated_updates_consent",
        "u_none": "missing_federated_updates_consent",
    }
    assert len(res["aggregated_update"]) == 2
    assert mgr.get_budget("u_ok").epsilon_spent == pytest.approx(0.6)

    # Second round exceeds u_ok remaining budget (0.4 < 0.6)
    res2 = mgr.aggregate_federated_updates(
        [("u_ok", [1.0, 0.0])], epsilon_per_user=0.6, seed=8
    )
    assert res2["accepted_count"] == 0
    assert res2["rejected_users"]["u_ok"] == "privacy_budget_exhausted"


def test_privacy_state_persistence(tmp_path: Path) -> None:
    """FederatedPrivacyManager should save and load state snapshots cleanly."""
    mgr_a = FederatedPrivacyManager()
    mgr_a.set_consent(
        "u5", [ConsentScope.PERSONALIZATION, ConsentScope.FEDERATED_UPDATES]
    )
    mgr_a.consume_privacy_budget("u5", epsilon=1.2)

    snapshot = tmp_path / "privacy_state.json"
    mgr_a.save_state(snapshot)

    mgr_b = FederatedPrivacyManager()
    assert mgr_b.load_state(snapshot) is True
    assert mgr_b.has_consent("u5", ConsentScope.FEDERATED_UPDATES) is True
    assert mgr_b.get_budget("u5").epsilon_spent == pytest.approx(1.2)
