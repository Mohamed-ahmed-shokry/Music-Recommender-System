"""Tests for federated privacy, DP budgets, and consent ledger."""

from __future__ import annotations

import pytest

from music_recommender.privacy import (
    ConsentRecord,
    ConsentScope,
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
