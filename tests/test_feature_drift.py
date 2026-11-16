"""Tests for feature store drift monitoring, embedding staleness, and retraining."""

from __future__ import annotations

import pytest

from music_recommender.feature_drift import (
    DriftMonitorConfig,
    DriftSeverity,
    FeatureBaseline,
    RetrainTriggerDecision,
)


def test_drift_config_and_baseline_roundtrip() -> None:
    """DriftMonitorConfig and FeatureBaseline should validate and roundtrip JSON."""
    cfg = DriftMonitorConfig(
        psi_warning_threshold=0.1,
        psi_critical_threshold=0.25,
        z_shift_threshold=2.0,
        embedding_max_age_hours=72.0,
        embedding_min_cosine_retention=0.9,
        retrain_cooldown_seconds=600.0,
        min_samples=10,
    )
    assert DriftMonitorConfig.from_dict(cfg.to_dict()) == cfg

    with pytest.raises(ValueError, match="psi_critical_threshold"):
        DriftMonitorConfig(psi_warning_threshold=0.3, psi_critical_threshold=0.1)
    with pytest.raises(ValueError, match="embedding_min_cosine_retention"):
        DriftMonitorConfig(embedding_min_cosine_retention=1.2)

    baseline = FeatureBaseline(
        name="log_plays",
        mean=2.5,
        std=0.8,
        min_val=0.0,
        max_val=5.0,
        histogram_probs=(0.5, 0.5),
        bin_edges=(0.0, 2.5, 5.0),
        sample_count=100,
    )
    assert FeatureBaseline.from_dict(baseline.to_dict()) == baseline

    decision = RetrainTriggerDecision(
        should_retrain=True,
        severity=DriftSeverity.CRITICAL,
        reasons=["PSI exceeded 0.25"],
        drifted_features=["log_plays"],
        embedding_stale=False,
        cooldown_active=False,
    )
    payload = decision.to_dict()
    assert payload["should_retrain"] is True
    assert payload["severity"] == "critical"
    assert payload["drifted_features"] == ["log_plays"]
