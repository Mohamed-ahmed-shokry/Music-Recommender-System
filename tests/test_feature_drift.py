"""Tests for feature store drift monitoring, embedding staleness, and retraining."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.feature_drift import (
    DriftMonitorConfig,
    DriftSeverity,
    FeatureBaseline,
    FeatureDriftMonitor,
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


def test_psi_and_zshift_drift_detection() -> None:
    """FeatureDriftMonitor should distinguish stable vs drifted distributions."""
    now = [1000.0]
    monitor = FeatureDriftMonitor(
        config=DriftMonitorConfig(min_samples=10),
        time_fn=lambda: now[0],
    )
    baseline_vals = [float(i % 10) for i in range(100)]
    monitor.register_baseline("log_plays", baseline_vals, bins=5)

    # Matching distribution -> no drift
    monitor.record_observations("log_plays", [float(i % 10) for i in range(50)])
    stable = monitor.evaluate_drift()
    assert stable["features"][0]["severity"] == "none"
    assert stable["decision"]["should_retrain"] is False

    # Shifted distribution -> critical drift & retrain recommended
    monitor.acknowledge_retrain()
    now[0] += 4000.0
    monitor.record_observations("log_plays", [25.0 + float(i % 3) for i in range(50)])
    drifted = monitor.evaluate_drift()
    assert drifted["features"][0]["severity"] == "critical"
    assert drifted["max_psi"] > 0.25
    assert drifted["decision"]["should_retrain"] is True
    assert "log_plays" in drifted["decision"]["drifted_features"]


def test_embedding_staleness_and_retrain_cooldown() -> None:
    """Stale embeddings should trigger retrain unless cooldown is active."""
    now = [10_000.0]
    monitor = FeatureDriftMonitor(
        config=DriftMonitorConfig(
            embedding_max_age_hours=24.0,
            embedding_min_cosine_retention=0.85,
            retrain_cooldown_seconds=3600.0,
        ),
        time_fn=lambda: now[0],
    )
    monitor.record_embedding_snapshot(updated_at=10_000.0, cosine_retention=0.95)
    fresh = monitor.evaluate_drift()
    assert fresh["embedding"]["stale"] is False
    assert fresh["decision"]["should_retrain"] is False

    # Advance 30h & drop cosine retention -> stale & retrain triggered
    now[0] += 30.0 * 3600.0
    monitor.record_embedding_snapshot(updated_at=10_000.0, cosine_retention=0.72)
    stale = monitor.evaluate_drift()
    assert stale["embedding"]["stale"] is True
    assert stale["decision"]["should_retrain"] is True

    # Acknowledge retrain -> enters cooldown
    monitor.acknowledge_retrain()
    now[0] += 100.0
    monitor.record_embedding_snapshot(updated_at=10_000.0, cosine_retention=0.72)
    during_cooldown = monitor.evaluate_drift()
    assert during_cooldown["embedding"]["stale"] is True
    assert during_cooldown["decision"]["cooldown_active"] is True
    assert during_cooldown["decision"]["should_retrain"] is False


def test_drift_state_persistence(tmp_path: Path) -> None:
    """FeatureDriftMonitor should save and load state snapshots cleanly."""
    monitor_a = FeatureDriftMonitor(config=DriftMonitorConfig(min_samples=5))
    monitor_a.register_baseline("pop_rank", [1.0, 2.0, 3.0, 4.0, 5.0], bins=5)
    monitor_a.record_observations("pop_rank", [1.0, 2.0, 3.0, 4.0, 5.0])
    monitor_a.evaluate_drift()

    snapshot = tmp_path / "drift_state.json"
    monitor_a.save_state(snapshot)

    monitor_b = FeatureDriftMonitor()
    assert monitor_b.load_state(snapshot) is True
    status = monitor_b.get_status()
    assert status["baseline_count"] == 1
    assert status["observation_counts"]["pop_rank"] == 5
