"""Tests for bandit state snapshotting, persistence, and drift tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.bandit import (
    DriftSafetyResult,
    DriftSafetyThresholds,
    LinUCBContextualBandit,
    compute_arm_thetas,
    compute_bandit_drift,
    evaluate_drift_safety,
    get_champion_snapshot,
    list_bandit_snapshots,
    load_bandit_snapshot,
    prune_bandit_snapshots,
    restore_bandit_snapshot,
    rollback_bandit_state,
    save_bandit_snapshot,
    snapshot_bandit_state,
    tag_champion_snapshot,
)


def _sample_bandit_state(*, selections: int = 0) -> dict:
    bandit = LinUCBContextualBandit(["popular", "balanced", "long_tail"], context_dim=3)
    if selections > 0:
        bandit.update("popular", [1.0, 0.5, 0.0], 1.0)
        bandit.update("balanced", [0.0, 1.0, 0.5], 0.5)
    return snapshot_bandit_state(bandit)


def test_compute_arm_thetas_zero_state() -> None:
    state = _sample_bandit_state(selections=0)
    thetas = compute_arm_thetas(state)
    assert set(thetas) == {"popular", "balanced", "long_tail"}
    for _arm, theta in thetas.items():
        assert len(theta) == 3
        assert theta == [0.0, 0.0, 0.0]


def test_compute_arm_thetas_after_updates() -> None:
    state = _sample_bandit_state(selections=1)
    thetas = compute_arm_thetas(state)
    # popular was updated with context [1.0, 0.5, 0.0] and reward 1.0
    assert any(val != 0.0 for val in thetas["popular"])
    assert thetas["long_tail"] == [0.0, 0.0, 0.0]


def test_compute_bandit_drift_identical_states() -> None:
    state = _sample_bandit_state(selections=1)
    drift = compute_bandit_drift(state, state)
    assert "summary" in drift
    assert "arms" in drift
    assert drift["summary"]["has_drift"] is False
    assert drift["summary"]["max_l2_drift"] == 0.0
    assert drift["summary"]["dominant_arm_changed"] is False
    for arm_data in drift["arms"].values():
        assert arm_data["l2_drift"] == 0.0
        assert arm_data["delta_selections"] == 0
        assert arm_data["delta_rewards"] == 0.0
        assert arm_data["cosine_similarity"] == 1.0


def test_compute_bandit_drift_detects_shift() -> None:
    state_a = _sample_bandit_state(selections=0)
    state_b = _sample_bandit_state(selections=1)

    drift = compute_bandit_drift(state_a, state_b)
    assert drift["summary"]["has_drift"] is True
    assert drift["summary"]["max_l2_drift"] > 0.0
    assert drift["arms"]["popular"]["delta_selections"] == 1
    assert drift["arms"]["popular"]["delta_rewards"] == 1.0
    assert drift["arms"]["popular"]["l2_drift"] > 0.0
    assert drift["arms"]["long_tail"]["delta_selections"] == 0


def test_compute_bandit_drift_validates_mismatch() -> None:
    bandit_3d = LinUCBContextualBandit(["popular", "balanced"], context_dim=3)
    bandit_2d = LinUCBContextualBandit(["popular", "balanced"], context_dim=2)
    bandit_arms = LinUCBContextualBandit(["popular", "long_tail"], context_dim=3)

    state_3d = snapshot_bandit_state(bandit_3d)
    state_2d = snapshot_bandit_state(bandit_2d)
    state_arms = snapshot_bandit_state(bandit_arms)

    with pytest.raises(ValueError, match="context dimensions do not match"):
        compute_bandit_drift(state_3d, state_2d)

    with pytest.raises(ValueError, match="state arms do not match"):
        compute_bandit_drift(state_3d, state_arms)


def test_save_and_load_bandit_snapshot(tmp_path: Path) -> None:
    state = _sample_bandit_state(selections=1)
    snapshot_path = save_bandit_snapshot(state, snapshot_dir=tmp_path, label="test-run")

    assert snapshot_path.exists()
    assert "test-run" in snapshot_path.name
    assert snapshot_path.suffix == ".json"

    loaded = load_bandit_snapshot(snapshot_path)
    assert loaded["config"]["arms"] == ["popular", "balanced", "long_tail"]
    assert "snapshot" in loaded
    assert loaded["snapshot"]["label"] == "test-run"


def test_list_bandit_snapshots(tmp_path: Path) -> None:
    state = _sample_bandit_state(selections=0)
    save_bandit_snapshot(state, snapshot_dir=tmp_path, label="first")
    save_bandit_snapshot(state, snapshot_dir=tmp_path, label="second")

    # non-matching file should be ignored
    (tmp_path / "other_file.json").write_text("{}", encoding="utf-8")

    snapshots = list_bandit_snapshots(tmp_path)
    assert len(snapshots) == 2
    assert all("filename" in s for s in snapshots)
    assert all("label" in s for s in snapshots)
    labels = {s["label"] for s in snapshots}
    assert labels == {"first", "second"}


def test_list_bandit_snapshots_empty_or_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing_dir"
    assert list_bandit_snapshots(missing_dir) == []


def test_restore_bandit_snapshot(tmp_path: Path) -> None:
    state = _sample_bandit_state(selections=1)
    snapshot_path = save_bandit_snapshot(
        state, snapshot_dir=tmp_path, label="restore-src"
    )

    target_state_path = tmp_path / "active" / "bandit_state.json"
    restored_path = restore_bandit_snapshot(
        snapshot_path, target_state_path=target_state_path
    )

    assert restored_path.exists()
    restored_state = load_bandit_snapshot(restored_path)
    assert restored_state["arms"]["popular"]["selections"] == 1


def test_prune_bandit_snapshots(tmp_path: Path) -> None:
    state = _sample_bandit_state(selections=0)
    # create 5 snapshots
    paths = []
    for i in range(5):
        path = save_bandit_snapshot(state, snapshot_dir=tmp_path, label=f"snap_{i}")
        paths.append(path)

    with pytest.raises(ValueError, match="max_keep must be a positive integer"):
        prune_bandit_snapshots(tmp_path, max_keep=0)

    # prune keeping 3 newest
    deleted = prune_bandit_snapshots(tmp_path, max_keep=3)
    assert len(deleted) == 2
    for p in deleted:
        assert not p.exists()

    remaining = list_bandit_snapshots(tmp_path)
    assert len(remaining) == 3


def test_drift_safety_thresholds_validation_and_serialization() -> None:
    thresh = DriftSafetyThresholds(
        max_l2_drift=1.2,
        min_cosine_similarity=0.3,
        max_reward_drop=0.2,
        allow_dominant_arm_change=False,
    )
    d = thresh.to_dict()
    assert d["max_l2_drift"] == 1.2
    assert d["min_cosine_similarity"] == 0.3
    assert d["max_reward_drop"] == 0.2
    assert d["allow_dominant_arm_change"] is False

    restored = DriftSafetyThresholds.from_dict(d)
    assert restored == thresh

    with pytest.raises(ValueError, match="max_l2_drift must be a non-negative"):
        DriftSafetyThresholds(max_l2_drift=-0.1)

    with pytest.raises(ValueError, match="min_cosine_similarity must be between"):
        DriftSafetyThresholds(min_cosine_similarity=1.5)

    with pytest.raises(ValueError, match="max_reward_drop must be a non-negative"):
        DriftSafetyThresholds(max_reward_drop=-0.5)


def test_evaluate_drift_safety_identical_states() -> None:
    state = _sample_bandit_state(selections=1)
    result = evaluate_drift_safety(state, state)
    assert isinstance(result, DriftSafetyResult)
    assert result.is_safe is True
    assert len(result.violations) == 0
    assert result.metrics["max_l2_drift"] == 0.0
    assert result.metrics["min_cosine_similarity"] == 1.0

    d = result.to_dict()
    assert d["is_safe"] is True
    assert "metrics" in d
    assert "evaluated_at" in d


def test_evaluate_drift_safety_max_l2_breach() -> None:
    state_a = _sample_bandit_state(selections=0)
    state_b = _sample_bandit_state(selections=1)
    # Drift has max_l2_drift > 0. Set threshold very low
    thresh = DriftSafetyThresholds(max_l2_drift=0.001)
    result = evaluate_drift_safety(state_b, state_a, thresholds=thresh)
    assert result.is_safe is False
    assert any("Maximum L2 parameter drift" in v for v in result.violations)


def test_evaluate_drift_safety_cosine_similarity_breach() -> None:
    # Construct a synthetic drift summary with low cosine similarity
    synthetic_drift = {
        "summary": {
            "max_l2_drift": 0.1,
            "mean_l2_drift": 0.05,
            "dominant_arm_a": "popular",
            "dominant_arm_b": "popular",
            "dominant_arm_changed": False,
        },
        "arms": {
            "popular": {
                "cosine_similarity": -0.5,
                "delta_mean_reward": 0.0,
            }
        },
    }
    thresh = DriftSafetyThresholds(min_cosine_similarity=0.0)
    result = evaluate_drift_safety(synthetic_drift, thresholds=thresh)
    assert result.is_safe is False
    assert any("cosine similarity" in v for v in result.violations)


def test_evaluate_drift_safety_reward_drop_breach() -> None:
    synthetic_drift = {
        "summary": {
            "max_l2_drift": 0.1,
            "mean_l2_drift": 0.05,
            "dominant_arm_a": "popular",
            "dominant_arm_b": "popular",
            "dominant_arm_changed": False,
        },
        "arms": {
            "popular": {
                "cosine_similarity": 0.9,
                "delta_mean_reward": -0.6,
            }
        },
    }
    thresh = DriftSafetyThresholds(max_reward_drop=0.2)
    result = evaluate_drift_safety(synthetic_drift, thresholds=thresh)
    assert result.is_safe is False
    assert any("mean reward drop" in v for v in result.violations)


def test_evaluate_drift_safety_dominant_arm_changed() -> None:
    synthetic_drift = {
        "summary": {
            "max_l2_drift": 0.1,
            "mean_l2_drift": 0.05,
            "dominant_arm_a": "popular",
            "dominant_arm_b": "long_tail",
            "dominant_arm_changed": True,
        },
        "arms": {
            "popular": {
                "cosine_similarity": 0.9,
                "delta_mean_reward": 0.0,
            },
            "long_tail": {
                "cosine_similarity": 0.9,
                "delta_mean_reward": 0.1,
            },
        },
    }
    # When allow_dominant_arm_change is False, flag violation
    thresh = DriftSafetyThresholds(allow_dominant_arm_change=False)
    result = evaluate_drift_safety(synthetic_drift, thresholds=thresh)
    assert result.is_safe is False
    assert any("Dominant arm changed" in v for v in result.violations)


def test_tag_champion_snapshot_and_get_champion(tmp_path: Path) -> None:
    state = _sample_bandit_state(selections=1)
    snap1 = save_bandit_snapshot(state, snapshot_dir=tmp_path, label="snap_1")
    snap2 = save_bandit_snapshot(state, snapshot_dir=tmp_path, label="snap_2")

    # Before tagging, get_champion returns the newest snapshot
    champion_initial = get_champion_snapshot(tmp_path)
    assert champion_initial is not None
    assert champion_initial.name == snap2.name

    # Tag snap1 as champion
    tagged = tag_champion_snapshot(snap1, snapshot_dir=tmp_path)
    assert tagged.resolve() == snap1.resolve()

    # Now get_champion resolves to snap1
    resolved_champion = get_champion_snapshot(tmp_path)
    assert resolved_champion is not None
    assert resolved_champion.name == snap1.name

    # list_bandit_snapshots shows is_champion = True for snap1
    snaps = list_bandit_snapshots(tmp_path)
    champion_entries = [s for s in snaps if s.get("is_champion")]
    assert len(champion_entries) == 1
    assert champion_entries[0]["filename"] == snap1.name


def test_prune_bandit_snapshots_protects_champion(tmp_path: Path) -> None:
    state = _sample_bandit_state(selections=0)
    snaps = [
        save_bandit_snapshot(state, snapshot_dir=tmp_path, label=f"snap_{i}")
        for i in range(5)
    ]
    # Tag the oldest snapshot as champion
    tag_champion_snapshot(snaps[0], snapshot_dir=tmp_path)

    # Prune keeping 2 newest
    deleted = prune_bandit_snapshots(tmp_path, max_keep=2)
    # Out of 3 excess snapshots, snaps[0] should be spared because it is champion
    assert snaps[0].exists()
    assert snaps[0] not in deleted
    remaining = list_bandit_snapshots(tmp_path)
    assert any(s["filename"] == snaps[0].name and s["is_champion"] for s in remaining)


def test_rollback_bandit_state_to_champion(tmp_path: Path) -> None:
    state_champ = _sample_bandit_state(selections=1)
    snap_champ = save_bandit_snapshot(
        state_champ, snapshot_dir=tmp_path, label="champion"
    )
    tag_champion_snapshot(snap_champ, snapshot_dir=tmp_path)

    # Create active state that has drifted/degraded
    active_state_path = tmp_path / "active_bandit_state.json"
    degraded_state = _sample_bandit_state(selections=0)
    restore_bandit_snapshot(
        save_bandit_snapshot(degraded_state, snapshot_dir=tmp_path, label="bad"),
        target_state_path=active_state_path,
    )
    loaded_active = load_bandit_snapshot(active_state_path)
    assert loaded_active["arms"]["popular"]["selections"] == 0

    # Roll back to champion
    rollback_result = rollback_bandit_state(
        target_state_path=active_state_path,
        snapshot_dir=tmp_path,
    )
    assert rollback_result["restored"] is True
    assert rollback_result["is_champion"] is True
    assert rollback_result["arm_selections"]["popular"] == 1

    restored = load_bandit_snapshot(active_state_path)
    assert restored["arms"]["popular"]["selections"] == 1


def test_rollback_bandit_state_explicit_path(tmp_path: Path) -> None:
    state = _sample_bandit_state(selections=1)
    snap = save_bandit_snapshot(state, snapshot_dir=tmp_path, label="specific_target")

    active_state_path = tmp_path / "active.json"
    res = rollback_bandit_state(
        target_state_path=active_state_path,
        snapshot_path=snap,
        snapshot_dir=tmp_path,
    )
    assert res["restored"] is True
    assert active_state_path.exists()
    assert res["label"] == "specific_target"


def test_rollback_bandit_state_missing_snapshot_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No snapshot found"):
        rollback_bandit_state(
            target_state_path=tmp_path / "active.json",
            snapshot_dir=tmp_path / "empty_dir",
        )

