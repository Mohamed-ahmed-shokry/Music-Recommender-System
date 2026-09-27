"""Tests for bandit state snapshotting, persistence, and drift tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.bandit import (
    LinUCBContextualBandit,
    compute_arm_thetas,
    compute_bandit_drift,
    list_bandit_snapshots,
    load_bandit_snapshot,
    prune_bandit_snapshots,
    restore_bandit_snapshot,
    save_bandit_snapshot,
    snapshot_bandit_state,
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
