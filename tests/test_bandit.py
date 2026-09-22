"""Tests for the cold-start exploration contextual bandit simulation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from music_recommender import cli
from music_recommender.bandit import (
    DEFAULT_COLD_START_ARMS,
    LinUCBContextualBandit,
    build_cold_start_context,
    derive_cold_start_policy,
    load_bandit_report,
    load_bandit_state,
    load_cold_start_policy,
    rank_cold_start_arm,
    rank_cold_start_bandit,
    simulate_cold_start_exploration,
    snapshot_bandit_state,
    write_bandit_report,
    write_bandit_state,
    write_cold_start_policy,
)
from music_recommender.baselines import popular_artists

runner = CliRunner()


def _interactions_df(n_users: int = 12, n_artists: int = 20) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    rng = np.random.default_rng(0)
    for user_index in range(n_users):
        for artist_index in range(n_artists):
            rows.append(
                {
                    "user_id": f"u{user_index}",
                    "artist_id": f"a{artist_index}",
                    "artist_name": f"artist_{artist_index}",
                    "play_count": int(rng.integers(1, 80)),
                }
            )
    return pd.DataFrame(rows)


def _artist_stats() -> dict[str, dict[str, object]]:
    return {
        artist_id: {
            "artist_id": artist_id,
            "artist_name": f"Artist {artist_id}",
            "total_plays": 900 - int(artist_id[1:]) * 100,
            "listener_count": 50 - int(artist_id[1:]),
            "interaction_count": 60 - int(artist_id[1:]),
            "popularity_rank": int(artist_id[1:]),
        }
        for artist_id in [f"a{i}" for i in range(1, 9)]
    }


class TestLinUCBContextualBandit:
    def test_select_and_update_arms(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "balanced", "long_tail"), 2)
        context = [1.0, 0.5]
        for _ in range(30):
            arm = bandit.select_arm(context)
            bandit.update(arm, context, 0.2)

        assert set(bandit.selections) == set(DEFAULT_COLD_START_ARMS)
        assert sum(bandit.selections.values()) == 30
        assert sum(bandit.rewards.values()) == pytest.approx(6.0, abs=1e-6)
        assert all(value >= 0 for value in bandit.selections.values())

    def test_select_prefers_highest_reward_arm(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "long_tail"), 1)
        for _ in range(100):
            context = [1.0]
            arm = bandit.select_arm(context)
            reward = 0.9 if arm == "popular" else 0.1
            bandit.update(arm, context, reward)

        assert bandit.rewards["popular"] > bandit.rewards["long_tail"]

    def test_updates_are_deterministic(self) -> None:
        first = LinUCBContextualBandit(("popular", "long_tail"), 2)
        second = LinUCBContextualBandit(("popular", "long_tail"), 2)
        context = [0.5, 1.0]
        for reward in [0.1, 0.4, 0.9, 0.7]:
            first.update(first.select_arm(context), context, reward)
            second.update(second.select_arm(context), context, reward)

        assert first.selections == second.selections
        assert first.rewards == second.rewards

    def test_select_and_update_validate_inputs(self) -> None:
        bandit = LinUCBContextualBandit(("popular",), 2)

        with pytest.raises(ValueError, match="2-dimensional"):
            bandit.select_arm([1.0, 0.5, 0.2])

        with pytest.raises(ValueError, match="finite"):
            bandit.update("popular", [1.0, 0.5], float("nan"))

        with pytest.raises(ValueError, match="2-dimensional"):
            bandit.update("popular", [1.0], 0.5)

        with pytest.raises(ValueError, match="positive"):
            LinUCBContextualBandit(("popular",), 2, alpha=-1.0)

        with pytest.raises(ValueError, match="Unknown arm"):
            bandit.update("bad_arm", [1.0, 0.5], 0.5)


class TestRankColdStartArm:
    def test_popular_arm_matches_popular_artists(self) -> None:
        stats = _artist_stats()
        assert rank_cold_start_arm("popular", stats, 5) == popular_artists(stats, 5)

    def test_arm_ordering_and_top_k(self) -> None:
        stats = _artist_stats()
        for arm in DEFAULT_COLD_START_ARMS:
            recs = rank_cold_start_arm(arm, stats, 4)
            assert len(recs) == 4
            assert all("artist_id" in rec for rec in recs)
            assert all(1 <= rec["popularity_rank"] <= 8 for rec in recs)
            assert len({rec["artist_id"] for rec in recs}) == 4

    def test_long_tail_differs_from_popular(self) -> None:
        stats = _artist_stats()
        popular_ids = [r["artist_id"] for r in rank_cold_start_arm("popular", stats, 5)]
        long_tail_ids = [
            r["artist_id"] for r in rank_cold_start_arm("long_tail", stats, 5)
        ]
        assert long_tail_ids != popular_ids

    def test_empty_stats_and_validation(self) -> None:
        assert rank_cold_start_arm("popular", {}, 5) == []
        with pytest.raises(ValueError, match="top_k"):
            rank_cold_start_arm("popular", _artist_stats(), 0)
        with pytest.raises(ValueError, match="Unknown arm"):
            rank_cold_start_arm("mystery", _artist_stats(), 5)


class TestBuildColdStartContext:
    def test_context_features_present(self) -> None:
        context = build_cold_start_context(
            _interactions_df(1, 5).assign(user_id="u0"),
            {"a1": 1, "a2": 2, "a3": 3},
        )
        assert len(context) == 3
        assert context[0] > 0
        assert context[2] >= 0

    def test_empty_frame_returns_zeros(self) -> None:
        context = build_cold_start_context(
            pd.DataFrame(columns=["user_id", "artist_id", "play_count"]),
            {},
            features=("log_plays",),
        )
        assert context == [0.0]


def _run_simulation(**overrides: object) -> dict[str, object]:
    df = _interactions_df()
    params: dict[str, object] = {"top_k": 5, "rounds": 30, "seed": 1}
    params.update(overrides)
    return simulate_cold_start_exploration(df, **params)  # type: ignore[arg-type]


class TestSimulateColdStartExploration:
    def test_report_shape(self) -> None:
        report = _run_simulation(seed=1)
        assert "config" in report
        assert "arms" in report
        assert "summary" in report
        assert "rounds" in report

        summary = report["summary"]
        assert summary["rounds_completed"] == 30
        assert summary["regret"] >= 0
        assert summary["mean_reward"] == pytest.approx(
            summary["cumulative_reward"] / 30, abs=1e-6
        )

    def test_selection_counts_sum_to_rounds(self) -> None:
        report = _run_simulation(seed=1)
        total_selected = sum(
            arm_stats["selections"] for arm_stats in report["arms"].values()
        )
        assert total_selected == report["config"]["rounds"]

    def test_rewards_within_bounds(self) -> None:
        report = _run_simulation(seed=1)
        for round_record in report["rounds"]:
            assert 0.0 <= round_record["reward"] <= 1.0
            assert 0.0 <= round_record["best_reward"] <= 1.0

    def test_determinism_and_seed_sensitivity(self) -> None:
        first = copy.deepcopy(_run_simulation(seed=1))
        second = copy.deepcopy(_run_simulation(seed=1))
        third = copy.deepcopy(_run_simulation(seed=99))

        for report in (first, second, third):
            report.pop("generated_at", None)

        assert first == second
        assert first != third

    def test_all_arms_selected_given_temperature(self) -> None:
        report = _run_simulation(seed=1)
        assert all(arm_stats["selections"] > 0 for arm_stats in report["arms"].values())

    def test_validation_errors(self) -> None:
        df = _interactions_df()
        with pytest.raises(ValueError, match="rounds"):
            simulate_cold_start_exploration(df, top_k=5, rounds=0)
        with pytest.raises(ValueError, match="Unknown arm"):
            simulate_cold_start_exploration(df, top_k=5, rounds=1, arms=("fake",))
        with pytest.raises(ValueError, match="holdout_ratio"):
            simulate_cold_start_exploration(df, top_k=5, rounds=1, holdout_ratio=1.2)


class TestBanditReportIO:
    def test_report_roundtrip(self, tmp_path: Path) -> None:
        report = _run_simulation(seed=1)
        written = write_bandit_report(report, tmp_path, report_name="cold_start")
        assert written.exists()
        assert written.name == "cold_start.json"

        loaded = load_bandit_report(written)
        assert loaded["config"] == report["config"]
        assert loaded["arms"] == report["arms"]
        assert loaded["summary"] == report["summary"]

    def test_default_report_name(self, tmp_path: Path) -> None:
        report = _run_simulation(seed=1)
        written = write_bandit_report(report, tmp_path)
        assert written.name == "bandit_simulation.json"

    def test_load_bandit_report_validates_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_bandit_report(missing)

        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{nope", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_bandit_report(corrupt)

        wrong_schema = tmp_path / "wrong.json"
        wrong_schema.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid bandit simulation report"):
            load_bandit_report(wrong_schema)


class TestDeriveColdStartPolicy:
    def test_weights_track_mean_rewards(self) -> None:
        report = {
            "arms": {
                "popular": {"mean_reward": 0.1},
                "balanced": {"mean_reward": 0.5},
                "long_tail": {"mean_reward": 0.9},
            }
        }
        policy = derive_cold_start_policy(report)
        assert set(policy) == set(DEFAULT_COLD_START_ARMS)
        assert sum(policy.values()) == pytest.approx(1.0, abs=1e-6)
        assert policy["long_tail"] > policy["balanced"] > policy["popular"]

    def test_deterministic_and_temperature_sensitive(self) -> None:
        report = {
            "arms": {
                "popular": {"mean_reward": 0.1},
                "long_tail": {"mean_reward": 0.9},
            }
        }
        first = derive_cold_start_policy(report)
        second = derive_cold_start_policy(report)
        assert first == second

        sharper = derive_cold_start_policy(report, temperature=0.5)
        assert sharper["long_tail"] > first["long_tail"]

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            derive_cold_start_policy({"arms": {}}, temperature=0)
        with pytest.raises(ValueError, match="'arms'"):
            derive_cold_start_policy({"summary": {}})
        with pytest.raises(ValueError, match="Unknown arm"):
            derive_cold_start_policy({"arms": {"fake": {"mean_reward": 0.5}}})
        with pytest.raises(ValueError, match="finite"):
            derive_cold_start_policy(
                {"arms": {"popular": {"mean_reward": float("nan")}}}
            )


class TestRankColdStartBandit:
    def test_pure_popular_policy_matches_popular_artists(self) -> None:
        stats = _artist_stats()
        bandit_recs = rank_cold_start_bandit({"popular": 1.0}, stats, 5)
        baseline_recs = popular_artists(stats, 5)
        assert [r["artist_id"] for r in bandit_recs] == [
            r["artist_id"] for r in baseline_recs
        ]

    def test_blend_includes_long_tail_artists(self) -> None:
        stats = _artist_stats()
        pure_popular = [r["artist_id"] for r in popular_artists(stats, 5)]
        blended = rank_cold_start_bandit({"long_tail": 1.0}, stats, 5)
        assert [r["artist_id"] for r in blended] != pure_popular

    def test_balanced_blend_respects_top_k(self) -> None:
        stats = _artist_stats()
        blended = rank_cold_start_bandit({"popular": 0.5, "long_tail": 0.5}, stats, 4)
        assert len(blended) == 4
        assert len({r["artist_id"] for r in blended}) == 4

    def test_deterministic(self) -> None:
        stats = _artist_stats()
        policy = {"popular": 0.4, "balanced": 0.3, "long_tail": 0.3}
        assert rank_cold_start_bandit(policy, stats, 6) == rank_cold_start_bandit(
            policy, stats, 6
        )

    def test_empty_stats_and_validation(self) -> None:
        assert rank_cold_start_bandit({"popular": 1.0}, {}, 5) == []
        with pytest.raises(ValueError, match="top_k"):
            rank_cold_start_bandit({"popular": 1.0}, _artist_stats(), 0)
        with pytest.raises(ValueError, match="empty"):
            rank_cold_start_bandit({}, _artist_stats(), 5)
        with pytest.raises(ValueError, match="positive"):
            rank_cold_start_bandit(
                {"popular": 0.0, "long_tail": 0.0}, _artist_stats(), 5
            )
        with pytest.raises(ValueError, match="Unknown arm"):
            rank_cold_start_bandit({"mystery": 1.0}, _artist_stats(), 5)


class TestColdStartPolicyIO:
    def test_policy_roundtrip(self, tmp_path: Path) -> None:
        policy = {"popular": 0.1, "balanced": 0.3, "long_tail": 0.6}
        written = write_cold_start_policy(policy, tmp_path, policy_name="learned")
        assert written.exists()
        assert written.name == "learned.json"

        loaded = load_cold_start_policy(written)
        assert loaded == policy

    def test_default_policy_name(self, tmp_path: Path) -> None:
        policy = {"popular": 1.0}
        written = write_cold_start_policy(policy, tmp_path)
        assert written.name == "cold_start_policy.json"

    def test_load_validates_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_cold_start_policy(missing)

        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{nope", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_cold_start_policy(corrupt)

        invalid = tmp_path / "invalid.json"
        invalid.write_text(json.dumps({"popular": -1.0}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-negative"):
            load_cold_start_policy(invalid)

    def test_write_validates_policy(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown arm"):
            write_cold_start_policy({"fake": 1.0}, tmp_path)


def _trained_bandit(context_dim: int = 2) -> LinUCBContextualBandit:
    bandit = LinUCBContextualBandit(DEFAULT_COLD_START_ARMS, context_dim, alpha=0.5)
    for _ in range(40):
        context = [1.0, 0.5] + [0.0] * (context_dim - 2)
        arm = bandit.select_arm(context)
        bandit.update(arm, context, 0.2)
    return bandit


class TestBanditStateSnapshotRestore:
    def test_snapshot_restore_roundtrip(self) -> None:
        bandit = _trained_bandit()
        state = snapshot_bandit_state(bandit)
        restored = LinUCBContextualBandit.from_state(state)

        assert restored.arms == bandit.arms
        assert restored.context_dim == bandit.context_dim
        assert restored.alpha == bandit.alpha
        assert restored.selections == bandit.selections
        assert restored.rewards == bandit.rewards
        for arm in bandit.arms:
            assert np.allclose(restored._a[arm], bandit._a[arm])
            assert np.allclose(restored._b[arm], bandit._b[arm])

    def test_restored_engine_selects_identically(self) -> None:
        bandit = _trained_bandit()
        restored = LinUCBContextualBandit.from_state(snapshot_bandit_state(bandit))
        assert bandit.select_arm([1.0, 0.5]) == restored.select_arm([1.0, 0.5])

    def test_fresh_state_has_identity_statistics(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "long_tail"), 2)
        state = snapshot_bandit_state(bandit)
        for arm in bandit.arms:
            assert state["arms"][arm]["selections"] == 0
            assert state["arms"][arm]["rewards"] == 0.0
            assert np.allclose(state["arms"][arm]["a"], np.eye(2))

    def test_from_state_validation(self) -> None:
        bandit = _trained_bandit()
        state = snapshot_bandit_state(bandit)

        with pytest.raises(ValueError, match="config"):
            LinUCBContextualBandit.from_state({"arms": {}})
        with pytest.raises(ValueError, match="arms"):
            LinUCBContextualBandit.from_state(
                {"config": {"arms": [], "context_dim": 2, "alpha": 0.5}, "arms": {}}
            )
        with pytest.raises(ValueError, match="context_dim"):
            LinUCBContextualBandit.from_state(
                {
                    "config": {
                        "arms": list(bandit.arms),
                        "context_dim": 0,
                        "alpha": 0.5,
                    },
                    "arms": {},
                }
            )
        with pytest.raises(ValueError, match="match the configured arm"):
            mismatch = dict(state)
            mismatch["arms"] = {"popular": state["arms"]["popular"]}
            LinUCBContextualBandit.from_state(mismatch)
        with pytest.raises(ValueError, match="shape"):
            bad_shape = dict(state)
            bad_shape["arms"]["popular"]["a"] = [[1.0, 0.0]]
            LinUCBContextualBandit.from_state(bad_shape)


class TestBanditStateIO:
    def test_state_roundtrip(self, tmp_path: Path) -> None:
        bandit = _trained_bandit()
        state = snapshot_bandit_state(bandit)
        written = write_bandit_state(state, tmp_path, state_name="prior")
        assert written.exists()
        assert written.name == "prior.json"

        loaded = load_bandit_state(written)
        assert loaded["config"] == state["config"]
        assert loaded["arms"] == state["arms"]

    def test_default_state_name(self, tmp_path: Path) -> None:
        bandit = _trained_bandit()
        written = write_bandit_state(snapshot_bandit_state(bandit), tmp_path)
        assert written.name == "bandit_state.json"

    def test_load_state_validates_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_bandit_state(missing)

        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{nope", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_bandit_state(corrupt)

        wrong_schema = tmp_path / "wrong.json"
        wrong_schema.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid bandit state"):
            load_bandit_state(wrong_schema)


class TestCLISimulateBandit:
    def test_cli_simulate_bandit(self) -> None:
        result = runner.invoke(
            cli.app,
            ["simulate-bandit", "--top-k", "5", "--rounds", "20", "--seed", "1"],
        )

        assert result.exit_code == 0
        assert "Cold-start exploration bandit simulation (top_k=5):" in result.output
        assert "Selected" in result.output
        assert "Mean Reward" in result.output
        assert "Bandit simulation report written to:" in result.output

    def test_cli_simulate_bandit_custom_report(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            [
                "simulate-bandit",
                "--top-k",
                "3",
                "--rounds",
                "10",
                "--report-dir",
                str(tmp_path),
                "--report-name",
                "custom_bandit",
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "custom_bandit.json").exists()

    def test_cli_simulate_bandit_handles_missing_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "RAW_DATA_PATH", Path("non_existent_data.csv"))
        result = runner.invoke(
            cli.app, ["simulate-bandit", "--top-k", "5", "--rounds", "5"]
        )
        assert result.exit_code == 1
        assert "Error:" in result.output


def _write_bandit_report(tmp_path: Path) -> Path:
    report = {
        "config": {"rounds": 2},
        "arms": {
            "popular": {"selections": 1, "mean_reward": 0.2},
            "balanced": {"selections": 1, "mean_reward": 0.4},
            "long_tail": {"selections": 1, "mean_reward": 0.6},
        },
        "summary": {"rounds_completed": 2},
        "rounds": [],
    }
    report_path = tmp_path / "bandit_simulation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


class TestCLIBanditPolicy:
    def test_cli_bandit_policy_writes_policy(self, tmp_path: Path) -> None:
        report_path = _write_bandit_report(tmp_path)

        result = runner.invoke(
            cli.app,
            [
                "bandit-policy",
                "--report-path",
                str(report_path),
                "--policy-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "Learned cold-start policy arm weights:" in result.output
        assert (tmp_path / "cold_start_policy.json").exists()
        assert "Cold-start policy written to:" in result.output

    def test_cli_bandit_policy_custom_name(self, tmp_path: Path) -> None:
        report_path = _write_bandit_report(tmp_path)

        result = runner.invoke(
            cli.app,
            [
                "bandit-policy",
                "--report-path",
                str(report_path),
                "--policy-name",
                "learned_serving",
                "--policy-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert (tmp_path / "learned_serving.json").exists()

    def test_cli_bandit_policy_handles_missing_report(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            [
                "bandit-policy",
                "--report-path",
                str(tmp_path / "missing.json"),
                "--policy-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
