"""Tests for the cold-start exploration contextual bandit simulation."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from music_recommender import cli
from music_recommender.bandit import (
    CONTEXT_FEATURE_EXTRACTORS,
    DEFAULT_COLD_START_ARMS,
    DEFAULT_CONTEXT_FEATURES,
    SUPPORTED_CONTEXT_FEATURES,
    BackpressureStrategy,
    BaseContextualBandit,
    EpsilonGreedyContextualBandit,
    FeedbackQueueFullError,
    LinUCBContextualBandit,
    StreamingFeedbackQueue,
    StreamingQueueMetrics,
    ThompsonSamplingContextualBandit,
    append_bandit_feedback,
    append_bandit_feedback_batch,
    build_cold_start_context,
    compare_bandit_simulation_policies,
    compute_off_policy_evaluation,
    derive_cold_start_policy,
    dominant_policy_arm,
    feedback_from_bandit_serve,
    feedback_from_report,
    feedback_records_from_bandit_serve,
    fold_bandit_state,
    load_bandit_comparison_report,
    load_bandit_context_features,
    load_bandit_feedback,
    load_bandit_report,
    load_bandit_state,
    load_cold_start_policy,
    load_ope_report,
    neutral_serve_context,
    pending_feedback_count,
    rank_cold_start_arm,
    rank_cold_start_bandit,
    resolve_context_features,
    save_bandit_context_features,
    simulate_cold_start_exploration,
    snapshot_bandit_state,
    summarize_bandit_lifecycle,
    sweep_bandit_journal,
    validate_context_features,
    validate_serve_context,
    write_bandit_comparison_report,
    write_bandit_report,
    write_bandit_state,
    write_cold_start_policy,
    write_ope_report,
)
from music_recommender.baselines import popular_artists
from music_recommender.config import CONTEXT_FEATURES_ENV_VAR

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

    def test_feature_subset_and_reordering(self) -> None:
        df = _interactions_df(1, 5).assign(user_id="u0")
        ranks = {"a1": 1, "a2": 2, "a3": 3}
        full = build_cold_start_context(df, ranks)
        reordered = build_cold_start_context(
            df, ranks, features=("mean_popularity_rank", "log_plays")
        )
        assert reordered == [full[2], full[0]]

    def test_unknown_feature_raises_with_supported_names(self) -> None:
        df = _interactions_df(1, 5).assign(user_id="u0")
        with pytest.raises(ValueError, match="Unknown context feature"):
            build_cold_start_context(df, {"a1": 1}, features=("log_plays", "magic"))


class TestContextFeatureRegistry:
    def test_registry_ships_default_features(self) -> None:
        assert set(SUPPORTED_CONTEXT_FEATURES) == set(DEFAULT_CONTEXT_FEATURES)
        for name in DEFAULT_CONTEXT_FEATURES:
            assert name in CONTEXT_FEATURE_EXTRACTORS

    def test_validate_context_features_returns_canonical_tuple(self) -> None:
        assert validate_context_features(["log_plays", "mean_popularity_rank"]) == (
            "log_plays",
            "mean_popularity_rank",
        )
        assert validate_context_features(("log_plays",)) == ("log_plays",)

    def test_validate_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            validate_context_features([])

    def test_validate_rejects_unknown_feature(self) -> None:
        with pytest.raises(ValueError, match="Unknown context feature"):
            validate_context_features(["log_plays", "genre_match"])

    def test_validate_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="must not contain duplicates"):
            validate_context_features(["log_plays", "log_plays"])

    def test_resolve_context_features_defaults(self) -> None:
        assert resolve_context_features(None) == DEFAULT_CONTEXT_FEATURES
        assert resolve_context_features() == DEFAULT_CONTEXT_FEATURES

    def test_resolve_context_features_validates_override(self) -> None:
        assert resolve_context_features(("log_plays",)) == ("log_plays",)
        with pytest.raises(ValueError, match="Unknown context feature"):
            resolve_context_features(("nope",))


class TestBanditContextFeaturesConfig:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        save_bandit_context_features(("log_plays", "mean_popularity_rank"), path)
        assert load_bandit_context_features(path) == (
            "log_plays",
            "mean_popularity_rank",
        )

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_bandit_context_features(tmp_path / "missing.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_bandit_context_features(path)

    def test_load_rejects_non_list(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        path.write_text(json.dumps("log_plays"), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON list"):
            load_bandit_context_features(path)

    def test_load_rejects_non_string_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        path.write_text(json.dumps(["log_plays", 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="only feature names"):
            load_bandit_context_features(path)

    def test_load_rejects_unknown_feature(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        save_bandit_context_features(("log_plays",), path)
        path.write_text(json.dumps(["log_plays", "magic"]), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown context feature"):
            load_bandit_context_features(path)

    def test_resolve_prefers_explicit_over_env(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        save_bandit_context_features(("mean_popularity_rank",), path)
        assert resolve_context_features(
            ("log_plays",), env="log_unique_artists", config_path=path
        ) == ("log_plays",)

    def test_resolve_uses_env_when_explicit_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        save_bandit_context_features(("mean_popularity_rank",), path)
        assert resolve_context_features(
            env="log_plays,log_unique_artists", config_path=path
        ) == ("log_plays", "log_unique_artists")

    def test_resolve_uses_persisted_config_when_present(self, tmp_path: Path) -> None:
        path = tmp_path / "features.json"
        save_bandit_context_features(("log_plays",), path)
        assert resolve_context_features(config_path=path) == ("log_plays",)

    def test_resolve_falls_back_to_default(self, tmp_path: Path) -> None:
        assert resolve_context_features(config_path=tmp_path / "missing.json") == (
            DEFAULT_CONTEXT_FEATURES
        )

    def test_resolve_env_empty_string_ignored(self) -> None:
        assert resolve_context_features(env="   ") == DEFAULT_CONTEXT_FEATURES

    def test_resolve_env_unknown_feature_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown context feature"):
            resolve_context_features(env="log_plays,meme")


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

    def test_context_features_subset_used_end_to_end(self) -> None:
        report = _run_simulation(
            seed=1, context_features=("log_plays", "mean_popularity_rank")
        )
        assert report["config"]["context_features"] == [
            "log_plays",
            "mean_popularity_rank",
        ]
        assert all(
            len(round_record["context"]) == 2 for round_record in report["rounds"]
        )

    def test_context_features_validated_upfront(self) -> None:
        with pytest.raises(ValueError, match="Unknown context feature"):
            _run_simulation(context_features=("log_plays", "magic"))


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

    def test_derives_policy_from_bandit_state(self) -> None:
        state = {
            "config": {"arms": ["popular", "balanced", "long_tail"]},
            "arms": {
                "popular": {"selections": 10, "rewards": 1.0},
                "balanced": {"selections": 10, "rewards": 5.0},
                "long_tail": {"selections": 10, "rewards": 9.0},
            },
        }
        policy = derive_cold_start_policy(state)
        assert set(policy) == set(DEFAULT_COLD_START_ARMS)
        assert policy["long_tail"] > policy["balanced"] > policy["popular"]

    def test_temperature_annealing_decay(self) -> None:
        state = {
            "config": {"arms": ["popular", "balanced"]},
            "arms": {
                "popular": {"selections": 50, "rewards": 10.0},
                "balanced": {"selections": 50, "rewards": 40.0},
            },
        }
        fixed = derive_cold_start_policy(state, temperature=1.0)
        annealed = derive_cold_start_policy(
            state, temperature=1.0, temperature_decay=0.05
        )
        assert annealed["balanced"] > fixed["balanced"]

    def test_temperature_annealing_validation(self) -> None:
        report = {"arms": {"popular": {"mean_reward": 0.5}}}
        with pytest.raises(ValueError, match="temperature_decay"):
            derive_cold_start_policy(report, temperature_decay=-0.1)
        with pytest.raises(ValueError, match="min_temperature"):
            derive_cold_start_policy(report, min_temperature=0)


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


class TestBanditStateContextFeatures:
    def test_engine_context_features_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match the number of"):
            LinUCBContextualBandit(
                ("popular",),
                2,
                context_features=(
                    "log_plays",
                    "mean_popularity_rank",
                    "log_unique_artists",
                ),
            )

    def test_snapshot_records_context_features(self) -> None:
        bandit = LinUCBContextualBandit(
            ("popular",),
            2,
            context_features=("log_plays", "mean_popularity_rank"),
        )
        state = snapshot_bandit_state(bandit)
        assert state["config"]["context_features"] == [
            "log_plays",
            "mean_popularity_rank",
        ]

    def test_snapshot_without_features_omits_key(self) -> None:
        state = snapshot_bandit_state(LinUCBContextualBandit(("popular",), 2))
        assert "context_features" not in state["config"]

    def test_from_state_restores_recorded_features(self) -> None:
        bandit = LinUCBContextualBandit(
            ("popular",),
            2,
            context_features=("log_plays", "mean_popularity_rank"),
        )
        restored = LinUCBContextualBandit.from_state(snapshot_bandit_state(bandit))
        assert restored.context_features == ("log_plays", "mean_popularity_rank")

    def test_from_state_rejects_feature_dimension_mismatch(self) -> None:
        state = snapshot_bandit_state(LinUCBContextualBandit(("popular",), 2))
        state["config"]["context_features"] = [
            "log_plays",
            "mean_popularity_rank",
            "log_unique_artists",
        ]
        with pytest.raises(ValueError, match="length must match"):
            LinUCBContextualBandit.from_state(state)

    def test_from_state_rejects_bad_feature_types(self) -> None:
        state = snapshot_bandit_state(LinUCBContextualBandit(("popular",), 2))
        state["config"]["context_features"] = ["log_plays", 3]
        with pytest.raises(ValueError, match="list of feature names"):
            LinUCBContextualBandit.from_state(state)

    def test_from_state_unknown_feature_rejected(self) -> None:
        state = snapshot_bandit_state(LinUCBContextualBandit(("popular",), 2))
        state["config"]["context_features"] = ["log_plays", "magic"]
        with pytest.raises(ValueError, match="Unknown context feature"):
            LinUCBContextualBandit.from_state(state)


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
        assert (
            "Cold-start exploration bandit simulation (policy=linucb, top_k=5):"
            in result.output
        )
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

    def test_cli_simulate_bandit_context_features_flag(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            [
                "simulate-bandit",
                "--top-k",
                "3",
                "--rounds",
                "6",
                "--context-features",
                "log_plays,mean_popularity_rank",
                "--report-dir",
                str(tmp_path),
                "--report-name",
                "feat_report",
            ],
        )
        assert result.exit_code == 0
        assert "Context features: log_plays, mean_popularity_rank" in result.output
        report_path = tmp_path / "feat_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["config"]["context_features"] == [
            "log_plays",
            "mean_popularity_rank",
        ]

    def test_cli_simulate_bandit_unknown_feature_fails(self) -> None:
        result = runner.invoke(
            cli.app,
            [
                "simulate-bandit",
                "--top-k",
                "5",
                "--rounds",
                "5",
                "--context-features",
                "log_plays,magic",
            ],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Unknown context feature" in result.output


def _write_bandit_report(tmp_path: Path) -> Path:
    report = {
        "config": {"rounds": 2},
        "arms": {
            "popular": {"selections": 1, "mean_reward": 0.2},
            "balanced": {"selections": 1, "mean_reward": 0.4},
            "long_tail": {"selections": 1, "mean_reward": 0.6},
        },
        "summary": {"rounds_completed": 2},
        "rounds": [
            {
                "round": 1,
                "arm": "popular",
                "context": [1.0, 0.5, 0.2],
                "reward": 0.2,
            },
            {
                "round": 2,
                "arm": "long_tail",
                "context": [0.2, 0.8, 0.4],
                "reward": 0.8,
            },
        ],
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


class TestFeedbackJournal:
    def test_append_and_load(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        entry = {"context": [1.0, 0.5], "arm": "popular", "reward": 0.5}

        first = append_bandit_feedback(entry, journal)
        assert first == journal
        append_bandit_feedback(
            {"context": [0.2, 0.8], "arm": "long_tail", "reward": 1.0},
            journal,
        )

        records = load_bandit_feedback(journal)
        assert len(records) == 2
        assert records[0]["arm"] == "popular"
        assert records[1]["arm"] == "long_tail"
        assert records[1]["reward"] == 1.0

    def test_append_batch_and_load(self, tmp_path: Path) -> None:
        journal = tmp_path / "nested" / "feedback.json"
        entries = [
            {"context": [1.0, 0.5], "arm": "popular", "reward": 0.5, "user_id": "u1"},
            {"context": [0.2, 0.8], "arm": "long_tail", "reward": 1.0, "user_id": "u2"},
        ]
        res = append_bandit_feedback_batch(entries, journal)
        assert res == journal
        assert journal.exists()

        records = load_bandit_feedback(journal)
        assert len(records) == 2
        assert records[0]["user_id"] == "u1"
        assert records[1]["user_id"] == "u2"

        # Append another batch to existing journal
        more_entries = [
            {"context": [0.0, 0.0], "arm": "balanced", "reward": 0.25},
        ]
        append_bandit_feedback_batch(more_entries, journal)
        updated = load_bandit_feedback(journal)
        assert len(updated) == 3
        assert updated[2]["arm"] == "balanced"

    def test_append_batch_empty_noop(self, tmp_path: Path) -> None:
        journal = tmp_path / "nonexistent.json"
        res = append_bandit_feedback_batch([], journal)
        assert res == journal
        assert not journal.exists()

    def test_append_batch_validates_all_records(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        entries = [
            {"context": [1.0], "arm": "popular", "reward": 1.0},
            {"context": [1.0], "arm": "invalid_arm", "reward": 1.0},
        ]
        with pytest.raises(ValueError, match="Unknown arm"):
            append_bandit_feedback_batch(entries, journal)
        assert not journal.exists()

    def test_append_preserves_optional_fields(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        append_bandit_feedback(
            {
                "context": [1.0],
                "arm": "balanced",
                "reward": 0.0,
                "user_id": "u1",
                "occurred_at": "2024-01-01T00:00:00+00:00",
            },
            journal,
        )
        records = load_bandit_feedback(journal)
        assert records[0]["user_id"] == "u1"
        assert records[0]["occurred_at"] == "2024-01-01T00:00:00+00:00"

    def test_append_validates_record(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        with pytest.raises(ValueError, match="arm"):
            append_bandit_feedback({"context": [1.0], "reward": 0.5}, journal)
        with pytest.raises(ValueError, match="finite"):
            append_bandit_feedback(
                {"context": [1.0], "arm": "popular", "reward": float("nan")},
                journal,
            )
        with pytest.raises(ValueError, match="Unknown arm"):
            append_bandit_feedback(
                {"context": [1.0], "arm": "fake", "reward": 0.5}, journal
            )

    def test_append_rejects_corrupt_journal(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        journal.write_text("{nope", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            append_bandit_feedback(
                {"context": [1.0], "arm": "popular", "reward": 0.5}, journal
            )

    def test_load_rejects_invalid(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        journal.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON list"):
            load_bandit_feedback(journal)

        with pytest.raises(FileNotFoundError, match="not found"):
            load_bandit_feedback(tmp_path / "missing.json")


class TestStreamingFeedbackQueue:
    def test_init_validation(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            StreamingFeedbackQueue(journal, max_queue_size=0, auto_start=False)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            StreamingFeedbackQueue(journal, batch_size=0, auto_start=False)
        with pytest.raises(
            ValueError, match="flush_interval_seconds must be positive"
        ):
            StreamingFeedbackQueue(
                journal, flush_interval_seconds=0.0, auto_start=False
            )
        with pytest.raises(ValueError, match="enqueue_timeout must be non-negative"):
            StreamingFeedbackQueue(journal, enqueue_timeout=-1.0, auto_start=False)
        with pytest.raises(ValueError, match="Invalid backpressure strategy"):
            StreamingFeedbackQueue(journal, backpressure="unknown", auto_start=False)
        with pytest.raises(
            TypeError, match="backpressure must be BackpressureStrategy"
        ):
            StreamingFeedbackQueue(
                journal, backpressure=123, auto_start=False  # type: ignore[arg-type]
            )

    def test_enqueue_validates_record(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(journal) as queue:
            with pytest.raises(ValueError, match="arm"):
                queue.enqueue({"context": [1.0], "reward": 0.5})
            with pytest.raises(ValueError, match="finite"):
                queue.enqueue(
                    {"context": [1.0], "arm": "popular", "reward": float("nan")}
                )

    def test_enqueue_and_batch_flushing(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(
            journal,
            batch_size=3,
            flush_interval_seconds=60.0,
        ) as queue:
            assert queue.is_running
            records = [
                {"context": [1.0, 0.0], "arm": "popular", "reward": 0.5},
                {"context": [0.0, 1.0], "arm": "long_tail", "reward": 1.0},
            ]
            for r in records:
                assert queue.enqueue(r)

            time.sleep(0.05)
            assert not journal.exists()

            assert queue.enqueue(
                {"context": [0.5, 0.5], "arm": "balanced", "reward": 0.75}
            )

            for _ in range(50):
                if journal.exists() and len(load_bandit_feedback(journal)) == 3:
                    break
                time.sleep(0.05)

            flushed = load_bandit_feedback(journal)
            assert len(flushed) == 3
            assert flushed[0]["arm"] == "popular"
            assert flushed[1]["arm"] == "long_tail"
            assert flushed[2]["arm"] == "balanced"

    def test_enqueue_time_based_flushing(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(
            journal,
            batch_size=100,
            flush_interval_seconds=0.1,
        ) as queue:
            queue.enqueue({"context": [1.0], "arm": "popular", "reward": 0.5})
            for _ in range(40):
                if journal.exists() and len(load_bandit_feedback(journal)) == 1:
                    break
                time.sleep(0.05)

            assert journal.exists()
            assert len(load_bandit_feedback(journal)) == 1

    def test_enqueue_batch_and_manual_flush(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(journal, auto_start=False) as queue:
            assert not queue.is_running
            records = [
                {"context": [1.0], "arm": "popular", "reward": 0.1},
                {"context": [2.0], "arm": "long_tail", "reward": 0.2},
                {"context": [3.0], "arm": "balanced", "reward": 0.3},
            ]
            accepted = queue.enqueue_batch(records)
            assert accepted == 3
            assert queue.queue_depth == 3
            assert not journal.exists()

            queue.flush()
            assert queue.queue_depth == 0
            assert journal.exists()
            assert len(load_bandit_feedback(journal)) == 3

    def test_backpressure_drop_oldest(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(
            journal,
            max_queue_size=2,
            backpressure=BackpressureStrategy.DROP_OLDEST,
            auto_start=False,
        ) as queue:
            queue.enqueue(
                {
                    "context": [1.0],
                    "arm": "popular",
                    "reward": 1.0,
                    "user_id": "u1",
                }
            )
            queue.enqueue(
                {
                    "context": [2.0],
                    "arm": "long_tail",
                    "reward": 2.0,
                    "user_id": "u2",
                }
            )
            queue.enqueue(
                {
                    "context": [3.0],
                    "arm": "balanced",
                    "reward": 3.0,
                    "user_id": "u3",
                }
            )

            metrics = queue.get_metrics()
            assert metrics.enqueued_count == 3
            assert metrics.dropped_count == 1
            assert metrics.queue_depth == 2

            queue.flush()
            records = load_bandit_feedback(journal)
            assert len(records) == 2
            assert records[0]["user_id"] == "u2"
            assert records[1]["user_id"] == "u3"

    def test_backpressure_reject(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(
            journal,
            max_queue_size=2,
            backpressure=BackpressureStrategy.REJECT,
            auto_start=False,
        ) as queue:
            assert queue.enqueue({"context": [1.0], "arm": "popular", "reward": 1.0})
            assert queue.enqueue({"context": [2.0], "arm": "long_tail", "reward": 2.0})

            assert not queue.enqueue(
                {"context": [3.0], "arm": "balanced", "reward": 3.0}
            )
            assert queue.get_metrics().dropped_count == 1

            with pytest.raises(FeedbackQueueFullError, match="full"):
                queue.enqueue(
                    {"context": [4.0], "arm": "balanced", "reward": 4.0},
                    raise_on_drop=True,
                )

    def test_backpressure_block_timeout(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(
            journal,
            max_queue_size=1,
            backpressure=BackpressureStrategy.BLOCK,
            enqueue_timeout=0.05,
            auto_start=False,
        ) as queue:
            assert queue.enqueue({"context": [1.0], "arm": "popular", "reward": 1.0})
            assert not queue.enqueue(
                {"context": [2.0], "arm": "long_tail", "reward": 2.0}
            )
            assert queue.get_metrics().dropped_count == 1

            with pytest.raises(FeedbackQueueFullError, match="timeout"):
                queue.enqueue(
                    {"context": [3.0], "arm": "balanced", "reward": 3.0},
                    raise_on_drop=True,
                )

    def test_graceful_close_and_post_close_error(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        queue = StreamingFeedbackQueue(journal, max_queue_size=10, auto_start=True)
        queue.enqueue({"context": [1.0], "arm": "popular", "reward": 0.5})
        queue.close()

        assert queue.is_closed
        assert not queue.is_running
        assert journal.exists()
        assert len(load_bandit_feedback(journal)) == 1

        with pytest.raises(RuntimeError, match="closed"):
            queue.enqueue({"context": [2.0], "arm": "popular", "reward": 0.5})

    def test_metrics_snapshot_to_dict(self, tmp_path: Path) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(journal, auto_start=False) as queue:
            queue.enqueue({"context": [1.0], "arm": "popular", "reward": 0.5})
            queue.flush()
            metrics = queue.get_metrics()
            assert isinstance(metrics, StreamingQueueMetrics)
            data = metrics.to_dict()
            assert data["queue_depth"] == 0
            assert data["enqueued_count"] == 1
            assert data["flushed_count"] == 1
            assert data["dropped_count"] == 0
            assert data["flush_errors"] == 0
            assert data["last_flush_at"] is not None

    def test_concurrent_enqueue_thread_safety(self, tmp_path: Path) -> None:
        import concurrent.futures

        journal = tmp_path / "concurrent_journal.json"
        with StreamingFeedbackQueue(
            journal,
            batch_size=20,
            flush_interval_seconds=0.05,
        ) as queue:
            num_threads = 8
            records_per_thread = 25

            def worker(thread_idx: int) -> None:
                for i in range(records_per_thread):
                    queue.enqueue(
                        {
                            "context": [float(thread_idx), float(i)],
                            "arm": "popular",
                            "reward": 1.0,
                            "user_id": f"t{thread_idx}_r{i}",
                        }
                    )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                futures = [
                    executor.submit(worker, idx) for idx in range(num_threads)
                ]
                concurrent.futures.wait(futures)

            queue.flush()

        records = load_bandit_feedback(journal)
        assert len(records) == num_threads * records_per_thread

    def test_flush_error_handling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        journal = tmp_path / "journal.json"
        with StreamingFeedbackQueue(journal, auto_start=False) as queue:
            queue.enqueue({"context": [1.0], "arm": "popular", "reward": 0.5})

            def failing_append(records: Any, path: Any) -> Any:
                raise OSError("Disk full")

            monkeypatch.setattr(
                "music_recommender.bandit.append_bandit_feedback_batch",
                failing_append,
            )

            queue.flush()
            metrics = queue.get_metrics()
            assert metrics.flush_errors == 1
            assert metrics.dropped_count == 1
            assert metrics.last_error == "Disk full"


class TestFoldBanditState:
    def test_fold_matches_direct_updates(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "long_tail"), 2, alpha=0.5)
        feedback = [
            {"context": [1.0, 0.5], "arm": "popular", "reward": 0.8},
            {"context": [0.1, 0.9], "arm": "long_tail", "reward": 0.2},
            {"context": [1.0, 0.5], "arm": "popular", "reward": 1.0},
        ]
        for record in feedback:
            bandit.update(record["arm"], record["context"], record["reward"])

        fresh = snapshot_bandit_state(
            LinUCBContextualBandit(("popular", "long_tail"), 2, alpha=0.5)
        )
        folded = fold_bandit_state(fresh, feedback)

        assert folded["config"]["alpha"] == 0.5
        for arm in ("popular", "long_tail"):
            assert folded["arms"][arm]["selections"] == bandit.selections[arm]
            assert folded["arms"][arm]["rewards"] == pytest.approx(
                bandit.rewards[arm], abs=1e-9
            )
            assert np.allclose(folded["arms"][arm]["a"], bandit._a[arm])
            assert np.allclose(folded["arms"][arm]["b"], bandit._b[arm])

    def test_fold_accumulates_over_prior_state(self) -> None:
        bandit = _trained_bandit()
        prior = snapshot_bandit_state(bandit)
        feedback = [
            {"context": [1.0, 0.5], "arm": "popular", "reward": 0.5},
            {"context": [0.4, 0.6], "arm": "long_tail", "reward": 1.0},
        ]
        updated = fold_bandit_state(prior, feedback)
        assert updated["arms"]["popular"]["selections"] == (
            prior["arms"]["popular"]["selections"] + 1
        )
        assert updated["arms"]["long_tail"]["selections"] == (
            prior["arms"]["long_tail"]["selections"] + 1
        )
        assert updated["arms"]["long_tail"]["rewards"] == pytest.approx(
            prior["arms"]["long_tail"]["rewards"] + 1.0, abs=1e-9
        )

    def test_fold_empty_feedback_is_identity(self) -> None:
        bandit = _trained_bandit()
        prior = snapshot_bandit_state(bandit)
        updated = fold_bandit_state(prior, [])
        assert updated["arms"] == prior["arms"]

    def test_fold_validation(self) -> None:
        bandit = LinUCBContextualBandit(("popular",), 2, alpha=0.5)
        prior = snapshot_bandit_state(bandit)
        with pytest.raises(ValueError, match="not in the state"):
            fold_bandit_state(
                prior,
                [{"context": [1.0, 0.5], "arm": "long_tail", "reward": 0.5}],
            )
        with pytest.raises(ValueError, match="expected 2"):
            fold_bandit_state(
                prior, [{"context": [1.0], "arm": "popular", "reward": 0.5}]
            )

    def test_fold_preserves_recorded_context_features(self) -> None:
        prior = snapshot_bandit_state(
            LinUCBContextualBandit(
                ("popular",),
                2,
                context_features=("log_plays", "mean_popularity_rank"),
            )
        )
        folded = fold_bandit_state(
            prior, [{"context": [1.0, 0.5], "arm": "popular", "reward": 0.5}]
        )
        assert folded["config"]["context_features"] == [
            "log_plays",
            "mean_popularity_rank",
        ]

    def test_fold_dimension_mismatch_names_recorded_features(self) -> None:
        prior = snapshot_bandit_state(
            LinUCBContextualBandit(
                ("popular",),
                2,
                context_features=("log_plays", "mean_popularity_rank"),
            )
        )
        with pytest.raises(ValueError, match="mean_popularity_rank"):
            fold_bandit_state(
                prior, [{"context": [1.0, 0.5, 0.2], "arm": "popular", "reward": 0.5}]
            )


def _fresh_bandit_state() -> dict[str, object]:
    return snapshot_bandit_state(
        LinUCBContextualBandit(
            DEFAULT_COLD_START_ARMS,
            len(DEFAULT_CONTEXT_FEATURES),
            alpha=0.5,
        )
    )


def _feedback_records(n: int) -> list[dict[str, object]]:
    return [
        {"context": [1.0, 0.5, 0.2], "arm": "popular", "reward": 0.7} for _ in range(n)
    ]


class TestSweepBanditJournal:
    def test_sweep_folds_pending_then_nothing(self) -> None:
        state = _fresh_bandit_state()
        updated, summary = sweep_bandit_journal(state, _feedback_records(3))
        assert summary["folded_count"] == 3
        assert summary["journal_length"] == 3
        assert updated["arms"]["popular"]["selections"] == 3
        assert updated["config"]["journal_fold_offset"] == 3
        assert updated["config"]["journal_folded_at"]

        again, second = sweep_bandit_journal(updated, _feedback_records(3))
        assert second["folded_count"] == 0
        assert again["arms"]["popular"]["selections"] == 3
        assert again["config"]["journal_fold_offset"] == 3

    def test_sweep_folds_only_new_records(self) -> None:
        state = _fresh_bandit_state()
        updated, _ = sweep_bandit_journal(state, _feedback_records(2))
        again, summary = sweep_bandit_journal(updated, _feedback_records(5))
        assert summary["folded_count"] == 3
        assert again["arms"]["popular"]["selections"] == 5

    def test_sweep_is_accumulative_over_prior_state(self) -> None:
        prior = fold_bandit_state(
            _fresh_bandit_state(),
            [{"context": [1.0, 0.5, 0.2], "arm": "long_tail", "reward": 1.0}],
        )
        updated, summary = sweep_bandit_journal(prior, _feedback_records(2))
        assert summary["folded_count"] == 2
        assert (
            updated["arms"]["long_tail"]["selections"]
            == (prior["arms"]["long_tail"]["selections"])
        )
        assert updated["arms"]["popular"]["selections"] == (
            prior["arms"]["popular"]["selections"] + 2
        )

    def test_sweep_resets_offset_when_journal_shrinks(self) -> None:
        state = _fresh_bandit_state()
        updated, _ = sweep_bandit_journal(state, _feedback_records(4))
        assert updated["config"]["journal_fold_offset"] == 4
        shrunken = _feedback_records(2)
        again, summary = sweep_bandit_journal(updated, shrunken)
        assert summary["folded_count"] == 2
        assert again["config"]["journal_fold_offset"] == 2
        assert again["arms"]["popular"]["selections"] == 6

    def test_sweep_rejects_invalid_watermark(self) -> None:
        state = _fresh_bandit_state()
        state["config"]["journal_fold_offset"] = -1
        with pytest.raises(ValueError, match="non-negative int"):
            sweep_bandit_journal(state, _feedback_records(1))


class TestSummarizeBanditLifecycle:
    def test_summary_with_no_files(self) -> None:
        summary = summarize_bandit_lifecycle(
            state=None,
            policy=None,
            feedback=[],
        )
        assert summary["available"] is False
        assert summary["state"] is None
        assert summary["policy"] is None
        assert summary["journal"] == {"length": 0, "pending": 0}
        assert summary["last_fold"] is None

    def test_summary_reports_state_policy_and_journal(self) -> None:
        bandit = _trained_bandit()
        state = snapshot_bandit_state(bandit)
        feedback = [{"context": [1.0, 0.5], "arm": "popular", "reward": 0.8}]
        summary = summarize_bandit_lifecycle(
            state=state,
            policy={"popular": 0.5, "long_tail": 0.5},
            feedback=feedback,
        )
        assert summary["available"] is True
        assert summary["policy"] == {"popular": 0.5, "long_tail": 0.5}
        assert summary["journal"] == {"length": 1, "pending": 1}
        assert summary["state"]["total_selections"] == sum(
            state["arms"][arm]["selections"] for arm in state["arms"]
        )
        for arm, stats in state["arms"].items():
            assert summary["state"]["arms"][arm]["selections"] == stats["selections"]
            assert summary["state"]["arms"][arm]["total_reward"] == pytest.approx(
                stats["rewards"], abs=1e-6
            )
        assert summary["last_fold"] is None

    def test_summary_pending_uses_fold_watermark(self) -> None:
        state = _fresh_bandit_state()
        swept, _ = sweep_bandit_journal(state, _feedback_records(2))
        appended = _feedback_records(3)
        summary = summarize_bandit_lifecycle(
            state=swept,
            policy={"popular": 1.0},
            feedback=appended,
        )
        assert summary["journal"] == {"length": 3, "pending": 1}
        assert summary["last_fold"]["offset"] == 2
        assert summary["last_fold"]["at"]

    def test_pending_feedback_count_without_state(self) -> None:
        assert pending_feedback_count(None, [{"arm": "popular"}]) == 1
        assert pending_feedback_count(None, []) == 0


class TestFeedbackFromReport:
    def test_extracts_rounds_with_context(self) -> None:
        report = {
            "rounds": [
                {
                    "arm": "popular",
                    "reward": 0.5,
                    "context": [1.0, 2.0],
                    "user_id": "u1",
                },
                {"arm": "long_tail", "reward": 1.0, "context": [0.5, 0.5]},
            ]
        }
        feedback = feedback_from_report(report)
        assert len(feedback) == 2
        assert feedback[0]["arm"] == "popular"
        assert feedback[0]["user_id"] == "u1"
        assert feedback[1]["reward"] == 1.0

    def test_requires_context_in_rounds(self) -> None:
        report = {"rounds": [{"arm": "popular", "reward": 0.5}]}
        with pytest.raises(ValueError, match="missing 'context'"):
            feedback_from_report(report)
        with pytest.raises(ValueError, match="'rounds' list"):
            feedback_from_report({"summary": {}})


class TestSimulateFromPrior:
    def test_prior_augments_arm_stats(self) -> None:
        df = _interactions_df()
        first = simulate_cold_start_exploration(df, top_k=5, rounds=30, seed=1)
        bandit = _trained_bandit(context_dim=len(DEFAULT_CONTEXT_FEATURES))
        state = snapshot_bandit_state(bandit)

        seeded = simulate_cold_start_exploration(
            df,
            top_k=5,
            rounds=30,
            seed=1,
            arms=DEFAULT_COLD_START_ARMS,
            alpha=0.5,
            initial_state=state,
        )
        for arm in DEFAULT_COLD_START_ARMS:
            assert seeded["arms"][arm]["selections"] >= state["arms"][arm]["selections"]
        assert seeded["config"]["prior"]["selections"] == sum(
            state["arms"][arm]["selections"] for arm in DEFAULT_COLD_START_ARMS
        )
        assert seeded["config"]["prior"] != first["config"]["prior"]

    def test_round_records_include_context(self) -> None:
        report = _run_simulation(seed=1)
        for round_record in report["rounds"]:
            assert len(round_record["context"]) == len(DEFAULT_CONTEXT_FEATURES)
            assert all(np.isfinite(value) for value in round_record["context"])

    def test_seeded_simulation_is_deterministic(self) -> None:
        df = _interactions_df()
        state = snapshot_bandit_state(
            _trained_bandit(context_dim=len(DEFAULT_CONTEXT_FEATURES))
        )
        first = simulate_cold_start_exploration(
            df, top_k=5, rounds=30, seed=1, alpha=0.5, initial_state=state
        )
        second = simulate_cold_start_exploration(
            df, top_k=5, rounds=30, seed=1, alpha=0.5, initial_state=state
        )
        first.pop("generated_at", None)
        second.pop("generated_at", None)
        assert first == second

    def test_prior_state_mismatch_validation(self) -> None:
        df = _interactions_df()
        state = snapshot_bandit_state(_trained_bandit())
        with pytest.raises(ValueError, match="arms must match"):
            simulate_cold_start_exploration(
                df,
                top_k=5,
                rounds=30,
                seed=1,
                arms=("popular",),
                initial_state=state,
            )


class TestCLISimulateBanditState:
    def test_cli_simulate_bandit_write_state(self, tmp_path: Path) -> None:
        state_out = tmp_path / "sim_state.json"
        result = runner.invoke(
            cli.app,
            [
                "simulate-bandit",
                "--top-k",
                "5",
                "--rounds",
                "10",
                "--context-features",
                "log_plays,mean_popularity_rank",
                "--report-dir",
                str(tmp_path),
                "--write-state",
                str(state_out),
            ],
        )
        assert result.exit_code == 0
        assert "Bandit state written to:" in result.output
        state = load_bandit_state(state_out)
        assert state["config"]["context_features"] == [
            "log_plays",
            "mean_popularity_rank",
        ]
        assert state["config"]["context_dim"] == 2

    def test_cli_simulate_bandit_from_state_resumes(self, tmp_path: Path) -> None:
        state_path = tmp_path / "prior_state.json"
        prior = snapshot_bandit_state(
            LinUCBContextualBandit(
                DEFAULT_COLD_START_ARMS,
                len(DEFAULT_CONTEXT_FEATURES),
                alpha=0.5,
            )
        )
        write_bandit_state(prior, tmp_path, state_name="prior_state")

        result = runner.invoke(
            cli.app,
            [
                "simulate-bandit",
                "--top-k",
                "5",
                "--rounds",
                "20",
                "--seed",
                "1",
                "--alpha",
                "0.5",
                "--from-state",
                str(state_path),
                "--report-dir",
                str(tmp_path),
                "--report-name",
                "from_prior",
            ],
        )
        assert result.exit_code == 0
        assert "Resumed from a prior state" in result.output

        report = load_bandit_report(tmp_path / "from_prior.json")
        assert report["config"]["prior"]["selections"] == 0


class TestCLIBanditUpdate:
    def test_cli_bandit_update_from_report(self, tmp_path: Path) -> None:
        state_path = tmp_path / "prior_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                )
            ),
            tmp_path,
            state_name="prior_state",
        )
        report_path = _write_bandit_report(tmp_path)
        output = tmp_path / "updated_state.json"

        result = runner.invoke(
            cli.app,
            [
                "bandit-update",
                "--state-path",
                str(state_path),
                "--report-path",
                str(report_path),
                "--journal-path",
                str(tmp_path / "missing_journal.json"),
                "--output-state",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert "Bandit state after folding feedback:" in result.output
        assert output.exists()
        assert "Updated bandit state written to:" in result.output

    def test_cli_bandit_update_from_journal(self, tmp_path: Path) -> None:
        state_path = tmp_path / "prior_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                )
            ),
            tmp_path,
            state_name="prior_state",
        )
        journal = tmp_path / "feedback.json"
        append_bandit_feedback(
            {
                "context": [1.0, 0.5, 0.2],
                "arm": "popular",
                "reward": 0.7,
            },
            journal,
        )

        result = runner.invoke(
            cli.app,
            [
                "bandit-update",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(journal),
            ],
        )
        assert result.exit_code == 0
        updated = load_bandit_state(state_path)
        assert updated["arms"]["popular"]["selections"] == 1

    def test_cli_bandit_update_no_feedback(self, tmp_path: Path) -> None:
        state_path = tmp_path / "prior_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                )
            ),
            tmp_path,
            state_name="prior_state",
        )
        result = runner.invoke(
            cli.app,
            [
                "bandit-update",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(tmp_path / "missing_journal.json"),
            ],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_cli_bandit_update_context_features_mismatch(self, tmp_path: Path) -> None:
        state_path = tmp_path / "prior_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                    context_features=DEFAULT_CONTEXT_FEATURES,
                )
            ),
            tmp_path,
            state_name="prior_state",
        )
        journal = tmp_path / "feedback.json"
        append_bandit_feedback(
            {
                "context": [1.0, 0.5, 0.2],
                "arm": "popular",
                "reward": 0.7,
            },
            journal,
        )
        result = runner.invoke(
            cli.app,
            [
                "bandit-update",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(journal),
                "--context-features",
                "log_plays,mean_popularity_rank",
            ],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "do not match the state's recorded feature set" in result.output

    def test_cli_bandit_update_context_features_match(self, tmp_path: Path) -> None:
        state_path = tmp_path / "prior_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                    context_features=DEFAULT_CONTEXT_FEATURES,
                )
            ),
            tmp_path,
            state_name="prior_state",
        )
        journal = tmp_path / "feedback.json"
        append_bandit_feedback(
            {
                "context": [1.0, 0.5, 0.2],
                "arm": "popular",
                "reward": 0.7,
            },
            journal,
        )
        result = runner.invoke(
            cli.app,
            [
                "bandit-update",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(journal),
                "--context-features",
                ",".join(DEFAULT_CONTEXT_FEATURES),
            ],
        )
        assert result.exit_code == 0
        assert "Bandit state after folding feedback:" in result.output

    def test_cli_bandit_update_from_journal_is_idempotent(self, tmp_path: Path) -> None:
        state_path = tmp_path / "prior_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                )
            ),
            tmp_path,
            state_name="prior_state",
        )
        journal = tmp_path / "feedback.json"
        append_bandit_feedback(
            {
                "context": [1.0, 0.5, 0.2],
                "arm": "popular",
                "reward": 0.7,
            },
            journal,
        )
        args = [
            "bandit-update",
            "--state-path",
            str(state_path),
            "--journal-path",
            str(journal),
        ]

        first = runner.invoke(cli.app, args)
        second = runner.invoke(cli.app, args)

        assert first.exit_code == 0
        assert second.exit_code == 0
        assert "Folded 0 pending journal record(s)" in second.output
        updated = load_bandit_state(state_path)
        assert updated["arms"]["popular"]["selections"] == 1


class TestCLIBanditStatus:
    def test_cli_bandit_status_without_files(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            [
                "bandit-status",
                "--state-path",
                str(tmp_path / "missing_state.json"),
                "--journal-path",
                str(tmp_path / "missing_journal.json"),
                "--policy-path",
                str(tmp_path / "missing_policy.json"),
            ],
        )
        assert result.exit_code == 0
        assert "State: none" in result.output
        assert "Policy: none" in result.output
        assert "Journal: 0 record(s), 0 pending" in result.output
        assert (
            "Context features (3): log_plays, log_unique_artists, mean_popularity_rank"
            in result.output
        )

    def test_cli_bandit_status_reports_recorded_features(self, tmp_path: Path) -> None:
        state_path = tmp_path / "bandit_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    ("popular",),
                    2,
                    alpha=0.5,
                    context_features=("log_plays", "mean_popularity_rank"),
                )
            ),
            tmp_path,
            state_name="bandit_state",
        )
        result = runner.invoke(
            cli.app,
            [
                "bandit-status",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(tmp_path / "missing_journal.json"),
            ],
        )
        assert result.exit_code == 0
        assert "Context features (2): log_plays, mean_popularity_rank" in result.output

    def test_cli_bandit_status_reports_lifecycle(self, tmp_path: Path) -> None:
        state_path = tmp_path / "bandit_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                )
            ),
            tmp_path,
            state_name="bandit_state",
        )
        journal = tmp_path / "feedback.json"
        append_bandit_feedback(
            {
                "context": [1.0, 0.5, 0.2],
                "arm": "popular",
                "reward": 0.7,
            },
            journal,
        )
        write_cold_start_policy(
            {"popular": 0.7, "long_tail": 0.3},
            tmp_path,
            policy_name="policy",
        )

        result = runner.invoke(
            cli.app,
            [
                "bandit-status",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(journal),
                "--policy-path",
                str(tmp_path / "policy.json"),
            ],
        )
        assert result.exit_code == 0
        assert "Cold-start bandit lifecycle:" in result.output
        assert "State: available" in result.output
        assert "popular" in result.output
        assert "long_tail" in result.output
        assert "Journal: 1 record(s), 1 pending" in result.output
        assert "no fold yet" in result.output

    def test_cli_bandit_status_reports_last_fold(self, tmp_path: Path) -> None:
        state_path = tmp_path / "bandit_state.json"
        write_bandit_state(
            snapshot_bandit_state(
                LinUCBContextualBandit(
                    DEFAULT_COLD_START_ARMS,
                    len(DEFAULT_CONTEXT_FEATURES),
                    alpha=0.5,
                )
            ),
            tmp_path,
            state_name="bandit_state",
        )
        journal = tmp_path / "feedback.json"
        append_bandit_feedback(
            {
                "context": [1.0, 0.5, 0.2],
                "arm": "popular",
                "reward": 0.7,
            },
            journal,
        )
        runner.invoke(
            cli.app,
            [
                "bandit-update",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(journal),
            ],
        )
        append_bandit_feedback(
            {
                "context": [1.0, 0.5, 0.2],
                "arm": "long_tail",
                "reward": 0.4,
            },
            journal,
        )

        result = runner.invoke(
            cli.app,
            [
                "bandit-status",
                "--state-path",
                str(state_path),
                "--journal-path",
                str(journal),
            ],
        )
        assert result.exit_code == 0
        assert "Journal: 2 record(s), 1 pending" in result.output
        assert "last fold" in result.output

    def test_cli_bandit_status_corrupt_state(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt_state.json"
        corrupt.write_text("{nope", encoding="utf-8")
        result = runner.invoke(
            cli.app,
            [
                "bandit-status",
                "--state-path",
                str(corrupt),
            ],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output


class TestCLIBanditContext:
    def test_cli_bandit_context_shows_default(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            ["bandit-context", "--config-path", str(tmp_path / "missing.json")],
        )
        assert result.exit_code == 0
        assert "Bandit context features (3): log_plays, log_unique_artists" in (
            result.output
        )
        assert "Source: project default." in result.output

    def test_cli_bandit_context_set_and_reload(self, tmp_path: Path) -> None:
        config = tmp_path / "features.json"
        set_result = runner.invoke(
            cli.app,
            [
                "bandit-context",
                "--set",
                "log_plays,mean_popularity_rank",
                "--config-path",
                str(config),
            ],
        )
        assert set_result.exit_code == 0
        assert "Bandit context features saved to:" in set_result.output
        reloaded = runner.invoke(
            cli.app,
            ["bandit-context", "--config-path", str(config)],
        )
        assert reloaded.exit_code == 0
        assert "Bandit context features (2): log_plays, mean_popularity_rank" in (
            reloaded.output
        )
        assert "Source: persisted config" in reloaded.output

    def test_cli_bandit_context_unknown_feature_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            ["bandit-context", "--set", "log_plays,magic"],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Unknown context feature" in result.output

    def test_cli_bandit_context_env_override(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            ["bandit-context"],
            env={CONTEXT_FEATURES_ENV_VAR: "log_plays"},
        )
        assert result.exit_code == 0
        assert "Bandit context features (1): log_plays" in result.output
        assert "Source: environment variable" in result.output


class TestCLIRecordBanditFeedback:
    def test_cli_record_feedback(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        result = runner.invoke(
            cli.app,
            [
                "record-bandit-feedback",
                "--context",
                "1.0, 0.5, 0.2",
                "--arm",
                "popular",
                "--reward",
                "0.7",
                "--journal-path",
                str(journal),
            ],
        )
        assert result.exit_code == 0
        assert "Recorded reward 0.7000" in result.output
        records = load_bandit_feedback(journal)
        assert len(records) == 1
        assert records[0]["arm"] == "popular"

    def test_cli_record_feedback_invalid_context(self, tmp_path: Path) -> None:
        journal = tmp_path / "feedback.json"
        result = runner.invoke(
            cli.app,
            [
                "record-bandit-feedback",
                "--context",
                "1.0, nope",
                "--arm",
                "popular",
                "--reward",
                "0.5",
                "--journal-path",
                str(journal),
            ],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output


class TestServeFeedback:
    POLICY = {"popular": 0.6, "balanced": 0.3, "long_tail": 0.1}

    def test_dominant_policy_arm(self) -> None:
        assert dominant_policy_arm({"popular": 0.4, "balanced": 0.6}) == "balanced"
        assert dominant_policy_arm({"popular": 0.5, "balanced": 0.5}) == "balanced"

    def test_dominant_policy_arm_validation(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            dominant_policy_arm({})
        with pytest.raises(ValueError, match="non-negative"):
            dominant_policy_arm({"popular": -0.1})
        with pytest.raises(ValueError, match="Unknown arm"):
            dominant_policy_arm({"fake": 1.0})

    def test_neutral_serve_context(self) -> None:
        context = neutral_serve_context()
        assert context == [0.0, 0.0, 0.0]
        assert len(context) == len(DEFAULT_CONTEXT_FEATURES)

    def test_pure_popular_policy_scores_full_fidelity(self) -> None:
        record = feedback_from_bandit_serve(
            {"popular": 1.0},
            _artist_stats(),
            top_k=5,
            user_id="new_user",
        )
        assert record["arm"] == "popular"
        assert record["reward"] == pytest.approx(1.0)
        assert record["context"] == [0.0, 0.0, 0.0]
        assert record["user_id"] == "new_user"

    def test_blended_policy_record_is_valid_and_deterministic(self) -> None:
        first = feedback_from_bandit_serve(
            dict(self.POLICY),
            _artist_stats(),
            top_k=5,
        )
        second = feedback_from_bandit_serve(
            dict(self.POLICY),
            _artist_stats(),
            top_k=5,
        )
        assert first == second
        assert first["arm"] == "popular"
        assert 0.0 < first["reward"] <= 1.0
        assert len(first["context"]) == len(DEFAULT_CONTEXT_FEATURES)

    def test_custom_context_and_user_id(self) -> None:
        record = feedback_from_bandit_serve(
            dict(self.POLICY),
            _artist_stats(),
            top_k=3,
            context=[1.0, 0.5, 2.0],
            user_id="u7",
        )
        assert record["context"] == [1.0, 0.5, 2.0]

    def test_feedback_is_foldable_into_state(self) -> None:
        record = feedback_from_bandit_serve(
            {"popular": 1.0},
            _artist_stats(),
            top_k=5,
        )
        bandit = LinUCBContextualBandit(
            DEFAULT_COLD_START_ARMS,
            len(DEFAULT_CONTEXT_FEATURES),
            alpha=0.5,
        )
        state = snapshot_bandit_state(bandit)
        updated = fold_bandit_state(state, [record])
        assert updated["arms"]["popular"]["selections"] == 1
        bandit.update(record["arm"], record["context"], record["reward"])
        direct = snapshot_bandit_state(bandit)
        assert np.allclose(
            updated["arms"]["popular"]["a"], direct["arms"]["popular"]["a"]
        )


class TestPerArmServeFeedback:
    POLICY = {"popular": 0.6, "balanced": 0.3, "long_tail": 0.1}

    def test_every_policy_arm_gets_a_record(self) -> None:
        records = feedback_records_from_bandit_serve(
            dict(self.POLICY),
            _artist_stats(),
            top_k=5,
        )
        assert [r["arm"] for r in records] == ["balanced", "long_tail", "popular"]

    def test_zero_weight_arms_are_skipped(self) -> None:
        records = feedback_records_from_bandit_serve(
            {"popular": 1.0, "balanced": 0.0},
            _artist_stats(),
            top_k=5,
        )
        assert [r["arm"] for r in records] == ["popular"]

    def test_dominant_record_matches_single_arm_contract(self) -> None:
        batch = feedback_records_from_bandit_serve(
            dict(self.POLICY),
            _artist_stats(),
            top_k=5,
            user_id="u1",
            context=[0.2, 0.4, 0.6],
        )
        single = feedback_from_bandit_serve(
            dict(self.POLICY),
            _artist_stats(),
            top_k=5,
            user_id="u1",
            context=[0.2, 0.4, 0.6],
        )
        assert single in batch
        assert single["arm"] == "popular"

    def test_per_arm_rewards_are_foldable(self) -> None:
        records = feedback_records_from_bandit_serve(
            dict(self.POLICY),
            _artist_stats(),
            top_k=5,
        )
        bandit = LinUCBContextualBandit(
            DEFAULT_COLD_START_ARMS,
            len(DEFAULT_CONTEXT_FEATURES),
            alpha=0.5,
        )
        state = snapshot_bandit_state(bandit)
        updated = fold_bandit_state(state, records)
        assert (
            sum(updated["arms"][record["arm"]]["selections"] for record in records) == 3
        )


class TestValidateServeContext:
    def test_valid_context_normalized(self) -> None:
        assert validate_serve_context([1, 0.5, 2]) == [1.0, 0.5, 2.0]

    def test_empty_context_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_serve_context([])

    def test_wrong_dimension_rejected(self) -> None:
        with pytest.raises(ValueError, match="2 features; expected 3"):
            validate_serve_context([0.0, 0.0])

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            validate_serve_context([0.0, "nope", 2.0])

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            validate_serve_context([float("nan"), 0.0, 0.0])

    def test_records_reject_invalid_context(self) -> None:
        with pytest.raises(ValueError, match="expected 3"):
            feedback_records_from_bandit_serve(
                {"popular": 1.0},
                _artist_stats(),
                top_k=5,
                context=[0.0, 0.0],
            )

    def test_records_respect_custom_features(self) -> None:
        records = feedback_records_from_bandit_serve(
            {"popular": 1.0},
            _artist_stats(),
            top_k=5,
            context=[1.0, 0.5],
            features=("log_plays", "mean_popularity_rank"),
        )
        assert records[0]["context"] == [1.0, 0.5]

    def test_records_use_neutral_context_with_custom_features(self) -> None:
        records = feedback_records_from_bandit_serve(
            {"popular": 1.0},
            _artist_stats(),
            top_k=5,
            features=("log_plays", "mean_popularity_rank"),
        )
        assert records[0]["context"] == [0.0, 0.0]

    def test_records_reject_mismatched_custom_features(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            feedback_records_from_bandit_serve(
                {"popular": 1.0},
                _artist_stats(),
                top_k=5,
                context=[1.0, 0.5, 0.2],
                features=("log_plays", "mean_popularity_rank"),
            )


class TestThompsonSamplingContextualBandit:
    def test_select_and_update_arms(self) -> None:
        bandit = ThompsonSamplingContextualBandit(
            ("popular", "balanced", "long_tail"), 2, seed=42
        )
        context = [1.0, 0.5]
        for _ in range(30):
            arm = bandit.select_arm(context)
            bandit.update(arm, context, 0.2)

        assert set(bandit.selections) == set(DEFAULT_COLD_START_ARMS)
        assert sum(bandit.selections.values()) == 30
        assert sum(bandit.rewards.values()) == pytest.approx(6.0, abs=1e-6)

    def test_reproducible_with_seed(self) -> None:
        b1 = ThompsonSamplingContextualBandit(("popular", "balanced"), 2, seed=123)
        b2 = ThompsonSamplingContextualBandit(("popular", "balanced"), 2, seed=123)
        context = [0.8, -0.2]
        selections1 = [b1.select_arm(context) for _ in range(10)]
        selections2 = [b2.select_arm(context) for _ in range(10)]
        assert selections1 == selections2

    def test_state_serialization_roundtrip(self) -> None:
        bandit = ThompsonSamplingContextualBandit(
            ("popular", "balanced"), 2, alpha=1.5, alpha_decay=0.01, seed=7
        )
        context = [1.0, 2.0]
        bandit.update("popular", context, 1.0)
        state = snapshot_bandit_state(bandit)
        assert state["config"]["policy_type"] == "thompson_sampling"
        assert state["config"]["alpha_decay"] == 0.01

        restored = BaseContextualBandit.from_state(state)
        assert isinstance(restored, ThompsonSamplingContextualBandit)
        assert restored.policy_type == "thompson_sampling"
        assert restored.alpha == 1.5
        assert restored.alpha_decay == 0.01
        assert restored.selections["popular"] == 1


class TestExponentialDiscountAndCooling:
    def test_alpha_decay_cooling(self) -> None:
        bandit = LinUCBContextualBandit(
            ("popular", "balanced"), 2, alpha=2.0, alpha_decay=0.1
        )
        assert bandit.effective_alpha == pytest.approx(2.0)
        bandit.update("popular", [1.0, 0.0], 1.0)
        bandit.update("balanced", [0.0, 1.0], 0.5)
        # total_selections = 2 => effective_alpha = 2.0 / (1.0 + 0.1 * 2) = 1.6666...
        assert bandit.effective_alpha == pytest.approx(2.0 / 1.2)

    def test_gamma_recency_discount_math(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "balanced"), 2)
        context = [1.0, 0.0]
        bandit.update("popular", context, 1.0, gamma=0.5)
        # 1st update with gamma=0.5: A = 0.5*I + xx^T = [[1.5, 0], [0, 0.5]]
        np.testing.assert_allclose(bandit._a["popular"], [[1.5, 0.0], [0.0, 0.5]])
        np.testing.assert_allclose(bandit._b["popular"], [1.0, 0.0])

        bandit.update("popular", context, 2.0, gamma=0.5)
        # 2nd update: A = 0.5*[[1.5, 0], [0, 0.5]] + xx^T = [[1.75, 0], [0, 0.25]]
        # b = 0.5*[1, 0] + 2*[1, 0] = [2.5, 0.0]
        np.testing.assert_allclose(bandit._a["popular"], [[1.75, 0.0], [0.0, 0.25]])
        np.testing.assert_allclose(bandit._b["popular"], [2.5, 0.0])

    def test_invalid_gamma(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "balanced"), 2)
        with pytest.raises(ValueError, match="gamma must be in the range"):
            bandit.update("popular", [1.0, 0.0], 1.0, gamma=0.0)
        with pytest.raises(ValueError, match="gamma must be in the range"):
            bandit.update("popular", [1.0, 0.0], 1.0, gamma=1.5)

    def test_fold_state_with_gamma(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "balanced"), 2)
        state = snapshot_bandit_state(bandit)
        records = [
            {"arm": "popular", "context": [1.0, 0.0], "reward": 1.0},
        ]
        folded = fold_bandit_state(state, records, gamma=0.8)
        np.testing.assert_allclose(
            folded["arms"]["popular"]["a"], [[1.8, 0.0], [0.0, 0.8]]
        )

    def test_sweep_journal_with_gamma(self) -> None:
        bandit = LinUCBContextualBandit(("popular", "balanced"), 2)
        state = snapshot_bandit_state(bandit)
        records = [
            {"arm": "popular", "context": [1.0, 0.0], "reward": 1.0},
        ]
        updated, summary = sweep_bandit_journal(state, records, gamma=0.9)
        assert summary["gamma"] == 0.9
        assert summary["folded_count"] == 1


class TestPhase118Simulation:
    def test_simulation_with_thompson_sampling_and_decay(self) -> None:
        df = _interactions_df()
        res = simulate_cold_start_exploration(
            df,
            5,
            10,
            policy_type="thompson_sampling",
            alpha_decay=0.05,
            gamma=0.9,
            seed=42,
        )
        assert res["config"]["policy_type"] == "thompson_sampling"
        assert res["config"]["alpha_decay"] == 0.05
        assert res["config"]["gamma"] == 0.9
        assert "regret" in res["summary"]


class TestEpsilonGreedyContextualBandit:
    def test_select_and_update_arms(self) -> None:
        bandit = EpsilonGreedyContextualBandit(
            ("popular", "balanced", "long_tail"), 2, alpha=0.2, seed=42
        )
        context = [1.0, 0.5]
        for _ in range(30):
            arm = bandit.select_arm(context)
            bandit.update(arm, context, 0.2)

        assert set(bandit.selections) == set(DEFAULT_COLD_START_ARMS)
        assert sum(bandit.selections.values()) == 30
        assert sum(bandit.rewards.values()) == pytest.approx(6.0, abs=1e-6)

    def test_reproducible_with_seed(self) -> None:
        b1 = EpsilonGreedyContextualBandit(
            ("popular", "balanced"), 2, alpha=0.5, seed=123
        )
        b2 = EpsilonGreedyContextualBandit(
            ("popular", "balanced"), 2, alpha=0.5, seed=123
        )
        context = [0.8, -0.2]
        selections1 = [b1.select_arm(context) for _ in range(15)]
        selections2 = [b2.select_arm(context) for _ in range(15)]
        assert selections1 == selections2

    def test_pure_greedy_selects_highest_point_estimate(self) -> None:
        bandit = EpsilonGreedyContextualBandit(
            ("popular", "balanced"), 1, alpha=0.0000001, seed=42
        )
        for _ in range(5):
            bandit.update("popular", [1.0], 1.0)
            bandit.update("balanced", [1.0], 0.1)

        selections = [bandit.select_arm([1.0]) for _ in range(10)]
        assert selections == ["popular"] * 10

    def test_state_serialization_roundtrip(self) -> None:
        bandit = EpsilonGreedyContextualBandit(
            ("popular", "balanced"), 2, alpha=0.3, alpha_decay=0.01, seed=7
        )
        context = [1.0, 2.0]
        bandit.update("popular", context, 1.0)
        state = snapshot_bandit_state(bandit)
        assert state["config"]["policy_type"] == "epsilon_greedy"
        assert state["config"]["alpha"] == 0.3
        assert state["config"]["alpha_decay"] == 0.01

        restored = BaseContextualBandit.from_state(state)
        assert isinstance(restored, EpsilonGreedyContextualBandit)
        assert restored.policy_type == "epsilon_greedy"
        assert restored.alpha == 0.3
        assert restored.alpha_decay == 0.01
        assert restored.selections["popular"] == 1


class TestPhase124Simulation:
    def test_simulation_with_epsilon_greedy(self) -> None:
        df = _interactions_df()
        res = simulate_cold_start_exploration(
            df,
            5,
            10,
            policy_type="epsilon_greedy",
            alpha=0.2,
            alpha_decay=0.05,
            gamma=0.9,
            seed=42,
        )
        assert res["config"]["policy_type"] == "epsilon_greedy"
        assert res["config"]["alpha"] == 0.2
        assert res["config"]["alpha_decay"] == 0.05
        assert res["config"]["gamma"] == 0.9
        assert "regret" in res["summary"]
        assert res["summary"]["rounds_completed"] == 10


class TestOffPolicyEvaluation:
    def test_compute_ope_with_bandit_policy(self) -> None:
        records = [
            {"context": [1.0, 0.5], "arm": "popular", "reward": 0.8},
            {"context": [0.5, 1.0], "arm": "balanced", "reward": 0.4},
            {"context": [1.2, 0.2], "arm": "popular", "reward": 1.0},
            {"context": [0.1, 0.9], "arm": "long_tail", "reward": 0.2},
        ]
        bandit = LinUCBContextualBandit(("popular", "balanced", "long_tail"), 2)
        ope = compute_off_policy_evaluation(records, bandit)

        assert ope["summary"]["records_evaluated"] == 4
        assert "ips" in ope["metrics"]
        assert "snips" in ope["metrics"]
        assert "direct_method" in ope["metrics"]
        assert "doubly_robust" in ope["metrics"]
        assert ope["metrics"]["ips"]["value"] >= 0.0
        assert ope["summary"]["match_rate"] >= 0.0
        assert ope["summary"]["effective_sample_size"] > 0.0

    def test_compute_ope_with_policy_dict(self) -> None:
        records = [
            {"context": [1.0, 0.5], "arm": "popular", "reward": 0.8},
            {"context": [0.5, 1.0], "arm": "balanced", "reward": 0.4},
        ]
        policy = {"popular": 0.7, "balanced": 0.3}
        ope = compute_off_policy_evaluation(records, policy)
        assert ope["target_policy"] == "custom_policy"
        assert "lift_over_logging" in ope["metrics"]["doubly_robust"]

    def test_compute_ope_with_single_arm(self) -> None:
        records = [
            {"context": [1.0, 0.5], "arm": "popular", "reward": 0.8},
            {"context": [0.5, 1.0], "arm": "balanced", "reward": 0.4},
        ]
        ope = compute_off_policy_evaluation(records, "popular")
        assert ope["target_policy"] == "arm_popular"
        assert ope["arms"]["popular"]["target_action_share"] == 1.0

    def test_compute_ope_with_behavior_propensities(self) -> None:
        records = [
            {"context": [1.0, 0.5], "arm": "popular", "reward": 0.8},
            {"context": [0.5, 1.0], "arm": "balanced", "reward": 0.4},
        ]
        bandit = EpsilonGreedyContextualBandit(("popular", "balanced"), 2, alpha=0.1)
        ope = compute_off_policy_evaluation(
            records,
            bandit,
            behavior_propensities={"popular": 0.5, "balanced": 0.5},
        )
        assert ope["summary"]["records_evaluated"] == 2

    def test_ope_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compute_off_policy_evaluation([], "popular")
        with pytest.raises(ValueError, match="min_propensity"):
            compute_off_policy_evaluation(
                [{"context": [1.0], "arm": "popular", "reward": 1.0}],
                "popular",
                min_propensity=0.0,
            )
        with pytest.raises(ValueError, match="Inconsistent context dimension"):
            compute_off_policy_evaluation(
                [
                    {"context": [1.0, 2.0], "arm": "popular", "reward": 1.0},
                    {"context": [1.0], "arm": "popular", "reward": 0.5},
                ],
                "popular",
            )


class TestMultiPolicyComparison:
    def test_compare_policies_default_suite(self) -> None:
        df = _interactions_df()
        comp = compare_bandit_simulation_policies(df, top_k=5, rounds=10, seed=42)

        assert comp["config"]["rounds"] == 10
        assert set(comp["policies"].keys()) == {
            "linucb",
            "thompson_sampling",
            "epsilon_greedy",
        }
        assert comp["summary"]["champion"] in comp["policies"]
        assert len(comp["summary"]["leaderboard"]) == 3
        for item in comp["summary"]["leaderboard"]:
            assert "mean_reward" in item
            assert "win_rate" in item
            assert "regret" in item

    def test_compare_policies_custom_configs(self) -> None:
        df = _interactions_df()
        comp = compare_bandit_simulation_policies(
            df,
            policies=[
                {"name": "linucb_fast", "policy_type": "linucb", "alpha": 0.5},
                {"name": "linucb_slow", "policy_type": "linucb", "alpha": 2.0},
            ],
            rounds=8,
        )
        assert set(comp["policies"].keys()) == {"linucb_fast", "linucb_slow"}
        assert len(comp["summary"]["leaderboard"]) == 2

    def test_compare_policies_validation(self) -> None:
        df = _interactions_df()
        with pytest.raises(ValueError, match="At least one policy"):
            compare_bandit_simulation_policies(df, policies=[])
        with pytest.raises(ValueError, match="string or dictionary"):
            compare_bandit_simulation_policies(df, policies=[123])  # type: ignore[list-item]


class TestPhase125Reports:
    def test_bandit_comparison_report_roundtrip(self, tmp_path: Path) -> None:
        report = {
            "config": {"rounds": 5},
            "policies": {"linucb": {"mean_reward": 0.4}},
            "summary": {"champion": "linucb"},
        }
        path = write_bandit_comparison_report(report, tmp_path, report_name="test_comp")
        assert path.exists()
        loaded = load_bandit_comparison_report(path)
        assert loaded == report

    def test_ope_report_roundtrip(self, tmp_path: Path) -> None:
        report = {
            "metrics": {"ips": {"value": 0.5}},
            "summary": {"records_evaluated": 10},
        }
        path = write_ope_report(report, tmp_path, report_name="test_ope")
        assert path.exists()
        loaded = load_ope_report(path)
        assert loaded == report

    def test_report_validation_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid bandit comparison report"):
            write_bandit_comparison_report({"invalid": True}, tmp_path)
        with pytest.raises(ValueError, match="Invalid OPE report"):
            write_ope_report({"invalid": True}, tmp_path)
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid bandit comparison report"):
            load_bandit_comparison_report(bad_json)
        with pytest.raises(ValueError, match="not a valid OPE report"):
            load_ope_report(bad_json)
