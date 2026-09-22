"""Contextual bandit simulation and online feedback for cold-start exploration.

Cold-start users have no listening history, so the system currently serves a
static ``popular_artists`` fallback with no exploration. This module simulates
an online contextual bandit that decides which *arm* (cold-start strategy) to
serve per user context, learns from a precision@k-style engagement reward, and
reports cumulative reward and regret against always-popular and best-in-
hindsight policies.

The learned bandit state (ridge-regression statistics ``a``/``b`` plus
selection and reward tallies per arm) can be snapshotted to a persistent
``bandit_state.json``, folded together with new observations from a live
serving feedback journal, and restored as the prior of a future simulation so
the cold-start policy keeps improving across deployments.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from music_recommender.artifacts import build_artist_stats
from music_recommender.baselines import popular_artists
from music_recommender.data import normalize_interactions
from music_recommender.evaluate import precision_at_k
from music_recommender.ranking import validate_ranking_parameters

DEFAULT_COLD_START_ARMS: tuple[str, ...] = ("popular", "balanced", "long_tail")
DEFAULT_CONTEXT_FEATURES = ("log_plays", "log_unique_artists", "mean_popularity_rank")


def _validate_arms(arms: Sequence[str]) -> None:
    if not arms:
        raise ValueError("arms must not be empty.")
    for arm in arms:
        if arm not in DEFAULT_COLD_START_ARMS:
            raise ValueError(
                f"Unknown arm '{arm}'. Expected one of {DEFAULT_COLD_START_ARMS}."
            )


def _arm_full_scores(
    arm: str, artist_stats: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Compute a per-artist raw score for an arm across the whole catalog.

    ``popular`` uses training-set plays; ``balanced`` and ``long_tail`` apply an
    increasingly aggressive popularity penalty so lower-ranked artists can rise.
    """
    if not artist_stats:
        return {}

    max_rank = max(len(artist_stats), 1)
    penalty = {
        "popular": 0.0,
        "balanced": 0.4,
        "long_tail": 0.85,
    }[arm]

    scores: dict[str, float] = {}
    for stats in artist_stats.values():
        artist_id = str(stats["artist_id"])
        popularity_rank = int(stats["popularity_rank"])
        total_plays = float(stats["total_plays"])
        weight = 1.0 if max_rank == 1 else 1 - (popularity_rank - 1) / (max_rank - 1)
        scores[artist_id] = total_plays - penalty * weight * total_plays
    return scores


class LinUCBContextualBandit:
    """Contextual multi-armed bandit using LinUCB (linear upper confidence bound).

    Each arm maintains a ridge regression over context features; at selection
    time the arm with the highest upper confidence bound for the context is
    chosen, balancing exploitation (point estimate) and exploration
    (uncertainty term scaled by ``alpha``).
    """

    def __init__(
        self,
        arms: Sequence[str],
        context_dim: int,
        *,
        alpha: float = 1.0,
    ) -> None:
        _validate_arms(arms)
        if type(context_dim) is not int or context_dim < 1:
            raise ValueError("context_dim must be a positive integer.")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number.")

        self.arms = list(arms)
        self.context_dim = context_dim
        self.alpha = float(alpha)
        self._a = {arm: np.eye(context_dim) for arm in self.arms}
        self._b = {arm: np.zeros(context_dim) for arm in self.arms}
        self.selections = dict.fromkeys(self.arms, 0)
        self.rewards = dict.fromkeys(self.arms, 0.0)

    def _arm_score(self, arm: str, context: np.ndarray) -> float:
        matrix = self._a[arm]
        inv_matrix = np.linalg.inv(matrix)
        theta = inv_matrix @ self._b[arm]
        mean = float(context @ theta)
        uncertainty = self.alpha * float(np.sqrt(context @ inv_matrix @ context))
        return float(np.clip(mean + uncertainty, -1e9, 1e9))

    def select_arm(self, context: Sequence[float]) -> str:
        """Return the arm with the highest UCB for the given context."""
        context_array = np.asarray(context, dtype=float)
        if context_array.ndim != 1 or context_array.size != self.context_dim:
            raise ValueError(
                f"context must be a {self.context_dim}-dimensional vector."
            )
        best_arm = max(
            self.arms,
            key=lambda arm: (self._arm_score(arm, context_array), arm),
        )
        return best_arm

    def update(self, arm: str, context: Sequence[float], reward: float) -> None:
        """Update the selected arm's ridge regression with an observed reward."""
        _validate_arms([arm])
        if not np.isfinite(reward):
            raise ValueError("reward must be finite.")
        context_array = np.asarray(context, dtype=float)
        if context_array.ndim != 1 or context_array.size != self.context_dim:
            raise ValueError(
                f"context must be a {self.context_dim}-dimensional vector."
            )

        self._a[arm] = self._a[arm] + np.outer(context_array, context_array)
        self._b[arm] = self._b[arm] + reward * context_array
        self.selections[arm] += 1
        self.rewards[arm] += float(reward)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LinUCBContextualBandit:
        """Build an engine from a persisted bandit state snapshot."""
        if not isinstance(state, dict) or "config" not in state or "arms" not in state:
            raise ValueError("Bandit state must contain 'config' and 'arms' sections.")

        config = state["config"]
        if not isinstance(config, dict):
            raise ValueError("Bandit state 'config' must be a dictionary.")
        arms = config.get("arms")
        context_dim = config.get("context_dim")
        alpha = config.get("alpha")
        if not isinstance(arms, list) or not arms:
            raise ValueError("Bandit state 'config.arms' must be a non-empty list.")
        if type(context_dim) is not int or context_dim < 1:
            raise ValueError(
                "Bandit state 'config.context_dim' must be a positive int."
            )
        if (
            not isinstance(alpha, (int, float))
            or not np.isfinite(float(alpha))
            or float(alpha) <= 0
        ):
            raise ValueError(
                "Bandit state 'config.alpha' must be a finite positive number."
            )
        alpha_value = float(alpha)

        _validate_arms([str(arm) for arm in arms])
        bandit = cls(
            [str(arm) for arm in arms],
            context_dim,
            alpha=alpha_value,
        )

        recorded = state["arms"]
        if not isinstance(recorded, dict) or set(recorded) != set(bandit.arms):
            raise ValueError(
                "Bandit state 'arms' must match the configured arm names exactly."
            )
        for arm in bandit.arms:
            arm_state = recorded[arm]
            if not isinstance(arm_state, dict):
                raise ValueError(f"Bandit state for arm '{arm}' must be a dictionary.")
            matrix = arm_state.get("a")
            vector = arm_state.get("b")
            selections = arm_state.get("selections")
            rewards = arm_state.get("rewards")
            try:
                matrix_array = np.asarray(matrix, dtype=float)
                vector_array = np.asarray(vector, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Bandit state for arm '{arm}' has non-numeric statistics."
                ) from error
            if matrix_array.shape != (context_dim, context_dim):
                raise ValueError(
                    f"Bandit state for arm '{arm}' has an invalid 'a' matrix shape."
                )
            if vector_array.shape != (context_dim,):
                raise ValueError(
                    f"Bandit state for arm '{arm}' has an invalid 'b' vector shape."
                )
            if type(selections) is not int or selections < 0:
                raise ValueError(
                    f"Bandit state for arm '{arm}' has invalid "
                    "selection/reward tallies."
                )
            if not isinstance(rewards, (int, float)) or not np.isfinite(rewards):
                raise ValueError(
                    f"Bandit state for arm '{arm}' has invalid "
                    "selection/reward tallies."
                )
            reward_value = float(rewards)

            bandit._a[arm] = matrix_array
            bandit._b[arm] = vector_array
            bandit.selections[arm] = selections
            bandit.rewards[arm] = reward_value
        return bandit


def snapshot_bandit_state(bandit: LinUCBContextualBandit) -> dict[str, Any]:
    """Serialize a bandit engine's learned state for persistence."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "arms": list(bandit.arms),
            "context_dim": bandit.context_dim,
            "alpha": bandit.alpha,
        },
        "arms": {
            arm: {
                "a": [[float(value) for value in row] for row in bandit._a[arm]],
                "b": [float(value) for value in bandit._b[arm]],
                "selections": int(bandit.selections[arm]),
                "rewards": float(bandit.rewards[arm]),
            }
            for arm in bandit.arms
        },
    }


def write_bandit_state(
    state: dict[str, Any],
    state_dir: Path | str,
    *,
    state_name: str | None = None,
) -> Path:
    """Persist a bandit engine state as JSON."""
    target_dir = Path(state_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    state_path = target_dir / f"{state_name or 'bandit_state'}.json"
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state_path


def load_bandit_state(state_path: Path | str) -> dict[str, Any]:
    """Load and validate a persisted bandit engine state."""
    path = Path(state_path)
    if not path.exists():
        raise FileNotFoundError(f"Bandit state not found: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Failed to parse bandit state '{path}': {error}") from error
    if not isinstance(state, dict) or "config" not in state or "arms" not in state:
        raise ValueError(
            f"'{path}' is not a valid bandit state (missing 'config' or 'arms')."
        )
    LinUCBContextualBandit.from_state(state)
    return state


def _feedback_record_context(
    record: dict[str, Any], dim: int | None = None
) -> np.ndarray:
    """Extract and validate the context vector from a feedback record."""
    context = record.get("context")
    try:
        context_array = np.asarray(context, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Feedback record 'context' must be a numeric vector."
        ) from error
    if context_array.ndim != 1 or context_array.size == 0:
        raise ValueError("Feedback record 'context' must be a non-empty vector.")
    if dim is not None and context_array.size != dim:
        raise ValueError(
            f"Feedback context has {context_array.size} features; expected {dim}."
        )
    return context_array


def _validate_feedback_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("Feedback records must be dictionaries.")
    if "context" not in record:
        raise ValueError("Feedback record is missing 'context'.")
    if "arm" not in record:
        raise ValueError("Feedback record is missing 'arm'.")
    if "reward" not in record:
        raise ValueError("Feedback record is missing 'reward'.")
    _feedback_record_context(record)
    _validate_arms([str(record["arm"])])
    reward = record["reward"]
    if not isinstance(reward, (int, float)) or not np.isfinite(reward):
        raise ValueError("Feedback record 'reward' must be a finite number.")


def append_bandit_feedback(
    record: dict[str, Any],
    path: Path | str,
) -> Path:
    """Append a served-request observation to the feedback journal.

    The journal is a JSON list of ``{context, arm, reward}`` records, matching
    the per-round observations recorded by the simulation, so live serving
    feedback and offline report rounds can be folded into a state together.
    """
    _validate_feedback_record(record)
    target = Path(path)
    existing: list[dict[str, Any]] = []
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Failed to parse bandit feedback journal '{target}': {error}"
            ) from error
        if not isinstance(existing, list):
            raise ValueError(
                f"'{target}' is not a bandit feedback journal (JSON list)."
            )

    entry = {
        "context": [float(value) for value in _feedback_record_context(record)],
        "arm": str(record["arm"]),
        "reward": float(record["reward"]),
    }
    if "occurred_at" in record:
        entry["occurred_at"] = str(record["occurred_at"])
    if "user_id" in record:
        entry["user_id"] = str(record["user_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    existing.append(entry)
    target.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_bandit_feedback(path: Path | str) -> list[dict[str, Any]]:
    """Load and validate a bandit feedback journal."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Bandit feedback journal not found: {target}")
    try:
        records = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Failed to parse bandit feedback journal '{target}': {error}"
        ) from error
    if not isinstance(records, list):
        raise ValueError(f"'{target}' is not a bandit feedback journal (JSON list).")
    for record in records:
        _validate_feedback_record(record)
    return records


def fold_bandit_state(
    state: dict[str, Any],
    feedback: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Fold served-request observations into a bandit state, returning the update.

    Each feedback record is applied to the prior state by replaying the
    engine's additive ridge-regression update (``A += x x^T``, ``b += r x``,
    tally increments), exactly as a live run would have learned it. The
    returned state reflects the accumulated experience and is ready to act as
    the next simulation's prior.
    """
    bandit = LinUCBContextualBandit.from_state(state)
    for record in feedback:
        _validate_feedback_record(record)
        _validate_arms([str(record["arm"])])
        if str(record["arm"]) not in set(bandit.arms):
            raise ValueError(
                f"Feedback names arm '{record['arm']}' which is not in the state."
            )
        context = _feedback_record_context(record, dim=bandit.context_dim)
        bandit.update(
            str(record["arm"]),
            [float(value) for value in context],
            float(record["reward"]),
        )
    return snapshot_bandit_state(bandit)


def feedback_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-round served observations from a bandit report.

    Requires the report to carry the per-round ``context`` vector, which the
    simulation records so reports can be replayed as offline feedback.
    """
    rounds = report.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("Bandit report must contain a 'rounds' list.")
    extracted: list[dict[str, Any]] = []
    for item in rounds:
        if not isinstance(item, dict) or "context" not in item:
            raise ValueError(
                "Bandit report rounds are missing 'context'; re-run the "
                "simulation with a version that records per-round contexts."
            )
        record = {
            "context": list(item["context"]),
            "arm": str(item["arm"]),
            "reward": float(item["reward"]),
        }
        if "user_id" in item:
            record["user_id"] = str(item["user_id"])
        extracted.append(record)
    return extracted


def rank_cold_start_arm(
    arm: str,
    artist_stats: dict[str, dict[str, Any]],
    top_k: int,
    *,
    rng: np.random.Generator | None = None,
) -> list[dict[str, str | float | int]]:
    """Rank the cold-start catalog using the given arm strategy.

    Arms:
    - ``popular``: training-set popularity (the current production fallback).
    - ``balanced``: popularity with a mild popularity penalty to favor the
      mid-tail.
    - ``long_tail``: popularity with a strong penalty toward the catalog long
      tail, exploring niche artists.
    """
    validate_ranking_parameters(top_k)
    _validate_arms([arm])

    if arm == "popular":
        return popular_artists(artist_stats, top_k)

    if not artist_stats:
        return []

    scores = _arm_full_scores(arm, artist_stats)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    recommendations: list[dict[str, str | float | int]] = []
    for artist_id, adjusted in ranked:
        stats = artist_stats[artist_id]
        recommendations.append(
            {
                "artist_id": artist_id,
                "artist_name": str(stats["artist_name"]),
                "score": adjusted,
                "popularity_rank": int(stats["popularity_rank"]),
                "total_plays": float(stats["total_plays"]),
            }
        )
    return recommendations


def build_cold_start_context(
    user_df: pd.DataFrame,
    rank_by_artist_id: dict[str, int],
    *,
    features: Sequence[str] = DEFAULT_CONTEXT_FEATURES,
) -> list[float]:
    """Build a context vector from a cold-start user's bootstrap interactions.

    The bootstrap window stands in for the signals a real system observes
    during the first moments of a new user's arrival (a short observation
    window before full personalization kicks in). Available features:

    - ``log_plays``: log1p of total play count.
    - ``log_unique_artists``: log1p of distinct artists played.
    - ``mean_popularity_rank``: mean popularity rank of the played artists,
       falling back to the median catalog rank for artists absent from the
       training catalog.
    """
    if user_df.empty:
        return [0.0] * len(features)

    artist_ranks = [
        rank_by_artist_id.get(str(artist_id), len(rank_by_artist_id) + 1)
        for artist_id in user_df["artist_id"].unique()
    ]
    total_plays = float(user_df["play_count"].sum())
    unique_artists = int(user_df["artist_id"].nunique())
    mean_rank = float(np.mean(artist_ranks)) if artist_ranks else 0.0

    values: dict[str, float] = {
        "log_plays": float(np.log1p(total_plays)),
        "log_unique_artists": float(np.log1p(unique_artists)),
        "mean_popularity_rank": mean_rank,
    }
    return [values[feature] for feature in features]


def simulate_cold_start_exploration(
    df: pd.DataFrame,
    top_k: int,
    rounds: int,
    *,
    seed: int = 42,
    holdout_seed: int = 7,
    holdout_ratio: float = 0.25,
    bootstrap_ratio: float = 0.5,
    arms: Sequence[str] = DEFAULT_COLD_START_ARMS,
    alpha: float = 1.0,
    context_features: Sequence[str] = DEFAULT_CONTEXT_FEATURES,
) -> dict[str, Any]:
    """Simulate an online LinUCB bandit serving cold-start users.

    Splits users into a warm pool (builds catalog popularity stats) and a held
    out ``rounds``-sized cold pool treated as new arrivals. Each cold user gets
    a bootstrap window (first ``bootstrap_ratio`` of their rows) used only for
    the context vector; the remaining rows are the engagement target. Per
    round the bandit selects an arm for the user's context, serves top-k
    artists, and receives a reward of ``precision@k`` against the target
    artists.

    Alongside bandit learning, the always-popular policy (current production
    fallback) and the in-hindsight best arm per round are evaluated so regret
    and exploration gains can be quantified. The simulation is deterministic
    for a given seed.
    """
    validate_ranking_parameters(top_k)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer.")
    _validate_arms(arms)
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1 exclusive.")
    if not 0 < bootstrap_ratio < 1:
        raise ValueError("bootstrap_ratio must be between 0 and 1 exclusive.")

    normalized = normalize_interactions(df)
    users = sorted(normalized["user_id"].unique())
    if len(users) < 2:
        raise ValueError("At least two users are required for the simulation.")

    rng = np.random.default_rng(holdout_seed)
    pool = rng.permutation(users)
    warm_users = pool[: max(1, round(len(pool) * (1 - holdout_ratio)))]
    cold_candidates = [user for user in pool if user not in set(warm_users)]
    if not cold_candidates:
        raise ValueError("holdout_ratio produced no cold-start users.")
    cold_users = list(cold_candidates[: min(rounds, len(cold_candidates))])

    warm_df = normalized[normalized["user_id"].isin(warm_users)]
    artist_stats = build_artist_stats(warm_df)
    rank_by_artist_id = {
        artist_id: int(stats["popularity_rank"])
        for artist_id, stats in artist_stats.items()
    }

    bandit = LinUCBContextualBandit(arms, len(context_features), alpha=alpha)

    round_records: list[dict[str, Any]] = []
    cumulative_reward = 0.0
    cumulative_best = 0.0
    always_popular_reward = 0.0

    for round_index in range(rounds):
        user_id = cold_users[round_index % len(cold_users)]
        user_df = normalized[normalized["user_id"] == user_id]
        boundary = max(1, round(len(user_df) * bootstrap_ratio))
        bootstrap_df = user_df.iloc[:boundary]
        target_artist_ids = user_df.iloc[boundary:]["artist_id"].astype(str).tolist()

        context = build_cold_start_context(
            bootstrap_df,
            rank_by_artist_id,
            features=context_features,
        )

        chosen_arm = bandit.select_arm(context)

        served = rank_cold_start_arm(chosen_arm, artist_stats, top_k)
        served_ids = [str(rec["artist_id"]) for rec in served]
        reward = precision_at_k(served_ids, target_artist_ids, top_k)
        bandit.update(chosen_arm, context, reward)
        cumulative_reward += reward

        best_reward = max(
            precision_at_k(
                [
                    str(rec["artist_id"])
                    for rec in rank_cold_start_arm(arm, artist_stats, top_k)
                ],
                target_artist_ids,
                top_k,
            )
            for arm in arms
        )
        cumulative_best += best_reward

        if round_index < len(cold_users):
            popular_recs = rank_cold_start_arm("popular", artist_stats, top_k)
            always_popular_reward += precision_at_k(
                [str(rec["artist_id"]) for rec in popular_recs],
                target_artist_ids,
                top_k,
            )

        round_records.append(
            {
                "round": round_index + 1,
                "user_id": str(user_id),
                "arm": chosen_arm,
                "reward": round(reward, 6),
                "best_reward": round(best_reward, 6),
            }
        )

    num_cold = len(cold_users)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "top_k": top_k,
            "rounds": len(round_records),
            "seed": seed,
            "holdout_seed": holdout_seed,
            "holdout_ratio": holdout_ratio,
            "bootstrap_ratio": bootstrap_ratio,
            "alpha": alpha,
            "arms": list(arms),
            "context_features": list(context_features),
            "catalog_size": len(artist_stats),
            "cold_users": num_cold,
        },
        "arms": {
            arm: {
                "selections": bandit.selections[arm],
                "total_reward": round(bandit.rewards[arm], 6),
                "mean_reward": round(bandit.rewards[arm] / bandit.selections[arm], 6)
                if bandit.selections[arm]
                else 0.0,
            }
            for arm in arms
        },
        "summary": {
            "rounds_completed": len(round_records),
            "cumulative_reward": round(cumulative_reward, 6),
            "mean_reward": round(cumulative_reward / len(round_records), 6),
            "best_in_hindsight": round(cumulative_best, 6),
            "regret": round(cumulative_best - cumulative_reward, 6),
            "always_popular": {
                "rounds_evaluated": min(num_cold, len(round_records)),
                "cumulative_reward": round(always_popular_reward, 6),
                "mean_reward": round(
                    always_popular_reward / min(num_cold, len(round_records)), 6
                ),
            },
        },
        "rounds": round_records,
    }


def write_bandit_report(
    report: dict[str, Any],
    report_dir: Path | str,
    *,
    report_name: str | None = None,
) -> Path:
    """Persist a cold-start bandit simulation report as JSON."""
    target_dir = Path(report_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / f"{report_name or 'bandit_simulation'}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def load_bandit_report(report_path: Path | str) -> dict[str, Any]:
    """Load and validate a persisted cold-start bandit simulation report."""
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Bandit simulation report not found: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Failed to parse bandit simulation report '{path}': {error}"
        ) from error
    if (
        not isinstance(report, dict)
        or "config" not in report
        or "arms" not in report
        or "summary" not in report
    ):
        raise ValueError(
            f"'{path}' is not a valid bandit simulation report "
            "(missing 'config', 'arms', or 'summary')."
        )
    return report


def derive_cold_start_policy(
    report: dict[str, Any],
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Convert a bandit report into per-arm serving weights (softmax).

    Weights are computed as the softmax over each arm's learned mean reward,
    so higher-reward arms dominate the served blend while lower-reward arms
    retain a small exploration share. Deterministic for a given report.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number.")
    if "arms" not in report or not isinstance(report["arms"], dict):
        raise ValueError(
            "Report must contain an 'arms' dictionary with learned arm weights."
        )

    arm_rewards: dict[str, float] = {}
    for arm, stats in report["arms"].items():
        if not isinstance(stats, dict):
            raise ValueError(f"Arm '{arm}' stats must be a dictionary.")
        mean_reward = stats.get("mean_reward")
        if not isinstance(mean_reward, (int, float)) or not np.isfinite(mean_reward):
            raise ValueError(f"Arm '{arm}' is missing a finite 'mean_reward'.")
        arm_rewards[arm] = float(mean_reward)

    _validate_arms(list(arm_rewards))
    rewards = np.asarray([arm_rewards[arm] for arm in arm_rewards], dtype=float)
    shifted = rewards - rewards.max()
    exponentials = np.exp(shifted / temperature)
    weights = exponentials / exponentials.sum()
    return {
        arm: round(float(weight), 8)
        for arm, weight in zip(arm_rewards, weights, strict=True)
    }


def _validate_policy(policy: dict[str, float]) -> None:
    if not policy:
        raise ValueError("Cold-start policy must not be empty.")
    _validate_arms(list(policy))
    if any(not np.isfinite(weight) for weight in policy.values()):
        raise ValueError("Cold-start policy weights must all be finite.")
    if any(weight < 0 for weight in policy.values()):
        raise ValueError("Cold-start policy weights must be non-negative.")
    if not any(weight > 0 for weight in policy.values()):
        raise ValueError("Cold-start policy must contain at least one positive weight.")


def rank_cold_start_bandit(
    policy: dict[str, float],
    artist_stats: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, str | float | int]]:
    """Serve top-k cold-start artists by a Borda-style weighted blend of arms.

    Each arm ranks the catalog by its own score; an artist's blended score is
    the policy-weighted position it picks up across arms. Purely popularity
    policies (``{"popular": 1.0}``) reduce to the ``popular_artists`` ranking.
    """
    validate_ranking_parameters(top_k)
    _validate_policy(policy)

    if not artist_stats:
        return []

    candidate_scores: dict[str, float] = {}
    for arm, weight in policy.items():
        if weight <= 0:
            continue
        for position, rec in enumerate(rank_cold_start_arm(arm, artist_stats, top_k)):
            artist_id = str(rec["artist_id"])
            position_score = 1.0 - position / top_k
            candidate_scores[artist_id] = (
                candidate_scores.get(artist_id, 0.0) + weight * position_score
            )

    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))[
        :top_k
    ]
    recommendations: list[dict[str, str | float | int]] = []
    for artist_id, blended_score in ranked:
        stats = artist_stats[artist_id]
        recommendations.append(
            {
                "artist_id": artist_id,
                "artist_name": str(stats["artist_name"]),
                "score": blended_score,
                "popularity_rank": int(stats["popularity_rank"]),
                "total_plays": float(stats["total_plays"]),
            }
        )
    return recommendations


def write_cold_start_policy(
    policy: dict[str, float],
    policy_dir: Path | str,
    *,
    policy_name: str | None = None,
) -> Path:
    """Persist a cold-start bandit policy as JSON."""
    _validate_policy(policy)
    target_dir = Path(policy_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    policy_path = target_dir / f"{policy_name or 'cold_start_policy'}.json"
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return policy_path


def load_cold_start_policy(policy_path: Path | str) -> dict[str, float]:
    """Load and validate a persisted cold-start bandit policy."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Cold-start policy not found: {path}")
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Failed to parse cold-start policy '{path}': {error}"
        ) from error
    if not isinstance(policy, dict):
        raise ValueError(f"'{path}' is not a valid cold-start policy.")
    try:
        normalized = {str(arm): float(weight) for arm, weight in policy.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"'{path}' contains non-numeric policy weights.") from error
    _validate_policy(normalized)
    return normalized
