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

Live serving closes the loop too: ``feedback_from_bandit_serve`` turns a
``rank_cold_start_bandit`` response into a ``{context, arm, reward}`` record
(mapped to the dominant policy arm) that is appended to the same journal, so
served traffic contributes directly to the next policy.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from music_recommender.artifacts import build_artist_stats
from music_recommender.baselines import popular_artists
from music_recommender.config import (
    BANDIT_CONTEXT_FEATURES_PATH,
    BANDIT_SNAPSHOTS_DIR,
    BANDIT_STATE_PATH,
)
from music_recommender.data import normalize_interactions
from music_recommender.evaluate import precision_at_k
from music_recommender.ranking import validate_ranking_parameters

DEFAULT_COLD_START_ARMS: tuple[str, ...] = ("popular", "balanced", "long_tail")
DEFAULT_CONTEXT_FEATURES = ("log_plays", "log_unique_artists", "mean_popularity_rank")
SUPPORTED_BANDIT_POLICIES: tuple[str, ...] = (
    "linucb",
    "thompson_sampling",
    "epsilon_greedy",
)
DEFAULT_BANDIT_POLICY: str = "linucb"


def _context_feature_log_plays(
    user_df: pd.DataFrame,
    rank_by_artist_id: dict[str, int],
) -> float:
    """Context feature: log1p of the user's total observe-window play count."""
    return float(np.log1p(float(user_df["play_count"].sum())))


def _context_feature_log_unique_artists(
    user_df: pd.DataFrame,
    rank_by_artist_id: dict[str, int],
) -> float:
    """Context feature: log1p of the distinct artists observed for the user."""
    return float(np.log1p(int(user_df["artist_id"].nunique())))


def _context_feature_mean_popularity_rank(
    user_df: pd.DataFrame,
    rank_by_artist_id: dict[str, int],
) -> float:
    """Context feature: mean popularity rank of the observed artists.

    Artists absent from the training catalog take the worst-known rank plus
    one, so the mean still rewards breadth toward popular artists.
    """
    artist_ranks = [
        rank_by_artist_id.get(str(artist_id), len(rank_by_artist_id) + 1)
        for artist_id in user_df["artist_id"].unique()
    ]
    return float(np.mean(artist_ranks)) if artist_ranks else 0.0


CONTEXT_FEATURE_EXTRACTORS: dict[
    str, Callable[[pd.DataFrame, dict[str, int]], float]
] = {
    "log_plays": _context_feature_log_plays,
    "log_unique_artists": _context_feature_log_unique_artists,
    "mean_popularity_rank": _context_feature_mean_popularity_rank,
}
# Registry mapping context feature names to their value extractors. Each
# extractor computes one feature value from a user's bootstrap observation
# window (a dataframe of interactions) and the catalog's popularity-rank
# lookup. Adding a new feature is a registry-only change; enabling it is
# configuration.

SUPPORTED_CONTEXT_FEATURES: tuple[str, ...] = tuple(CONTEXT_FEATURE_EXTRACTORS)


def validate_context_features(features: Sequence[str]) -> tuple[str, ...]:
    """Validate a bandit context feature set, returning the canonical tuple.

    Rejects empty, unknown, or duplicated feature names with an error that
    names every offending feature and lists the supported ones, so a
    misconfigured feature set fails fast at config time instead of surfacing a
    raw lookup error mid-flight.
    """
    resolved = tuple(features)
    if not resolved:
        raise ValueError("Context features must not be empty.")
    unknown = [f for f in resolved if f not in CONTEXT_FEATURE_EXTRACTORS]
    if unknown:
        raise ValueError(
            "Unknown context feature(s): "
            + ", ".join(repr(f) for f in unknown)
            + ". Supported features: "
            + ", ".join(SUPPORTED_CONTEXT_FEATURES)
            + "."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("Context features must not contain duplicates.")
    return resolved


def resolve_context_features(
    features: Sequence[str] | None = None,
    *,
    env: str | None = None,
    config_path: Path | str | None = None,
) -> tuple[str, ...]:
    """Resolve the active bandit context feature set by precedence.

    Order: an explicit ``features`` argument, else a comma-separated ``env``
    override, else a persisted JSON ``config_path`` (consulted only when the
    file exists), else the project default (``DEFAULT_CONTEXT_FEATURES``).
    Invalid feature names fail fast regardless of which source supplies them.
    """
    if features is not None:
        return validate_context_features(features)
    if env is not None and env.strip():
        names = [part.strip() for part in env.split(",") if part.strip()]
        return validate_context_features(names)
    if config_path is not None:
        config = Path(config_path)
        if config.exists():
            return load_bandit_context_features(config)
    return DEFAULT_CONTEXT_FEATURES


def save_bandit_context_features(
    features: Sequence[str],
    path: Path | str = BANDIT_CONTEXT_FEATURES_PATH,
) -> Path:
    """Persist a validated context feature set as a JSON list and return its path."""
    resolved = validate_context_features(features)
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(list(resolved)), encoding="utf-8")
    return config_path


def load_bandit_context_features(path: Path | str) -> tuple[str, ...]:
    """Load and validate a persisted bandit context feature config.

    A missing file raises ``FileNotFoundError``; malformed content or unknown
    feature names raise ``ValueError`` with a description of the problem, so a
    broken persisted config is impossible to silently ignore.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Bandit context features not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(
            f"Failed to parse bandit context features '{config_path}': {error}"
        ) from error
    if not isinstance(raw, list):
        raise ValueError(
            f"Bandit context features '{config_path}' must be a JSON list of "
            "feature names."
        )
    non_string = [item for item in raw if not isinstance(item, str)]
    if non_string:
        raise ValueError(
            f"Bandit context features '{config_path}' must contain only feature names."
        )
    return validate_context_features(raw)


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


T = TypeVar("T", bound="BaseContextualBandit")


class BaseContextualBandit:
    """Base class for linear contextual multi-armed bandits.

    Each arm maintains a regularized ridge regression over context features
    (statistics ``A`` and ``b``) along with selection and cumulative reward tallies.
    Subclasses implement arm selection strategies (e.g. LinUCB upper confidence
    bounds or Thompson Sampling posterior sampling).
    """

    policy_type: str = "linucb"

    def __init__(
        self,
        arms: Sequence[str],
        context_dim: int,
        *,
        alpha: float = 1.0,
        alpha_decay: float = 0.0,
        context_features: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        _validate_arms(arms)
        if type(context_dim) is not int or context_dim < 1:
            raise ValueError("context_dim must be a positive integer.")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number.")
        if not np.isfinite(alpha_decay) or alpha_decay < 0:
            raise ValueError("alpha_decay must be a non-negative finite number.")
        if context_features is not None:
            resolved_features = validate_context_features(context_features)
            if len(resolved_features) != context_dim:
                raise ValueError(
                    f"context_dim ({context_dim}) must match the number of "
                    f"context features ({len(resolved_features)})."
                )
        else:
            resolved_features = None

        self.arms = list(arms)
        self.context_dim = context_dim
        self.alpha = float(alpha)
        self.alpha_decay = float(alpha_decay)
        self.context_features = resolved_features
        self._a = {arm: np.eye(context_dim) for arm in self.arms}
        self._b = {arm: np.zeros(context_dim) for arm in self.arms}
        self.selections = dict.fromkeys(self.arms, 0)
        self.rewards = dict.fromkeys(self.arms, 0.0)
        self._rng = np.random.default_rng(seed)

    @property
    def total_selections(self) -> int:
        """Total number of selections made across all arms."""
        return sum(self.selections.values())

    @property
    def effective_alpha(self) -> float:
        """Dynamic exploration cooling: alpha(t) = alpha / (1 + alpha_decay * t)."""
        return float(self.alpha / (1.0 + self.alpha_decay * self.total_selections))

    def update(
        self,
        arm: str,
        context: Sequence[float],
        reward: float,
        *,
        gamma: float = 1.0,
    ) -> None:
        """Update selected arm ridge statistics with reward and decay."""
        _validate_arms([arm])
        if not np.isfinite(reward):
            raise ValueError("reward must be finite.")
        if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"gamma must be in the range (0, 1], got {gamma}.")
        context_array = np.asarray(context, dtype=float)
        if context_array.ndim != 1 or context_array.size != self.context_dim:
            raise ValueError(
                f"context must be a {self.context_dim}-dimensional vector."
            )

        decay = float(gamma)
        self._a[arm] = decay * self._a[arm] + np.outer(context_array, context_array)
        self._b[arm] = decay * self._b[arm] + reward * context_array
        self.selections[arm] += 1
        self.rewards[arm] += float(reward)

    def select_arm(self, context: Sequence[float]) -> str:
        """Select an arm for the given context vector."""
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BaseContextualBandit:
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

        alpha_decay_raw = config.get("alpha_decay", 0.0)
        if (
            not isinstance(alpha_decay_raw, (int, float))
            or not np.isfinite(float(alpha_decay_raw))
            or float(alpha_decay_raw) < 0
        ):
            raise ValueError(
                "Bandit state 'config.alpha_decay' must be non-negative finite."
            )
        alpha_decay_value = float(alpha_decay_raw)

        policy_type_raw = config.get("policy_type", "linucb")
        if policy_type_raw not in SUPPORTED_BANDIT_POLICIES:
            raise ValueError(
                f"Unknown policy_type '{policy_type_raw}' in state config. "
                f"Expected one of {SUPPORTED_BANDIT_POLICIES}."
            )

        recorded_features = config.get("context_features")
        if recorded_features is not None:
            if (
                not isinstance(recorded_features, list)
                or not recorded_features
                or not all(isinstance(name, str) for name in recorded_features)
            ):
                raise ValueError(
                    "Bandit state 'config.context_features' must be a "
                    "non-empty list of feature names."
                )
            features = validate_context_features(recorded_features)
            if len(features) != context_dim:
                raise ValueError(
                    "Bandit state 'config.context_features' length must match "
                    "'config.context_dim'."
                )
        else:
            features = None

        _validate_arms([str(arm) for arm in arms])

        if cls is BaseContextualBandit:
            target_cls: type[BaseContextualBandit]
            if policy_type_raw == "thompson_sampling":
                target_cls = ThompsonSamplingContextualBandit
            elif policy_type_raw == "epsilon_greedy":
                target_cls = EpsilonGreedyContextualBandit
            else:
                target_cls = LinUCBContextualBandit
        else:
            target_cls = cls

        bandit = target_cls(
            [str(arm) for arm in arms],
            context_dim,
            alpha=alpha_value,
            alpha_decay=alpha_decay_value,
            context_features=features,
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


class LinUCBContextualBandit(BaseContextualBandit):
    """Contextual multi-armed bandit using LinUCB (linear upper confidence bound).

    Each arm maintains a ridge regression over context features; at selection
    time the arm with the highest upper confidence bound for the context is
    chosen, balancing exploitation (point estimate) and exploration
    (uncertainty term scaled by dynamic ``effective_alpha``).
    """

    policy_type = "linucb"

    def _arm_score(self, arm: str, context: np.ndarray) -> float:
        matrix = self._a[arm]
        inv_matrix = np.linalg.inv(matrix)
        theta = inv_matrix @ self._b[arm]
        mean = float(context @ theta)
        uncertainty = self.effective_alpha * float(
            np.sqrt(context @ inv_matrix @ context)
        )
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


class ThompsonSamplingContextualBandit(BaseContextualBandit):
    """Contextual multi-armed bandit using contextual Thompson Sampling.

    For each arm, parameters are sampled from the posterior distribution
    theta_a ~ N(A_a^{-1} b_a, v^2 A_a^{-1}), where v is scaled by the
    effective exploration parameter. The arm with the highest sampled
    expected reward x^T theta_a is selected.
    """

    policy_type = "thompson_sampling"

    def select_arm(self, context: Sequence[float]) -> str:
        """Return the arm with the highest sampled expected reward."""
        context_array = np.asarray(context, dtype=float)
        if context_array.ndim != 1 or context_array.size != self.context_dim:
            raise ValueError(
                f"context must be a {self.context_dim}-dimensional vector."
            )
        v = self.effective_alpha
        sampled_scores: dict[str, float] = {}
        for arm in self.arms:
            matrix = self._a[arm]
            inv_matrix = np.linalg.inv(matrix)
            mu = inv_matrix @ self._b[arm]
            cov = (v**2) * inv_matrix
            cov = 0.5 * (cov + cov.T)
            sampled_theta = self._rng.multivariate_normal(mu, cov)
            score = float(np.clip(context_array @ sampled_theta, -1e9, 1e9))
            sampled_scores[arm] = score

        best_arm = max(
            self.arms,
            key=lambda arm: (sampled_scores[arm], arm),
        )
        return best_arm


class EpsilonGreedyContextualBandit(BaseContextualBandit):
    """Contextual multi-armed bandit using contextual epsilon-greedy exploration.

    Each arm maintains a ridge regression over context features. With
    probability epsilon(t) = epsilon / (1 + alpha_decay * t) (where epsilon
    is parameterized via the alpha exploration parameter), an arm is chosen
    uniformly at random. With probability 1 - epsilon(t), the arm with the
    highest estimated reward x^T theta (where theta = A^{-1} b) is selected
    greedily, with deterministic lexicographic tie-breaking.
    """

    policy_type = "epsilon_greedy"

    def select_arm(self, context: Sequence[float]) -> str:
        """Select an arm using contextual epsilon-greedy exploration."""
        context_array = np.asarray(context, dtype=float)
        if context_array.ndim != 1 or context_array.size != self.context_dim:
            raise ValueError(
                f"context must be a {self.context_dim}-dimensional vector."
            )
        eps = float(np.clip(self.effective_alpha, 0.0, 1.0))
        if self._rng.random() < eps:
            arm_index = int(self._rng.integers(0, len(self.arms)))
            return self.arms[arm_index]

        scores: dict[str, float] = {}
        for arm in self.arms:
            matrix = self._a[arm]
            inv_matrix = np.linalg.inv(matrix)
            theta = inv_matrix @ self._b[arm]
            score = float(np.clip(context_array @ theta, -1e9, 1e9))
            scores[arm] = score

        best_arm = max(
            self.arms,
            key=lambda arm: (scores[arm], arm),
        )
        return best_arm


def snapshot_bandit_state(bandit: BaseContextualBandit) -> dict[str, Any]:
    """Serialize a bandit engine's learned state for persistence."""
    config: dict[str, Any] = {
        "policy_type": bandit.policy_type,
        "arms": list(bandit.arms),
        "context_dim": bandit.context_dim,
        "alpha": bandit.alpha,
        "alpha_decay": bandit.alpha_decay,
    }
    if bandit.context_features is not None:
        config["context_features"] = list(bandit.context_features)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
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
    BaseContextualBandit.from_state(state)
    return state


def compute_arm_thetas(state: dict[str, Any]) -> dict[str, list[float]]:
    """Compute ridge regression coefficients theta = A^{-1} b for each arm in state."""
    BaseContextualBandit.from_state(state)
    thetas: dict[str, list[float]] = {}
    for arm, arm_data in state["arms"].items():
        matrix_a = np.asarray(arm_data["a"], dtype=float)
        vector_b = np.asarray(arm_data["b"], dtype=float)
        theta = np.linalg.inv(matrix_a) @ vector_b
        thetas[str(arm)] = [round(float(value), 6) for value in theta]
    return thetas


def compute_bandit_drift(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
) -> dict[str, Any]:
    """Compute parameter and reward drift between two bandit states.

    Compares the ridge regression coefficients (theta = A^{-1} b), selections,
    and cumulative rewards per arm between state_a (baseline/prior) and state_b
    (comparison/current), quantifying parameter drift via L2 norm and cosine
    similarity.
    """
    bandit_a = BaseContextualBandit.from_state(state_a)
    bandit_b = BaseContextualBandit.from_state(state_b)
    if set(bandit_a.arms) != set(bandit_b.arms):
        raise ValueError(
            f"Cannot compute drift: state arms do not match ({bandit_a.arms} vs "
            f"{bandit_b.arms})."
        )
    if bandit_a.context_dim != bandit_b.context_dim:
        raise ValueError(
            f"Cannot compute drift: context dimensions do not match "
            f"({bandit_a.context_dim} vs {bandit_b.context_dim})."
        )

    thetas_a = compute_arm_thetas(state_a)
    thetas_b = compute_arm_thetas(state_b)
    arms_drift: dict[str, Any] = {}
    for arm in bandit_a.arms:
        vec_a = np.asarray(thetas_a[arm], dtype=float)
        vec_b = np.asarray(thetas_b[arm], dtype=float)
        delta_theta = vec_b - vec_a
        l2_drift = float(np.linalg.norm(delta_theta))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a > 1e-9 and norm_b > 1e-9:
            cosine_similarity = float(
                np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
            )
        elif norm_a <= 1e-9 and norm_b <= 1e-9:
            cosine_similarity = 1.0
        else:
            cosine_similarity = 0.0

        sel_a = int(state_a["arms"][arm]["selections"])
        sel_b = int(state_b["arms"][arm]["selections"])
        rew_a = float(state_a["arms"][arm]["rewards"])
        rew_b = float(state_b["arms"][arm]["rewards"])
        mean_rew_a = rew_a / sel_a if sel_a > 0 else 0.0
        mean_rew_b = rew_b / sel_b if sel_b > 0 else 0.0

        arms_drift[arm] = {
            "theta_a": thetas_a[arm],
            "theta_b": thetas_b[arm],
            "delta_theta": [round(float(v), 6) for v in delta_theta],
            "l2_drift": round(l2_drift, 6),
            "cosine_similarity": round(cosine_similarity, 6),
            "selections_a": sel_a,
            "selections_b": sel_b,
            "delta_selections": sel_b - sel_a,
            "rewards_a": round(rew_a, 6),
            "rewards_b": round(rew_b, 6),
            "delta_rewards": round(rew_b - rew_a, 6),
            "mean_reward_a": round(mean_rew_a, 6),
            "mean_reward_b": round(mean_rew_b, 6),
            "delta_mean_reward": round(mean_rew_b - mean_rew_a, 6),
        }

    l2_drifts = [data["l2_drift"] for data in arms_drift.values()]
    max_l2_drift = max(l2_drifts) if l2_drifts else 0.0
    mean_l2_drift = float(np.mean(l2_drifts)) if l2_drifts else 0.0

    dom_a = max(bandit_a.arms, key=lambda a: (int(state_a["arms"][a]["selections"]), a))
    dom_b = max(bandit_b.arms, key=lambda a: (int(state_b["arms"][a]["selections"]), a))

    has_drift = max_l2_drift > 1e-6 or any(
        data["delta_selections"] != 0 for data in arms_drift.values()
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "max_l2_drift": round(max_l2_drift, 6),
            "mean_l2_drift": round(mean_l2_drift, 6),
            "dominant_arm_a": dom_a,
            "dominant_arm_b": dom_b,
            "dominant_arm_changed": dom_a != dom_b,
            "has_drift": bool(has_drift),
        },
        "arms": arms_drift,
    }


def save_bandit_snapshot(
    state: dict[str, Any],
    snapshot_dir: Path | str = BANDIT_SNAPSHOTS_DIR,
    *,
    label: str | None = None,
) -> Path:
    """Save a timestamped snapshot of a bandit engine state."""
    BaseContextualBandit.from_state(state)

    target_dir = Path(snapshot_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if label and label.strip():
        clean_label = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in label.strip()
        )
        filename = f"bandit_state_{timestamp}_{clean_label}.json"
    else:
        clean_label = None
        filename = f"bandit_state_{timestamp}.json"

    snapshot_path = target_dir / filename
    state_to_write = json.loads(json.dumps(state))
    state_to_write["snapshot"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "label": clean_label,
        "filename": filename,
    }
    snapshot_path.write_text(
        json.dumps(state_to_write, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot_path


def list_bandit_snapshots(
    snapshot_dir: Path | str = BANDIT_SNAPSHOTS_DIR,
) -> list[dict[str, Any]]:
    """List persisted bandit state snapshots sorted newest first."""
    target_dir = Path(snapshot_dir)
    if not target_dir.exists():
        return []
    snapshots: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("bandit_state_*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(state, dict)
                or "arms" not in state
                or "config" not in state
            ):
                continue
            snapshot_meta = state.get("snapshot", {})
            total_selections = sum(
                int(arm_data.get("selections", 0))
                for arm_data in state["arms"].values()
                if isinstance(arm_data, dict)
            )
            created_at = (
                snapshot_meta.get("created_at") or state.get("generated_at") or ""
            )
            label = snapshot_meta.get("label")
            snapshots.append(
                {
                    "filename": path.name,
                    "path": str(path.resolve()),
                    "created_at": created_at,
                    "label": label,
                    "arms": list(state["config"].get("arms", [])),
                    "context_dim": state["config"].get("context_dim"),
                    "total_selections": total_selections,
                }
            )
        except (OSError, ValueError):
            continue
    snapshots.sort(key=lambda s: s["filename"], reverse=True)
    return snapshots


def load_bandit_snapshot(snapshot_path: Path | str) -> dict[str, Any]:
    """Load and validate a persisted bandit state snapshot."""
    return load_bandit_state(snapshot_path)


def restore_bandit_snapshot(
    snapshot_path: Path | str,
    target_state_path: Path | str = BANDIT_STATE_PATH,
) -> Path:
    """Restore a saved snapshot into the active bandit state path."""
    state = load_bandit_snapshot(snapshot_path)
    target = Path(target_state_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def prune_bandit_snapshots(
    snapshot_dir: Path | str = BANDIT_SNAPSHOTS_DIR,
    max_keep: int = 10,
) -> list[Path]:
    """Prune older snapshots in snapshot_dir, keeping the max_keep newest.

    Returns the list of deleted snapshot paths.
    """
    if type(max_keep) is not int or max_keep < 1:
        raise ValueError("max_keep must be a positive integer.")
    target_dir = Path(snapshot_dir)
    if not target_dir.exists():
        return []
    existing = sorted(target_dir.glob("bandit_state_*.json"))
    if len(existing) <= max_keep:
        return []
    excess = existing[:-max_keep]
    deleted: list[Path] = []
    for path in excess:
        try:
            path.unlink()
            deleted.append(path)
        except OSError:
            continue
    return deleted


def validate_state_context_features(
    state: dict[str, Any],
    requested: Sequence[str],
) -> tuple[str, ...]:
    """Cross-check a requested context feature set against a state's records.

    When the state records ``config.context_features``, the requested set must
    match exactly; folding with a different context dimension would silently
    misalign ridge statistics. Raises an actionable error on mismatch.
    """
    resolved = validate_context_features(requested)
    recorded = state["config"].get("context_features")
    if recorded is not None and list(resolved) != recorded:
        raise ValueError(
            f"Requested context features {list(resolved)} do not match the "
            f"state's recorded feature set {recorded}; use the state's "
            "context dimension or retrain the state."
        )
    return resolved


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
    *,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Fold served-request observations into a bandit state, returning the update.

    Each feedback record is applied to the prior state by replaying the
    engine's additive ridge-regression update (``A = gamma * A + x x^T``,
    ``b = gamma * b + r x``, tally increments), discounted by ``gamma``. The
    returned state reflects the accumulated experience and is ready to act as
    the next simulation's prior.
    """
    if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in the range (0, 1], got {gamma}.")
    bandit = BaseContextualBandit.from_state(state)
    for record in feedback:
        _validate_feedback_record(record)
        _validate_arms([str(record["arm"])])
        if str(record["arm"]) not in set(bandit.arms):
            raise ValueError(
                f"Feedback names arm '{record['arm']}' which is not in the state."
            )
        try:
            context = _feedback_record_context(record, dim=bandit.context_dim)
        except ValueError as error:
            if bandit.context_features is not None:
                raise ValueError(
                    f"{error} The state's context features are "
                    f"{list(bandit.context_features)}."
                ) from error
            raise
        bandit.update(
            str(record["arm"]),
            [float(value) for value in context],
            float(record["reward"]),
            gamma=gamma,
        )
    return snapshot_bandit_state(bandit)


def _journal_fold_offset(state: dict[str, Any]) -> int:
    """Read the fold watermark (folded journal record count) off a state."""
    config = state.get("config")
    if not isinstance(config, dict):
        return 0
    value = config.get("journal_fold_offset", 0)
    if type(value) is not int or value < 0:
        raise ValueError(
            "Bandit state 'config.journal_fold_offset' must be a non-negative int."
        )
    return value


def sweep_bandit_journal(
    state: dict[str, Any],
    feedback: Sequence[dict[str, Any]],
    *,
    gamma: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold only the pending records of a feedback journal into a bandit state.

    The state's config records the journal fold watermark
    (``journal_fold_offset`` and ``journal_folded_at``), so a repeated sweep
    folds only the records appended since the last one instead of double
    counting the whole journal. A journal that shrank below the recorded
    offset (e.g. it was reset) starts over from the top. Returns the updated
    state together with a ``{"folded_count", "journal_length", "offset", "gamma"}``
    summary.
    """
    records = list(feedback)
    offset = _journal_fold_offset(state)
    if len(records) < offset:
        offset = 0
    pending = records[offset:]
    updated = fold_bandit_state(state, pending, gamma=gamma)
    updated["config"]["journal_fold_offset"] = len(records)
    updated["config"]["journal_folded_at"] = datetime.now(UTC).isoformat()
    return updated, {
        "folded_count": len(pending),
        "journal_length": len(records),
        "offset": len(records),
        "gamma": gamma,
    }


def pending_feedback_count(
    state: dict[str, Any] | None,
    feedback: Sequence[dict[str, Any]],
) -> int:
    """Return how many journal records are not yet folded into the state.

    Uses the state's fold watermark; a journal that shrank below the recorded
    offset counts the whole journal as pending, mirroring the sweep semantics.
    """
    records = list(feedback)
    if state is None:
        return len(records)
    offset = _journal_fold_offset(state)
    if len(records) < offset:
        return len(records)
    return len(records) - offset


def summarize_bandit_lifecycle(
    *,
    state: dict[str, Any] | None,
    policy: dict[str, float] | None,
    feedback: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build a readable summary of the cold-start bandit lifecycle.

    Consumes the optional persisted state, active policy, and feedback journal
    and reports per-arm statistics, policy weights, journal length and pending
    count (via the state's fold watermark), and the last fold time. Any
    component may be absent (e.g. no state file yet), reported cleanly so the
    summary stays useful while the loop is still warming up.
    """
    state_summary: dict[str, Any] | None = None
    last_fold: dict[str, Any] | None = None
    if state is not None:
        arms_summary: dict[str, Any] = {}
        for arm, stats in state["arms"].items():
            selections = int(stats["selections"])
            total_reward = float(stats["rewards"])
            arms_summary[str(arm)] = {
                "selections": selections,
                "total_reward": round(total_reward, 6),
                "mean_reward": round(
                    total_reward / selections if selections else 0.0, 6
                ),
            }
        config = state.get("config", {})
        alpha_val = float(config.get("alpha", 1.0))
        alpha_decay_val = float(config.get("alpha_decay", 0.0))
        total_sel = sum(item["selections"] for item in arms_summary.values())
        eff_alpha = float(alpha_val / (1.0 + alpha_decay_val * total_sel))
        state_summary = {
            "generated_at": str(state.get("generated_at", "")),
            "context_dim": int(config["context_dim"]),
            "alpha": alpha_val,
            "alpha_decay": alpha_decay_val,
            "effective_alpha": round(eff_alpha, 4),
            "policy_type": str(config.get("policy_type", "linucb")),
            "total_selections": total_sel,
            "arms": arms_summary,
        }
        if isinstance(config, dict) and "context_features" in config:
            state_summary["context_features"] = list(config["context_features"])
        if isinstance(config, dict) and "journal_folded_at" in config:
            last_fold = {
                "offset": int(config["journal_fold_offset"]),
                "at": str(config["journal_folded_at"]),
            }

    records = list(feedback)
    return {
        "available": state is not None or policy is not None,
        "state": state_summary,
        "policy": dict(policy) if policy is not None else None,
        "journal": {
            "length": len(records),
            "pending": pending_feedback_count(state, records),
        },
        "last_fold": last_fold,
    }


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


def neutral_serve_context(
    features: Sequence[str] = DEFAULT_CONTEXT_FEATURES,
) -> list[float]:
    """Return the neutral context of a brand-new, unknown user.

    An unknown user has no interaction history, so none of the context features
    (plays, artist breadth, popularity exposure) is observable yet; the serve
    context is the zero vector matching the simulation's cold-start features.
    """
    return [0.0] * len(features)


def validate_serve_context(
    context: Sequence[float],
    *,
    features: Sequence[str] = DEFAULT_CONTEXT_FEATURES,
) -> list[float]:
    """Validate a caller-supplied serve context and normalize it to floats.

    Live serving is expected to record contexts with the same feature set the
    simulation uses (``DEFAULT_CONTEXT_FEATURES`` by default), so the vector
    must be non-empty, fully numeric, finite, and of the expected dimension.
    Returns the normalized float vector.
    """
    values = list(context)
    if not values:
        raise ValueError("Serve context must be a non-empty vector.")
    try:
        floats = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("Serve context must contain only numeric values.") from error
    if any(not np.isfinite(value) for value in floats):
        raise ValueError("Serve context values must all be finite.")
    if len(floats) != len(features):
        raise ValueError(
            f"Serve context has {len(floats)} features; expected {len(features)}."
        )
    return floats


def dominant_policy_arm(policy: dict[str, float]) -> str:
    """Return the arm with the highest policy weight (deterministic).

    Ties are broken by the lexicographically smallest arm name so repeated
    calls agree even when weights are equal.
    """
    _validate_policy(policy)
    max_weight = max(policy.values())
    top_arms = sorted(arm for arm, weight in policy.items() if weight == max_weight)
    return top_arms[0]


def feedback_records_from_bandit_serve(
    policy: dict[str, float],
    artist_stats: dict[str, dict[str, Any]],
    top_k: int,
    *,
    context: Sequence[float] | None = None,
    user_id: str | None = None,
    features: Sequence[str] = DEFAULT_CONTEXT_FEATURES,
) -> list[dict[str, Any]]:
    """Build one feedback record per policy arm for a live bandit serve.

    The blended serving response (``rank_cold_start_bandit``) is rewarded per
    arm: each record's reward is the precision@k of the served artist ids
    against that arm's own ranking, an engagement proxy computed entirely from
    the serve. Crediting every arm (not just the dominant one) lets
    ``bandit-update`` fold influence across the whole policy. Unknown users
    carry the neutral (zero) context unless one is supplied; a supplied context
    is validated against ``features`` (the active context feature set).
    """
    _validate_policy(policy)
    validate_ranking_parameters(top_k)
    served = rank_cold_start_bandit(policy, artist_stats, top_k)
    served_ids = [str(rec["artist_id"]) for rec in served]
    records: list[dict[str, Any]] = []
    for arm in sorted(policy):
        weight = policy[arm]
        if weight <= 0:
            continue
        arm_served = rank_cold_start_arm(arm, artist_stats, top_k)
        reward = precision_at_k(
            served_ids,
            [str(rec["artist_id"]) for rec in arm_served],
            top_k,
        )
        record: dict[str, Any] = {
            "context": (
                validate_serve_context(context, features=features)
                if context is not None
                else neutral_serve_context(features)
            ),
            "arm": arm,
            "reward": reward,
        }
        if user_id is not None:
            record["user_id"] = user_id
        _validate_feedback_record(record)
        records.append(record)
    return records


def feedback_from_bandit_serve(
    policy: dict[str, float],
    artist_stats: dict[str, dict[str, Any]],
    top_k: int,
    *,
    context: Sequence[float] | None = None,
    user_id: str | None = None,
    features: Sequence[str] = DEFAULT_CONTEXT_FEATURES,
) -> dict[str, Any]:
    """Build a feedback record for a live bandit-fallback serve.

    The blended serving response (``rank_cold_start_bandit``) is rewarded by
    how closely it matches the dominant arm's own ranking: the reward is
    precision@k of the served artist ids against the dominant arm's top-k, an
    engagement proxy computed entirely from the serve itself. Unknown users
    carry the neutral (zero) context unless one is supplied.
    """
    records = feedback_records_from_bandit_serve(
        policy,
        artist_stats,
        top_k,
        context=context,
        user_id=user_id,
        features=features,
    )
    dominant = dominant_policy_arm(policy)
    for record in records:
        if record["arm"] == dominant:
            return record
    raise ValueError(f"Policy has no feedback record for dominant arm '{dominant}'.")


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
    window before full personalization kicks in). Values are computed through
    the context-feature registry, so any subset or reordering of the supported
    features is valid and the vector follows the requested order:

    - ``log_plays``: log1p of total play count.
    - ``log_unique_artists``: log1p of distinct artists played.
    - ``mean_popularity_rank``: mean popularity rank of the played artists,
      using the worst-known rank plus one for artists absent from the training
      catalog.

    Unknown feature names are rejected with a helpful error listing the
    supported features.
    """
    resolved = validate_context_features(features)
    if user_df.empty:
        return [0.0] * len(resolved)

    values = {
        name: CONTEXT_FEATURE_EXTRACTORS[name](user_df, rank_by_artist_id)
        for name in resolved
    }
    return [values[name] for name in resolved]


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
    alpha_decay: float = 0.0,
    gamma: float = 1.0,
    policy_type: str = "linucb",
    context_features: Sequence[str] = DEFAULT_CONTEXT_FEATURES,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate an online contextual bandit serving cold-start users.

    Splits users into a warm pool (builds catalog popularity stats) and a held
    out ``rounds``-sized cold pool treated as new arrivals. Each cold user gets
    a bootstrap window (first ``bootstrap_ratio`` of their rows) used only for
    the context vector; the remaining rows are the engagement target. Per
    round the bandit selects an arm for the user's context (via LinUCB or
    Thompson Sampling), serves top-k artists, and receives a reward of
    ``precision@k`` against the target artists.

    Alongside bandit learning, the always-popular policy (current production
    fallback) and the in-hindsight best arm per round are evaluated so regret
    and exploration gains can be quantified. The simulation is deterministic
    for a given seed.

    ``initial_state`` restores a persisted engine (e.g. from a prior run or a
    folded feedback journal) so the simulation continues learning from
    accumulated experience instead of starting fresh; when given, it must
    match ``arms``, ``context_features``, and ``alpha``.
    """
    validate_ranking_parameters(top_k)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer.")
    _validate_arms(arms)
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1 exclusive.")
    if not 0 < bootstrap_ratio < 1:
        raise ValueError("bootstrap_ratio must be between 0 and 1 exclusive.")
    if policy_type not in SUPPORTED_BANDIT_POLICIES:
        raise ValueError(
            f"Unknown policy_type '{policy_type}'. Expected one of "
            f"{SUPPORTED_BANDIT_POLICIES}."
        )
    if not np.isfinite(alpha_decay) or alpha_decay < 0:
        raise ValueError("alpha_decay must be a non-negative finite number.")
    if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in the range (0, 1], got {gamma}.")
    resolved_context_features = validate_context_features(context_features)

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

    if initial_state is not None:
        bandit = BaseContextualBandit.from_state(initial_state)
        if list(bandit.arms) != list(arms):
            raise ValueError(
                "initial_state arms must match the simulation arms "
                f"({', '.join(arms)})."
            )
        if bandit.context_dim != len(resolved_context_features):
            raise ValueError(
                "initial_state context_dim must match the number of context_features."
            )
        if bandit.alpha != float(alpha):
            raise ValueError("initial_state alpha must match the simulation alpha.")
        if bandit.policy_type != policy_type:
            raise ValueError(
                f"initial_state policy_type '{bandit.policy_type}' must match "
                f"simulation policy_type '{policy_type}'."
            )
        prior_selections = sum(bandit.selections.values())
        prior_rewards = sum(bandit.rewards.values())
    else:
        target_cls: Any = (
            ThompsonSamplingContextualBandit
            if policy_type == "thompson_sampling"
            else (
                EpsilonGreedyContextualBandit
                if policy_type == "epsilon_greedy"
                else LinUCBContextualBandit
            )
        )
        bandit = target_cls(
            arms,
            len(resolved_context_features),
            alpha=alpha,
            alpha_decay=alpha_decay,
            context_features=resolved_context_features,
            seed=seed,
        )
        prior_selections = 0
        prior_rewards = 0.0

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
            features=resolved_context_features,
        )

        chosen_arm = bandit.select_arm(context)

        served = rank_cold_start_arm(chosen_arm, artist_stats, top_k)
        served_ids = [str(rec["artist_id"]) for rec in served]
        reward = precision_at_k(served_ids, target_artist_ids, top_k)
        bandit.update(chosen_arm, context, reward, gamma=gamma)
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
                "context": [round(float(value), 6) for value in context],
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
            "alpha_decay": alpha_decay,
            "gamma": gamma,
            "policy_type": policy_type,
            "arms": list(arms),
            "context_features": list(resolved_context_features),
            "catalog_size": len(artist_stats),
            "cold_users": num_cold,
            "prior": {
                "selections": int(prior_selections),
                "total_reward": round(prior_rewards, 6),
            },
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
    temperature_decay: float = 0.0,
    min_temperature: float = 0.01,
) -> dict[str, float]:
    """Convert a bandit report or state snapshot into per-arm serving weights (softmax).

    Weights are computed as the softmax over each arm's learned mean reward,
    so higher-reward arms dominate the served blend while lower-reward arms
    retain an exploration share. Supports temperature annealing
    tau(t) = max(min_temperature, tau_0 / (1 + temperature_decay * t)),
    where t is the total round or selection count.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number.")
    if not np.isfinite(temperature_decay) or temperature_decay < 0:
        raise ValueError("temperature_decay must be a non-negative finite number.")
    if not np.isfinite(min_temperature) or min_temperature <= 0:
        raise ValueError("min_temperature must be a finite positive number.")
    if "arms" not in report or not isinstance(report["arms"], dict):
        raise ValueError(
            "Report must contain an 'arms' dictionary with learned arm weights."
        )

    arm_rewards: dict[str, float] = {}
    for arm, stats in report["arms"].items():
        if not isinstance(stats, dict):
            raise ValueError(f"Arm '{arm}' stats must be a dictionary.")
        if "mean_reward" in stats:
            mean_reward = stats.get("mean_reward")
        elif "rewards" in stats and "selections" in stats:
            selections = stats.get("selections")
            rewards = stats.get("rewards")
            if type(selections) is not int or selections < 0:
                raise ValueError(f"Arm '{arm}' has invalid selections tally.")
            if not isinstance(rewards, (int, float)) or not np.isfinite(rewards):
                raise ValueError(f"Arm '{arm}' has invalid rewards tally.")
            mean_reward = float(rewards) / selections if selections > 0 else 0.0
        else:
            raise ValueError(f"Arm '{arm}' is missing a finite 'mean_reward'.")

        if not isinstance(mean_reward, (int, float)) or not np.isfinite(mean_reward):
            raise ValueError(f"Arm '{arm}' is missing a finite 'mean_reward'.")
        arm_rewards[arm] = float(mean_reward)

    _validate_arms(list(arm_rewards))

    effective_temperature = float(temperature)
    if temperature_decay > 0.0:
        t = 0
        if "summary" in report and isinstance(report["summary"], dict):
            rounds_val = report["summary"].get("rounds_completed", 0)
            if isinstance(rounds_val, (int, float)):
                t = int(rounds_val)
        if t == 0:
            t = sum(
                int(s.get("selections", 0))
                for s in report["arms"].values()
                if isinstance(s, dict) and isinstance(s.get("selections"), int)
            )
        effective_temperature = max(
            min_temperature,
            float(temperature / (1.0 + temperature_decay * t)),
        )

    rewards = np.asarray([arm_rewards[arm] for arm in arm_rewards], dtype=float)
    shifted = rewards - rewards.max()
    exponentials = np.exp(shifted / effective_temperature)
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
