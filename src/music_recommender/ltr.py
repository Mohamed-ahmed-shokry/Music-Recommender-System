"""A lightweight pointwise learning-to-rank re-ranker.

The re-ranker learns to order the candidate list produced by the ALS model. It
is trained on the same implicit-feedback interactions as the collaborative
model: for every observed (user, artist) pair a set of feature values is
extracted and paired with the interaction play count, and a small set of
sampled negative candidates is paired with zero. A ridge regressor then predicts
a relevance score for any candidate so the served list can be re-ranked.

This keeps the recommender's serving path fast and dependency-light: the model
is a single linear regressor over a handful of interpretable features, and no
candidate-search is required because it only re-ranks artists the collaborative
model already surfaced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from music_recommender.artifacts import ArtistStats
from music_recommender.ranking import validate_ranking_parameters

if TYPE_CHECKING:
    from music_recommender.tracks import TrackServingResources


def _validate_ltr_inputs(
    user_item_matrix: csr_matrix,
    model: Any,
) -> None:
    if not isinstance(user_item_matrix, csr_matrix):
        raise TypeError("user_item_matrix must be a CSR matrix.")
    if (
        model is None
        or not hasattr(model, "user_factors")
        or not hasattr(model, "item_factors")
    ):
        raise ValueError("model must expose user_factors and item_factors.")


def _artist_popularity_features(
    artist_index: int,
    artist_stats: dict[str, ArtistStats],
    index_to_artist_id: dict[int, str],
    num_artists: int,
) -> tuple[float, float]:
    """Return (log-total-plays, normalized popularity-rank) for an artist."""
    artist_id = index_to_artist_id[artist_index]
    stats = artist_stats.get(artist_id)
    if stats is None:
        return 0.0, 0.0
    total_plays = float(stats["total_plays"])
    log_plays = float(np.log1p(total_plays))
    rank = float(stats["popularity_rank"]) - 1.0
    normalized_rank = rank / num_artists if num_artists else 0.0
    return log_plays, normalized_rank


def _user_feature_vector(
    user_index: int,
    model: Any,
    user_item_matrix: csr_matrix,
    artist_stats: dict[str, ArtistStats],
    index_to_artist_id: dict[int, str],
    num_artists: int,
    artist_indices: list[int],
) -> np.ndarray:
    """Build a feature matrix (one row per candidate artist) for a user."""
    user_latent = model.item_factors
    artist_latent = model.user_factors
    collaborative_scores = artist_latent[artist_indices] @ user_latent[user_index]
    user_interaction_count = float(user_item_matrix[user_index].nnz)

    log_plays = np.empty(len(artist_indices), dtype=float)
    normalized_ranks = np.empty(len(artist_indices), dtype=float)
    for position, artist_index in enumerate(artist_indices):
        log_plays[position], normalized_ranks[position] = _artist_popularity_features(
            artist_index,
            artist_stats,
            index_to_artist_id,
            num_artists,
        )

    features = np.column_stack(
        [
            collaborative_scores,
            log_plays,
            normalized_ranks,
            np.full(len(artist_indices), user_interaction_count),
        ]
    )
    return cast(np.ndarray, np.nan_to_num(features))


def train_ltr_ranker(
    *,
    train_df: Any,
    mappings: dict[str, Any],
    user_item_matrix: csr_matrix,
    model: Any,
    artist_stats: dict[str, ArtistStats],
    negatives_per_positive: int = 3,
    random_state: int = 42,
    alpha: float = 1.0,
) -> Ridge:
    """Fit a pointwise learning-to-rank re-ranker on the training fold.

    Positive examples come from observed interactions; negatives are sampled
    uniformly from the artists the user did not interact with. The returned
    regressor predicts an artist's relevance to a user.
    """
    _validate_ltr_inputs(user_item_matrix, model)
    if type(negatives_per_positive) is not int or negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be a positive integer.")
    if type(random_state) is not int:
        raise ValueError("random_state must be an integer.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than 0.")

    user_id_to_index = mappings["user_id_to_index"]
    index_to_artist_id = mappings["index_to_artist_id"]
    num_artists = len(mappings["artist_id_to_index"])
    rng = np.random.default_rng(random_state)

    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for user_id, user_df in train_df.groupby("user_id"):
        user_index = user_id_to_index[user_id]
        interacted_indices = set(user_item_matrix[user_index].indices)
        if not interacted_indices:
            continue

        positive_indices = list(interacted_indices)
        user_features = _user_feature_vector(
            user_index=user_index,
            model=model,
            user_item_matrix=user_item_matrix,
            artist_stats=artist_stats,
            index_to_artist_id=index_to_artist_id,
            num_artists=num_artists,
            artist_indices=positive_indices,
        )
        feature_rows.append(user_features)

        summed_plays = user_df.groupby("artist_id")["play_count"].sum()
        play_by_artist = {
            artist_id: float(plays) for artist_id, plays in summed_plays.items()
        }
        positive_labels = np.array(
            [
                play_by_artist.get(index_to_artist_id[idx], 0.0)
                for idx in positive_indices
            ],
            dtype=float,
        )
        labels.append(positive_labels)

        negative_candidates = [
            idx for idx in range(num_artists) if idx not in interacted_indices
        ]
        negative_draws = rng.choice(
            negative_candidates,
            size=min(
                negatives_per_positive * len(positive_indices),
                len(negative_candidates),
            ),
            replace=False,
        )
        negative_features = _user_feature_vector(
            user_index=user_index,
            model=model,
            user_item_matrix=user_item_matrix,
            artist_stats=artist_stats,
            index_to_artist_id=index_to_artist_id,
            num_artists=num_artists,
            artist_indices=[int(idx) for idx in negative_draws],
        )
        feature_rows.append(negative_features)
        labels.append(np.zeros(len(negative_draws)))

    features = np.vstack(feature_rows)
    targets = np.concatenate(labels)

    ranker = Ridge(alpha=alpha)
    ranker.fit(features, targets)
    return ranker


def rank_with_ltr(
    ranker: Ridge,
    *,
    user_id: str,
    user_item_matrix: csr_matrix,
    mappings: dict[str, Any],
    model: Any,
    artist_stats: dict[str, ArtistStats],
    recommendations: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-rank an ALS candidate list with the fitted learning-to-rank model."""
    validate_ranking_parameters(top_k)
    user_id_to_index = mappings["user_id_to_index"]
    index_to_artist_id = mappings["index_to_artist_id"]
    num_artists = len(mappings["artist_id_to_index"])

    artist_indices: list[int] = []
    for recommendation in recommendations:
        artist_id = str(recommendation["artist_id"])
        # Candidates the fitted model has never seen get ordered last so the
        # collaborative scores remain a sensible fallback ordering.
        if artist_id not in mappings["artist_id_to_index"]:
            artist_indices.append(-1)
        else:
            artist_indices.append(mappings["artist_id_to_index"][artist_id])

    user_index = user_id_to_index[user_id]
    known_positions = [i for i in artist_indices if i >= 0]
    if not known_positions:
        return recommendations

    candidate_features = _user_feature_vector(
        user_index=user_index,
        model=model,
        user_item_matrix=user_item_matrix,
        artist_stats=artist_stats,
        index_to_artist_id=index_to_artist_id,
        num_artists=num_artists,
        artist_indices=[idx for idx in artist_indices if idx >= 0],
    )
    predicted = ranker.predict(candidate_features)
    scored_pairs: list[tuple[float, dict[str, Any]]] = []
    prediction_position = 0
    for artist_index, recommendation in zip(
        artist_indices,
        recommendations,
        strict=True,
    ):
        if artist_index >= 0:
            scored_pairs.append((float(predicted[prediction_position]), recommendation))
            prediction_position += 1
        else:
            scored_pairs.append((float("-inf"), recommendation))
    scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
    re_ranked = [recommendation for _, recommendation in scored_pairs]
    return re_ranked[:top_k]


def _track_popularity_features(
    track_id: str,
    track_stats: dict[str, Any],
    num_tracks: int,
) -> tuple[float, float]:
    """Return (log-total-plays, normalized popularity-rank) for a track."""
    stats = track_stats.get(track_id)
    if stats is None:
        return 0.0, 0.0
    total_plays = float(stats["total_plays"])
    log_plays = float(np.log1p(total_plays))
    rank = float(stats["popularity_rank"]) - 1.0
    normalized_rank = rank / num_tracks if num_tracks else 0.0
    return log_plays, normalized_rank


def _validate_track_ltr_inputs(
    train_df: Any,
    resources: Any,
) -> None:
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame.")
    if train_df.empty:
        raise ValueError("train_df must not be empty.")
    required_cols = ("user_id", "track_id", "play_count")
    missing = [col for col in required_cols if col not in train_df.columns]
    if missing:
        raise ValueError(f"train_df missing required columns: {missing}")
    if (
        resources is None
        or not hasattr(resources, "user_track_matrix")
        or not hasattr(resources, "similarity_matrix")
        or not hasattr(resources, "track_id_to_index")
        or not hasattr(resources, "track_stats")
        or not hasattr(resources, "track_ids")
    ):
        raise ValueError("resources must provide track serving structures.")


def train_track_ltr_ranker(
    *,
    train_df: pd.DataFrame,
    resources: TrackServingResources,
    negatives_per_positive: int = 3,
    random_state: int = 42,
    alpha: float = 1.0,
) -> Ridge:
    """Fit a pointwise learning-to-rank re-ranker on track training interactions.

    Positive examples come from observed user-track interactions; negative
    candidates are sampled uniformly from tracks the user has not interacted with.
    The returned regressor predicts a track's relevance score to a user based on
    base content similarity to user history, log total plays, normalized popularity
    rank, and user interaction count.
    """
    _validate_track_ltr_inputs(train_df, resources)
    if type(negatives_per_positive) is not int or negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be a positive integer.")
    if type(random_state) is not int:
        raise ValueError("random_state must be an integer.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than 0.")

    num_tracks = len(resources.track_ids)
    if num_tracks == 0:
        raise ValueError("resources.track_ids must not be empty.")
    rng = np.random.default_rng(random_state)
    index_to_track_id = {v: k for k, v in resources.track_id_to_index.items()}

    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for _, user_df in train_df.groupby("user_id"):
        user_track_ids = set(user_df["track_id"])
        interacted_indices = {
            resources.track_id_to_index[tid]
            for tid in user_track_ids
            if tid in resources.track_id_to_index
        }
        if not interacted_indices:
            continue

        listened_indices = sorted(interacted_indices)
        similarity_scores = resources.similarity_matrix[:, listened_indices].mean(
            axis=1
        )
        user_interaction_count = float(len(interacted_indices))

        positive_indices = list(interacted_indices)
        pos_sim = similarity_scores[positive_indices]
        pos_pop = [
            _track_popularity_features(
                index_to_track_id[idx], resources.track_stats, num_tracks
            )
            for idx in positive_indices
        ]
        pos_log_plays = [p[0] for p in pos_pop]
        pos_norm_ranks = [p[1] for p in pos_pop]

        pos_features = np.column_stack(
            [
                pos_sim,
                pos_log_plays,
                pos_norm_ranks,
                np.full(len(positive_indices), user_interaction_count),
            ]
        )
        feature_rows.append(np.nan_to_num(pos_features))

        summed_plays = user_df.groupby("track_id")["play_count"].sum()
        play_by_track = {
            track_id: float(plays) for track_id, plays in summed_plays.items()
        }
        pos_labels = np.array(
            [
                play_by_track.get(index_to_track_id[idx], 0.0)
                for idx in positive_indices
            ],
            dtype=float,
        )
        labels.append(pos_labels)

        negative_candidates = [
            idx for idx in range(num_tracks) if idx not in interacted_indices
        ]
        if negative_candidates:
            negative_draws = rng.choice(
                negative_candidates,
                size=min(
                    negatives_per_positive * len(positive_indices),
                    len(negative_candidates),
                ),
                replace=False,
            )
            neg_indices = [int(idx) for idx in negative_draws]
            neg_sim = similarity_scores[neg_indices]
            neg_pop = [
                _track_popularity_features(
                    index_to_track_id[idx], resources.track_stats, num_tracks
                )
                for idx in neg_indices
            ]
            neg_log_plays = [p[0] for p in neg_pop]
            neg_norm_ranks = [p[1] for p in neg_pop]

            neg_features = np.column_stack(
                [
                    neg_sim,
                    neg_log_plays,
                    neg_norm_ranks,
                    np.full(len(neg_indices), user_interaction_count),
                ]
            )
            feature_rows.append(np.nan_to_num(neg_features))
            labels.append(np.zeros(len(neg_indices), dtype=float))

    if not feature_rows:
        raise ValueError(
            "No valid training interactions found to train track LTR ranker."
        )

    features = np.vstack(feature_rows)
    targets = np.concatenate(labels)

    ranker = Ridge(alpha=alpha)
    ranker.fit(features, targets)
    return ranker


def rank_tracks_with_ltr(
    ranker: Ridge,
    *,
    recommendations: list[dict[str, Any]],
    user_id: str,
    resources: TrackServingResources,
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-rank candidate track recommendations with the fitted ranker."""
    validate_ranking_parameters(top_k)
    if not recommendations:
        return []
    if ranker is None:
        raise ValueError("ranker must not be None.")

    num_tracks = len(resources.track_ids)
    index_to_track_id = {v: k for k, v in resources.track_id_to_index.items()}

    track_indices: list[int] = []
    for recommendation in recommendations:
        track_id = str(recommendation["track_id"])
        if track_id not in resources.track_id_to_index:
            track_indices.append(-1)
        else:
            track_indices.append(resources.track_id_to_index[track_id])

    known_positions = [i for i in track_indices if i >= 0]
    if not known_positions:
        return recommendations[:top_k]

    # Compute similarity to user history if user is known
    similarity_scores: np.ndarray | None = None
    user_interaction_count = 0.0
    if user_id in resources.user_track_matrix.index:
        user_row = resources.user_track_matrix.loc[user_id]
        listened_tids = user_row[user_row > 0].index.tolist()
        listened_indices = [
            resources.track_id_to_index[tid]
            for tid in listened_tids
            if tid in resources.track_id_to_index
        ]
        if listened_indices:
            similarity_scores = resources.similarity_matrix[:, listened_indices].mean(
                axis=1
            )
            user_interaction_count = float(len(listened_indices))

    candidate_rows: list[list[float]] = []
    for track_index, recommendation in zip(track_indices, recommendations, strict=True):
        if track_index >= 0:
            if similarity_scores is not None:
                base_score = float(similarity_scores[track_index])
            else:
                base_score = float(recommendation.get("score", 0.0))
            track_id = index_to_track_id[track_index]
            log_plays, norm_rank = _track_popularity_features(
                track_id, resources.track_stats, num_tracks
            )
            candidate_rows.append(
                [base_score, log_plays, norm_rank, user_interaction_count]
            )

    candidate_features = np.nan_to_num(np.array(candidate_rows, dtype=float))
    predicted = ranker.predict(candidate_features)

    scored_pairs: list[tuple[float, dict[str, Any]]] = []
    pred_idx = 0
    for track_index, recommendation in zip(track_indices, recommendations, strict=True):
        if track_index >= 0:
            scored_pairs.append((float(predicted[pred_idx]), recommendation))
            pred_idx += 1
        else:
            scored_pairs.append((float("-inf"), recommendation))

    scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
    re_ranked = [recommendation for _, recommendation in scored_pairs]
    return re_ranked[:top_k]
