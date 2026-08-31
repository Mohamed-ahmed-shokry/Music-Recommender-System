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

from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from music_recommender.artifacts import ArtistStats
from music_recommender.ranking import validate_ranking_parameters


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

        play_by_artist = {
            artist_id: float(plays)
            for artist_id, plays in user_df.groupby("artist_id")["play_count"]
            .sum()
            .items()  # noqa: E501
        }
        positive_labels = np.array(
            [
                play_by_artist.get(index_to_artist_id[idx], 0.0)  # noqa: E501
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
