"""Recommendation helpers for users and artists."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix

from music_recommender.config import (
    DEFAULT_MIN_ARTIST_INTERACTIONS,
    DEFAULT_MIN_USER_INTERACTIONS,
    MAPPINGS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)
from music_recommender.data import load_and_validate_interactions
from music_recommender.model import load_model
from music_recommender.preprocessing import (
    Mappings,
    build_user_item_matrix,
    filter_interactions,
    load_mappings,
)

if TYPE_CHECKING:
    from implicit.als import AlternatingLeastSquares

Recommendation = dict[str, str | float]


def recommend_artists_for_user(
    model: AlternatingLeastSquares,
    user_id: str,
    user_item_matrix: csr_matrix,
    mappings: Mappings,
    top_k: int,
) -> list[Recommendation]:
    """Recommend artists for a user by original user ID."""
    user_id_to_index = mappings["user_id_to_index"]
    index_to_artist_id = mappings["index_to_artist_id"]
    artist_id_to_name = mappings["artist_id_to_name"]

    if user_id not in user_id_to_index:
        raise ValueError(f"Unknown user_id: {user_id}")

    user_index = user_id_to_index[user_id]
    user_factors = model.item_factors
    artist_factors = model.user_factors
    scores = artist_factors @ user_factors[user_index]
    listened_artist_indices = set(user_item_matrix[user_index].indices)
    ranked_artist_indices = np.argsort(scores)[::-1]

    recommendations: list[Recommendation] = []
    for artist_index in ranked_artist_indices:
        artist_index = int(artist_index)
        if artist_index in listened_artist_indices:
            continue

        artist_id = index_to_artist_id[int(artist_index)]
        recommendations.append(
            {
                "artist_id": artist_id,
                "artist_name": artist_id_to_name[artist_id],
                "score": float(scores[artist_index]),
            }
        )
        if len(recommendations) == top_k:
            break

    return recommendations


def get_similar_artists(
    model: AlternatingLeastSquares,
    artist_id: str,
    mappings: Mappings,
    top_k: int,
) -> list[Recommendation]:
    """Find artists similar to an original artist ID."""
    artist_id_to_index = mappings["artist_id_to_index"]
    index_to_artist_id = mappings["index_to_artist_id"]
    artist_id_to_name = mappings["artist_id_to_name"]

    if artist_id not in artist_id_to_index:
        raise ValueError(f"Unknown artist_id: {artist_id}")

    artist_index = artist_id_to_index[artist_id]
    artist_factors = model.user_factors
    query_vector = artist_factors[artist_index]
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return []

    norms = np.linalg.norm(artist_factors, axis=1)
    denominator = norms * query_norm
    scores = np.divide(
        artist_factors @ query_vector,
        denominator,
        out=np.zeros_like(norms),
        where=denominator != 0,
    )
    ranked_artist_indices = np.argsort(scores)[::-1]

    similar_artists: list[Recommendation] = []
    for similar_index in ranked_artist_indices:
        similar_index = int(similar_index)
        similar_artist_id = index_to_artist_id[int(similar_index)]
        if similar_artist_id == artist_id:
            continue
        similar_artists.append(
            {
                "artist_id": similar_artist_id,
                "artist_name": artist_id_to_name[similar_artist_id],
                "score": float(scores[similar_index]),
            }
        )
        if len(similar_artists) == top_k:
            break

    return similar_artists


def load_recommender_artifacts(
    model_path: str | Path = MODEL_PATH,
    mappings_path: str | Path = MAPPINGS_PATH,
    raw_data_path: str | Path = RAW_DATA_PATH,
) -> tuple[AlternatingLeastSquares, csr_matrix, Mappings]:
    """Load the saved model, mappings, and matching user-item matrix."""
    if not Path(model_path).exists():
        raise FileNotFoundError("Model artifact not found. Train the model first.")
    if not Path(mappings_path).exists():
        raise FileNotFoundError("Mappings artifact not found. Train the model first.")

    model = load_model(model_path)
    mappings = load_mappings(mappings_path)
    df = load_and_validate_interactions(raw_data_path)
    filtered_df = filter_interactions(
        df,
        min_user_interactions=DEFAULT_MIN_USER_INTERACTIONS,
        min_artist_interactions=DEFAULT_MIN_ARTIST_INTERACTIONS,
    )
    user_item_matrix = build_user_item_matrix(
        filtered_df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    return model, user_item_matrix, mappings


def format_recommendations(recommendations: list[dict[str, Any]]) -> str:
    """Format recommendations as human-readable CLI output."""
    if not recommendations:
        return "No recommendations found."

    lines = []
    for index, recommendation in enumerate(recommendations, start=1):
        lines.append(
            f"{index}. {recommendation['artist_name']} "
            f"({recommendation['artist_id']}) - score: {recommendation['score']:.4f}"
        )
    return "\n".join(lines)
