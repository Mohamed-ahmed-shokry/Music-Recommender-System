"""Track-level data loading, validation, and recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

TRACK_REQUIRED_COLUMNS = (
    "user_id",
    "track_id",
    "track_name",
    "artist_id",
    "artist_name",
    "play_count",
)
TRACK_TEXT_COLUMNS = (
    "user_id",
    "track_id",
    "track_name",
    "artist_id",
    "artist_name",
)


def load_track_interactions(path: str | Path) -> pd.DataFrame:
    """Load track-level interaction data from a CSV file."""
    return pd.read_csv(
        path,
        dtype=dict.fromkeys(TRACK_TEXT_COLUMNS, "string"),
    )


def validate_track_interactions(df: pd.DataFrame) -> None:
    """Validate that a track interactions dataframe has the expected schema."""
    if df.empty:
        raise ValueError("Track interactions dataframe is empty.")

    missing_columns = [
        column for column in TRACK_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in TRACK_TEXT_COLUMNS:
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")
        if (df[column].astype("string").str.strip() == "").any():
            raise ValueError(f"Column '{column}' contains empty values.")

    if not pd.api.types.is_numeric_dtype(
        df["play_count"]
    ) or pd.api.types.is_bool_dtype(df["play_count"]):
        raise ValueError("Column 'play_count' must be numeric.")
    if not np.isfinite(df["play_count"].to_numpy(dtype=float)).all():
        raise ValueError("Column 'play_count' must contain finite values.")
    if (df["play_count"] <= 0).any():
        raise ValueError("Column 'play_count' must contain values greater than 0.")

    # Validate track-artist consistency
    normalized = df.loc[:, ["track_id", "artist_id", "artist_name"]].copy()
    for column in ["track_id", "artist_id", "artist_name"]:
        normalized[column] = normalized[column].astype("string").str.strip()

    # Check that each track maps to a single artist
    artists_per_track = normalized.groupby("track_id")["artist_id"].nunique()
    conflicting_track_ids = sorted(
        str(track_id) for track_id in artists_per_track[artists_per_track > 1].index
    )
    if conflicting_track_ids:
        raise ValueError(
            f"Track IDs map to multiple artist IDs: {conflicting_track_ids}"
        )

    # Check artist name consistency per artist_id
    names_per_artist = normalized.groupby("artist_id")["artist_name"].nunique()
    conflicting_artist_ids = sorted(
        str(artist_id) for artist_id in names_per_artist[names_per_artist > 1].index
    )
    if conflicting_artist_ids:
        raise ValueError(
            f"Artist IDs map to multiple artist names: {conflicting_artist_ids}"
        )


def normalize_track_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers and combine repeated user-track interactions."""
    validate_track_interactions(df)
    normalized = df.loc[:, TRACK_REQUIRED_COLUMNS].copy()
    for column in TRACK_TEXT_COLUMNS:
        normalized[column] = normalized[column].astype("string").str.strip()

    normalized = (
        normalized.groupby(
            ["user_id", "track_id", "track_name", "artist_id", "artist_name"],
            as_index=False,
            sort=False,
        )["play_count"]
        .sum()
        .loc[:, TRACK_REQUIRED_COLUMNS]
    )
    validate_track_interactions(normalized)
    return normalized


def load_and_validate_track_interactions(path: str | Path) -> pd.DataFrame:
    """Load, validate, and normalize track interactions from disk."""
    df = load_track_interactions(path)
    return normalize_track_interactions(df)


TRACK_METADATA_REQUIRED_COLUMNS = (
    "track_id",
    "track_name",
    "artist_id",
    "artist_name",
    "album_id",
    "album_name",
    "duration_ms",
    "popularity",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
)
TRACK_METADATA_TEXT_COLUMNS = (
    "track_id",
    "track_name",
    "artist_id",
    "artist_name",
    "album_id",
    "album_name",
)


def load_track_metadata(path: str | Path) -> pd.DataFrame:
    """Load track metadata from a CSV file."""
    return pd.read_csv(
        path,
        dtype=dict.fromkeys(TRACK_METADATA_TEXT_COLUMNS, "string"),
    )


def validate_track_metadata(
    df: pd.DataFrame, interactions_df: pd.DataFrame | None = None
) -> None:
    """Validate track metadata dataframe."""
    if df.empty:
        raise ValueError("Track metadata dataframe is empty.")

    missing_columns = [
        column for column in TRACK_METADATA_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in TRACK_METADATA_TEXT_COLUMNS:
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")
        if (df[column].astype("string").str.strip() == "").any():
            raise ValueError(f"Column '{column}' contains empty values.")

    # Check for duplicate track IDs
    if df["track_id"].duplicated().any():
        raise ValueError("Duplicate track IDs found in metadata.")

    # If interactions provided, check coverage
    if interactions_df is not None:
        interaction_tracks = set(
            interactions_df["track_id"].astype("string").str.strip()
        )
        metadata_tracks = set(df["track_id"].astype("string").str.strip())
        missing_tracks = interaction_tracks - metadata_tracks
        if missing_tracks:
            raise ValueError(
                "Tracks in interactions not found in metadata: "
                f"{sorted(missing_tracks)}"
            )


def load_and_validate_track_metadata(
    metadata_path: str | Path, interactions_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Load, validate track metadata."""
    df = load_track_metadata(metadata_path)
    validate_track_metadata(df, interactions_df)
    return df


def build_track_content_matrix(
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build content feature matrix for tracks using audio features."""
    # Use audio features as content
    audio_feature_cols = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]

    # Normalize audio features
    feature_df = metadata_df.set_index("track_id")[audio_feature_cols].copy()

    # Handle loudness (typically negative, shift to positive)
    if "loudness" in feature_df.columns:
        feature_df["loudness"] = feature_df["loudness"] + 60

    # Normalize tempo to 0-1 range
    if "tempo" in feature_df.columns:
        max_tempo = feature_df["tempo"].max()
        if max_tempo > 0:
            feature_df["tempo"] = feature_df["tempo"] / max_tempo

    # Normalize key to 0-1 (0-11)
    if "key" in feature_df.columns:
        feature_df["key"] = feature_df["key"] / 11.0

    # Normalize time_signature (typically 1-7)
    if "time_signature" in feature_df.columns:
        feature_df["time_signature"] = feature_df["time_signature"] / 7.0

    feature_names = audio_feature_cols
    return feature_df, feature_names


def recommend_tracks_for_user(
    user_id: str,
    user_track_matrix: pd.DataFrame,
    track_similarity_matrix: np.ndarray,
    track_id_to_index: dict[str, int],
    top_k: int = 10,
    include_listened: bool = False,
) -> list[dict[str, Any]]:
    """Recommend tracks for a user based on track similarity.

    Uses a simple collaborative filtering approach: find similar tracks
    to the user's listening history and rank by similarity score.
    """
    if user_id not in user_track_matrix.index:
        return []

    user_tracks = user_track_matrix.loc[user_id]
    listened_tracks = user_tracks[user_tracks > 0].index.tolist()

    if not listened_tracks:
        return []

    # Get indices of listened tracks
    listened_indices = [
        track_id_to_index[tid] for tid in listened_tracks if tid in track_id_to_index
    ]

    if not listened_indices:
        return []

    # Compute average similarity to listened tracks for all tracks
    similarity_scores = track_similarity_matrix[:, listened_indices].mean(axis=1)

    # Create ranked list
    ranked_indices = np.argsort(similarity_scores)[::-1]

    recommendations: list[dict[str, Any]] = []
    index_to_track_id = {v: k for k, v in track_id_to_index.items()}

    for idx in ranked_indices:
        track_id = index_to_track_id[idx]
        if not include_listened and track_id in listened_tracks:
            continue
        if len(recommendations) >= top_k:
            break
        recommendations.append(
            {
                "track_id": track_id,
                "score": float(similarity_scores[idx]),
            }
        )

    return recommendations


def get_similar_tracks(
    track_id: str,
    track_similarity_matrix: np.ndarray,
    track_id_to_index: dict[str, int],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Get tracks similar to a given track."""
    if track_id not in track_id_to_index:
        return []

    track_idx = track_id_to_index[track_id]
    similarities = track_similarity_matrix[track_idx]

    ranked_indices = np.argsort(similarities)[::-1]
    index_to_track_id = {v: k for k, v in track_id_to_index.items()}

    recommendations: list[dict[str, Any]] = []
    for idx in ranked_indices:
        if idx == track_idx:
            continue
        if len(recommendations) >= top_k:
            break
        recommendations.append(
            {
                "track_id": index_to_track_id[idx],
                "score": float(similarities[idx]),
            }
        )

    return recommendations


@dataclass
class TrackServingResources:
    """Precomputed track data for serving recommendations."""

    interactions: pd.DataFrame
    metadata: pd.DataFrame
    feature_names: list[str]
    track_ids: list[str]
    track_id_to_index: dict[str, int]
    similarity_matrix: np.ndarray
    user_track_matrix: pd.DataFrame
    track_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_track_serving_resources(
    interactions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> TrackServingResources:
    """Build cached serving resources from validated track dataframes."""
    validate_track_interactions(interactions_df)
    validate_track_metadata(metadata_df, interactions_df)
    feature_df, feature_names = build_track_content_matrix(metadata_df)
    track_ids = [str(track_id) for track_id in feature_df.index.tolist()]
    track_id_to_index = {track_id: i for i, track_id in enumerate(track_ids)}
    similarity_matrix = cosine_similarity(feature_df.values)
    user_track_matrix = interactions_df.pivot_table(
        index="user_id",
        columns="track_id",
        values="play_count",
        fill_value=0,
    )
    lookup: dict[str, dict[str, Any]] = {}
    for row in metadata_df.itertuples(index=False):
        lookup[str(row.track_id)] = {
            "track_id": str(row.track_id),
            "track_name": str(row.track_name),
            "artist_id": str(row.artist_id),
            "artist_name": str(row.artist_name),
            "popularity": row.popularity,
        }
    return TrackServingResources(
        interactions=interactions_df,
        metadata=metadata_df,
        feature_names=feature_names,
        track_ids=track_ids,
        track_id_to_index=track_id_to_index,
        similarity_matrix=similarity_matrix,
        user_track_matrix=user_track_matrix,
        track_lookup=lookup,
    )


def load_track_serving_resources(
    data_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> TrackServingResources:
    """Load track CSVs and build serving resources."""
    from music_recommender.config import RAW_TRACK_DATA_PATH, RAW_TRACK_METADATA_PATH

    interactions = load_and_validate_track_interactions(
        data_path or RAW_TRACK_DATA_PATH
    )
    metadata = load_and_validate_track_metadata(
        metadata_path or RAW_TRACK_METADATA_PATH, interactions
    )
    return build_track_serving_resources(interactions, metadata)
