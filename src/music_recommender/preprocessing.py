"""Preprocessing utilities for ALS training."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from music_recommender.data import load_and_validate_interactions

Mappings = dict[str, dict[Any, Any]]


def filter_interactions(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_artist_interactions: int,
) -> pd.DataFrame:
    """Remove users and artists with too few interactions."""
    filtered = df.copy()

    while True:
        start_count = len(filtered)
        user_counts = filtered.groupby("user_id")["artist_id"].transform("count")
        filtered = filtered[user_counts >= min_user_interactions]

        artist_counts = filtered.groupby("artist_id")["user_id"].transform("count")
        filtered = filtered[artist_counts >= min_artist_interactions]

        if len(filtered) == start_count:
            break

    return filtered.reset_index(drop=True)


def create_id_mappings(df: pd.DataFrame) -> Mappings:
    """Create stable user and artist ID mappings."""
    user_ids = list(pd.unique(df["user_id"]))
    artist_ids = list(pd.unique(df["artist_id"]))

    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    index_to_user_id = {index: user_id for user_id, index in user_id_to_index.items()}
    artist_id_to_index = {
        artist_id: index for index, artist_id in enumerate(artist_ids)
    }
    index_to_artist_id = {
        index: artist_id for artist_id, index in artist_id_to_index.items()
    }
    artist_id_to_name = (
        df[["artist_id", "artist_name"]]
        .drop_duplicates(subset="artist_id")
        .set_index("artist_id")["artist_name"]
        .to_dict()
    )

    return {
        "user_id_to_index": user_id_to_index,
        "index_to_user_id": index_to_user_id,
        "artist_id_to_index": artist_id_to_index,
        "index_to_artist_id": index_to_artist_id,
        "artist_id_to_name": artist_id_to_name,
    }


def build_user_item_matrix(
    df: pd.DataFrame,
    user_id_to_index: dict[str, int],
    artist_id_to_index: dict[str, int],
) -> csr_matrix:
    """Build a sparse user-item matrix from play counts."""
    row_indices = df["user_id"].map(user_id_to_index).to_numpy()
    column_indices = df["artist_id"].map(artist_id_to_index).to_numpy()
    values = df["play_count"].to_numpy(dtype=np.float32)

    matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(user_id_to_index), len(artist_id_to_index)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


def save_mappings(mappings: Mappings, path: str | Path) -> None:
    """Save ID mappings to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mappings, output_path)


def load_mappings(path: str | Path) -> Mappings:
    """Load ID mappings from disk."""
    return joblib.load(path)


def prepare_training_data(
    raw_data_path: str | Path,
    mappings_path: str | Path,
    min_user_interactions: int,
    min_artist_interactions: int,
) -> tuple[pd.DataFrame, csr_matrix, Mappings]:
    """Load, validate, filter, map, and persist training data."""
    df = load_and_validate_interactions(raw_data_path)
    filtered_df = filter_interactions(
        df,
        min_user_interactions=min_user_interactions,
        min_artist_interactions=min_artist_interactions,
    )
    mappings = create_id_mappings(filtered_df)
    user_item_matrix = build_user_item_matrix(
        filtered_df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    save_mappings(mappings, mappings_path)
    return filtered_df, user_item_matrix, mappings
