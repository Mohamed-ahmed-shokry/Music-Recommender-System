"""Data loading and validation for music listening interactions."""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("user_id", "artist_id", "artist_name", "play_count")


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load interaction data from a CSV file."""
    return pd.read_csv(path)


def validate_interactions(df: pd.DataFrame) -> None:
    """Validate that an interactions dataframe has the expected schema."""
    if df.empty:
        raise ValueError("Interactions dataframe is empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["user_id"].isna().any():
        raise ValueError("Column 'user_id' contains missing values.")
    if df["artist_id"].isna().any():
        raise ValueError("Column 'artist_id' contains missing values.")
    if df["artist_name"].isna().any():
        raise ValueError("Column 'artist_name' contains missing values.")
    if df["play_count"].isna().any():
        raise ValueError("Column 'play_count' contains missing values.")

    for column in ("user_id", "artist_id", "artist_name"):
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

    normalized_names = pd.DataFrame(
        {
            "artist_id": df["artist_id"].astype("string").str.strip(),
            "artist_name": df["artist_name"].astype("string").str.strip(),
        }
    )
    names_per_artist = normalized_names.groupby("artist_id")["artist_name"].nunique()
    conflicting_artist_ids = sorted(
        str(artist_id) for artist_id in names_per_artist[names_per_artist > 1].index
    )
    if conflicting_artist_ids:
        raise ValueError(
            f"Artist IDs map to multiple artist names: {conflicting_artist_ids}"
        )


def load_and_validate_interactions(path: str | Path) -> pd.DataFrame:
    """Load interactions from disk and validate them."""
    df = load_interactions(path)
    validate_interactions(df)
    return df
