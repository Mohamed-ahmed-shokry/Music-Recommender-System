"""Data loading and validation for music listening interactions."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("user_id", "artist_id", "artist_name", "play_count")


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load interaction data from a CSV file."""
    return pd.read_csv(path)


def validate_interactions(df: pd.DataFrame) -> None:
    """Validate that an interactions dataframe has the expected schema."""
    if df.empty:
        raise ValueError("Interactions dataframe is empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
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

    if not pd.api.types.is_numeric_dtype(df["play_count"]):
        raise ValueError("Column 'play_count' must be numeric.")
    if (df["play_count"] <= 0).any():
        raise ValueError("Column 'play_count' must contain values greater than 0.")


def load_and_validate_interactions(path: str | Path) -> pd.DataFrame:
    """Load interactions from disk and validate them."""
    df = load_interactions(path)
    validate_interactions(df)
    return df
