"""Artist metadata loading and validation."""

from pathlib import Path

import pandas as pd

METADATA_COLUMNS = (
    "artist_id",
    "artist_name",
    "genres",
    "mood_tags",
    "country",
    "era",
)


def load_artist_metadata(path: str | Path) -> pd.DataFrame:
    """Load artist metadata from a CSV file."""
    return pd.read_csv(path)


def validate_artist_metadata(df: pd.DataFrame) -> None:
    """Validate artist metadata required for content-based recommendations."""
    if df.empty:
        raise ValueError("Artist metadata dataframe is empty.")

    missing_columns = [column for column in METADATA_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")

    if df["artist_id"].isna().any():
        raise ValueError("Column 'artist_id' contains missing values.")
    if df["artist_id"].duplicated().any():
        raise ValueError("Column 'artist_id' contains duplicate values.")

    required_text_columns = ("artist_name", "genres", "mood_tags", "country", "era")
    for column in required_text_columns:
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values.")
        if (df[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Column '{column}' contains empty values.")


def validate_metadata_coverage(
    interactions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    """Ensure every interaction artist has metadata."""
    interaction_artist_ids = set(interactions_df["artist_id"])
    metadata_artist_ids = set(metadata_df["artist_id"])
    missing_artist_ids = sorted(interaction_artist_ids - metadata_artist_ids)
    if missing_artist_ids:
        raise ValueError(f"Missing metadata for artist IDs: {missing_artist_ids}")


def load_and_validate_artist_metadata(
    path: str | Path,
    interactions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load metadata and validate schema plus optional interaction coverage."""
    metadata_df = load_artist_metadata(path)
    validate_artist_metadata(metadata_df)
    if interactions_df is not None:
        validate_metadata_coverage(interactions_df, metadata_df)
    return metadata_df
