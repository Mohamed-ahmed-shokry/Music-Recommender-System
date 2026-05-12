import pandas as pd
import pytest

from music_recommender.metadata import (
    validate_artist_metadata,
    validate_metadata_coverage,
)


def valid_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artist_id": ["artist_1", "artist_2"],
            "artist_name": ["Artist One", "Artist Two"],
            "genres": ["pop;dance", "rock;indie"],
            "mood_tags": ["bright;fun", "cool;raw"],
            "country": ["United States", "United Kingdom"],
            "era": ["2020s", "2000s"],
        }
    )


def test_valid_metadata_passes_validation() -> None:
    validate_artist_metadata(valid_metadata_df())


def test_missing_metadata_column_raises_value_error() -> None:
    df = valid_metadata_df().drop(columns=["genres"])

    with pytest.raises(ValueError, match="Missing metadata columns"):
        validate_artist_metadata(df)


def test_empty_metadata_raises_value_error() -> None:
    df = pd.DataFrame(columns=valid_metadata_df().columns)

    with pytest.raises(ValueError, match="empty"):
        validate_artist_metadata(df)


def test_duplicate_artist_ids_raise_value_error() -> None:
    df = valid_metadata_df()
    df.loc[1, "artist_id"] = "artist_1"

    with pytest.raises(ValueError, match="duplicate"):
        validate_artist_metadata(df)


def test_missing_genres_raise_value_error() -> None:
    df = valid_metadata_df()
    df.loc[0, "genres"] = ""

    with pytest.raises(ValueError, match="genres"):
        validate_artist_metadata(df)


def test_missing_metadata_coverage_raises_value_error() -> None:
    interactions = pd.DataFrame(
        {
            "artist_id": ["artist_1", "artist_3"],
        }
    )

    with pytest.raises(ValueError, match="artist_3"):
        validate_metadata_coverage(interactions, valid_metadata_df())
