import pandas as pd
import pytest

from music_recommender.data import validate_interactions


def valid_interactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_2"],
            "artist_id": ["artist_1", "artist_2"],
            "artist_name": ["The Weeknd", "Drake"],
            "play_count": [10, 20],
        }
    )


def test_valid_dataframe_passes_validation() -> None:
    validate_interactions(valid_interactions_df())


def test_missing_column_raises_value_error() -> None:
    df = valid_interactions_df().drop(columns=["artist_name"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_interactions(df)


def test_negative_play_count_raises_value_error() -> None:
    df = valid_interactions_df()
    df.loc[0, "play_count"] = -1

    with pytest.raises(ValueError, match="greater than 0"):
        validate_interactions(df)


def test_empty_dataframe_raises_value_error() -> None:
    df = pd.DataFrame(columns=["user_id", "artist_id", "artist_name", "play_count"])

    with pytest.raises(ValueError, match="empty"):
        validate_interactions(df)


@pytest.mark.parametrize("column", ["user_id", "artist_id", "artist_name"])
def test_empty_identifiers_and_names_raise_value_error(column: str) -> None:
    df = valid_interactions_df()
    df.loc[0, column] = "   "

    with pytest.raises(ValueError, match=rf"Column '{column}' contains empty values"):
        validate_interactions(df)


@pytest.mark.parametrize("play_count", [float("inf"), float("-inf")])
def test_non_finite_play_count_raises_value_error(play_count: float) -> None:
    df = valid_interactions_df()
    df["play_count"] = df["play_count"].astype(float)
    df.loc[0, "play_count"] = play_count

    with pytest.raises(ValueError, match="finite values"):
        validate_interactions(df)


def test_boolean_play_count_raises_value_error() -> None:
    df = valid_interactions_df()
    df["play_count"] = [True, False]

    with pytest.raises(ValueError, match="must be numeric"):
        validate_interactions(df)


def test_conflicting_artist_names_raise_value_error() -> None:
    df = valid_interactions_df()
    df.loc[1, "artist_id"] = "artist_1"

    with pytest.raises(ValueError, match="multiple artist names"):
        validate_interactions(df)
