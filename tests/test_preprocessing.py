import pandas as pd
import pytest

from music_recommender.preprocessing import (
    build_user_item_matrix,
    create_id_mappings,
    filter_interactions,
    load_mappings,
)


def interactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "artist_id": ["artist_1", "artist_2", "artist_2"],
            "artist_name": ["The Weeknd", "Drake", "Drake"],
            "play_count": [10, 5, 20],
        }
    )


def test_mappings_are_created_correctly() -> None:
    mappings = create_id_mappings(interactions_df())

    assert mappings["user_id_to_index"] == {"user_1": 0, "user_2": 1}
    assert mappings["index_to_user_id"] == {0: "user_1", 1: "user_2"}
    assert mappings["artist_id_to_index"] == {"artist_1": 0, "artist_2": 1}
    assert mappings["index_to_artist_id"] == {0: "artist_1", 1: "artist_2"}
    assert mappings["artist_id_to_name"] == {
        "artist_1": "The Weeknd",
        "artist_2": "Drake",
    }


def test_sparse_matrix_has_correct_shape() -> None:
    df = interactions_df()
    mappings = create_id_mappings(df)

    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )

    assert matrix.shape == (2, 2)


def test_sparse_matrix_contains_expected_play_counts() -> None:
    df = interactions_df()
    mappings = create_id_mappings(df)

    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )

    assert matrix[0, 0] == 10
    assert matrix[0, 1] == 5
    assert matrix[1, 1] == 20


@pytest.mark.parametrize(
    ("min_user_interactions", "min_artist_interactions", "message"),
    [
        (0, 1, "min_user_interactions"),
        (1, 0, "min_artist_interactions"),
        (-1, 1, "min_user_interactions"),
        (1, 1.5, "min_artist_interactions"),
        (True, 1, "min_user_interactions"),
    ],
)
def test_filter_interactions_rejects_invalid_thresholds(
    min_user_interactions,
    min_artist_interactions,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        filter_interactions(
            interactions_df(),
            min_user_interactions=min_user_interactions,
            min_artist_interactions=min_artist_interactions,
        )


def test_filter_interactions_rejects_empty_result() -> None:
    with pytest.raises(ValueError, match="No interactions remain"):
        filter_interactions(
            interactions_df(),
            min_user_interactions=3,
            min_artist_interactions=3,
        )


def test_load_mappings_raises_when_artifact_is_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Mappings artifact not found"):
        load_mappings(tmp_path / "missing_mappings.joblib")


def test_load_mappings_raises_on_corrupt_artifact(tmp_path) -> None:
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"not a valid joblib payload")

    with pytest.raises(ValueError, match="could not be loaded"):
        load_mappings(corrupt_path)
