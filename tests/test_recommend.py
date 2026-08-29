import numpy as np
import pandas as pd
import pytest

from music_recommender.model import train_als_model
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings
from music_recommender.recommend import (
    format_recommendations,
    get_similar_artists,
    load_recommender_artifacts,
    recommend_artists_for_user,
)


def recommender_artifacts():
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_2", "user_3", "user_3"],
            "artist_id": [
                "artist_1",
                "artist_2",
                "artist_2",
                "artist_3",
                "artist_3",
                "artist_4",
            ],
            "artist_name": [
                "The Weeknd",
                "Drake",
                "Drake",
                "Kendrick Lamar",
                "Kendrick Lamar",
                "SZA",
            ],
            "play_count": [10, 8, 9, 7, 8, 6],
        }
    )
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(
        user_item_matrix=matrix,
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=False,
    )
    return model, matrix, mappings


def test_recommend_artists_for_user_returns_list() -> None:
    model, matrix, mappings = recommender_artifacts()

    recommendations = recommend_artists_for_user(
        model=model,
        user_id="user_1",
        user_item_matrix=matrix,
        mappings=mappings,
        top_k=2,
    )

    assert isinstance(recommendations, list)
    assert recommendations


def test_recommendations_contain_expected_fields() -> None:
    model, matrix, mappings = recommender_artifacts()

    recommendations = recommend_artists_for_user(
        model=model,
        user_id="user_1",
        user_item_matrix=matrix,
        mappings=mappings,
        top_k=1,
    )

    assert {"artist_id", "artist_name", "score"} <= recommendations[0].keys()


def test_unknown_user_raises_value_error() -> None:
    model, matrix, mappings = recommender_artifacts()

    with pytest.raises(ValueError, match="Unknown user_id"):
        recommend_artists_for_user(
            model=model,
            user_id="missing_user",
            user_item_matrix=matrix,
            mappings=mappings,
            top_k=1,
        )


def test_get_similar_artists_returns_list() -> None:
    model, _, mappings = recommender_artifacts()

    similar_artists = get_similar_artists(
        model=model,
        artist_id="artist_2",
        mappings=mappings,
        top_k=2,
    )

    assert isinstance(similar_artists, list)
    assert similar_artists
    assert similar_artists[0]["artist_id"] != "artist_2"


def test_unknown_artist_raises_value_error() -> None:
    model, _, mappings = recommender_artifacts()

    with pytest.raises(ValueError, match="Unknown artist_id"):
        get_similar_artists(
            model=model,
            artist_id="missing_artist",
            mappings=mappings,
            top_k=1,
        )


def test_get_similar_artists_returns_empty_for_zero_query_vector() -> None:
    model, _, mappings = recommender_artifacts()
    model.user_factors = np.zeros_like(model.user_factors)

    similar_artists = get_similar_artists(
        model=model,
        artist_id="artist_1",
        mappings=mappings,
        top_k=1,
    )

    assert similar_artists == []


def test_format_recommendations_returns_message_when_empty() -> None:
    assert format_recommendations([]) == "No recommendations found."


def test_load_recommender_artifacts_raises_when_model_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_recommender_artifacts(model_path=tmp_path / "missing_model.joblib")


def test_load_recommender_artifacts_raises_when_mappings_missing(tmp_path) -> None:
    import joblib

    model_path = tmp_path / "model.joblib"
    joblib.dump(object(), model_path)

    with pytest.raises(FileNotFoundError, match="Mappings artifact not found"):
        load_recommender_artifacts(
            model_path=model_path,
            mappings_path=tmp_path / "missing_mappings.joblib",
        )


def test_load_recommender_artifacts_loads_model_matrix_and_mappings(
    tmp_path,
) -> None:
    import joblib

    df = pd.DataFrame(
        {
            "user_id": [
                "user_1",
                "user_1",
                "user_1",
                "user_2",
                "user_2",
                "user_2",
                "user_3",
                "user_3",
            ],
            "artist_id": [
                "artist_1",
                "artist_2",
                "artist_3",
                "artist_2",
                "artist_3",
                "artist_4",
                "artist_1",
                "artist_4",
            ],
            "artist_name": [
                "The Weeknd",
                "Drake",
                "Kendrick Lamar",
                "Drake",
                "Kendrick Lamar",
                "SZA",
                "The Weeknd",
                "SZA",
            ],
            "play_count": [10, 8, 7, 9, 8, 6, 5, 4],
        }
    )
    df.to_csv(tmp_path / "interactions.csv", index=False)
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(
        user_item_matrix=matrix,
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=False,
    )
    model_path = tmp_path / "model.joblib"
    mappings_path = tmp_path / "mappings.joblib"
    joblib.dump(model, model_path)
    joblib.dump(mappings, mappings_path)

    loaded_model, loaded_matrix, loaded_mappings = load_recommender_artifacts(
        model_path=model_path,
        mappings_path=mappings_path,
        raw_data_path=tmp_path / "interactions.csv",
    )

    assert loaded_model.user_factors.shape == model.user_factors.shape
    assert loaded_matrix.shape == matrix.shape
    assert loaded_mappings["user_id_to_index"] == mappings["user_id_to_index"]
    assert loaded_mappings["index_to_artist_id"] == mappings["index_to_artist_id"]
