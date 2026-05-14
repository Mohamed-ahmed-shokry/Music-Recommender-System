from pathlib import Path

import pandas as pd

from music_recommender.artifacts import build_recommender_artifact, save_artifact
from music_recommender.content import build_content_artifacts
from music_recommender.model import train_als_model
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings
from music_recommender.service import RecommenderService


def service_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
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
            "artist_name": ["A", "B", "B", "C", "C", "D"],
            "play_count": [10, 8, 9, 7, 8, 6],
        }
    )


def service_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artist_id": ["artist_1", "artist_2", "artist_3", "artist_4"],
            "artist_name": ["A", "B", "C", "D"],
            "genres": ["pop", "pop;dance", "rock", "soul"],
            "mood_tags": ["bright", "bright;fun", "raw", "warm"],
            "country": ["United States", "United States", "United Kingdom", "Canada"],
            "era": ["2020s", "2020s", "2000s", "2010s"],
        }
    )


def create_service(tmp_path: Path) -> RecommenderService:
    df = service_dataframe()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(matrix, 4, 0.01, 1, 10.0, use_gpu=False)
    content_artifacts = build_content_artifacts(
        service_metadata_df(),
        ["artist_1", "artist_2", "artist_3", "artist_4"],
    )
    artifact = build_recommender_artifact(
        model=model,
        mappings=mappings,
        user_item_matrix=matrix,
        filtered_df=df,
        content_artifacts=content_artifacts,
        raw_data_path=tmp_path / "missing.csv",
        metadata_path=tmp_path / "metadata.csv",
        training_config={"factors": 4},
        hybrid_config={"default_content_weight": 0.25},
    )
    artifact_path = tmp_path / "artifact.joblib"
    save_artifact(artifact, artifact_path)
    return RecommenderService.from_artifacts(artifact_path)


def test_known_user_returns_hybrid_strategy(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    response = service.recommend_user("user_1", top_k=2, explain=True)

    assert response["strategy"] == "hybrid_personalized"
    assert response["content_weight"] == 0.25
    assert response["recommendations"]
    assert "score_components" in response["recommendations"][0]
    assert "reasons" in response["recommendations"][0]


def test_unknown_user_returns_popular_fallback(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    response = service.recommend_user("missing_user", top_k=2)

    assert response["strategy"] == "popular_fallback"
    assert "Unknown user_id" in response["message"]
    assert response["recommendations"][0]["popularity_rank"] == 1


def test_profile_recommendations_return_content_strategy(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    response = service.recommend_profile(
        artist_ids=["artist_1"],
        genres=["pop"],
        top_k=2,
        explain=True,
    )

    assert response["strategy"] == "content_profile"
    assert response["recommendations"]
    assert response["recommendations"][0]["artist_id"] != "artist_1"
    assert response["recommendations"][0]["reasons"]


def test_session_recommendations_blend_user_and_session_preferences(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    response = service.recommend_session(
        user_id="user_1",
        artist_ids=["artist_1"],
        genres=["pop"],
        top_k=2,
        explain=True,
    )

    assert response["strategy"] == "session_hybrid"
    assert response["seed_artist_ids"] == ["artist_1"]
    assert response["recommendations"]
    assert all(
        recommendation["artist_id"] not in {"artist_1", "artist_2"}
        for recommendation in response["recommendations"]
    )
    score_components = response["recommendations"][0]["score_components"]
    assert "collaborative_score" in score_components
    assert "session_content_score" in score_components
    assert response["recommendations"][0]["reasons"]


def test_session_recommendations_fall_back_for_unknown_user(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    response = service.recommend_session(
        user_id="missing_user",
        artist_ids=["artist_1"],
        exclude_artist_ids=["artist_2"],
        top_k=2,
    )

    assert response["strategy"] == "session_content"
    assert "Unknown user_id" in response["message"]
    assert "artist_2" in response["excluded_artist_ids"]
    assert all(
        recommendation["artist_id"] != "artist_2"
        for recommendation in response["recommendations"]
    )


def test_content_similar_artists_return_content_strategy(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    response = service.content_similar_artists("artist_1", top_k=2, explain=True)

    assert response["strategy"] == "content_similarity"
    assert response["similar_artists"]
    assert response["similar_artists"][0]["artist_id"] != "artist_1"


def test_service_metadata_is_available(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    metadata = service.metadata()

    assert metadata["version"] == "4.0"
    assert metadata["metadata"]["num_users"] == 3
    assert metadata["training_config"]["factors"] == 4
