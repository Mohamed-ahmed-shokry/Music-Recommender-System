from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from music_recommender.artifacts import (
    build_artist_stats,
    build_recommender_artifact,
    save_artifact,
)
from music_recommender.content import build_content_artifacts
from music_recommender.ltr import train_ltr_ranker
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


def create_service(
    tmp_path: Path,
    ranking_config: dict[str, object] | None = None,
    ltr_model: object | None = None,
) -> RecommenderService:
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
        training_config={
            "raw_data_path": str(tmp_path / "missing.csv"),
            "metadata_path": str(tmp_path / "metadata.csv"),
            "min_user_interactions": 1,
            "min_artist_interactions": 1,
            "factors": 4,
            "regularization": 0.01,
            "iterations": 1,
            "alpha": 10.0,
            "use_gpu": False,
            "content_weight": 0.25,
        },
        hybrid_config={"default_content_weight": 0.25},
        ranking_config=ranking_config
        or {
            "include_listened": False,
            "popularity_penalty": 0.0,
            "diversity": 0.0,
        },
        ltr_model=ltr_model,
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


def test_service_metadata_exposes_ranking_config(tmp_path: Path) -> None:
    service = create_service(
        tmp_path,
        ranking_config={
            "include_listened": True,
            "popularity_penalty": 0.2,
            "diversity": 0.4,
        },
    )

    metadata = service.metadata()

    assert metadata["ranking_config"] == {
        "include_listened": True,
        "popularity_penalty": 0.2,
        "diversity": 0.4,
    }


def test_ranking_overrides_fall_back_to_champion_settings(tmp_path: Path) -> None:
    service = create_service(
        tmp_path,
        ranking_config={
            "include_listened": True,
            "popularity_penalty": 0.3,
            "diversity": 0.5,
        },
    )

    assert service._ranking_config() == {
        "include_listened": True,
        "popularity_penalty": 0.3,
        "diversity": 0.5,
    }
    assert service._ranking_overrides(None, None, None) == (True, 0.3, 0.5)
    assert service._ranking_overrides(False, None, 0.1) == (False, 0.3, 0.1)


def test_recommend_user_als_applies_champion_settings(tmp_path: Path) -> None:
    service = create_service(
        tmp_path,
        ranking_config={
            "include_listened": True,
            "popularity_penalty": 0.3,
            "diversity": 0.5,
        },
    )

    response = service.recommend_user_als("user_1", top_k=2)

    assert response["strategy"] == "als_personalized"
    assert response["recommendations"]


def test_recommend_user_ltr_reranks_with_bundled_model(tmp_path: Path) -> None:
    df = service_dataframe()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(matrix, 4, 0.01, 1, 10.0, use_gpu=False)
    stats = build_artist_stats(df)
    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=9,
    )
    service = create_service(tmp_path, ltr_model=ranker)

    response = service.recommend_user_ltr("user_1", top_k=2)

    assert response["strategy"] == "ltr_personalized"
    assert response["recommendations"]
    assert service.metadata()["ltr"]["available"] is True
    assert service.health()["ltr_available"] is True


def test_recommend_user_ltr_falls_back_when_no_ranker_bundled(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    response = service.recommend_user_ltr("user_1", top_k=2)

    assert response["strategy"] == "ltr_personalized"
    assert response["recommendations"]
    assert service.metadata()["ltr"]["available"] is False
    assert service.health()["ltr_available"] is False


def test_recommend_user_ltr_forwards_champion_ranking_settings(tmp_path: Path) -> None:
    service = create_service(
        tmp_path,
        ranking_config={
            "include_listened": True,
            "popularity_penalty": 0.3,
            "diversity": 0.5,
        },
    )

    response = service.recommend_user_ltr("user_1", top_k=2)

    assert response["strategy"] == "ltr_personalized"
    assert response["recommendations"]


def test_recommendations_without_diversity_do_not_densify_content_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = create_service(tmp_path)

    def fail_if_densified(*args, **kwargs) -> None:
        raise AssertionError("content matrix should remain sparse")

    monkeypatch.setattr("scipy.sparse.csr_matrix.toarray", fail_if_densified)

    response = service.recommend_user("user_1", top_k=1)

    assert response["recommendations"]


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


def test_service_health_reports_counts_and_status(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    health = service.health()

    assert health["status"] == "ok"
    assert health["artifact_version"] == "4.0"
    assert health["num_users"] == 3
    assert health["num_artists"] == 4
    assert health["num_interactions"] == 6


def test_recommend_user_als_returns_als_strategy(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    response = service.recommend_user_als("user_1", top_k=2)

    assert response["strategy"] == "als_personalized"
    assert response["user_id"] == "user_1"
    assert response["recommendations"]


def test_recommend_user_als_raises_for_unknown_user(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError, match="Unknown user_id"):
        service.recommend_user_als("missing_user", top_k=2)


@pytest.mark.parametrize(
    ("method", "strategy"),
    [
        ("als", "als_similarity"),
        ("content", "content_similarity"),
        ("hybrid", "hybrid_similarity"),
    ],
)
def test_similar_artists_supports_all_methods(
    tmp_path: Path,
    method: str,
    strategy: str,
) -> None:
    service = create_service(tmp_path)

    response = service.similar_artists("artist_1", top_k=2, method=method)

    assert response["strategy"] == strategy
    assert response["similar_artists"]
    assert response["similar_artists"][0]["artist_id"] != "artist_1"


def test_similar_artists_rejects_unknown_method(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError, match="method"):
        service.similar_artists("artist_1", top_k=2, method="hybrd")  # type: ignore[arg-type]


def test_hybrid_similar_artists_rejects_unknown_artist(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError, match="Unknown artist_id"):
        service.similar_artists("missing_artist", top_k=2, method="hybrid")


def test_popular_artists_returns_baseline_strategy(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    response = service.popular_artists(top_k=2)

    assert response["strategy"] == "popular_baseline"
    assert [artist["popularity_rank"] for artist in response["recommendations"]] == [
        1,
        2,
    ]


def test_hybrid_similar_artists_zero_norm_returns_zero_scores(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.artifact.model.user_factors = np.zeros_like(
        service.artifact.model.user_factors
    )

    response = service.similar_artists("artist_1", top_k=2, method="hybrid")

    assert response["strategy"] == "hybrid_similarity"


def test_recommend_user_reports_adjusted_score_with_penalty(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    response = service.recommend_user(
        "user_1",
        top_k=2,
        popularity_penalty=0.5,
        explain=True,
    )

    score_components = response["recommendations"][0]["score_components"]
    assert "adjusted_score" in score_components


def test_artist_catalog_excludes_country_and_era_mismatches(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    result = service.browse_artists(country="Canada", era="2020s")

    assert result["total"] == 0
    assert result["artists"] == []


def test_service_metadata_is_available(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    metadata = service.metadata()

    assert metadata["version"] == "4.0"
    assert metadata["metadata"]["num_users"] == 3
    assert metadata["training_config"]["factors"] == 4


def test_artist_catalog_is_popularity_sorted_and_paginated(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    first_page = service.browse_artists(limit=2)
    second_page = service.browse_artists(offset=2, limit=2)

    assert first_page["total"] == 4
    assert first_page["has_more"] is True
    assert [artist["popularity_rank"] for artist in first_page["artists"]] == [1, 2]
    assert second_page["has_more"] is False
    assert [artist["popularity_rank"] for artist in second_page["artists"]] == [3, 4]
    assert first_page["artists"][0]["genres"]
    assert first_page["artists"][0]["mood_tags"]


def test_artist_catalog_searches_and_filters_metadata(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    search_result = service.browse_artists(query="CANADA")
    filtered_result = service.browse_artists(genre="POP", mood_tag="fun")

    assert [artist["artist_id"] for artist in search_result["artists"]] == ["artist_4"]
    assert [artist["artist_id"] for artist in filtered_result["artists"]] == [
        "artist_2"
    ]


def test_artist_catalog_all_filters_narrow_results(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    result = service.browse_artists(
        genre="pop",
        mood_tag="fun",
        country="United States",
        era="2020s",
    )

    assert result["total"] == 1
    assert result["artists"][0]["artist_id"] == "artist_2"


def test_artist_catalog_prefers_curated_metadata_names(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    service.artifact.artist_stats["artist_1"]["artist_name"] = "Stale interaction name"

    result = service.browse_artists(query="artist_1")

    assert result["artists"][0]["artist_name"] == "A"


@pytest.mark.parametrize(
    ("offset", "limit", "message"),
    [
        (-1, 10, "offset must be a non-negative integer"),
        (0, 0, "limit must be a positive integer"),
    ],
)
def test_artist_catalog_rejects_invalid_pagination(
    tmp_path: Path,
    offset: int,
    limit: int,
    message: str,
) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError, match=message):
        service.browse_artists(offset=offset, limit=limit)


def test_recommend_tracks_returns_enriched_recommendations(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    result = service.recommend_tracks(user_id="user_1", top_k=3)

    assert result["strategy"] == "track_similarity"
    assert result["user_id"] == "user_1"
    assert len(result["recommendations"]) == 3
    first = result["recommendations"][0]
    assert first["track_name"]
    assert first["artist_name"]


def test_recommend_tracks_rejects_unknown_user_and_top_k(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError, match="Unknown user_id"):
        service.recommend_tracks(user_id="ghost", top_k=5)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        service.recommend_tracks(user_id="user_1", top_k=0)


def test_similar_tracks_returns_enriched_similarity(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    result = service.similar_tracks(track_id="track_1", top_k=3)

    assert result["strategy"] == "track_similarity"
    assert result["track_id"] == "track_1"
    assert len(result["similar_tracks"]) == 3
    assert all(item["track_id"] != "track_1" for item in result["similar_tracks"])


def test_similar_tracks_rejects_unknown_track(tmp_path: Path) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError, match="Unknown track_id"):
        service.similar_tracks(track_id="track_missing", top_k=5)


def test_recommend_tracks_prefers_bundled_resources(tmp_path: Path) -> None:
    from music_recommender.tracks import build_track_serving_resources

    service = create_service(tmp_path)
    track_df = pd.DataFrame(
        {
            "user_id": ["user_9"],
            "track_id": ["track_9"],
            "track_name": ["Rare Song"],
            "artist_id": ["artist_9"],
            "artist_name": ["Rare Artist"],
            "play_count": [12],
        }
    )
    meta_df = pd.DataFrame(
        {
            "track_id": ["track_9"],
            "track_name": ["Rare Song"],
            "artist_id": ["artist_9"],
            "artist_name": ["Rare Artist"],
            "album_id": ["album_9"],
            "album_name": ["Rare Album"],
            "duration_ms": [200000],
            "popularity": [10],
            "explicit": [False],
            "danceability": [0.5],
            "energy": [0.5],
            "key": [0],
            "loudness": [-6.0],
            "mode": [1],
            "speechiness": [0.05],
            "acousticness": [0.1],
            "instrumentalness": [0.0],
            "liveness": [0.1],
            "valence": [0.5],
            "tempo": [110.0],
            "time_signature": [4],
        }
    )
    service.artifact.track_bundle = build_track_serving_resources(track_df, meta_df)

    result = service.recommend_tracks(user_id="user_9", top_k=5)

    assert result["strategy"] == "track_similarity"
    assert result["recommendations"] == []
