import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from music_recommender.content import (
    _weighted_profile,
    build_content_artifacts,
    content_similar_artists,
    content_similarity_scores,
    hybrid_scores,
    listened_artist_ids,
    metadata_tokens_for_artist,
    profile_content_scores,
    recommend_from_scores,
    user_content_scores,
    validate_content_weight,
)
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings


def metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artist_id": ["artist_1", "artist_2", "artist_3"],
            "artist_name": ["Pop One", "Pop Two", "Rock Three"],
            "genres": ["pop;dance", "pop;dance", "rock;indie"],
            "mood_tags": ["bright;fun", "bright;party", "raw;guitar"],
            "country": ["United States", "United States", "United Kingdom"],
            "era": ["2020s", "2020s", "2000s"],
        }
    )


def interactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "artist_id": ["artist_1", "artist_2", "artist_3"],
            "artist_name": ["Pop One", "Pop Two", "Rock Three"],
            "play_count": [10, 5, 7],
        }
    )


def test_content_matrix_shape_matches_artist_count() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    assert content.content_matrix.shape[0] == 3
    assert content.content_matrix.shape[1] == len(content.feature_names)


def test_content_similar_artists_excludes_query_artist() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    recommendations = content_similar_artists("artist_1", content, None, top_k=2)

    assert recommendations[0]["artist_id"] == "artist_2"
    assert all(item["artist_id"] != "artist_1" for item in recommendations)


def test_user_content_scores_build_profile() -> None:
    df = interactions_df()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    scores, listened_artist_ids = user_content_scores(
        "user_1",
        matrix,
        mappings,
        content,
    )

    assert scores.shape == (3,)
    assert set(listened_artist_ids) == {"artist_1", "artist_2"}


def test_profile_content_scores_use_artist_and_genre_preferences() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    scores, artist_ids, preference_tokens = profile_content_scores(
        content,
        artist_ids=["artist_1"],
        genres=["pop"],
    )

    assert (
        scores[content.artist_id_to_content_index["artist_2"]]
        > scores[content.artist_id_to_content_index["artist_3"]]
    )
    assert artist_ids == ["artist_1"]
    assert "pop" in preference_tokens


def test_recommend_from_scores_returns_reasons_when_explained() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )
    scores, artist_ids, preference_tokens = profile_content_scores(
        content,
        artist_ids=["artist_1"],
        genres=["pop"],
    )

    recommendations = recommend_from_scores(
        scores,
        content,
        artist_stats=None,
        top_k=1,
        exclude_artist_ids=set(artist_ids),
        reference_artist_ids=artist_ids,
        preference_tokens=preference_tokens,
        explain=True,
    )

    assert recommendations[0]["artist_id"] == "artist_2"
    assert recommendations[0]["reasons"]


def test_hybrid_weight_extremes_match_expected_scores() -> None:
    collaborative_scores = pd.Series([0.2, 0.9]).to_numpy()
    content_scores = pd.Series([0.8, 0.1]).to_numpy()

    collaborative_only = hybrid_scores(collaborative_scores, content_scores, 0.0)
    content_only = hybrid_scores(collaborative_scores, content_scores, 1.0)

    assert collaborative_only[1] > collaborative_only[0]
    assert content_only[0] > content_only[1]


@pytest.mark.parametrize(
    "content_weight",
    [-0.1, 1.5, float("nan"), float("inf"), True],
)
def test_invalid_content_weight_raises_value_error(content_weight) -> None:
    with pytest.raises(ValueError, match="content_weight"):
        validate_content_weight(content_weight)


def test_content_artifacts_repr_reports_shape() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    assert "num_artists=3" in repr(content)


def test_content_similarity_scores_rejects_unknown_artist() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    with pytest.raises(ValueError, match="Unknown artist_id"):
        content_similarity_scores("missing_artist", content)


def test_listened_artist_ids_rejects_unknown_user() -> None:
    df = interactions_df()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )

    with pytest.raises(ValueError, match="Unknown user_id"):
        listened_artist_ids("missing_user", matrix, mappings)


def test_user_content_scores_rejects_unknown_user() -> None:
    df = interactions_df()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    with pytest.raises(ValueError, match="Unknown user_id"):
        user_content_scores("missing_user", matrix, mappings, content)


def test_user_content_scores_skips_artists_missing_from_content() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "artist_id": ["artist_1", "artist_4", "artist_3"],
            "artist_name": ["Pop One", "Untracked", "Rock Three"],
            "play_count": [10, 5, 7],
        }
    )
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    scores, listened = user_content_scores(
        "user_1",
        matrix,
        mappings,
        content,
    )

    assert "artist_4" in listened
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_user_content_scores_return_zeros_when_no_artists_in_content() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_2"],
            "artist_id": ["artist_4", "artist_4"],
            "artist_name": ["Untracked", "Untracked"],
            "play_count": [10, 7],
        }
    )
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    scores, _ = user_content_scores("user_1", matrix, mappings, content)

    assert (scores == 0).all()


def test_profile_content_scores_rejects_empty_inputs() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    with pytest.raises(ValueError, match="at least one"):
        profile_content_scores(
            content,
            artist_ids=[],
            genres=[],
            mood_tags=[],
        )


def test_profile_content_scores_rejects_unknown_artists() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    with pytest.raises(ValueError, match="Unknown artist IDs"):
        profile_content_scores(content, artist_ids=["missing_artist"])


def test_recommend_from_scores_ignores_unknown_reference_artists() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )
    scores, _, _ = profile_content_scores(content, genres=["pop"])

    recommendations = recommend_from_scores(
        scores,
        content,
        artist_stats=None,
        top_k=1,
        exclude_artist_ids=set(),
        reference_artist_ids=["unknown_reference"],
        explain=True,
    )

    assert recommendations


def test_metadata_tokens_for_artist_returns_token_set() -> None:
    content = build_content_artifacts(
        metadata_df(), ["artist_1", "artist_2", "artist_3"]
    )

    tokens = metadata_tokens_for_artist("artist_1", content)

    assert "pop" in tokens
    assert "2020s" in tokens


def test_weighted_profile_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="weights must not be empty"):
        _weighted_profile(csr_matrix([[1.0]]), np.array([]))
