"""Tests for the pointwise learning-to-rank re-ranker."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from music_recommender.artifacts import ArtistStats, build_artist_stats
from music_recommender.ltr import (
    _track_popularity_features,
    rank_tracks_with_ltr,
    rank_with_ltr,
    train_ltr_ranker,
    train_track_ltr_ranker,
)
from music_recommender.model import train_als_model
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings
from music_recommender.tracks import build_track_serving_resources


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "artist_id": ["a1", "a2", "a1", "a2", "a3", "a4", "a3", "a4"],
            "artist_name": ["A1", "A2", "A1", "A2", "A3", "A4", "A3", "A4"],
            "play_count": [10, 8, 9, 7, 8, 6, 5, 4],
        }
    )


def _fixtures() -> tuple[
    pd.DataFrame, dict[str, object], csr_matrix, object, dict[str, ArtistStats]
]:
    df = _sample_df()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(
        matrix,
        factors=4,
        regularization=0.01,
        iterations=5,
        alpha=10.0,
        use_gpu=False,
    )
    stats = build_artist_stats(df)
    return df, mappings, matrix, model, stats


def test_train_ltr_ranker_fits_a_ridge_model() -> None:
    df, mappings, matrix, model, stats = _fixtures()

    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=7,
    )

    assert isinstance(ranker, Ridge)
    assert ranker.coef_.shape[0] == 4


def test_train_ltr_ranker_returns_same_predictions_for_seeded_runs() -> None:
    df, mappings, matrix, model, stats = _fixtures()

    first = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=3,
    )
    second = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=3,
    )

    np.testing.assert_array_almost_equal(first.coef_, second.coef_)


def test_train_ltr_rejects_invalid_arguments() -> None:
    df, mappings, matrix, model, stats = _fixtures()

    with pytest.raises(TypeError, match="CSR matrix"):
        train_ltr_ranker(
            train_df=df,
            mappings=mappings,
            user_item_matrix=[[1, 0], [0, 1]],
            model=model,
            artist_stats=stats,
        )
    with pytest.raises(ValueError, match="model must expose"):
        train_ltr_ranker(
            train_df=df,
            mappings=mappings,
            user_item_matrix=matrix,
            model=SimpleNamespace(),
            artist_stats=stats,
        )
    with pytest.raises(ValueError, match="negatives_per_positive"):
        train_ltr_ranker(
            train_df=df,
            mappings=mappings,
            user_item_matrix=matrix,
            model=model,
            artist_stats=stats,
            negatives_per_positive=0,
        )
    with pytest.raises(ValueError, match="random_state"):
        train_ltr_ranker(
            train_df=df,
            mappings=mappings,
            user_item_matrix=matrix,
            model=model,
            artist_stats=stats,
            random_state="not-an-int",
        )
    with pytest.raises(ValueError, match="alpha"):
        train_ltr_ranker(
            train_df=df,
            mappings=mappings,
            user_item_matrix=matrix,
            model=model,
            artist_stats=stats,
            alpha=0.0,
        )


def _stats_without_artist(df: pd.DataFrame) -> dict[str, ArtistStats]:
    stats = build_artist_stats(df)
    first_id = df["artist_id"].iloc[0]
    return {artist_id: stats[artist_id] for artist_id in stats if artist_id != first_id}


def test_train_ltr_handles_artist_missing_from_stats() -> None:
    df = _sample_df()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(
        matrix,
        factors=4,
        regularization=0.01,
        iterations=5,
        alpha=10.0,
        use_gpu=False,
    )
    stats = _stats_without_artist(df)

    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=5,
    )

    assert isinstance(ranker, Ridge)


def test_train_ltr_skips_user_with_no_interaction_in_matrix() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "artist_id": ["a1", "a2", "a1", "a2", "a1"],
            "artist_name": ["A1", "A2", "A1", "A2", "A1"],
            "play_count": [10, 8, 9, 7, 4],
        }
    )
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    # Zero out user_3's interactions so the fold sees a user with no
    # interaction in the matrix.
    u3_index = mappings["user_id_to_index"]["u3"]
    matrix = matrix.tolil()
    matrix.data[u3_index] = []
    matrix.rows[u3_index] = []
    matrix = matrix.tocsr()
    model = train_als_model(
        matrix,
        factors=4,
        regularization=0.01,
        iterations=5,
        alpha=10.0,
        use_gpu=False,
    )
    stats = build_artist_stats(df)

    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=6,
    )

    assert isinstance(ranker, Ridge)


def test_rank_with_ltr_mixes_known_and_unknown_candidates() -> None:
    df, mappings, matrix, model, stats = _fixtures()
    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=2,
    )
    recommendations = [
        {"artist_id": "a1", "artist_name": "A1", "score": 0.5},
        {"artist_id": "ghost", "artist_name": "Ghost", "score": 0.1},
        {"artist_id": "a2", "artist_name": "A2", "score": 0.9},
    ]

    ranked = rank_with_ltr(
        ranker,
        user_id="u1",
        user_item_matrix=matrix,
        mappings=mappings,
        model=model,
        artist_stats=stats,
        recommendations=recommendations,
        top_k=3,
    )

    ranked_ids = [item["artist_id"] for item in ranked]
    assert len(ranked_ids) == 3
    # The unknown candidate must be pushed to the end.
    assert ranked_ids[-1] == "ghost"


def test_rank_with_ltr_reranks_and_respects_top_k() -> None:
    df, mappings, matrix, model, stats = _fixtures()
    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=2,
    )
    recommendations = [
        {"artist_id": "a3", "artist_name": "A3", "score": 0.5},
        {"artist_id": "a1", "artist_name": "A1", "score": 0.9},
        {"artist_id": "a2", "artist_name": "A2", "score": 0.8},
        {"artist_id": "a4", "artist_name": "A4", "score": 0.3},
    ]

    ranked = rank_with_ltr(
        ranker,
        user_id="u1",
        user_item_matrix=matrix,
        mappings=mappings,
        model=model,
        artist_stats=stats,
        recommendations=recommendations,
        top_k=2,
    )

    assert len(ranked) == 2
    ranked_ids = {item["artist_id"] for item in ranked}
    assert ranked_ids.issubset({"a1", "a2", "a3", "a4"})


def test_rank_with_ltr_validates_top_k() -> None:
    df, mappings, matrix, model, stats = _fixtures()
    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=2,
    )

    with pytest.raises(ValueError, match="top_k"):
        rank_with_ltr(
            ranker,
            user_id="u1",
            user_item_matrix=matrix,
            mappings=mappings,
            model=model,
            artist_stats=stats,
            recommendations=[{"artist_id": "a1", "artist_name": "A1", "score": 1.0}],
            top_k=0,
        )


def test_rank_with_ltr_returns_input_order_for_unknown_candidates() -> None:
    df, mappings, matrix, model, stats = _fixtures()
    ranker = train_ltr_ranker(
        train_df=df,
        mappings=mappings,
        user_item_matrix=matrix,
        model=model,
        artist_stats=stats,
        random_state=2,
    )
    recommendations = [
        {"artist_id": "ghost_1", "artist_name": "G1", "score": 0.9},
        {"artist_id": "ghost_2", "artist_name": "G2", "score": 0.1},
    ]

    ranked = rank_with_ltr(
        ranker,
        user_id="u1",
        user_item_matrix=matrix,
        mappings=mappings,
        model=model,
        artist_stats=stats,
        recommendations=recommendations,
        top_k=2,
    )

    assert [item["artist_id"] for item in ranked] == ["ghost_1", "ghost_2"]


def _track_fixtures():
    track_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "track_id": ["t1", "t2", "t2", "t3", "t1", "t4", "t3", "t4"],
            "track_name": ["T1", "T2", "T2", "T3", "T1", "T4", "T3", "T4"],
            "artist_id": ["a1", "a1", "a1", "a2", "a1", "a2", "a2", "a2"],
            "artist_name": ["A1", "A1", "A1", "A2", "A1", "A2", "A2", "A2"],
            "play_count": [10, 8, 9, 7, 8, 6, 5, 4],
        }
    )
    meta_df = pd.DataFrame(
        {
            "track_id": ["t1", "t2", "t3", "t4"],
            "track_name": ["T1", "T2", "T3", "T4"],
            "artist_id": ["a1", "a1", "a2", "a2"],
            "artist_name": ["A1", "A1", "A2", "A2"],
            "album_id": ["alb1"] * 4,
            "album_name": ["Alb"] * 4,
            "duration_ms": [200000] * 4,
            "popularity": [80, 70, 60, 50],
            "explicit": [False] * 4,
            "danceability": [0.8, 0.7, 0.3, 0.2],
            "energy": [0.9, 0.8, 0.4, 0.3],
            "key": [1, 2, 3, 4],
            "loudness": [-5.0, -6.0, -10.0, -12.0],
            "mode": [1, 0, 1, 0],
            "speechiness": [0.05, 0.04, 0.03, 0.02],
            "acousticness": [0.1, 0.2, 0.8, 0.9],
            "instrumentalness": [0.0, 0.0, 0.5, 0.6],
            "liveness": [0.1, 0.2, 0.1, 0.1],
            "valence": [0.9, 0.8, 0.3, 0.2],
            "tempo": [120.0, 115.0, 90.0, 85.0],
            "time_signature": [4, 4, 4, 4],
        }
    )
    resources = build_track_serving_resources(track_df, meta_df)
    return track_df, resources


def test_train_track_ltr_ranker_fits_a_ridge_model() -> None:
    track_df, resources = _track_fixtures()
    ranker = train_track_ltr_ranker(
        train_df=track_df,
        resources=resources,
        negatives_per_positive=2,
        random_state=42,
    )
    assert isinstance(ranker, Ridge)
    assert ranker.coef_.shape == (4,)


def test_train_track_ltr_ranker_validates_inputs() -> None:
    track_df, resources = _track_fixtures()

    with pytest.raises(TypeError, match="train_df must be a pandas DataFrame"):
        train_track_ltr_ranker(
            train_df=None,  # type: ignore[arg-type]
            resources=resources,
        )

    with pytest.raises(ValueError, match="train_df must not be empty"):
        train_track_ltr_ranker(
            train_df=pd.DataFrame(),
            resources=resources,
        )

    with pytest.raises(ValueError, match="missing required columns"):
        train_track_ltr_ranker(
            train_df=track_df.drop(columns=["play_count"]),
            resources=resources,
        )

    with pytest.raises(ValueError, match="resources must provide"):
        train_track_ltr_ranker(
            train_df=track_df,
            resources=None,  # type: ignore[arg-type]
        )

    broken_resources = SimpleNamespace(
        user_track_matrix=resources.user_track_matrix,
        similarity_matrix=resources.similarity_matrix,
        track_id_to_index=resources.track_id_to_index,
        track_stats=resources.track_stats,
        track_ids=[],
    )
    with pytest.raises(ValueError, match="resources.track_ids must not be empty"):
        train_track_ltr_ranker(
            train_df=track_df,
            resources=broken_resources,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="negatives_per_positive"):
        train_track_ltr_ranker(
            train_df=track_df,
            resources=resources,
            negatives_per_positive=0,
        )

    with pytest.raises(ValueError, match="random_state"):
        train_track_ltr_ranker(
            train_df=track_df,
            resources=resources,
            random_state="bad",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="alpha"):
        train_track_ltr_ranker(
            train_df=track_df,
            resources=resources,
            alpha=0.0,
        )


def test_train_track_ltr_ranker_skips_users_without_known_tracks() -> None:
    track_df, resources = _track_fixtures()
    # Add interaction with unknown track
    extra_df = pd.DataFrame(
        [
            {
                "user_id": "u99",
                "track_id": "unknown_track",
                "track_name": "Unknown",
                "artist_id": "a99",
                "artist_name": "A99",
                "play_count": 5,
            }
        ]
    )
    combined_df = pd.concat([track_df, extra_df], ignore_index=True)
    ranker = train_track_ltr_ranker(
        train_df=combined_df,
        resources=resources,
    )
    assert isinstance(ranker, Ridge)


def test_train_track_ltr_ranker_raises_when_no_valid_interactions() -> None:
    _, resources = _track_fixtures()
    ghost_df = pd.DataFrame(
        [
            {
                "user_id": "u99",
                "track_id": "unknown_track",
                "track_name": "Unknown",
                "artist_id": "a99",
                "artist_name": "A99",
                "play_count": 5,
            }
        ]
    )
    with pytest.raises(ValueError, match="No valid training interactions"):
        train_track_ltr_ranker(
            train_df=ghost_df,
            resources=resources,
        )


def test_rank_tracks_with_ltr_reranks_and_slices_top_k() -> None:
    track_df, resources = _track_fixtures()
    ranker = train_track_ltr_ranker(
        train_df=track_df,
        resources=resources,
        random_state=42,
    )
    recommendations = [
        {"track_id": "t3", "score": 0.1, "reasons": ["Because X"]},
        {"track_id": "t1", "score": 0.9, "reasons": ["Because Y"]},
        {"track_id": "t2", "score": 0.5, "reasons": ["Because Z"]},
    ]

    reranked = rank_tracks_with_ltr(
        ranker,
        recommendations=recommendations,
        user_id="u1",
        resources=resources,
        top_k=2,
    )
    assert len(reranked) == 2
    # Ensure all original keys are preserved
    for rec in reranked:
        assert "track_id" in rec
        assert "score" in rec
        assert "reasons" in rec


def test_rank_tracks_with_ltr_handles_empty_recommendations() -> None:
    track_df, resources = _track_fixtures()
    ranker = train_track_ltr_ranker(
        train_df=track_df,
        resources=resources,
    )
    assert (
        rank_tracks_with_ltr(
            ranker,
            recommendations=[],
            user_id="u1",
            resources=resources,
            top_k=5,
        )
        == []
    )


def test_rank_tracks_with_ltr_validates_inputs() -> None:
    track_df, resources = _track_fixtures()
    ranker = train_track_ltr_ranker(
        train_df=track_df,
        resources=resources,
    )

    with pytest.raises(ValueError, match="top_k"):
        rank_tracks_with_ltr(
            ranker,
            recommendations=[{"track_id": "t1", "score": 1.0}],
            user_id="u1",
            resources=resources,
            top_k=0,
        )

    with pytest.raises(ValueError, match="ranker must not be None"):
        rank_tracks_with_ltr(
            None,  # type: ignore[arg-type]
            recommendations=[{"track_id": "t1", "score": 1.0}],
            user_id="u1",
            resources=resources,
            top_k=5,
        )


def test_rank_tracks_with_ltr_handles_unknown_candidates() -> None:
    track_df, resources = _track_fixtures()
    ranker = train_track_ltr_ranker(
        train_df=track_df,
        resources=resources,
    )
    # All unknown: returns original order
    all_unknown = [
        {"track_id": "ghost_1", "score": 0.9},
        {"track_id": "ghost_2", "score": 0.8},
    ]
    assert rank_tracks_with_ltr(
        ranker,
        recommendations=all_unknown,
        user_id="u1",
        resources=resources,
        top_k=2,
    ) == all_unknown

    # Mixed known and unknown: known comes first
    mixed = [
        {"track_id": "ghost_1", "score": 0.9},
        {"track_id": "t1", "score": 0.5},
    ]
    reranked = rank_tracks_with_ltr(
        ranker,
        recommendations=mixed,
        user_id="u1",
        resources=resources,
        top_k=2,
    )
    assert reranked[0]["track_id"] == "t1"
    assert reranked[1]["track_id"] == "ghost_1"


def test_rank_tracks_with_ltr_cold_start_user() -> None:
    track_df, resources = _track_fixtures()
    ranker = train_track_ltr_ranker(
        train_df=track_df,
        resources=resources,
    )
    recommendations = [
        {"track_id": "t3", "score": 0.2},
        {"track_id": "t1", "score": 0.8},
    ]
    reranked = rank_tracks_with_ltr(
        ranker,
        recommendations=recommendations,
        user_id="cold_user_999",
        resources=resources,
        top_k=2,
    )
    assert len(reranked) == 2


def test_track_popularity_features_helper() -> None:
    stats = {
        "t1": {
            "track_id": "t1",
            "total_plays": 99,
            "popularity_rank": 1,
        }
    }
    log_plays, norm_rank = _track_popularity_features("t1", stats, 10)
    assert log_plays == pytest.approx(float(np.log1p(99)))
    assert norm_rank == pytest.approx(0.0)

    # Missing stats returns 0, 0
    log_plays_missing, norm_rank_missing = _track_popularity_features(
        "missing", stats, 10
    )
    assert log_plays_missing == 0.0
    assert norm_rank_missing == 0.0

