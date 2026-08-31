"""Tests for the pointwise learning-to-rank re-ranker."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from music_recommender.artifacts import ArtistStats, build_artist_stats
from music_recommender.ltr import rank_with_ltr, train_ltr_ranker
from music_recommender.model import train_als_model
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings


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
