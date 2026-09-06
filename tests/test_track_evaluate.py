import pandas as pd
import pytest

from music_recommender.track_evaluate import (
    evaluate_track_holdout,
    train_test_split_tracks_by_user,
)


def track_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_1", "user_2", "user_2"],
            "track_id": ["track_1", "track_2", "track_3", "track_1", "track_2"],
            "track_name": ["A1", "A2", "A3", "A1", "A2"],
            "artist_id": ["artist_1"] * 5,
            "artist_name": ["Artist A"] * 5,
            "play_count": [5, 3, 4, 7, 2],
        }
    )


def track_meta_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": ["track_1", "track_2", "track_3"],
            "track_name": ["A1", "A2", "A3"],
            "artist_id": ["artist_1"] * 3,
            "artist_name": ["Artist A"] * 3,
            "album_id": ["album_1"] * 3,
            "album_name": ["Album A"] * 3,
            "duration_ms": [200000, 210000, 220000],
            "popularity": [80, 70, 60],
            "explicit": [False, False, False],
            "danceability": [0.7, 0.6, 0.5],
            "energy": [0.8, 0.5, 0.6],
            "key": [5, 2, 7],
            "loudness": [-5.0, -8.0, -6.0],
            "mode": [1, 0, 1],
            "speechiness": [0.05, 0.04, 0.06],
            "acousticness": [0.1, 0.3, 0.2],
            "instrumentalness": [0.0, 0.1, 0.0],
            "liveness": [0.1, 0.2, 0.15],
            "valence": [0.9, 0.4, 0.6],
            "tempo": [120.0, 100.0, 110.0],
            "time_signature": [4, 3, 4],
        }
    )


def test_split_keeps_single_track_users_in_train() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "track_id": ["track_1", "track_2", "track_1"],
            "track_name": ["A1", "A2", "A1"],
            "artist_id": ["artist_1"] * 3,
            "artist_name": ["Artist A"] * 3,
            "play_count": [5, 3, 7],
        }
    )

    train_df, test_df = train_test_split_tracks_by_user(df, random_state=7)

    assert len(train_df) == 2
    assert len(test_df) == 1
    assert test_df.loc[0, "user_id"] == "user_1"


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5, True])
def test_split_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="test_ratio must be between 0 and 1"):
        train_test_split_tracks_by_user(track_df(), test_ratio=ratio)


def test_split_rejects_invalid_random_state() -> None:
    with pytest.raises(ValueError, match="random_state must be an integer"):
        train_test_split_tracks_by_user(track_df(), random_state="42")  # type: ignore[arg-type]


def test_evaluate_track_holdout_returns_bounded_metrics() -> None:
    metrics = evaluate_track_holdout(track_df(), track_meta_df(), top_k=2, folds=2)

    assert set(metrics) == {
        "precision_at_k",
        "recall_at_k",
        "map_at_k",
        "ndcg_at_k",
        "catalog_coverage",
        "average_popularity",
        "novelty_at_k",
    }
    assert all(value >= 0.0 for value in metrics.values())
    assert metrics["catalog_coverage"] <= 1.0
    assert metrics["novelty_at_k"] <= 1.0


def test_evaluate_track_holdout_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        evaluate_track_holdout(track_df(), track_meta_df(), top_k=0)
    with pytest.raises(ValueError, match="folds must be a positive integer"):
        evaluate_track_holdout(track_df(), track_meta_df(), folds=0)
    with pytest.raises(ValueError, match="include_listened must be a boolean"):
        evaluate_track_holdout(track_df(), track_meta_df(), include_listened="yes")  # type: ignore[arg-type]


def test_evaluate_track_holdout_requires_held_out_tracks() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_2"],
            "track_id": ["track_1", "track_2"],
            "track_name": ["A1", "A2"],
            "artist_id": ["artist_1", "artist_1"],
            "artist_name": ["Artist A", "Artist A"],
            "play_count": [5, 3],
        }
    )

    with pytest.raises(ValueError, match="No held-out track interactions"):
        evaluate_track_holdout(df, track_meta_df(), top_k=2)
