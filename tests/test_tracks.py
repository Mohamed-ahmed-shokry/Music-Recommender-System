from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from music_recommender.tracks import (
    build_track_content_matrix,
    build_track_serving_resources,
    get_similar_tracks,
    load_and_validate_track_interactions,
    load_and_validate_track_metadata,
    load_track_serving_resources,
    normalize_track_interactions,
    recommend_tracks_for_user,
    validate_track_interactions,
    validate_track_metadata,
)


def valid_track_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "track_id": ["track_1", "track_2", "track_1"],
            "track_name": ["Song A1", "Song A2", "Song A1"],
            "artist_id": ["artist_1", "artist_1", "artist_1"],
            "artist_name": ["Artist A", "Artist A", "Artist A"],
            "play_count": [5, 3, 7],
        }
    )


def valid_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": ["track_1", "track_2"],
            "track_name": ["Song A1", "Song A2"],
            "artist_id": ["artist_1", "artist_1"],
            "artist_name": ["Artist A", "Artist A"],
            "album_id": ["album_1", "album_1"],
            "album_name": ["Album A", "Album A"],
            "duration_ms": [200000, 210000],
            "popularity": [80, 70],
            "explicit": [False, False],
            "danceability": [0.7, 0.6],
            "energy": [0.8, 0.5],
            "key": [5, 2],
            "loudness": [-5.0, -8.0],
            "mode": [1, 0],
            "speechiness": [0.05, 0.04],
            "acousticness": [0.1, 0.3],
            "instrumentalness": [0.0, 0.1],
            "liveness": [0.1, 0.2],
            "valence": [0.9, 0.4],
            "tempo": [120.0, 100.0],
            "time_signature": [4, 3],
        }
    )


def test_valid_track_interactions_pass() -> None:
    validate_track_interactions(valid_track_df())


def test_track_missing_column_raises() -> None:
    df = valid_track_df().drop(columns=["track_name"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_track_interactions(df)


def test_track_empty_raises() -> None:
    df = pd.DataFrame(
        columns=[
            "user_id",
            "track_id",
            "track_name",
            "artist_id",
            "artist_name",
            "play_count",
        ]
    )
    with pytest.raises(ValueError, match="empty"):
        validate_track_interactions(df)


def test_track_conflicting_artist_raises() -> None:
    df = valid_track_df()
    df.loc[1, "track_id"] = "track_1"
    df.loc[1, "artist_id"] = "artist_2"
    with pytest.raises(ValueError, match="multiple artist"):
        validate_track_interactions(df)


def test_track_normalization_aggregates() -> None:
    df = pd.DataFrame(
        {
            "user_id": [" user_1 ", "user_1"],
            "track_id": ["track_1", "track_1"],
            "track_name": ["Song A1 ", " Song A1"],
            "artist_id": ["artist_1", "artist_1 "],
            "artist_name": ["Artist A", "Artist A "],
            "play_count": [2, 3],
        }
    )
    out = normalize_track_interactions(df)
    assert len(out) == 1
    assert out.loc[0, "play_count"] == 5
    assert out.loc[0, "user_id"] == "user_1"


def test_track_metadata_validation_and_coverage() -> None:
    validate_track_metadata(valid_metadata_df(), valid_track_df())
    bad = valid_metadata_df().drop(columns=["tempo"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_track_metadata(bad)
    dup = pd.concat([valid_metadata_df(), valid_metadata_df().iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate track"):
        validate_track_metadata(dup)
    missing = valid_metadata_df().iloc[[0]]
    with pytest.raises(ValueError, match="not found in metadata"):
        validate_track_metadata(missing, valid_track_df())


def test_build_track_content_matrix_normalizes() -> None:
    feature_df, names = build_track_content_matrix(valid_metadata_df())
    assert len(names) == 12
    assert list(feature_df.index) == ["track_1", "track_2"]
    # loudness shifted positive, tempo normalized, key normalized
    assert (feature_df["loudness"] >= 0).all()
    assert (feature_df["tempo"] <= 1.0).all()
    assert (feature_df["key"] <= 1.0).all()


def test_recommend_tracks_for_user() -> None:
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0], [0.0, 4.0]],
        index=["user_1", "user_2"],
        columns=["track_1", "track_2"],
    )
    sim = np.array([[1.0, 0.2], [0.2, 1.0]])
    mapping = {"track_1": 0, "track_2": 1}
    recs = recommend_tracks_for_user("user_1", user_track_matrix, sim, mapping, top_k=1)
    assert recs[0]["track_id"] == "track_2"
    assert recommend_tracks_for_user("unknown", user_track_matrix, sim, mapping) == []
    empty_matrix = pd.DataFrame(
        [[0.0, 0.0]], index=["user_3"], columns=["track_1", "track_2"]
    )
    assert recommend_tracks_for_user("user_3", empty_matrix, sim, mapping) == []


def test_get_similar_tracks() -> None:
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    mapping = {"track_1": 0, "track_2": 1, "track_3": 2}
    out = get_similar_tracks("track_1", sim, mapping, top_k=1)
    assert out[0]["track_id"] == "track_2"
    assert get_similar_tracks("missing", sim, mapping) == []


def test_load_track_files_roundtrip(tmp_path: Path) -> None:
    inter_path = tmp_path / "inter.csv"
    meta_path = tmp_path / "meta.csv"
    valid_track_df().to_csv(inter_path, index=False)
    valid_metadata_df().to_csv(meta_path, index=False)
    df = load_and_validate_track_interactions(inter_path)
    assert len(df) == 3
    meta = load_and_validate_track_metadata(meta_path, df)
    assert len(meta) == 2


def test_build_track_serving_resources() -> None:
    resources = build_track_serving_resources(valid_track_df(), valid_metadata_df())
    assert resources.track_ids == ["track_1", "track_2"]
    assert resources.track_id_to_index == {"track_1": 0, "track_2": 1}
    assert resources.similarity_matrix.shape == (2, 2)
    assert resources.track_lookup["track_1"]["artist_name"] == "Artist A"
    assert list(resources.user_track_matrix.index) == ["user_1", "user_2"]


def test_load_track_serving_resources_roundtrip(tmp_path: Path) -> None:
    inter_path = tmp_path / "inter.csv"
    meta_path = tmp_path / "meta.csv"
    valid_track_df().to_csv(inter_path, index=False)
    valid_metadata_df().to_csv(meta_path, index=False)
    resources = load_track_serving_resources(inter_path, meta_path)
    assert resources.track_ids == ["track_1", "track_2"]
    assert resources.feature_names[:2] == ["danceability", "energy"]
