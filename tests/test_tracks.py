from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from music_recommender.tracks import (
    artist_affinity_to_track_scores,
    artist_taste_scores_for_user,
    build_track_content_matrix,
    build_track_serving_resources,
    build_track_stats,
    get_similar_tracks,
    load_and_validate_track_interactions,
    load_and_validate_track_metadata,
    load_track_serving_resources,
    normalize_track_interactions,
    popular_tracks,
    recommend_popular_tracks,
    recommend_tracks_for_user,
    train_track_artist_taste,
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


def test_track_artist_id_maps_to_multiple_names_raises() -> None:
    df = valid_track_df()
    df.loc[2, "artist_name"] = "Artist B"
    with pytest.raises(ValueError, match="multiple artist names"):
        validate_track_interactions(df)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda df: df.assign(user_id=["user_1", None, "user_2"]),
        lambda df: df.assign(track_name=["Song A1", " ", "Song A1"]),
        lambda df: df.assign(play_count=["5", "3", "7"]),
        lambda df: df.assign(play_count=[5.0, np.inf, 7.0]),
        lambda df: df.assign(play_count=[5, -3, 7]),
    ],
)
def test_track_interactions_reject_bad_values(
    mutator,
) -> None:
    with pytest.raises(ValueError):
        validate_track_interactions(mutator(valid_track_df()))


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
    with pytest.raises(ValueError, match="not found in metadata"):
        validate_track_metadata(
            valid_metadata_df().drop(index=1),
            valid_track_df(),
        )
    with pytest.raises(ValueError, match="empty"):
        validate_track_metadata(pd.DataFrame(columns=valid_metadata_df().columns))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda df: df.assign(album_name=["Album A", None]),
        lambda df: df.assign(album_name=["Album A", " "]),
    ],
)
def test_track_metadata_rejects_bad_values(mutator) -> None:
    with pytest.raises(ValueError):
        validate_track_metadata(mutator(valid_metadata_df()))
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


def test_recommend_tracks_for_user_hybrid_blend() -> None:
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0], [0.0, 4.0]],
        index=["user_1", "user_2"],
        columns=["track_1", "track_2"],
    )
    sim = np.array([[1.0, 0.2], [0.2, 1.0]])
    mapping = {"track_1": 0, "track_2": 1}
    # User 1 listens to track_1 but strongly prefers artists track_2 maps to.
    artist_taste_per_track = np.array([0.1, 0.9])
    recs = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=1,
        artist_taste_per_track=artist_taste_per_track,
        content_weight=0.25,
    )
    assert recs
    assert recs[0]["track_id"] == "track_2"
    components = recs[0]["score_components"]
    assert set(components) == {
        "content_score",
        "collaborative_score",
        "hybrid_score",
    }

    # A pure content weight reproduces the similarity ranking.
    pure = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=1,
        content_weight=1.0,
    )
    assert pure[0]["track_id"] == "track_2"
    assert "score_components" not in pure[0]


def test_recommend_tracks_for_user_hybrid_explains_artist_affinity() -> None:
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0], [0.0, 4.0]],
        index=["user_1", "user_2"],
        columns=["track_1", "track_2"],
    )
    sim = np.array([[1.0, 0.2], [0.2, 1.0]])
    mapping = {"track_1": 0, "track_2": 1}
    artist_taste_per_track = np.array([0.05, 0.95])
    recs = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=1,
        explain=True,
        track_name_lookup={"track_1": "Song A", "track_2": "Song B"},
        track_artist_lookup={"track_2": "Artist B"},
        artist_taste_per_track=artist_taste_per_track,
        content_weight=0.25,
    )
    assert "Artist affinity: Artist B" in recs[0]["reasons"]
    assert "Because you listened to Song A" in recs[0]["reasons"]

    # Without a lookup, hybrid explanations only carry content reasons.
    no_lookup = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=1,
        explain=True,
        artist_taste_per_track=artist_taste_per_track,
        content_weight=0.25,
    )
    assert all("Artist affinity" not in reason for reason in no_lookup[0]["reasons"])

    # A zero collaborative contribution is not presented as artist affinity.
    zero_taste = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=1,
        explain=True,
        track_artist_lookup={"track_2": "Artist B"},
        artist_taste_per_track=np.array([0.0, 0.0]),
        content_weight=0.25,
    )
    assert all("Artist affinity" not in reason for reason in zero_taste[0]["reasons"])


def test_recommend_tracks_for_user_rejects_invalid_hybrid_inputs() -> None:
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0]],
        index=["user_1"],
        columns=["track_1", "track_2"],
    )
    sim = np.array([[1.0, 0.2], [0.2, 1.0]])
    mapping = {"track_1": 0, "track_2": 1}
    with pytest.raises(ValueError, match="artist_taste_per_track must be a finite"):
        recommend_tracks_for_user(
            "user_1",
            user_track_matrix,
            sim,
            mapping,
            artist_taste_per_track=np.array([0.1, np.nan]),
            content_weight=0.5,
        )
    with pytest.raises(ValueError, match="artist_taste_per_track must be a finite"):
        recommend_tracks_for_user(
            "user_1",
            user_track_matrix,
            sim,
            mapping,
            artist_taste_per_track=np.array([0.1, 0.2, 0.3]),
            content_weight=0.5,
        )
    with pytest.raises(ValueError, match="content_weight"):
        recommend_tracks_for_user(
            "user_1",
            user_track_matrix,
            sim,
            mapping,
            artist_taste_per_track=np.array([0.1, 0.9]),
            content_weight=1.5,
        )


def test_artist_taste_scores_for_user_and_track_map() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "track_id": ["track_1", "track_2", "track_1"],
            "track_name": ["Song A1", "Song A2", "Song A1"],
            "artist_id": ["artist_1", "artist_2", "artist_1"],
            "artist_name": ["Artist One", "Artist Two", "Artist One"],
            "play_count": [3, 2, 5],
        }
    )
    model, user_ids, artist_ids = train_track_artist_taste(df)
    assert list(user_ids) == ["user_1", "user_2"]
    assert list(artist_ids) == ["artist_1", "artist_2"]

    artist_scores = artist_taste_scores_for_user(model, user_ids, artist_ids, "user_1")
    assert artist_scores is not None
    assert artist_scores.shape == (len(artist_ids),)

    track_artists = {"track_1": "artist_1", "track_2": "artist_3"}
    track_id_to_index = {"track_1": 0, "track_2": 1}
    per_track = artist_affinity_to_track_scores(
        artist_scores=artist_scores,
        artist_id_to_index=artist_ids,
        track_id_to_index=track_id_to_index,
        track_artists=track_artists,
    )
    assert per_track.shape == (2,)
    assert np.isfinite(per_track).all()

    unknown_user = artist_taste_scores_for_user(model, user_ids, artist_ids, "ghost")
    assert unknown_user is None
    assert np.allclose(
        artist_affinity_to_track_scores(
            artist_scores=np.array([0.25, 0.75]),
            artist_id_to_index={"artist_1": 0, "artist_2": 1},
            track_id_to_index=track_id_to_index,
            track_artists=track_artists,
        ),
        np.array([0.25, 0.0]),
    )


def test_recommend_tracks_for_user_explains_reasons() -> None:
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0, 0.0]],
        index=["user_1"],
        columns=["track_1", "track_2", "track_3"],
    )
    sim = np.array(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )
    mapping = {"track_1": 0, "track_2": 1, "track_3": 2}
    lookup = {"track_1": "Mirror", "track_2": "Spark", "track_3": "Ember"}

    recs = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=1,
        explain=True,
        track_name_lookup=lookup,
    )
    assert recs[0]["reasons"] == ["Because you listened to Mirror"]

    plain = recommend_tracks_for_user(
        "user_1", user_track_matrix, sim, mapping, top_k=1
    )
    assert "reasons" not in plain[0]

    with pytest.raises(ValueError, match="explain"):
        recommend_tracks_for_user(
            "user_1",
            user_track_matrix,
            sim,
            mapping,
            explain=1,
        )


def test_recommend_tracks_for_user_penalty_demotes_popular_track() -> None:
    # user_1 listened to track_1; track_2 (unheard) is highly similar but the
    # most popular. With a penalty, the less-identical track_3 should win.
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0, 0.0]],
        index=["user_1"],
        columns=["track_1", "track_2", "track_3"],
    )
    sim = np.array(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )
    mapping = {"track_1": 0, "track_2": 1, "track_3": 2}
    track_stats = {
        "track_1": {"total_plays": 5},
        "track_2": {"total_plays": 90},
        "track_3": {"total_plays": 4},
    }

    baseline = recommend_tracks_for_user(
        "user_1", user_track_matrix, sim, mapping, top_k=2
    )
    assert [rec["track_id"] for rec in baseline] == ["track_2", "track_3"]

    penalized = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=2,
        track_stats=track_stats,
        popularity_penalty=1.0,
    )
    assert [rec["track_id"] for rec in penalized] == ["track_3", "track_2"]


def test_recommend_tracks_for_user_diversity_breaks_near_duplicates() -> None:
    # track_2 and track_3 both score highly and share nearly identical audio
    # features; diversity should promote the distinct track_4 instead.
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0, 0.0, 0.0]],
        index=["user_1"],
        columns=["track_1", "track_2", "track_3", "track_4"],
    )
    sim = np.array(
        [
            [1.0, 0.9, 0.85, 0.1],
            [0.9, 1.0, 0.98, 0.1],
            [0.85, 0.98, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ]
    )
    mapping = {"track_1": 0, "track_2": 1, "track_3": 2, "track_4": 3}
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.98, 0.01, 0.02],
            [0.0, 1.0, 0.0],
        ]
    )

    diversed = recommend_tracks_for_user(
        "user_1",
        user_track_matrix,
        sim,
        mapping,
        top_k=3,
        feature_matrix=features,
        diversity=1.0,
    )
    assert diversed[0]["track_id"] == "track_2"
    assert diversed[1]["track_id"] == "track_4"


def test_recommend_tracks_for_user_rejects_invalid_knobs() -> None:
    user_track_matrix = pd.DataFrame(
        [[5.0, 0.0]],
        index=["user_1"],
        columns=["track_1", "track_2"],
    )
    sim = np.eye(2)
    mapping = {"track_1": 0, "track_2": 1}
    with pytest.raises(ValueError, match="popularity_penalty"):
        recommend_tracks_for_user(
            "user_1",
            user_track_matrix,
            sim,
            mapping,
            popularity_penalty=1.5,
        )
    with pytest.raises(ValueError, match="diversity"):
        recommend_tracks_for_user(
            "user_1",
            user_track_matrix,
            sim,
            mapping,
            diversity=-0.1,
        )


def test_get_similar_tracks() -> None:
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    mapping = {"track_1": 0, "track_2": 1, "track_3": 2}
    out = get_similar_tracks("track_1", sim, mapping, top_k=1)
    assert out[0]["track_id"] == "track_2"
    assert get_similar_tracks("missing", sim, mapping) == []


def test_popular_tracks_ranks_globally() -> None:
    resources = build_track_serving_resources(valid_track_df(), valid_metadata_df())

    top = popular_tracks(resources.track_stats, top_k=1)

    assert len(top) == 1
    assert top[0]["track_id"] == "track_1"
    assert top[0]["popularity_rank"] == 1
    assert top[0]["score"] == 12

    with pytest.raises(ValueError, match="top_k"):
        popular_tracks(resources.track_stats, top_k=0)


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
    assert resources.feature_matrix.shape == (2, 12)
    assert resources.track_lookup["track_1"]["artist_name"] == "Artist A"
    assert list(resources.user_track_matrix.index) == ["user_1", "user_2"]
    assert set(resources.track_stats) <= {"track_1", "track_2"}


def test_build_track_stats_ranks_by_plays() -> None:
    stats = build_track_stats(valid_track_df())

    assert stats["track_1"]["popularity_rank"] == 1
    assert stats["track_1"]["total_plays"] == 12
    assert stats["track_2"]["popularity_rank"] == 2
    assert stats["track_2"]["artist_name"] == "Artist A"


def test_recommend_popular_tracks_ranks_by_plays() -> None:
    resources = build_track_serving_resources(valid_track_df(), valid_metadata_df())

    recs = recommend_popular_tracks(
        user_id="user_2",
        user_track_matrix=resources.user_track_matrix,
        track_stats=resources.track_stats,
        top_k=2,
    )

    assert [rec["track_id"] for rec in recs] == ["track_2"]
    included = recommend_popular_tracks(
        user_id="user_2",
        user_track_matrix=resources.user_track_matrix,
        track_stats=resources.track_stats,
        top_k=2,
        include_listened=True,
    )
    assert [rec["track_id"] for rec in included] == ["track_1", "track_2"]
    assert included[0]["score"] == 12.0


def test_recommend_popular_tracks_excludes_listened() -> None:
    resources = build_track_serving_resources(valid_track_df(), valid_metadata_df())

    recs = recommend_popular_tracks(
        user_id="user_1",
        user_track_matrix=resources.user_track_matrix,
        track_stats=resources.track_stats,
        top_k=5,
    )

    assert recs == []
    included = recommend_popular_tracks(
        user_id="user_1",
        user_track_matrix=resources.user_track_matrix,
        track_stats=resources.track_stats,
        top_k=5,
        include_listened=True,
    )
    assert [rec["track_id"] for rec in included] == ["track_1", "track_2"]
    assert (
        recommend_popular_tracks(
            user_id="ghost",
            user_track_matrix=resources.user_track_matrix,
            track_stats=resources.track_stats,
        )
        == []
    )


def test_load_track_serving_resources_roundtrip(tmp_path: Path) -> None:
    inter_path = tmp_path / "inter.csv"
    meta_path = tmp_path / "meta.csv"
    valid_track_df().to_csv(inter_path, index=False)
    valid_metadata_df().to_csv(meta_path, index=False)
    resources = load_track_serving_resources(inter_path, meta_path)
    assert resources.track_ids == ["track_1", "track_2"]
    assert resources.feature_names[:2] == ["danceability", "energy"]
