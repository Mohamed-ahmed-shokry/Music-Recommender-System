from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import music_recommender.artifacts as artifacts_module
from music_recommender.artifacts import (
    RecommenderArtifact,
    _validate_content_artifacts,
    build_artist_stats,
    build_recommender_artifact,
    create_dataset_fingerprint,
    load_artifact,
    save_artifact,
)
from music_recommender.content import build_content_artifacts
from music_recommender.model import train_als_model
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings


def artifact_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_2"],
            "artist_id": ["artist_1", "artist_2", "artist_2", "artist_3"],
            "artist_name": ["A", "B", "B", "C"],
            "play_count": [5, 3, 7, 2],
        }
    )


def artifact_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artist_id": ["artist_1", "artist_2", "artist_3"],
            "artist_name": ["A", "B", "C"],
            "genres": ["pop", "pop;dance", "rock"],
            "mood_tags": ["bright", "bright;fun", "raw"],
            "country": ["United States", "United States", "United Kingdom"],
            "era": ["2020s", "2020s", "2000s"],
        }
    )


def create_test_artifact(tmp_path: Path) -> RecommenderArtifact:
    df = artifact_dataframe()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(matrix, 4, 0.01, 1, 10.0, use_gpu=False)
    content_artifacts = build_content_artifacts(
        artifact_metadata_df(),
        ["artist_1", "artist_2", "artist_3"],
    )
    return build_recommender_artifact(
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
        ranking_config={
            "include_listened": False,
            "popularity_penalty": 0.0,
            "diversity": 0.0,
        },
    )


def test_artifact_bundle_saves_and_loads(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact_path = tmp_path / "artifact.joblib"

    save_artifact(artifact, artifact_path)
    loaded_artifact = load_artifact(artifact_path)

    assert loaded_artifact.version == "4.0"
    assert loaded_artifact.user_item_matrix.shape == (2, 3)
    assert loaded_artifact.metadata["num_users"] == 2
    assert loaded_artifact.training_config["factors"] == 4
    assert loaded_artifact.hybrid_config["default_content_weight"] == 0.25
    assert loaded_artifact.ranking_config == {
        "include_listened": False,
        "popularity_penalty": 0.0,
        "diversity": 0.0,
    }
    assert loaded_artifact.content_artifacts.content_matrix.shape[0] == 3
    assert "artist_2" in loaded_artifact.artist_stats


def test_artifact_without_ranking_config_defaults_to_neutral(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    del artifact.ranking_config
    artifact_path = tmp_path / "legacy.joblib"

    save_artifact(artifact, artifact_path)
    loaded_artifact = load_artifact(artifact_path)

    assert loaded_artifact.ranking_config == {
        "include_listened": False,
        "popularity_penalty": 0.0,
        "diversity": 0.0,
    }


def test_artifact_rejects_invalid_ranking_config(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.ranking_config = {
        "include_listened": "yes",
        "popularity_penalty": 0.0,
        "diversity": 0.0,
    }
    artifact_path = tmp_path / "invalid-ranking.joblib"

    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="ranking configuration"):
        load_artifact(artifact_path)


def test_artifact_without_ltr_model_defaults_to_none(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    del artifact.ltr_model
    artifact_path = tmp_path / "legacy-ltr.joblib"

    save_artifact(artifact, artifact_path)
    loaded_artifact = load_artifact(artifact_path)

    assert loaded_artifact.ltr_model is None


def test_artifact_bundle_round_trips_ltr_model(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.ltr_model = {"kind": "ridge"}
    artifact_path = tmp_path / "ltr.joblib"

    save_artifact(artifact, artifact_path)
    loaded_artifact = load_artifact(artifact_path)

    assert loaded_artifact.ltr_model == {"kind": "ridge"}


def track_interactions_df() -> pd.DataFrame:
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


def track_metadata_df() -> pd.DataFrame:
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


def test_artifact_bundle_round_trips_track_bundle(tmp_path: Path) -> None:
    from music_recommender.tracks import build_track_serving_resources

    artifact = create_test_artifact(tmp_path)
    artifact.track_bundle = build_track_serving_resources(
        track_interactions_df(), track_metadata_df()
    )
    artifact_path = tmp_path / "tracks.joblib"

    save_artifact(artifact, artifact_path)
    loaded_artifact = load_artifact(artifact_path)

    assert loaded_artifact.track_bundle is not None
    assert loaded_artifact.track_bundle.track_ids == ["track_1", "track_2"]
    assert loaded_artifact.track_bundle.similarity_matrix.shape == (2, 2)


def test_artifact_without_track_bundle_defaults_to_none(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    del artifact.track_bundle
    artifact_path = tmp_path / "legacy-tracks.joblib"

    save_artifact(artifact, artifact_path)
    loaded_artifact = load_artifact(artifact_path)

    assert loaded_artifact.track_bundle is None


def test_artifact_rejects_inconsistent_track_bundle(tmp_path: Path) -> None:
    from music_recommender.tracks import build_track_serving_resources

    artifact = create_test_artifact(tmp_path)
    artifact.track_bundle = build_track_serving_resources(
        track_interactions_df(), track_metadata_df()
    )
    artifact.track_bundle.similarity_matrix = np.zeros((3, 3))
    artifact_path = tmp_path / "bad-tracks.joblib"

    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="track similarity matrix"):
        load_artifact(artifact_path)


def test_artifact_rejects_invalid_track_bundle_type(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.track_bundle = {"track_ids": ["track_1"]}
    artifact_path = tmp_path / "bad-track-type.joblib"

    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="track bundle has an invalid structure"):
        load_artifact(artifact_path)


def _track_bundle_artifact(tmp_path: Path):
    from music_recommender.tracks import build_track_serving_resources

    artifact = create_test_artifact(tmp_path)
    artifact.track_bundle = build_track_serving_resources(
        track_interactions_df(), track_metadata_df()
    )
    return artifact


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("track_ids", [], "invalid track identifiers"),
        ("track_ids", ["track_1", "track_1"], "invalid track identifiers"),
        ("track_id_to_index", {}, "mappings are inconsistent"),
        ("feature_names", ["danceability", "danceability"], "invalid feature names"),
        (
            "user_track_matrix",
            pd.DataFrame(columns=["unknown_track"]),
            "interaction matrix is inconsistent",
        ),
        ("track_lookup", {}, "metadata lookup is inconsistent"),
        ("track_stats", {}, "statistics are inconsistent"),
        (
            "track_stats",
            {"track_1": {"track_id": "track_1"}},
            "statistics are inconsistent",
        ),
    ],
)
def test_artifact_rejects_inconsistent_track_bundle_fields(
    tmp_path: Path, attribute: str, value: object, message: str
) -> None:
    artifact = _track_bundle_artifact(tmp_path)
    setattr(artifact.track_bundle, attribute, value)
    artifact_path = tmp_path / "bad-track-field.joblib"

    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match=message):
        load_artifact(artifact_path)


def test_corrupt_artifact_has_actionable_load_error(tmp_path: Path) -> None:
    artifact_path = tmp_path / "corrupt.joblib"
    artifact_path.write_bytes(b"not a joblib artifact")

    with pytest.raises(ValueError, match="could not be loaded.*Retrain"):
        load_artifact(artifact_path)


def test_non_bundle_artifact_is_rejected(tmp_path: Path) -> None:
    artifact_path = tmp_path / "wrong-type.joblib"
    joblib.dump({"version": "4.0"}, artifact_path)

    with pytest.raises(ValueError, match="not a valid recommender bundle"):
        load_artifact(artifact_path)


def test_inconsistent_artifact_dimensions_are_rejected(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.user_item_matrix = csr_matrix((1, 1))
    artifact_path = tmp_path / "inconsistent.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="matrix dimensions"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    "corrupt_mappings",
    [
        lambda mappings: mappings["index_to_user_id"].pop(0),
        lambda mappings: mappings["artist_id_to_index"].update({"artist_2": 0}),
        lambda mappings: mappings["artist_id_to_name"].pop("artist_1"),
        lambda mappings: mappings["user_id_to_index"].update({" user_1 ": 0}),
    ],
)
def test_inconsistent_artifact_mappings_are_rejected(
    tmp_path: Path,
    corrupt_mappings,
) -> None:
    artifact = create_test_artifact(tmp_path)
    corrupt_mappings(artifact.mappings)
    artifact_path = tmp_path / "inconsistent-mappings.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="mapping|artist names"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    "corrupt_artifact",
    [
        lambda artifact: artifact.model.user_factors.__setitem__(
            (0, 0),
            np.nan,
        ),
        lambda artifact: artifact.user_item_matrix.data.__setitem__(0, np.inf),
        lambda artifact: artifact.content_artifacts.content_matrix.data.__setitem__(
            0,
            np.nan,
        ),
        lambda artifact: setattr(
            artifact.model,
            "item_factors",
            artifact.model.item_factors[:, :-1],
        ),
    ],
)
def test_non_finite_or_inconsistent_numeric_artifacts_are_rejected(
    tmp_path: Path,
    corrupt_artifact,
) -> None:
    artifact = create_test_artifact(tmp_path)
    corrupt_artifact(artifact)
    artifact_path = tmp_path / "invalid-numeric-data.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="finite|latent dimensions"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    "corrupt_content",
    [
        lambda artifact: artifact.content_artifacts.artist_id_to_content_index.update(
            {"artist_2": 0}
        ),
        lambda artifact: setattr(
            artifact.content_artifacts,
            "metadata",
            artifact.content_artifacts.metadata.iloc[::-1].reset_index(drop=True),
        ),
        lambda artifact: artifact.content_artifacts.feature_names.pop(),
        lambda artifact: artifact.content_artifacts.metadata_lookup["artist_1"].update(
            {"genres": ["wrong"]}
        ),
    ],
)
def test_inconsistent_content_artifacts_are_rejected(
    tmp_path: Path,
    corrupt_content,
) -> None:
    artifact = create_test_artifact(tmp_path)
    corrupt_content(artifact)
    artifact_path = tmp_path / "invalid-content-data.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="content"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    "corrupt_stats",
    [
        lambda artifact: artifact.artist_stats["artist_1"].update(
            {"total_plays": np.nan}
        ),
        lambda artifact: artifact.artist_stats["artist_1"].update(
            {"artist_name": "Wrong artist"}
        ),
        lambda artifact: artifact.artist_stats["artist_1"].update(
            {"popularity_rank": artifact.artist_stats["artist_2"]["popularity_rank"]}
        ),
        lambda artifact: artifact.artist_stats["artist_1"].pop("listener_count"),
    ],
)
def test_inconsistent_artist_statistics_are_rejected(
    tmp_path: Path,
    corrupt_stats,
) -> None:
    artifact = create_test_artifact(tmp_path)
    corrupt_stats(artifact)
    artifact_path = tmp_path / "invalid-artist-stats.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="artist statistics|popularity ranks"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    "corrupt_metadata",
    [
        lambda artifact: artifact.metadata.update({"num_users": "2"}),
        lambda artifact: artifact.metadata.update({"created_at": "not-a-timestamp"}),
        lambda artifact: artifact.metadata["dataset"].update({"sha256": "invalid"}),
        lambda artifact: artifact.metadata["metadata_dataset"].update(
            {"row_count": 999}
        ),
        lambda artifact: artifact.metadata.update({"training_device": "quantum"}),
    ],
)
def test_inconsistent_artifact_metadata_is_rejected(
    tmp_path: Path,
    corrupt_metadata,
) -> None:
    artifact = create_test_artifact(tmp_path)
    corrupt_metadata(artifact)
    artifact_path = tmp_path / "invalid-metadata.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="metadata|fingerprint"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    "corrupt_configuration",
    [
        lambda artifact: artifact.training_config.update({"factors": 99}),
        lambda artifact: artifact.training_config.update({"alpha": np.nan}),
        lambda artifact: artifact.training_config.pop("iterations"),
        lambda artifact: artifact.hybrid_config.update(
            {"default_content_weight": 0.75}
        ),
    ],
)
def test_inconsistent_artifact_configuration_is_rejected(
    tmp_path: Path,
    corrupt_configuration,
) -> None:
    artifact = create_test_artifact(tmp_path)
    corrupt_configuration(artifact)
    artifact_path = tmp_path / "invalid-configuration.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="configuration|factor count"):
        load_artifact(artifact_path)


def test_artist_stats_include_popularity_rank() -> None:
    stats = build_artist_stats(artifact_dataframe())

    assert stats["artist_2"]["total_plays"] == 10
    assert stats["artist_2"]["listener_count"] == 2
    assert stats["artist_2"]["popularity_rank"] == 1


def test_dataset_fingerprint_changes_when_data_changes(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_path.write_text("user_id,artist_id,artist_name,play_count\nu,a,A,1\n")
    second_path.write_text("user_id,artist_id,artist_name,play_count\nu,a,A,2\n")

    first = create_dataset_fingerprint(first_path, artifact_dataframe())
    second = create_dataset_fingerprint(second_path, artifact_dataframe())

    assert first["sha256"] != second["sha256"]


def test_dataset_fingerprint_falls_back_when_path_cannot_be_read(
    tmp_path: Path,
) -> None:
    fingerprint = create_dataset_fingerprint(tmp_path, artifact_dataframe())

    assert fingerprint["path"] == str(tmp_path)
    assert fingerprint["row_count"] == 4
    assert len(fingerprint["sha256"]) == 64


def test_recommender_artifact_repr_summarizes_bundle() -> None:
    artifact = create_test_artifact(Path("."))

    text = repr(artifact)

    assert text == ("RecommenderArtifact(version='4.0', num_users=2, num_artists=3)")


def test_load_artifact_wraps_unexpected_validation_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact_path = tmp_path / "valid.joblib"
    save_artifact(artifact, artifact_path)

    def explode(_: object) -> None:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(artifacts_module, "_validate_loaded_artifact", explode)

    with pytest.raises(ValueError, match="structure is invalid"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_incompatible_version(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.version = "3.0"
    artifact_path = tmp_path / "wrong-version.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="not.*compatible with required version"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_non_dict_mappings(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.mappings = ["not-a-mapping"]
    artifact_path = tmp_path / "bad-mappings.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="mappings are not a dictionary"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_mappings_missing_required_fields(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.mappings.pop("index_to_user_id")
    artifact_path = tmp_path / "missing-mapping.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="missing fields"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_non_dict_forward_mapping(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.mappings["user_id_to_index"] = "users"
    artifact_path = tmp_path / "non-dict-mapping.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="user mappings are not dictionaries"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_non_csr_interaction_matrix(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.user_item_matrix = np.zeros((2, 3))
    artifact_path = tmp_path / "dense-matrix.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="not CSR sparse data"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_invalid_content_artifact_type(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.content_artifacts = {"not": "content"}
    artifact_path = tmp_path / "bad-content.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="content data has an invalid structure"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_invalid_factor_shapes(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.model.user_factors = np.array([1.0, 2.0])
    artifact_path = tmp_path / "flat-factors.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="invalid structure"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_artist_factor_row_mismatch(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.model.user_factors = np.zeros((5, 4))
    artifact_path = tmp_path / "extra-artist-factors.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="artist factors do not match"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_user_factor_row_mismatch(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.model.item_factors = np.zeros((1, 4))
    artifact_path = tmp_path / "fewer-user-factors.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="user factors do not match"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_content_matrix_row_mismatch(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    feature_count = artifact.content_artifacts.content_matrix.shape[1]
    artifact.content_artifacts.content_matrix = csr_matrix((2, feature_count))
    artifact_path = tmp_path / "short-content-matrix.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="content matrix does not match"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    "corrupt_content",
    [
        lambda content: setattr(
            content, "feature_names", content.feature_names[:-1] + ["zzz_new"]
        ),
        lambda content: setattr(content, "metadata", "not-a-frame"),
        lambda content: content.metadata.loc.__setitem__(
            (0, "genres"),
            "",
        ),
        lambda content: setattr(content, "metadata_lookup", {}),
    ],
)
def test_content_artifacts_validation_rejects_inconsistent_state(
    corrupt_content,
) -> None:
    artifact = create_test_artifact(Path("."))
    corrupt_content(artifact.content_artifacts)
    artist_ids = set(artifact.mappings["artist_id_to_index"])

    with pytest.raises(
        ValueError,
        match="vectorizer|invalid structure|empty values|metadata lookup",
    ):
        _validate_content_artifacts(
            artifact.content_artifacts,
            artist_ids,
            artifact.mappings["artist_id_to_name"],
        )


def test_content_artifacts_validation_rejects_artist_id_mismatch() -> None:
    artifact = create_test_artifact(Path("."))
    artist_ids = set(artifact.mappings["artist_id_to_index"])

    with pytest.raises(ValueError, match="content artists do not match"):
        _validate_content_artifacts(
            artifact.content_artifacts,
            artist_ids | {"artist_99"},
            artifact.mappings["artist_id_to_name"],
        )


def test_load_artifact_rejects_artist_stats_not_matching_mappings(
    tmp_path: Path,
) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.artist_stats["artist_99"] = {"total_plays": 1}
    artifact_path = tmp_path / "extra-stats.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="artist statistics do not match"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_non_dict_metadata(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.metadata = "not-a-dict"
    artifact_path = tmp_path / "bad-metadata.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="metadata is not a dictionary"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_metadata_missing_required_fields(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.metadata.pop("created_at")
    artifact_path = tmp_path / "missing-metadata.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="missing fields"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_invalid_configuration_paths(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.training_config["raw_data_path"] = "   "
    artifact_path = tmp_path / "blank-path.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="invalid data paths"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_invalid_configuration_integers(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.training_config["factors"] = "four"
    artifact_path = tmp_path / "string-int.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="invalid integer parameters"):
        load_artifact(artifact_path)


def test_load_artifact_rejects_invalid_dataset_fingerprint(tmp_path: Path) -> None:
    artifact = create_test_artifact(tmp_path)
    artifact.metadata["dataset"] = "not-a-fingerprint"
    artifact_path = tmp_path / "bad-fingerprint.joblib"
    save_artifact(artifact, artifact_path)

    with pytest.raises(ValueError, match="fingerprint is not a dictionary"):
        load_artifact(artifact_path)
