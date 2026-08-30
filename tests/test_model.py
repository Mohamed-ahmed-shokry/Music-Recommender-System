from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import music_recommender.model as model_module
from music_recommender.artifacts import load_artifact
from music_recommender.model import (
    load_model,
    save_model,
    train_als_model,
    train_and_save_model,
)


def tiny_matrix() -> csr_matrix:
    return csr_matrix(
        [
            [5, 0, 1],
            [0, 4, 1],
            [3, 0, 0],
        ],
        dtype="float32",
    )


def test_als_model_can_train_on_tiny_matrix() -> None:
    model = train_als_model(
        user_item_matrix=tiny_matrix(),
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=False,
    )

    assert model.user_factors.shape[0] == 3
    assert model.item_factors.shape[0] == 3


def test_saved_model_can_be_loaded(tmp_path: Path) -> None:
    model = train_als_model(
        user_item_matrix=tiny_matrix(),
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=False,
    )
    model_path = tmp_path / "model.joblib"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    assert loaded_model.user_factors.shape == model.user_factors.shape
    assert loaded_model.item_factors.shape == model.item_factors.shape


def test_load_model_raises_when_artifact_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model(tmp_path / "missing_model.joblib")


def test_load_model_raises_on_corrupt_artifact(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"not a valid joblib payload")

    with pytest.raises(ValueError, match="could not be loaded"):
        load_model(corrupt_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"factors": 0}, "factors"),
        ({"factors": True}, "factors"),
        ({"regularization": -0.1}, "regularization"),
        ({"regularization": np.nan}, "regularization"),
        ({"iterations": 0}, "iterations"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": np.inf}, "alpha"),
        ({"use_gpu": 1}, "use_gpu"),
    ],
)
def test_als_training_rejects_invalid_hyperparameters(
    overrides: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "user_item_matrix": tiny_matrix(),
        "factors": 4,
        "regularization": 0.01,
        "iterations": 2,
        "alpha": 10.0,
        "use_gpu": False,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        train_als_model(**parameters)


@pytest.mark.parametrize(
    "matrix",
    [
        csr_matrix((0, 0), dtype="float32"),
        csr_matrix([[0, 0], [0, 0]], dtype="float32"),
        csr_matrix([[1, np.nan]], dtype="float32"),
        csr_matrix([[1, -1]], dtype="float32"),
        csr_matrix([[1 + 0j, 2 + 3j]], dtype="complex128"),
    ],
)
def test_als_training_rejects_invalid_interaction_matrix(
    matrix: csr_matrix,
) -> None:
    with pytest.raises(ValueError, match="user_item_matrix"):
        train_als_model(
            user_item_matrix=matrix,
            factors=4,
            regularization=0.01,
            iterations=2,
            alpha=10.0,
            use_gpu=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"factors": 0}, "factors"),
        ({"content_weight": float("nan")}, "content_weight"),
        ({"popularity_penalty": 1.5}, "popularity_penalty"),
        ({"diversity": float("nan")}, "diversity"),
        ({"include_listened": 1}, "include_listened"),
    ],
)
def test_train_and_save_validates_configuration_before_reading_data(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "raw_data_path": tmp_path / "missing-interactions.csv",
        "metadata_path": tmp_path / "missing-metadata.csv",
        "model_path": tmp_path / "model.joblib",
        "mappings_path": tmp_path / "mappings.joblib",
        "artifact_path": tmp_path / "artifact.joblib",
        "use_gpu": False,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        train_and_save_model(**parameters)


def test_train_and_save_persists_model_and_artifact(tmp_path: Path) -> None:
    interactions = (
        "user_id,artist_id,artist_name,play_count\n"
        "user_1,artist_1,artist-one,10\n"
        "user_1,artist_2,artist-two,8\n"
        "user_2,artist_1,artist-one,6\n"
        "user_2,artist_2,artist-two,4\n"
    )
    metadata = (
        "artist_id,artist_name,genres,mood_tags,country,era\n"
        "artist_1,artist-one,pop;rock,bright;upbeat,US,2020s\n"
        "artist_2,artist-two,rock;indie,raw;dark,UK,2010s\n"
    )
    raw_data_path = tmp_path / "interactions.csv"
    raw_metadata_path = tmp_path / "metadata.csv"
    raw_data_path.write_text(interactions)
    raw_metadata_path.write_text(metadata)

    model, matrix, mappings = train_and_save_model(
        raw_data_path=raw_data_path,
        metadata_path=raw_metadata_path,
        model_path=tmp_path / "model.joblib",
        mappings_path=tmp_path / "mappings.joblib",
        artifact_path=tmp_path / "artifact.joblib",
        min_user_interactions=2,
        min_artist_interactions=2,
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=False,
        content_weight=0.25,
    )

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "mappings.joblib").exists()
    assert (tmp_path / "artifact.joblib").exists()
    assert len(mappings["user_id_to_index"]) == 2
    assert len(mappings["artist_id_to_index"]) == 2
    assert matrix.shape == (2, 2)
    assert model.user_factors.shape == (2, 4)


def test_train_and_save_stores_champion_ranking_settings(tmp_path: Path) -> None:
    interactions = (
        "user_id,artist_id,artist_name,play_count\n"
        "user_1,artist_1,artist-one,10\n"
        "user_1,artist_2,artist-two,8\n"
        "user_2,artist_1,artist-one,6\n"
        "user_2,artist_2,artist-two,4\n"
    )
    metadata = (
        "artist_id,artist_name,genres,mood_tags,country,era\n"
        "artist_1,artist-one,pop;rock,bright;upbeat,US,2020s\n"
        "artist_2,artist-two,rock;indie,raw;dark,UK,2010s\n"
    )
    raw_data_path = tmp_path / "interactions.csv"
    raw_metadata_path = tmp_path / "metadata.csv"
    raw_data_path.write_text(interactions)
    raw_metadata_path.write_text(metadata)

    _, _, _ = train_and_save_model(
        raw_data_path=raw_data_path,
        metadata_path=raw_metadata_path,
        model_path=tmp_path / "model.joblib",
        mappings_path=tmp_path / "mappings.joblib",
        artifact_path=tmp_path / "artifact.joblib",
        min_user_interactions=2,
        min_artist_interactions=2,
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=False,
        content_weight=0.25,
        popularity_penalty=0.2,
        diversity=0.5,
        include_listened=True,
    )

    artifact = load_artifact(tmp_path / "artifact.joblib")
    assert artifact.ranking_config == {
        "include_listened": True,
        "popularity_penalty": 0.2,
        "diversity": 0.5,
    }


def test_train_and_save_does_not_persist_model_on_metadata_failure(
    tmp_path: Path,
) -> None:
    interactions = (
        "user_id,artist_id,artist_name,play_count\n"
        "user_1,artist_1,artist-one,10\n"
        "user_1,artist_2,artist-two,8\n"
        "user_2,artist_1,artist-one,6\n"
        "user_2,artist_2,artist-two,4\n"
    )
    bad_metadata = (
        "artist_id,artist_name,mood_tags,country,era\n"
        "artist_1,artist-one,bright;upbeat,US,2020s\n"
        "artist_2,artist-two,raw;dark,UK,2010s\n"
    )
    raw_data_path = tmp_path / "interactions.csv"
    raw_metadata_path = tmp_path / "metadata.csv"
    raw_data_path.write_text(interactions)
    raw_metadata_path.write_text(bad_metadata)
    model_path = tmp_path / "model.joblib"
    artifact_path = tmp_path / "artifact.joblib"

    with pytest.raises(ValueError, match="Missing metadata columns"):
        train_and_save_model(
            raw_data_path=raw_data_path,
            metadata_path=raw_metadata_path,
            model_path=model_path,
            mappings_path=tmp_path / "mappings.joblib",
            artifact_path=artifact_path,
            min_user_interactions=2,
            min_artist_interactions=2,
            factors=4,
            regularization=0.01,
            iterations=2,
            alpha=10.0,
            use_gpu=False,
        )

    assert not model_path.exists()
    assert not artifact_path.exists()


def test_gpu_model_creation_failure_falls_back_to_cpu(monkeypatch) -> None:
    def fake_create(factors, regularization, iterations, use_gpu):
        if use_gpu:
            raise RuntimeError("cuda unavailable")
        model = SimpleNamespace()
        model.training_device = "cpu"
        model.gpu_fallback_reason = None
        model.fit = lambda *_args, **_kwargs: None
        return model

    monkeypatch.setattr(model_module, "_create_als_model", fake_create)

    model = train_als_model(
        user_item_matrix=tiny_matrix(),
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=True,
    )

    assert model.training_device == "cpu"
    assert model.gpu_fallback_reason == "cuda unavailable"


def test_gpu_model_creation_failure_without_gpu_raises(monkeypatch) -> None:
    def fake_create(*_args, **_kwargs) -> None:
        raise RuntimeError("cuda unavailable")

    monkeypatch.setattr(model_module, "_create_als_model", fake_create)

    with pytest.raises(RuntimeError, match="cuda unavailable"):
        train_als_model(
            user_item_matrix=tiny_matrix(),
            factors=4,
            regularization=0.01,
            iterations=2,
            alpha=10.0,
            use_gpu=False,
        )


def test_gpu_fit_failure_falls_back_to_cpu(monkeypatch) -> None:
    def make_model(*_args, use_gpu: bool, **_kwargs):
        model = SimpleNamespace()
        model.training_device = "gpu" if use_gpu else "cpu"
        model.gpu_fallback_reason = None

        def fit(*_fit_args, **_fit_kwargs) -> None:
            if model.training_device == "gpu":
                raise RuntimeError("gpu fit failed")

        model.fit = fit
        return model

    monkeypatch.setattr(model_module, "_create_als_model", make_model)

    model = train_als_model(
        user_item_matrix=tiny_matrix(),
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=True,
    )

    assert model.training_device == "cpu"
    assert model.gpu_fallback_reason == "gpu fit failed"


def test_gpu_fit_failure_without_gpu_raises(monkeypatch) -> None:
    model = SimpleNamespace()
    model.training_device = "cpu"
    model.gpu_fallback_reason = None

    def fit(*_args, **_kwargs) -> None:
        raise RuntimeError("fit exploded")

    model.fit = fit

    monkeypatch.setattr(
        model_module,
        "_create_als_model",
        lambda *_args, **_kwargs: model,
    )

    with pytest.raises(RuntimeError, match="fit exploded"):
        train_als_model(
            user_item_matrix=tiny_matrix(),
            factors=4,
            regularization=0.01,
            iterations=2,
            alpha=10.0,
            use_gpu=False,
        )


def test_gpu_model_moves_to_cpu_when_supported(monkeypatch) -> None:
    class GpuModel:
        def __init__(self) -> None:
            self.training_device = "gpu"
            self.gpu_fallback_reason = None

        def fit(self, *_args, **_kwargs) -> None:
            return None

        def to_cpu(self) -> "GpuModel":
            return self

    monkeypatch.setattr(
        model_module,
        "_create_als_model",
        lambda *_args, **_kwargs: GpuModel(),
    )

    model = train_als_model(
        user_item_matrix=tiny_matrix(),
        factors=4,
        regularization=0.01,
        iterations=2,
        alpha=10.0,
        use_gpu=True,
    )

    assert model.training_device == "gpu"
    assert model.gpu_fallback_reason is None
