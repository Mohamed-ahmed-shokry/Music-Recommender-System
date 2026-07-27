from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

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
