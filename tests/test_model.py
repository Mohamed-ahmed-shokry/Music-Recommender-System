from pathlib import Path

from scipy.sparse import csr_matrix

from music_recommender.model import load_model, save_model, train_als_model


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
