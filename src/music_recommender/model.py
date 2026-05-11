"""ALS model training and persistence."""

from pathlib import Path

import joblib
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from music_recommender.config import (
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
    DEFAULT_MIN_ARTIST_INTERACTIONS,
    DEFAULT_MIN_USER_INTERACTIONS,
    MAPPINGS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)
from music_recommender.preprocessing import Mappings, prepare_training_data


def train_als_model(
    user_item_matrix: csr_matrix,
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
) -> AlternatingLeastSquares:
    """Train an ALS model on a user-item interaction matrix."""
    item_user_matrix = (user_item_matrix * alpha).T.tocsr()
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        dtype=np.float32,
        use_gpu=False,
        random_state=42,
    )
    model.fit(item_user_matrix, show_progress=False)
    return model


def save_model(model: AlternatingLeastSquares, path: str | Path) -> None:
    """Save a trained ALS model to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str | Path) -> AlternatingLeastSquares:
    """Load a trained ALS model from disk."""
    return joblib.load(path)


def train_and_save_model(
    raw_data_path: str | Path = RAW_DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    mappings_path: str | Path = MAPPINGS_PATH,
    min_user_interactions: int = DEFAULT_MIN_USER_INTERACTIONS,
    min_artist_interactions: int = DEFAULT_MIN_ARTIST_INTERACTIONS,
    factors: int = DEFAULT_ALS_FACTORS,
    regularization: float = DEFAULT_ALS_REGULARIZATION,
    iterations: int = DEFAULT_ALS_ITERATIONS,
    alpha: float = DEFAULT_ALS_ALPHA,
) -> tuple[AlternatingLeastSquares, csr_matrix, Mappings]:
    """Prepare data, train ALS, and save model artifacts."""
    _, user_item_matrix, mappings = prepare_training_data(
        raw_data_path=raw_data_path,
        mappings_path=mappings_path,
        min_user_interactions=min_user_interactions,
        min_artist_interactions=min_artist_interactions,
    )
    model = train_als_model(
        user_item_matrix=user_item_matrix,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
    )
    save_model(model, model_path)
    return model, user_item_matrix, mappings
