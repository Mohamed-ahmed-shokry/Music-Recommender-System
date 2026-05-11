"""ALS model training and persistence."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from scipy.sparse import csr_matrix

from music_recommender.artifacts import build_recommender_artifact, save_artifact
from music_recommender.config import (
    ARTIFACT_BUNDLE_PATH,
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
    DEFAULT_MIN_ARTIST_INTERACTIONS,
    DEFAULT_MIN_USER_INTERACTIONS,
    DEFAULT_USE_GPU,
    MAPPINGS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)
from music_recommender.preprocessing import Mappings, prepare_training_data

if TYPE_CHECKING:
    from implicit.als import AlternatingLeastSquares


def _create_als_model(
    factors: int,
    regularization: float,
    iterations: int,
    use_gpu: bool,
) -> AlternatingLeastSquares:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Disabling GPU support because.*",
            category=UserWarning,
            module="implicit.gpu",
        )
        warnings.filterwarnings(
            "ignore",
            message="OpenBLAS is configured.*",
            category=RuntimeWarning,
            module="implicit.cpu.als",
        )
        from implicit.als import AlternatingLeastSquares

        return AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            dtype=np.float32,
            use_gpu=use_gpu,
            random_state=42,
        )


def train_als_model(
    user_item_matrix: csr_matrix,
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
    use_gpu: bool = DEFAULT_USE_GPU,
) -> AlternatingLeastSquares:
    """Train an ALS model on a user-item interaction matrix."""
    item_user_matrix = (user_item_matrix * alpha).T.tocsr()
    try:
        model = _create_als_model(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu,
        )
    except (ImportError, ValueError, RuntimeError) as error:
        if not use_gpu:
            raise
        model = _create_als_model(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False,
        )
        model.training_device = "cpu"
        model.gpu_fallback_reason = str(error)
    else:
        model.training_device = "gpu" if use_gpu else "cpu"
        model.gpu_fallback_reason = None

    try:
        model.fit(item_user_matrix, show_progress=False)
    except (ImportError, ValueError, RuntimeError) as error:
        if not use_gpu or getattr(model, "training_device", "cpu") != "gpu":
            raise
        model = _create_als_model(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False,
        )
        model.training_device = "cpu"
        model.gpu_fallback_reason = str(error)
        model.fit(item_user_matrix, show_progress=False)

    if getattr(model, "training_device", "cpu") == "gpu" and hasattr(model, "to_cpu"):
        model = model.to_cpu()
        model.training_device = "gpu"
        model.gpu_fallback_reason = None
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
    artifact_path: str | Path = ARTIFACT_BUNDLE_PATH,
    min_user_interactions: int = DEFAULT_MIN_USER_INTERACTIONS,
    min_artist_interactions: int = DEFAULT_MIN_ARTIST_INTERACTIONS,
    factors: int = DEFAULT_ALS_FACTORS,
    regularization: float = DEFAULT_ALS_REGULARIZATION,
    iterations: int = DEFAULT_ALS_ITERATIONS,
    alpha: float = DEFAULT_ALS_ALPHA,
    use_gpu: bool = DEFAULT_USE_GPU,
) -> tuple[AlternatingLeastSquares, csr_matrix, Mappings]:
    """Prepare data, train ALS, and save model artifacts."""
    filtered_df, user_item_matrix, mappings = prepare_training_data(
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
        use_gpu=use_gpu,
    )
    save_model(model, model_path)
    training_config = {
        "raw_data_path": str(raw_data_path),
        "min_user_interactions": min_user_interactions,
        "min_artist_interactions": min_artist_interactions,
        "factors": factors,
        "regularization": regularization,
        "iterations": iterations,
        "alpha": alpha,
        "use_gpu": use_gpu,
    }
    artifact = build_recommender_artifact(
        model=model,
        mappings=mappings,
        user_item_matrix=user_item_matrix,
        filtered_df=filtered_df,
        raw_data_path=raw_data_path,
        training_config=training_config,
    )
    save_artifact(artifact, artifact_path)
    return model, user_item_matrix, mappings
