"""Project configuration and default training settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV_VAR = "MUSIC_RECOMMENDER_ROOT"


def resolve_project_root() -> Path:
    """Resolve the runtime root for data and model artifacts."""
    configured_root = os.getenv(PROJECT_ROOT_ENV_VAR)
    if configured_root and configured_root.strip():
        return Path(configured_root.strip()).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = resolve_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sample_interactions.csv"
RAW_METADATA_PATH = DATA_DIR / "raw" / "sample_artist_metadata.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
MAPPINGS_DIR = ARTIFACTS_DIR / "mappings"
ARTIFACT_BUNDLE_PATH = ARTIFACTS_DIR / "recommender_artifact.joblib"
MODEL_PATH = MODEL_DIR / "als_model.joblib"
MAPPINGS_PATH = MAPPINGS_DIR / "id_mappings.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_MIN_USER_INTERACTIONS = 2
DEFAULT_MIN_ARTIST_INTERACTIONS = 2
DEFAULT_ALS_FACTORS = 32
DEFAULT_ALS_REGULARIZATION = 0.05
DEFAULT_ALS_ITERATIONS = 20
DEFAULT_ALS_ALPHA = 15.0
DEFAULT_USE_GPU = True
DEFAULT_CONTENT_WEIGHT = 0.25
DEFAULT_TOP_K = 10
