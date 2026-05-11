"""Project configuration and default training settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sample_interactions.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
MAPPINGS_DIR = ARTIFACTS_DIR / "mappings"
MODEL_PATH = MODEL_DIR / "als_model.joblib"
MAPPINGS_PATH = MAPPINGS_DIR / "id_mappings.joblib"

DEFAULT_MIN_USER_INTERACTIONS = 2
DEFAULT_MIN_ARTIST_INTERACTIONS = 2
DEFAULT_ALS_FACTORS = 32
DEFAULT_ALS_REGULARIZATION = 0.05
DEFAULT_ALS_ITERATIONS = 20
DEFAULT_ALS_ALPHA = 15.0
DEFAULT_USE_GPU = True
DEFAULT_TOP_K = 10
