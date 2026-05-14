"""Versioned artifact bundle helpers for serving recommendations."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.sparse import csr_matrix

from music_recommender.config import ARTIFACT_BUNDLE_PATH
from music_recommender.content import ContentArtifacts
from music_recommender.preprocessing import Mappings

ARTIFACT_VERSION = "4.0"

ArtistStats = dict[str, str | int | float]


@dataclass
class RecommenderArtifact:
    """All state needed to serve recommendations without reprocessing raw data."""

    version: str
    model: Any
    mappings: Mappings
    user_item_matrix: csr_matrix
    artist_stats: dict[str, ArtistStats]
    content_artifacts: ContentArtifacts
    metadata: dict[str, Any]
    training_config: dict[str, Any]
    hybrid_config: dict[str, Any]


def create_dataset_fingerprint(
    data_path: str | Path,
    df: pd.DataFrame,
) -> dict[str, str | int]:
    """Create a stable fingerprint for the source interaction data."""
    path = Path(data_path)
    if path.exists():
        content = path.read_bytes()
    else:
        content = df.to_csv(index=False).encode("utf-8")

    return {
        "path": str(path),
        "row_count": int(len(df)),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_artist_stats(df: pd.DataFrame) -> dict[str, ArtistStats]:
    """Compute popularity and explainability stats for each artist."""
    artist_names = (
        df[["artist_id", "artist_name"]]
        .drop_duplicates(subset="artist_id")
        .set_index("artist_id")["artist_name"]
    )
    stats_df = (
        df.groupby("artist_id")
        .agg(
            total_plays=("play_count", "sum"),
            listener_count=("user_id", "nunique"),
            interaction_count=("user_id", "count"),
        )
        .reset_index()
    )
    stats_df["artist_name"] = stats_df["artist_id"].map(artist_names)
    stats_df = stats_df.sort_values(
        by=["total_plays", "listener_count", "interaction_count", "artist_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    stats_df["popularity_rank"] = stats_df.index + 1

    return {
        str(row.artist_id): {
            "artist_id": str(row.artist_id),
            "artist_name": str(row.artist_name),
            "total_plays": int(row.total_plays),
            "listener_count": int(row.listener_count),
            "interaction_count": int(row.interaction_count),
            "popularity_rank": int(row.popularity_rank),
        }
        for row in stats_df.itertuples(index=False)
    }


def build_recommender_artifact(
    model: Any,
    mappings: Mappings,
    user_item_matrix: csr_matrix,
    filtered_df: pd.DataFrame,
    content_artifacts: ContentArtifacts,
    raw_data_path: str | Path,
    metadata_path: str | Path,
    training_config: dict[str, Any],
    hybrid_config: dict[str, Any],
) -> RecommenderArtifact:
    """Build a versioned artifact from trained model state."""
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "training_device": getattr(model, "training_device", "unknown"),
        "gpu_fallback_reason": getattr(model, "gpu_fallback_reason", None),
        "num_users": len(mappings["user_id_to_index"]),
        "num_artists": len(mappings["artist_id_to_index"]),
        "num_interactions": int(user_item_matrix.nnz),
        "dataset": create_dataset_fingerprint(raw_data_path, filtered_df),
        "metadata_dataset": create_dataset_fingerprint(
            metadata_path,
            content_artifacts.metadata,
        ),
    }

    return RecommenderArtifact(
        version=ARTIFACT_VERSION,
        model=model,
        mappings=mappings,
        user_item_matrix=user_item_matrix,
        artist_stats=build_artist_stats(filtered_df),
        content_artifacts=content_artifacts,
        metadata=metadata,
        training_config=training_config,
        hybrid_config=hybrid_config,
    )


def save_artifact(
    artifact: RecommenderArtifact,
    path: str | Path = ARTIFACT_BUNDLE_PATH,
) -> None:
    """Persist a recommender artifact bundle."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def load_artifact(path: str | Path = ARTIFACT_BUNDLE_PATH) -> RecommenderArtifact:
    """Load a recommender artifact bundle."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Recommender artifact not found. Train the model first."
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Disabling GPU support because.*",
            category=UserWarning,
            module="implicit.gpu",
        )
        artifact = joblib.load(artifact_path)

    if getattr(artifact, "version", None) != ARTIFACT_VERSION:
        raise ValueError(
            f"Artifact version {getattr(artifact, 'version', 'unknown')} is not "
            f"compatible with required version {ARTIFACT_VERSION}. Retrain the model."
        )
    required_fields = ("content_artifacts", "hybrid_config")
    missing_fields = [
        field for field in required_fields if not hasattr(artifact, field)
    ]
    if missing_fields:
        raise ValueError(
            f"Artifact is missing v3 fields {missing_fields}. Retrain the model."
        )

    return artifact
