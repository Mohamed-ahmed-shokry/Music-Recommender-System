"""Versioned artifact bundle helpers for serving recommendations."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from music_recommender.config import ARTIFACT_BUNDLE_PATH
from music_recommender.content import ContentArtifacts
from music_recommender.preprocessing import Mappings
from music_recommender.utils import atomic_joblib_dump

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
    atomic_joblib_dump(artifact, path)


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
        try:
            artifact = joblib.load(artifact_path)
        except Exception as error:
            raise ValueError(
                f"Recommender artifact at '{artifact_path}' could not be loaded. "
                "Retrain the model."
            ) from error

    try:
        artifact = _validate_loaded_artifact(artifact)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Artifact structure is invalid. Retrain the model.") from error
    return artifact


def _validate_loaded_artifact(artifact: Any) -> RecommenderArtifact:
    if not isinstance(artifact, RecommenderArtifact):
        raise ValueError(
            "Artifact is not a valid recommender bundle. Retrain the model."
        )
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(
            f"Artifact version {artifact.version} is not "
            f"compatible with required version {ARTIFACT_VERSION}. Retrain the model."
        )

    if not isinstance(artifact.mappings, dict):
        raise ValueError("Artifact mappings are not a dictionary. Retrain the model.")
    required_mapping_fields = {
        "user_id_to_index",
        "index_to_user_id",
        "artist_id_to_index",
        "index_to_artist_id",
        "artist_id_to_name",
    }
    missing_mapping_fields = sorted(required_mapping_fields - artifact.mappings.keys())
    if missing_mapping_fields:
        raise ValueError(
            f"Artifact mappings are missing fields {missing_mapping_fields}. "
            "Retrain the model."
        )

    _validate_mapping_pair(
        artifact.mappings["user_id_to_index"],
        artifact.mappings["index_to_user_id"],
        entity="user",
    )
    _validate_mapping_pair(
        artifact.mappings["artist_id_to_index"],
        artifact.mappings["index_to_artist_id"],
        entity="artist",
    )
    artist_ids = set(artifact.mappings["artist_id_to_index"])
    artist_id_to_name = artifact.mappings["artist_id_to_name"]
    if (
        not isinstance(artist_id_to_name, dict)
        or set(artist_id_to_name) != artist_ids
        or any(
            not isinstance(name, str) or not name.strip()
            for name in artist_id_to_name.values()
        )
    ):
        raise ValueError(
            "Artifact artist names do not match its artist mappings. Retrain the model."
        )

    num_users = len(artifact.mappings["user_id_to_index"])
    num_artists = len(artifact.mappings["artist_id_to_index"])
    if not isinstance(artifact.user_item_matrix, csr_matrix):
        raise ValueError(
            "Artifact interaction matrix is not CSR sparse data. Retrain the model."
        )
    if not isinstance(artifact.content_artifacts, ContentArtifacts):
        raise ValueError(
            "Artifact content data has an invalid structure. Retrain the model."
        )
    if artifact.user_item_matrix.shape != (num_users, num_artists):
        raise ValueError(
            "Artifact interaction matrix dimensions do not match its mappings. "
            "Retrain the model."
        )
    artist_factors = getattr(artifact.model, "user_factors", None)
    user_factors = getattr(artifact.model, "item_factors", None)
    if (
        not isinstance(artist_factors, np.ndarray)
        or not isinstance(user_factors, np.ndarray)
        or artist_factors.ndim != 2
        or user_factors.ndim != 2
    ):
        raise ValueError(
            "Artifact collaborative factors have an invalid structure. "
            "Retrain the model."
        )
    if artist_factors.shape[0] != num_artists:
        raise ValueError(
            "Artifact artist factors do not match its artist mappings. "
            "Retrain the model."
        )
    if user_factors.shape[0] != num_users:
        raise ValueError(
            "Artifact user factors do not match its user mappings. Retrain the model."
        )
    if artist_factors.shape[1] == 0 or artist_factors.shape[1] != user_factors.shape[1]:
        raise ValueError(
            "Artifact collaborative factors have inconsistent latent dimensions. "
            "Retrain the model."
        )
    if artifact.content_artifacts.content_matrix.shape[0] != num_artists:
        raise ValueError(
            "Artifact content matrix does not match its artist mappings. "
            "Retrain the model."
        )
    _validate_numeric_artifacts(
        artifact.user_item_matrix,
        artifact.content_artifacts.content_matrix,
        artist_factors,
        user_factors,
    )
    _validate_content_artifacts(
        artifact.content_artifacts,
        artist_ids,
        artist_id_to_name,
    )
    if artist_ids != set(artifact.artist_stats):
        raise ValueError(
            "Artifact artist statistics do not match its mappings. Retrain the model."
        )

    required_metadata_fields = {"num_users", "num_artists", "num_interactions"}
    missing_metadata_fields = sorted(
        required_metadata_fields - artifact.metadata.keys()
    )
    if missing_metadata_fields:
        raise ValueError(
            f"Artifact metadata is missing fields {missing_metadata_fields}. "
            "Retrain the model."
        )
    if (
        int(artifact.metadata["num_users"]) != num_users
        or int(artifact.metadata["num_artists"]) != num_artists
        or int(artifact.metadata["num_interactions"]) != artifact.user_item_matrix.nnz
    ):
        raise ValueError(
            "Artifact metadata dimensions do not match its model data. "
            "Retrain the model."
        )
    return artifact


def _validate_mapping_pair(
    forward: Any,
    reverse: Any,
    *,
    entity: str,
) -> None:
    if not isinstance(forward, dict) or not isinstance(reverse, dict):
        raise ValueError(
            f"Artifact {entity} mappings are not dictionaries. Retrain the model."
        )

    expected_indices = set(range(len(forward)))
    has_valid_identifiers = all(
        isinstance(identifier, str)
        and bool(identifier.strip())
        and identifier == identifier.strip()
        for identifier in forward
    )
    has_integer_indices = all(type(index) is int for index in forward.values())
    has_integer_reverse_indices = all(type(index) is int for index in reverse)
    is_contiguous_bijection = (
        set(forward.values()) == expected_indices
        and set(reverse) == expected_indices
        and all(reverse[index] == identifier for identifier, index in forward.items())
    )
    if not (
        has_valid_identifiers
        and has_integer_indices
        and has_integer_reverse_indices
        and is_contiguous_bijection
    ):
        raise ValueError(
            f"Artifact {entity} mappings are not a contiguous bijection. "
            "Retrain the model."
        )


def _validate_numeric_artifacts(
    user_item_matrix: csr_matrix,
    content_matrix: csr_matrix,
    artist_factors: np.ndarray,
    user_factors: np.ndarray,
) -> None:
    if (
        not np.issubdtype(user_item_matrix.dtype, np.number)
        or not np.all(np.isfinite(user_item_matrix.data))
        or np.any(user_item_matrix.data <= 0)
    ):
        raise ValueError(
            "Artifact interaction weights must be finite and positive. "
            "Retrain the model."
        )
    if (
        not isinstance(content_matrix, csr_matrix)
        or not np.issubdtype(content_matrix.dtype, np.number)
        or not np.all(np.isfinite(content_matrix.data))
        or np.any(content_matrix.data < 0)
    ):
        raise ValueError(
            "Artifact content values must be finite and non-negative. "
            "Retrain the model."
        )
    if not np.all(np.isfinite(artist_factors)) or not np.all(np.isfinite(user_factors)):
        raise ValueError(
            "Artifact collaborative factors must contain finite values. "
            "Retrain the model."
        )


def _validate_content_artifacts(
    content_artifacts: ContentArtifacts,
    artist_ids: set[str],
    artist_id_to_name: dict[Any, Any],
) -> None:
    _validate_mapping_pair(
        content_artifacts.artist_id_to_content_index,
        content_artifacts.content_index_to_artist_id,
        entity="content artist",
    )
    if set(content_artifacts.artist_id_to_content_index) != artist_ids:
        raise ValueError(
            "Artifact content artists do not match its artist mappings. "
            "Retrain the model."
        )

    feature_names = content_artifacts.feature_names
    if (
        not isinstance(feature_names, list)
        or any(not isinstance(name, str) or not name for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
        or content_artifacts.content_matrix.shape
        != (len(artist_ids), len(feature_names))
    ):
        raise ValueError(
            "Artifact content features do not match its content matrix. "
            "Retrain the model."
        )
    vectorizer_feature_names = (
        content_artifacts.vectorizer.get_feature_names_out().tolist()
    )
    if feature_names != vectorizer_feature_names:
        raise ValueError(
            "Artifact content features do not match its vectorizer. Retrain the model."
        )

    metadata = content_artifacts.metadata
    required_columns = {
        "artist_id",
        "artist_name",
        "genres",
        "mood_tags",
        "country",
        "era",
    }
    if not isinstance(metadata, pd.DataFrame) or not required_columns <= set(
        metadata.columns
    ):
        raise ValueError(
            "Artifact content metadata has an invalid structure. Retrain the model."
        )
    if any(
        not isinstance(value, str) or not value.strip()
        for column in required_columns
        for value in metadata[column]
    ):
        raise ValueError(
            "Artifact content metadata contains empty values. Retrain the model."
        )

    ordered_artist_ids = [
        content_artifacts.content_index_to_artist_id[index]
        for index in range(len(artist_ids))
    ]
    metadata_artist_ids = metadata["artist_id"].tolist()
    if metadata_artist_ids != ordered_artist_ids or any(
        metadata.loc[index, "artist_name"] != artist_id_to_name[artist_id]
        for index, artist_id in enumerate(ordered_artist_ids)
    ):
        raise ValueError(
            "Artifact content metadata is not aligned with its artist mappings. "
            "Retrain the model."
        )

    metadata_lookup = content_artifacts.metadata_lookup
    if not isinstance(metadata_lookup, dict) or set(metadata_lookup) != artist_ids:
        raise ValueError(
            "Artifact content metadata lookup does not match its artists. "
            "Retrain the model."
        )
    for row in metadata.itertuples(index=False):
        artist_id = str(row.artist_id)
        lookup = metadata_lookup[artist_id]
        expected_values = {
            field: _split_metadata_values(getattr(row, field))
            for field in ("genres", "mood_tags", "country", "era")
        }
        expected_tokens = {
            value for values in expected_values.values() for value in values
        }
        if (
            not isinstance(lookup, dict)
            or lookup.get("artist_id") != artist_id
            or lookup.get("artist_name") != str(row.artist_name)
            or any(
                lookup.get(field) != values for field, values in expected_values.items()
            )
            or lookup.get("token_values") != expected_tokens
        ):
            raise ValueError(
                "Artifact content metadata lookup is inconsistent. Retrain the model."
            )


def _split_metadata_values(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(";") if item.strip()]
