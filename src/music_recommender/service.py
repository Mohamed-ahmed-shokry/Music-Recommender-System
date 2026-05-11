"""Service layer for loading and serving recommender artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from music_recommender.artifacts import RecommenderArtifact, load_artifact
from music_recommender.baselines import popular_artists
from music_recommender.config import ARTIFACT_BUNDLE_PATH
from music_recommender.recommend import (
    get_similar_artists,
    recommend_artists_for_user,
)


class RecommenderService:
    """Thin serving layer around a loaded recommender artifact."""

    def __init__(self, artifact: RecommenderArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def from_artifacts(
        cls,
        artifact_path: str | Path = ARTIFACT_BUNDLE_PATH,
    ) -> RecommenderService:
        """Load a service from a saved artifact bundle."""
        return cls(load_artifact(artifact_path))

    def metadata(self) -> dict[str, Any]:
        """Return artifact and training metadata."""
        return {
            "version": self.artifact.version,
            "metadata": self.artifact.metadata,
            "training_config": self.artifact.training_config,
        }

    def health(self) -> dict[str, Any]:
        """Return lightweight service health details."""
        return {
            "status": "ok",
            "artifact_version": self.artifact.version,
            "num_users": self.artifact.metadata["num_users"],
            "num_artists": self.artifact.metadata["num_artists"],
            "num_interactions": self.artifact.metadata["num_interactions"],
        }

    def recommend_user(
        self,
        user_id: str,
        top_k: int,
        include_listened: bool = False,
        popularity_penalty: float = 0.0,
        diversity: float = 0.0,
    ) -> dict[str, Any]:
        """Recommend artists for a user, with a popularity fallback if unknown."""
        if user_id in self.artifact.mappings["user_id_to_index"]:
            recommendations = recommend_artists_for_user(
                model=self.artifact.model,
                user_id=user_id,
                user_item_matrix=self.artifact.user_item_matrix,
                mappings=self.artifact.mappings,
                top_k=top_k,
                include_listened=include_listened,
                artist_stats=self.artifact.artist_stats,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
            )
            return {
                "user_id": user_id,
                "strategy": "als_personalized",
                "recommendations": recommendations,
            }

        recommendations = popular_artists(self.artifact.artist_stats, top_k=top_k)
        return {
            "user_id": user_id,
            "strategy": "popular_fallback",
            "message": f"Unknown user_id '{user_id}'. Returning popular artists.",
            "recommendations": recommendations,
        }

    def popular_artists(self, top_k: int) -> dict[str, Any]:
        """Return globally popular artists from the training data."""
        return {
            "strategy": "popular_baseline",
            "recommendations": popular_artists(self.artifact.artist_stats, top_k=top_k),
        }

    def similar_artists(
        self,
        artist_id: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Find artists similar to a selected artist."""
        recommendations = get_similar_artists(
            model=self.artifact.model,
            artist_id=artist_id,
            mappings=self.artifact.mappings,
            top_k=top_k,
            artist_stats=self.artifact.artist_stats,
        )
        return {
            "artist_id": artist_id,
            "similar_artists": recommendations,
        }
