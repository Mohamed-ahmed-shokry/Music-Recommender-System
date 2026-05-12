"""Service layer for loading and serving recommender artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from music_recommender.artifacts import RecommenderArtifact, load_artifact
from music_recommender.baselines import popular_artists
from music_recommender.config import ARTIFACT_BUNDLE_PATH, DEFAULT_CONTENT_WEIGHT
from music_recommender.content import (
    build_content_recommendation,
    content_similar_artists,
    content_similarity_scores,
    hybrid_scores,
    profile_content_scores,
    recommend_from_scores,
    user_content_scores,
    validate_content_weight,
)
from music_recommender.ranking import (
    apply_popularity_penalty,
    rerank_with_diversity,
    validate_ranking_parameters,
)
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
            "hybrid_config": self.artifact.hybrid_config,
            "content": {
                "num_features": len(self.artifact.content_artifacts.feature_names),
                "feature_names": self.artifact.content_artifacts.feature_names,
            },
        }

    def health(self) -> dict[str, Any]:
        """Return lightweight service health details."""
        return {
            "status": "ok",
            "artifact_version": self.artifact.version,
            "num_users": self.artifact.metadata["num_users"],
            "num_artists": self.artifact.metadata["num_artists"],
            "num_interactions": self.artifact.metadata["num_interactions"],
            "content_features": len(self.artifact.content_artifacts.feature_names),
        }

    def recommend_user(
        self,
        user_id: str,
        top_k: int,
        include_listened: bool = False,
        popularity_penalty: float = 0.0,
        diversity: float = 0.0,
        content_weight: float | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Recommend artists for a user, with a popularity fallback if unknown."""
        content_weight = self._content_weight(content_weight)
        if user_id in self.artifact.mappings["user_id_to_index"]:
            collaborative_scores = self._collaborative_scores_for_user(user_id)
            content_scores, listened_artist_ids = user_content_scores(
                user_id=user_id,
                user_item_matrix=self.artifact.user_item_matrix,
                mappings=self.artifact.mappings,
                content_artifacts=self.artifact.content_artifacts,
            )
            blended_scores = hybrid_scores(
                collaborative_scores=collaborative_scores,
                content_scores=content_scores,
                content_weight=content_weight,
            )
            excluded_artist_ids = (
                set() if include_listened else set(listened_artist_ids)
            )
            recommendations = self._rank_content_scores(
                scores=blended_scores,
                top_k=top_k,
                exclude_artist_ids=excluded_artist_ids,
                reference_artist_ids=listened_artist_ids,
                explain=explain,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                score_components={
                    "collaborative_score": collaborative_scores,
                    "content_score": content_scores,
                    "hybrid_score": blended_scores,
                },
            )
            return {
                "user_id": user_id,
                "strategy": "hybrid_personalized",
                "content_weight": content_weight,
                "recommendations": recommendations,
            }

        recommendations = popular_artists(self.artifact.artist_stats, top_k=top_k)
        return {
            "user_id": user_id,
            "strategy": "popular_fallback",
            "message": f"Unknown user_id '{user_id}'. Returning popular artists.",
            "recommendations": recommendations,
        }

    def recommend_user_als(
        self,
        user_id: str,
        top_k: int,
        include_listened: bool = False,
        popularity_penalty: float = 0.0,
        diversity: float = 0.0,
    ) -> dict[str, Any]:
        """Return the v2-style ALS-only recommendation response."""
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

    def recommend_profile(
        self,
        artist_ids: list[str] | None = None,
        genres: list[str] | None = None,
        mood_tags: list[str] | None = None,
        top_k: int = 10,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Recommend artists from onboarding preferences."""
        scores, selected_artist_ids, preference_tokens = profile_content_scores(
            content_artifacts=self.artifact.content_artifacts,
            artist_ids=artist_ids,
            genres=genres,
            mood_tags=mood_tags,
        )
        recommendations = recommend_from_scores(
            scores=scores,
            content_artifacts=self.artifact.content_artifacts,
            artist_stats=self.artifact.artist_stats,
            top_k=top_k,
            exclude_artist_ids=set(selected_artist_ids),
            reference_artist_ids=selected_artist_ids,
            preference_tokens=preference_tokens,
            explain=explain,
        )
        for recommendation in recommendations:
            artist_index = self.artifact.content_artifacts.artist_id_to_content_index[
                str(recommendation["artist_id"])
            ]
            recommendation["score_components"] = {
                "content_score": float(scores[artist_index]),
                "hybrid_score": float(scores[artist_index]),
            }

        return {
            "strategy": "content_profile",
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
        method: str = "als",
        content_weight: float | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Find artists similar to a selected artist."""
        method = method.lower()
        if method == "als":
            recommendations = get_similar_artists(
                model=self.artifact.model,
                artist_id=artist_id,
                mappings=self.artifact.mappings,
                top_k=top_k,
                artist_stats=self.artifact.artist_stats,
            )
            strategy = "als_similarity"
        elif method == "content":
            recommendations = content_similar_artists(
                artist_id=artist_id,
                content_artifacts=self.artifact.content_artifacts,
                artist_stats=self.artifact.artist_stats,
                top_k=top_k,
                explain=explain,
            )
            strategy = "content_similarity"
        elif method == "hybrid":
            content_weight = self._content_weight(content_weight)
            collaborative_scores = self._collaborative_similarity_scores(artist_id)
            content_scores = content_similarity_scores(
                artist_id=artist_id,
                content_artifacts=self.artifact.content_artifacts,
            )
            blended_scores = hybrid_scores(
                collaborative_scores=collaborative_scores,
                content_scores=content_scores,
                content_weight=content_weight,
            )
            recommendations = self._rank_content_scores(
                scores=blended_scores,
                top_k=top_k,
                exclude_artist_ids={artist_id},
                reference_artist_ids=[artist_id],
                explain=explain,
                score_components={
                    "collaborative_score": collaborative_scores,
                    "content_score": content_scores,
                    "hybrid_score": blended_scores,
                },
            )
            strategy = "hybrid_similarity"
        else:
            raise ValueError("method must be one of: als, content, hybrid.")

        return {
            "artist_id": artist_id,
            "strategy": strategy,
            "similar_artists": recommendations,
        }

    def content_similar_artists(
        self,
        artist_id: str,
        top_k: int,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Find artists similar to a selected artist by metadata only."""
        recommendations = content_similar_artists(
            artist_id=artist_id,
            content_artifacts=self.artifact.content_artifacts,
            artist_stats=self.artifact.artist_stats,
            top_k=top_k,
            explain=explain,
        )
        return {
            "artist_id": artist_id,
            "strategy": "content_similarity",
            "similar_artists": recommendations,
        }

    def _content_weight(self, content_weight: float | None) -> float:
        if content_weight is None:
            content_weight = float(
                self.artifact.hybrid_config.get(
                    "default_content_weight",
                    DEFAULT_CONTENT_WEIGHT,
                )
            )
        validate_content_weight(content_weight)
        return content_weight

    def _collaborative_scores_for_user(self, user_id: str) -> np.ndarray:
        user_index = self.artifact.mappings["user_id_to_index"][user_id]
        user_factors = self.artifact.model.item_factors
        artist_factors = self.artifact.model.user_factors
        return artist_factors @ user_factors[user_index]

    def _collaborative_similarity_scores(self, artist_id: str) -> np.ndarray:
        artist_id_to_index = self.artifact.mappings["artist_id_to_index"]
        if artist_id not in artist_id_to_index:
            raise ValueError(f"Unknown artist_id: {artist_id}")

        artist_index = artist_id_to_index[artist_id]
        artist_factors = self.artifact.model.user_factors
        query_vector = artist_factors[artist_index]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(artist_factors.shape[0])

        norms = np.linalg.norm(artist_factors, axis=1)
        denominator = norms * query_norm
        return np.divide(
            artist_factors @ query_vector,
            denominator,
            out=np.zeros_like(norms),
            where=denominator != 0,
        )

    def _rank_content_scores(
        self,
        scores: np.ndarray,
        top_k: int,
        exclude_artist_ids: set[str] | None = None,
        reference_artist_ids: list[str] | None = None,
        preference_tokens: set[str] | None = None,
        explain: bool = False,
        popularity_penalty: float = 0.0,
        diversity: float = 0.0,
        score_components: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        validate_ranking_parameters(top_k, diversity, popularity_penalty)
        index_to_artist_id = self.artifact.content_artifacts.content_index_to_artist_id
        adjusted_scores = apply_popularity_penalty(
            scores=scores,
            index_to_artist_id=index_to_artist_id,
            artist_stats=self.artifact.artist_stats,
            popularity_penalty=popularity_penalty,
        )
        excluded = exclude_artist_ids or set()
        ranked_indices = np.argsort(adjusted_scores)[::-1]
        candidate_indices = [
            int(index)
            for index in ranked_indices
            if index_to_artist_id[int(index)] not in excluded
        ]
        final_indices = rerank_with_diversity(
            candidate_indices=candidate_indices,
            scores=adjusted_scores,
            artist_factors=self.artifact.content_artifacts.content_matrix.toarray(),
            top_k=top_k,
            diversity=diversity,
        )

        recommendations: list[dict[str, Any]] = []
        for artist_index in final_indices:
            artist_id = index_to_artist_id[int(artist_index)]
            recommendation = build_content_recommendation(
                artist_id=artist_id,
                score=float(adjusted_scores[artist_index]),
                content_artifacts=self.artifact.content_artifacts,
                artist_stats=self.artifact.artist_stats,
                reference_artist_ids=reference_artist_ids,
                preference_tokens=preference_tokens,
                explain=explain,
            )
            if score_components:
                recommendation["score_components"] = {
                    name: float(values[artist_index])
                    for name, values in score_components.items()
                }
                if popularity_penalty:
                    recommendation["score_components"]["adjusted_score"] = float(
                        adjusted_scores[artist_index]
                    )
            recommendations.append(recommendation)

        return recommendations
