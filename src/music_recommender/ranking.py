"""Ranking controls for recommendation post-processing."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_ranking_parameters(
    top_k: int,
    diversity: float = 0.0,
    popularity_penalty: float = 0.0,
) -> None:
    """Validate common recommendation ranking parameters."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if not 0 <= diversity <= 1:
        raise ValueError("diversity must be between 0 and 1.")
    if not 0 <= popularity_penalty <= 1:
        raise ValueError("popularity_penalty must be between 0 and 1.")


def apply_popularity_penalty(
    scores: np.ndarray,
    index_to_artist_id: dict[int, str],
    artist_stats: dict[str, dict[str, Any]] | None,
    popularity_penalty: float,
) -> np.ndarray:
    """Reduce scores for globally popular artists by a configurable amount."""
    if popularity_penalty == 0 or not artist_stats:
        return scores.copy()

    adjusted_scores = scores.astype(float).copy()
    max_rank = max(len(artist_stats), 1)
    score_scale = float(np.ptp(scores))
    if score_scale == 0:
        score_scale = max(float(np.max(np.abs(scores))), 1.0)

    for artist_index, artist_id in index_to_artist_id.items():
        stats = artist_stats.get(artist_id)
        if stats is None:
            continue

        rank = int(stats["popularity_rank"])
        if max_rank == 1:
            popularity_weight = 1.0
        else:
            popularity_weight = 1 - ((rank - 1) / (max_rank - 1))
        adjusted_scores[artist_index] -= (
            popularity_penalty * popularity_weight * score_scale
        )

    return adjusted_scores


def rerank_with_diversity(
    candidate_indices: list[int],
    scores: np.ndarray,
    artist_factors: np.ndarray,
    top_k: int,
    diversity: float,
) -> list[int]:
    """Apply a simple MMR-style diversity reranking pass."""
    if diversity == 0 or len(candidate_indices) <= 1:
        return candidate_indices[:top_k]

    candidate_scores = scores[candidate_indices]
    score_min = float(candidate_scores.min())
    score_range = float(candidate_scores.max() - score_min)
    if score_range == 0:
        normalized_scores = {artist_index: 1.0 for artist_index in candidate_indices}
    else:
        normalized_scores = {
            artist_index: float((scores[artist_index] - score_min) / score_range)
            for artist_index in candidate_indices
        }

    selected: list[int] = []
    remaining = list(candidate_indices)
    norms = np.linalg.norm(artist_factors, axis=1)

    while remaining and len(selected) < top_k:
        best_index = max(
            remaining,
            key=lambda artist_index: _mmr_score(
                artist_index=artist_index,
                selected_indices=selected,
                normalized_scores=normalized_scores,
                artist_factors=artist_factors,
                norms=norms,
                diversity=diversity,
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)

    return selected


def _mmr_score(
    artist_index: int,
    selected_indices: list[int],
    normalized_scores: dict[int, float],
    artist_factors: np.ndarray,
    norms: np.ndarray,
    diversity: float,
) -> float:
    relevance = normalized_scores[artist_index]
    if not selected_indices:
        return relevance

    max_similarity = max(
        _cosine_similarity(
            artist_factors[artist_index],
            artist_factors[selected_index],
            float(norms[artist_index]),
            float(norms[selected_index]),
        )
        for selected_index in selected_indices
    )
    return ((1 - diversity) * relevance) - (diversity * max_similarity)


def _cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
    left_norm: float,
    right_norm: float,
) -> float:
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return float((left @ right) / denominator)
