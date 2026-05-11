import numpy as np
import pytest

from music_recommender.ranking import (
    apply_popularity_penalty,
    rerank_with_diversity,
    validate_ranking_parameters,
)


def test_invalid_ranking_parameters_raise_value_error() -> None:
    with pytest.raises(ValueError, match="top_k"):
        validate_ranking_parameters(top_k=0)
    with pytest.raises(ValueError, match="diversity"):
        validate_ranking_parameters(top_k=1, diversity=1.2)
    with pytest.raises(ValueError, match="popularity_penalty"):
        validate_ranking_parameters(top_k=1, popularity_penalty=-0.1)


def test_popularity_penalty_reduces_popular_artist_score() -> None:
    scores = np.array([1.0, 0.8])
    index_to_artist_id = {0: "popular", 1: "niche"}
    artist_stats = {
        "popular": {"popularity_rank": 1},
        "niche": {"popularity_rank": 2},
    }

    adjusted = apply_popularity_penalty(
        scores,
        index_to_artist_id,
        artist_stats,
        popularity_penalty=1.0,
    )

    assert adjusted[0] < adjusted[1]


def test_diversity_reranking_reduces_near_duplicates() -> None:
    candidate_indices = [0, 1, 2]
    scores = np.array([1.0, 0.99, 0.5])
    artist_factors = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )

    reranked = rerank_with_diversity(
        candidate_indices,
        scores,
        artist_factors,
        top_k=2,
        diversity=1.0,
    )

    assert reranked == [0, 2]
