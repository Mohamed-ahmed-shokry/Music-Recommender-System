import numpy as np
import pytest

from music_recommender.ranking import (
    apply_popularity_penalty,
    rerank_with_diversity,
    validate_ranking_parameters,
)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_raises_value_error(top_k) -> None:
    with pytest.raises(ValueError, match="top_k"):
        validate_ranking_parameters(top_k=top_k)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("diversity", -0.1),
        ("diversity", 1.2),
        ("diversity", float("nan")),
        ("diversity", True),
        ("popularity_penalty", -0.1),
        ("popularity_penalty", float("inf")),
        ("popularity_penalty", False),
    ],
)
def test_invalid_unit_interval_ranking_parameter_raises_value_error(
    parameter: str,
    value,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        validate_ranking_parameters(top_k=1, **{parameter: value})


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
