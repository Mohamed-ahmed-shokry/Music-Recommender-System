"""Unit tests for multi-objective candidate re-ranking and Pareto frontiers."""

from __future__ import annotations

import numpy as np
import pytest

from music_recommender.multi_objective import (
    compute_pareto_frontier,
    is_pareto_efficient,
    rerank_multi_objective,
    rerank_recommendations_multi_objective,
    validate_multi_objective_weights,
)


def test_validate_multi_objective_weights_normalizes_sum() -> None:
    w1, w2, w3 = validate_multi_objective_weights(2.0, 1.0, 1.0)
    assert pytest.approx(w1) == 0.5
    assert pytest.approx(w2) == 0.25
    assert pytest.approx(w3) == 0.25


@pytest.mark.parametrize(
    ("w_rel", "w_div", "w_nov", "match"),
    [
        (-0.1, 0.5, 0.5, "relevance_weight must be a non-negative finite number"),
        (0.5, -0.1, 0.5, "diversity_weight must be a non-negative finite number"),
        (0.5, 0.5, -0.1, "novelty_weight must be a non-negative finite number"),
        (0.0, 0.0, 0.0, "At least one multi-objective weight must be"),
        ("0.5", 0.5, 0.5, "relevance_weight must be a non-negative finite number"),
        (True, 0.5, 0.5, "relevance_weight must be a non-negative finite number"),
    ],
)
def test_validate_multi_objective_weights_rejects_invalid(
    w_rel: object,
    w_div: object,
    w_nov: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_multi_objective_weights(w_rel, w_div, w_nov)  # type: ignore[arg-type]


def test_rerank_multi_objective_pure_relevance_preserves_order() -> None:
    candidates = [0, 1, 2]
    scores = np.array([0.9, 0.8, 0.7])
    features = np.eye(3)
    stats = {
        "item_0": {"popularity_rank": 1},
        "item_1": {"popularity_rank": 2},
        "item_2": {"popularity_rank": 3},
    }
    index_to_id = {0: "item_0", 1: "item_1", 2: "item_2"}

    result = rerank_multi_objective(
        candidate_indices=candidates,
        scores=scores,
        feature_matrix=features,
        item_stats=stats,
        index_to_id=index_to_id,
        top_k=3,
        relevance_weight=1.0,
        diversity_weight=0.0,
        novelty_weight=0.0,
    )
    assert result == [0, 1, 2]


def test_rerank_multi_objective_diversity_promotes_dissimilar_candidate() -> None:
    # Item 0 and 1 are almost identical vectors [1, 0]
    # Item 2 is orthogonal [0, 1]
    candidates = [0, 1, 2]
    scores = np.array([1.0, 0.95, 0.80])
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.01],
            [0.0, 1.0],
        ]
    )
    index_to_id = {0: "t0", 1: "t1", 2: "t2"}

    # With high diversity weight, item 2 should be chosen 2nd ahead of item 1
    result = rerank_multi_objective(
        candidate_indices=candidates,
        scores=scores,
        feature_matrix=features,
        item_stats=None,
        index_to_id=index_to_id,
        top_k=2,
        relevance_weight=0.2,
        diversity_weight=0.8,
        novelty_weight=0.0,
    )
    assert result == [0, 2]


def test_rerank_multi_objective_novelty_promotes_long_tail() -> None:
    candidates = [0, 1, 2]
    scores = np.array([0.9, 0.85, 0.80])
    # Item 0 is rank 1 (mainstream), item 2 is rank 100 (rare discovery)
    stats = {
        "item_0": {"popularity_rank": 1},
        "item_1": {"popularity_rank": 10},
        "item_2": {"popularity_rank": 100},
    }
    index_to_id = {0: "item_0", 1: "item_1", 2: "item_2"}

    result = rerank_multi_objective(
        candidate_indices=candidates,
        scores=scores,
        feature_matrix=np.empty((0, 0)),
        item_stats=stats,
        index_to_id=index_to_id,
        top_k=3,
        relevance_weight=0.1,
        diversity_weight=0.0,
        novelty_weight=0.9,
    )
    # The rare item 2 should be promoted first due to high novelty weight
    assert result[0] == 2


def test_rerank_multi_objective_edge_cases() -> None:
    assert rerank_multi_objective([], np.array([]), None, None, None, top_k=5) == []
    assert (
        rerank_multi_objective(
            [3], np.array([0, 0, 0, 1.0]), None, None, None, top_k=5
        )
        == [3]
    )
    assert (
        rerank_multi_objective(
            [1, 2], np.array([0, 0.5, 0.8]), None, None, None, top_k=1
        )
        == [1]
    )
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rerank_multi_objective([1], np.array([1.0]), None, None, None, top_k=0)


def test_rerank_recommendations_multi_objective() -> None:
    recs = [
        {"track_id": "t1", "score": 0.95},
        {"track_id": "t2", "score": 0.90},
        {"track_id": "t3", "score": 0.85},
    ]
    feature_matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    item_id_to_index = {"t1": 0, "t2": 1, "t3": 2}
    stats = {
        "t1": {"popularity_rank": 1},
        "t2": {"popularity_rank": 2},
        "t3": {"popularity_rank": 50},
    }

    reranked = rerank_recommendations_multi_objective(
        recs,
        feature_matrix=feature_matrix,
        item_id_to_index=item_id_to_index,
        item_stats=stats,
        top_k=2,
        relevance_weight=0.3,
        diversity_weight=0.7,
        novelty_weight=0.0,
        id_field="track_id",
    )
    assert len(reranked) == 2
    assert reranked[0]["track_id"] == "t1"
    assert reranked[1]["track_id"] == "t3"


def test_is_pareto_efficient_2d() -> None:
    # 4 points with 2 objectives (both maximized):
    # P0: [1.0, 1.0] (Dominates P1, P2)
    # P1: [0.5, 0.5] (Dominated by P0)
    # P2: [0.8, 0.2] (Dominated by P0)
    # P3: [0.2, 1.5] (Incomparable with P0: higher on obj 2, lower on obj 1)
    costs = np.array(
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.2, 1.5],
        ]
    )
    efficient = is_pareto_efficient(costs, maximize=True)
    assert efficient.tolist() == [True, False, False, True]


def test_is_pareto_efficient_all_dominated_except_one() -> None:
    costs = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
    efficient = is_pareto_efficient(costs, maximize=True)
    assert efficient.tolist() == [False, True, False]


def test_is_pareto_efficient_empty_and_1d() -> None:
    assert len(is_pareto_efficient(np.empty((0, 2)))) == 0
    assert len(is_pareto_efficient(np.array([1.0, 2.0]))) == 2


def test_compute_pareto_frontier() -> None:
    points = [
        {"id": "A", "accuracy": 0.90, "diversity": 0.20, "novelty": 0.10},
        {"id": "B", "accuracy": 0.85, "diversity": 0.70, "novelty": 0.30},
        {"id": "C", "accuracy": 0.60, "diversity": 0.80, "novelty": 0.90},
        # Dominated by B:
        {"id": "D", "accuracy": 0.50, "diversity": 0.40, "novelty": 0.10},
    ]
    objectives = ["accuracy", "diversity", "novelty"]

    frontier = compute_pareto_frontier(points, objectives)
    frontier_ids = [p["id"] for p in frontier]
    assert frontier_ids == ["A", "B", "C"]
    assert points[3]["is_pareto_optimal"] is False
    assert points[0]["is_pareto_optimal"] is True


def test_compute_pareto_frontier_empty_inputs() -> None:
    assert compute_pareto_frontier([], ["acc"]) == []
    single = [{"acc": 0.5}]
    result = compute_pareto_frontier(single, [])
    assert result[0]["is_pareto_optimal"] is True
