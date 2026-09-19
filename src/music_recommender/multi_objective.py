"""Multi-objective candidate re-ranking and Pareto frontier analysis.

Balances relevance, intra-list diversity, and novelty/information gain using
scalarized greedy utility and identifies Pareto-optimal frontiers across
competing recommendation criteria.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_multi_objective_weights(
    relevance_weight: float = 1.0,
    diversity_weight: float = 0.0,
    novelty_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Validate and normalize non-negative multi-objective weights.

    Raises:
        ValueError: If any weight is negative, non-numeric, or not finite,
            or if all weights are zero.
    """
    weights = [relevance_weight, diversity_weight, novelty_weight]
    names = ["relevance_weight", "diversity_weight", "novelty_weight"]
    for val, name in zip(weights, names, strict=True):
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float, np.number))
            or not np.isfinite(val)
            or val < 0.0
        ):
            raise ValueError(f"{name} must be a non-negative finite number.")

    total = float(sum(weights))
    if total == 0.0:
        raise ValueError(
            "At least one multi-objective weight must be strictly positive."
        )

    return (
        float(relevance_weight / total),
        float(diversity_weight / total),
        float(novelty_weight / total),
    )


def rerank_multi_objective(
    candidate_indices: list[int],
    scores: np.ndarray,
    feature_matrix: np.ndarray | None,
    item_stats: dict[str, dict[str, Any]] | None,
    index_to_id: dict[int, str] | None,
    top_k: int,
    relevance_weight: float = 1.0,
    diversity_weight: float = 0.0,
    novelty_weight: float = 0.0,
) -> list[int]:
    """Re-rank candidate indices balancing relevance, diversity, and novelty.

    Parameters:
        candidate_indices: List of candidate integer indices.
        scores: 1D array of relevance scores corresponding to entity indices.
        feature_matrix: 2D array of entity feature vectors for diversity.
        item_stats: Popularity statistics with 'popularity_rank'.
        index_to_id: Mapping from candidate index to string item_id.
        top_k: Number of recommendations to return.
        relevance_weight: Weight for base candidate score.
        diversity_weight: Weight for marginal intra-list feature distance.
        novelty_weight: Weight for normalized popularity long-tail rank.

    Returns:
        List of selected candidate indices up to top_k.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if not candidate_indices:
        return []

    if len(candidate_indices) == 1 or top_k == 1:
        return candidate_indices[:top_k]

    w_rel, w_div, w_nov = validate_multi_objective_weights(
        relevance_weight=relevance_weight,
        diversity_weight=diversity_weight,
        novelty_weight=novelty_weight,
    )

    # Pure relevance shortcut: preserve base ranking
    if w_div == 0.0 and w_nov == 0.0:
        return candidate_indices[:top_k]

    # Normalize candidate relevance scores to [0, 1]
    candidate_scores = scores[candidate_indices]
    score_min = float(candidate_scores.min())
    score_range = float(candidate_scores.max() - score_min)
    if score_range == 0.0:
        norm_scores = dict.fromkeys(candidate_indices, 1.0)
    else:
        norm_scores = {
            idx: float((scores[idx] - score_min) / score_range)
            for idx in candidate_indices
        }

    # Precompute novelty scores in [0, 1]
    novelty_scores: dict[int, float] = {}
    if w_nov > 0.0 and item_stats and index_to_id:
        max_rank = max(len(item_stats), 1)
        for idx in candidate_indices:
            item_id = index_to_id.get(idx)
            if item_id and item_id in item_stats:
                rank = int(item_stats[item_id].get("popularity_rank", 1))
                novelty_scores[idx] = (
                    0.0 if max_rank == 1 else (rank - 1) / (max_rank - 1)
                )
            else:
                novelty_scores[idx] = 0.5
    else:
        novelty_scores = dict.fromkeys(candidate_indices, 0.0)

    # Precompute feature norms for diversity cosine distance
    has_features = (
        w_div > 0.0
        and feature_matrix is not None
        and feature_matrix.size > 0
        and feature_matrix.ndim == 2
    )
    norms = (
        np.linalg.norm(feature_matrix, axis=1)
        if has_features and feature_matrix is not None
        else np.empty(0)
    )

    selected: list[int] = []
    remaining = list(candidate_indices)

    while remaining and len(selected) < top_k:
        if not selected:
            # First item: diversity is maximal
            best_idx = max(
                remaining,
                key=lambda idx: (
                    w_rel * norm_scores[idx]
                    + w_div * 1.0
                    + w_nov * novelty_scores[idx]
                ),
            )
        else:
            def marginal_utility(candidate_idx: int) -> float:
                u_rel = norm_scores[candidate_idx]
                u_nov = novelty_scores[candidate_idx]
                if (
                    has_features
                    and feature_matrix is not None
                    and norms[candidate_idx] > 0
                ):
                    cand_vec = feature_matrix[candidate_idx]
                    cand_norm = float(norms[candidate_idx])
                    similarities = [
                        float(
                            (cand_vec @ feature_matrix[sel_idx])
                            / (cand_norm * float(norms[sel_idx]))
                        )
                        if float(norms[sel_idx]) > 0
                        else 0.0
                        for sel_idx in selected
                    ]
                    max_sim = max(similarities) if similarities else 0.0
                    max_sim = max(-1.0, min(1.0, max_sim))
                    u_div = max(0.0, 1.0 - max_sim)
                else:
                    u_div = 1.0

                return (w_rel * u_rel) + (w_div * u_div) + (w_nov * u_nov)

            best_idx = max(remaining, key=marginal_utility)

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def rerank_recommendations_multi_objective(
    recommendations: list[dict[str, Any]],
    *,
    feature_matrix: np.ndarray,
    item_id_to_index: dict[str, int],
    item_stats: dict[str, dict[str, Any]] | None,
    top_k: int,
    relevance_weight: float = 1.0,
    diversity_weight: float = 0.0,
    novelty_weight: float = 0.0,
    id_field: str = "track_id",
) -> list[dict[str, Any]]:
    """Re-rank recommendation dictionaries with multi-objective criteria."""
    if not recommendations or top_k <= 0:
        return []
    if len(recommendations) == 1 or top_k == 1:
        return recommendations[:top_k]

    indices: list[int] = []
    valid_recs: list[dict[str, Any]] = []
    scores_list: list[float] = []
    for rec in recommendations:
        item_id = str(rec.get(id_field, ""))
        if item_id in item_id_to_index:
            idx = item_id_to_index[item_id]
            indices.append(idx)
            valid_recs.append(rec)
            scores_list.append(float(rec.get("score", 0.0)))

    if not indices:
        return recommendations[:top_k]

    max_idx = max(indices) + 1
    scores_arr = np.zeros(max_idx, dtype=float)
    for idx, sc in zip(indices, scores_list, strict=True):
        scores_arr[idx] = sc

    index_to_id = {v: k for k, v in item_id_to_index.items()}
    selected_indices = rerank_multi_objective(
        candidate_indices=indices,
        scores=scores_arr,
        feature_matrix=feature_matrix,
        item_stats=item_stats,
        index_to_id=index_to_id,
        top_k=top_k,
        relevance_weight=relevance_weight,
        diversity_weight=diversity_weight,
        novelty_weight=novelty_weight,
    )

    rec_by_idx = dict(zip(indices, valid_recs, strict=True))
    result = [rec_by_idx[idx] for idx in selected_indices if idx in rec_by_idx]
    if len(result) < top_k and len(valid_recs) < len(recommendations):
        unmapped = [r for r in recommendations if r not in valid_recs]
        result.extend(unmapped[: top_k - len(result)])
    return result


def is_pareto_efficient(costs: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Find the Pareto-efficient points in a 2D matrix of shape (N, M).

    Parameters:
        costs: 2D array where each row is an evaluation configuration and
            each column represents an objective value.
        maximize: If True, higher objective values dominate lower ones.

    Returns:
        1D boolean mask of shape (N,) where True indicates Pareto-optimal rows.
    """
    if costs.ndim != 2 or costs.shape[0] == 0:
        return np.ones(costs.shape[0], dtype=bool)

    data = np.asarray(costs, dtype=float)
    if not maximize:
        data = -data

    num_points = data.shape[0]
    is_efficient = np.ones(num_points, dtype=bool)

    for i in range(num_points):
        if not is_efficient[i]:
            continue
        all_ge = np.all(data[i] >= data, axis=1)
        any_gt = np.any(data[i] > data, axis=1)
        dominated = all_ge & any_gt
        is_efficient[dominated] = False

    return is_efficient


def compute_pareto_frontier(
    points: list[dict[str, Any]],
    objectives: list[str],
    maximize: bool = True,
) -> list[dict[str, Any]]:
    """Annotate evaluation points with Pareto efficiency and return the frontier.

    Parameters:
        points: List of configuration records (dictionaries).
        objectives: List of dictionary keys defining the optimization criteria.
        maximize: If True, higher values are preferred for all objectives.

    Returns:
        List of dictionaries that lie on the non-dominated Pareto frontier.
        Each point in the input list also has 'is_pareto_optimal' set in place.
    """
    if not points:
        return []

    if not objectives:
        for p in points:
            p["is_pareto_optimal"] = True
        return list(points)

    matrix = np.array(
        [[float(p.get(obj, 0.0)) for obj in objectives] for p in points],
        dtype=float,
    )
    mask = is_pareto_efficient(matrix, maximize=maximize)

    frontier: list[dict[str, Any]] = []
    for point, is_optimal in zip(points, mask, strict=True):
        point["is_pareto_optimal"] = bool(is_optimal)
        if is_optimal:
            frontier.append(point)

    return frontier
