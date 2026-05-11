"""Evaluation helpers for recommendation quality."""

import math

import numpy as np
import pandas as pd

from music_recommender.artifacts import build_artist_stats
from music_recommender.baselines import popular_artists
from music_recommender.config import (
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
    DEFAULT_USE_GPU,
)
from music_recommender.model import train_als_model
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings
from music_recommender.recommend import recommend_artists_for_user


def precision_at_k(
    recommended_items: list[str],
    relevant_items: set[str] | list[str],
    k: int,
) -> float:
    """Calculate Precision@K."""
    if k <= 0:
        return 0.0

    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    hits = sum(item in relevant_set for item in recommended_at_k)
    return hits / k


def recall_at_k(
    recommended_items: list[str],
    relevant_items: set[str] | list[str],
    k: int,
) -> float:
    """Calculate Recall@K."""
    relevant_set = set(relevant_items)
    if k <= 0 or not relevant_set:
        return 0.0

    recommended_at_k = recommended_items[:k]
    hits = sum(item in relevant_set for item in recommended_at_k)
    return hits / len(relevant_set)


def average_precision_at_k(
    recommended_items: list[str],
    relevant_items: set[str] | list[str],
    k: int,
) -> float:
    """Calculate Average Precision@K."""
    relevant_set = set(relevant_items)
    if k <= 0 or not relevant_set:
        return 0.0

    score = 0.0
    hits = 0
    for index, item in enumerate(recommended_items[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / index

    return score / min(len(relevant_set), k)


def map_at_k(
    list_of_recommended_items: list[list[str]],
    list_of_relevant_items: list[set[str] | list[str]],
    k: int,
) -> float:
    """Calculate Mean Average Precision@K."""
    if not list_of_recommended_items:
        return 0.0

    scores = [
        average_precision_at_k(recommended_items, relevant_items, k)
        for recommended_items, relevant_items in zip(
            list_of_recommended_items,
            list_of_relevant_items,
            strict=False,
        )
    ]
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    recommended_items: list[str],
    relevant_items: set[str] | list[str],
    k: int,
) -> float:
    """Calculate normalized discounted cumulative gain at K."""
    relevant_set = set(relevant_items)
    if k <= 0 or not relevant_set:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended_items[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(
    list_of_recommended_items: list[list[str]],
    catalog_items: set[str],
) -> float:
    """Calculate the share of catalog items that appear in recommendations."""
    if not catalog_items:
        return 0.0

    recommended_catalog_items = {
        item
        for recommended_items in list_of_recommended_items
        for item in recommended_items
    }
    return len(recommended_catalog_items & catalog_items) / len(catalog_items)


def average_popularity(
    list_of_recommended_items: list[list[str]],
    artist_stats: dict[str, dict[str, object]],
) -> float:
    """Calculate average total plays for recommended artists."""
    popularity_values = [
        float(artist_stats[item]["total_plays"])
        for recommended_items in list_of_recommended_items
        for item in recommended_items
        if item in artist_stats
    ]
    return float(np.mean(popularity_values)) if popularity_values else 0.0


def intra_list_diversity(
    recommended_items: list[str],
    artist_factors: np.ndarray,
    artist_id_to_index: dict[str, int],
) -> float:
    """Calculate average pairwise dissimilarity inside one recommendation list."""
    indices = [
        artist_id_to_index[item]
        for item in recommended_items
        if item in artist_id_to_index
    ]
    if len(indices) < 2:
        return 0.0

    diversities: list[float] = []
    norms = np.linalg.norm(artist_factors, axis=1)
    for left_position, left_index in enumerate(indices):
        for right_index in indices[left_position + 1 :]:
            denominator = norms[left_index] * norms[right_index]
            similarity = 0.0
            if denominator != 0:
                similarity = float(
                    (artist_factors[left_index] @ artist_factors[right_index])
                    / denominator
                )
            similarity = float(np.clip(similarity, -1.0, 1.0))
            diversities.append(1 - similarity)

    return float(np.mean(diversities)) if diversities else 0.0


def train_test_split_by_user(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions per user while keeping train interactions when possible."""
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, user_df in df.groupby("user_id"):
        indices = user_df.index.to_numpy()
        if len(indices) <= 1:
            train_indices.extend(indices)
            continue

        shuffled_indices = rng.permutation(indices)
        test_count = max(1, int(round(len(indices) * test_ratio)))
        test_count = min(test_count, len(indices) - 1)

        test_indices.extend(shuffled_indices[:test_count])
        train_indices.extend(shuffled_indices[test_count:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def evaluate_model(
    df: pd.DataFrame,
    top_k: int,
    use_gpu: bool = DEFAULT_USE_GPU,
) -> dict[str, float]:
    """Train and evaluate ALS using a simple per-user holdout split."""
    return evaluate_repeated_holdout(
        df=df,
        top_k=top_k,
        folds=1,
        use_gpu=use_gpu,
        compare_baseline=False,
    )


def evaluate_repeated_holdout(
    df: pd.DataFrame,
    top_k: int,
    folds: int = 1,
    use_gpu: bool = DEFAULT_USE_GPU,
    compare_baseline: bool = False,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Evaluate ALS and optionally compare it with a popularity baseline."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if folds <= 0:
        raise ValueError("folds must be greater than 0.")

    als_fold_metrics: list[dict[str, float]] = []
    popularity_fold_metrics: list[dict[str, float]] = []

    for fold in range(folds):
        fold_result = _evaluate_single_fold(
            df=df,
            top_k=top_k,
            random_state=42 + fold,
            use_gpu=use_gpu,
            compare_baseline=compare_baseline,
        )
        als_fold_metrics.append(fold_result["als"])
        if compare_baseline:
            popularity_fold_metrics.append(fold_result["popularity"])

    als_metrics = _average_metric_dicts(als_fold_metrics)
    if not compare_baseline:
        return als_metrics

    return {
        "als": als_metrics,
        "popularity": _average_metric_dicts(popularity_fold_metrics),
    }


def _evaluate_single_fold(
    df: pd.DataFrame,
    top_k: int,
    random_state: int,
    use_gpu: bool,
    compare_baseline: bool,
) -> dict[str, dict[str, float]]:
    train_df, test_df = train_test_split_by_user(df, random_state=random_state)
    mappings = create_id_mappings(train_df)
    user_item_matrix = build_user_item_matrix(
        train_df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(
        user_item_matrix=user_item_matrix,
        factors=DEFAULT_ALS_FACTORS,
        regularization=DEFAULT_ALS_REGULARIZATION,
        iterations=DEFAULT_ALS_ITERATIONS,
        alpha=DEFAULT_ALS_ALPHA,
        use_gpu=use_gpu,
    )
    artist_stats = build_artist_stats(train_df)

    all_recommended_items: list[list[str]] = []
    all_popularity_items: list[list[str]] = []
    all_relevant_items: list[set[str]] = []

    known_artists = set(mappings["artist_id_to_index"])
    for user_id, user_test_df in test_df.groupby("user_id"):
        if user_id not in mappings["user_id_to_index"]:
            continue

        relevant_items = set(user_test_df["artist_id"]) & known_artists
        if not relevant_items:
            continue

        recommendations = recommend_artists_for_user(
            model=model,
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            mappings=mappings,
            top_k=top_k,
            artist_stats=artist_stats,
        )
        recommended_items = [
            str(recommendation["artist_id"]) for recommendation in recommendations
        ]
        all_recommended_items.append(recommended_items)
        all_relevant_items.append(relevant_items)

        if compare_baseline:
            train_artist_ids = set(
                train_df.loc[train_df["user_id"] == user_id, "artist_id"]
            )
            baseline_recommendations = popular_artists(
                artist_stats,
                top_k=top_k,
                exclude_artist_ids=train_artist_ids,
            )
            all_popularity_items.append(
                [
                    str(recommendation["artist_id"])
                    for recommendation in baseline_recommendations
                ]
            )

    result = {
        "als": _summarize_recommendations(
            all_recommended_items,
            all_relevant_items,
            known_artists,
            artist_stats,
            model.user_factors,
            mappings["artist_id_to_index"],
            top_k,
        )
    }
    if compare_baseline:
        result["popularity"] = _summarize_recommendations(
            all_popularity_items,
            all_relevant_items,
            known_artists,
            artist_stats,
            model.user_factors,
            mappings["artist_id_to_index"],
            top_k,
        )
    return result


def _summarize_recommendations(
    list_of_recommended_items: list[list[str]],
    list_of_relevant_items: list[set[str]],
    catalog_items: set[str],
    artist_stats: dict[str, dict[str, object]],
    artist_factors: np.ndarray,
    artist_id_to_index: dict[str, int],
    top_k: int,
) -> dict[str, float]:
    if not list_of_recommended_items:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "catalog_coverage": 0.0,
            "average_popularity": 0.0,
            "intra_list_diversity": 0.0,
        }

    return {
        "precision_at_k": float(
            np.mean(
                [
                    precision_at_k(recommended_items, relevant_items, top_k)
                    for recommended_items, relevant_items in zip(
                        list_of_recommended_items,
                        list_of_relevant_items,
                        strict=False,
                    )
                ]
            )
        ),
        "recall_at_k": float(
            np.mean(
                [
                    recall_at_k(recommended_items, relevant_items, top_k)
                    for recommended_items, relevant_items in zip(
                        list_of_recommended_items,
                        list_of_relevant_items,
                        strict=False,
                    )
                ]
            )
        ),
        "map_at_k": map_at_k(
            list_of_recommended_items,
            list_of_relevant_items,
            top_k,
        ),
        "ndcg_at_k": float(
            np.mean(
                [
                    ndcg_at_k(recommended_items, relevant_items, top_k)
                    for recommended_items, relevant_items in zip(
                        list_of_recommended_items,
                        list_of_relevant_items,
                        strict=False,
                    )
                ]
            )
        ),
        "catalog_coverage": catalog_coverage(list_of_recommended_items, catalog_items),
        "average_popularity": average_popularity(
            list_of_recommended_items,
            artist_stats,
        ),
        "intra_list_diversity": float(
            np.mean(
                [
                    intra_list_diversity(
                        recommended_items,
                        artist_factors,
                        artist_id_to_index,
                    )
                    for recommended_items in list_of_recommended_items
                ]
            )
        ),
    }


def _average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}

    metric_names = metric_dicts[0].keys()
    return {
        metric_name: float(np.mean([metrics[metric_name] for metrics in metric_dicts]))
        for metric_name in metric_names
    }
