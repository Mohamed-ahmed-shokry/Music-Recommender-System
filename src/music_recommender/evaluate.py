"""Evaluation helpers for recommendation quality."""

import math

import numpy as np
import pandas as pd

from music_recommender.config import (
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
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


def evaluate_model(df: pd.DataFrame, top_k: int) -> dict[str, float]:
    """Train and evaluate ALS using a simple per-user holdout split."""
    train_df, test_df = train_test_split_by_user(df)
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
    )

    all_recommended_items: list[list[str]] = []
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
        )
        recommended_items = [
            str(recommendation["artist_id"]) for recommendation in recommendations
        ]
        all_recommended_items.append(recommended_items)
        all_relevant_items.append(relevant_items)

    if not all_recommended_items:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    return {
        "precision_at_k": float(
            np.mean(
                [
                    precision_at_k(recommended_items, relevant_items, top_k)
                    for recommended_items, relevant_items in zip(
                        all_recommended_items,
                        all_relevant_items,
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
                        all_recommended_items,
                        all_relevant_items,
                        strict=False,
                    )
                ]
            )
        ),
        "map_at_k": map_at_k(all_recommended_items, all_relevant_items, top_k),
        "ndcg_at_k": float(
            np.mean(
                [
                    ndcg_at_k(recommended_items, relevant_items, top_k)
                    for recommended_items, relevant_items in zip(
                        all_recommended_items,
                        all_relevant_items,
                        strict=False,
                    )
                ]
            )
        ),
    }
