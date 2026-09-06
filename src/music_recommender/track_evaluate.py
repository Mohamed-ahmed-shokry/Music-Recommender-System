"""Holdout evaluation for track similarity recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from music_recommender.evaluate import (
    average_popularity,
    catalog_coverage,
    map_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
)
from music_recommender.ranking import validate_ranking_parameters
from music_recommender.tracks import (
    build_track_serving_resources,
    normalize_track_interactions,
    recommend_tracks_for_user,
)


def train_test_split_tracks_by_user(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split track interactions per user, holding out tracks when possible."""
    if (
        isinstance(test_ratio, bool)
        or not isinstance(test_ratio, (int, float, np.number))
        or not np.isfinite(test_ratio)
        or not 0 < test_ratio < 1
    ):
        raise ValueError("test_ratio must be between 0 and 1.")
    if type(random_state) is not int:
        raise ValueError("random_state must be an integer.")
    df = normalize_track_interactions(df)
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, user_df in df.groupby("user_id"):
        indices = user_df.index.to_numpy()
        if len(indices) <= 1:
            train_indices.extend(indices)
            continue

        shuffled_indices = rng.permutation(indices)
        test_count = max(1, round(len(indices) * test_ratio))
        test_count = min(test_count, len(indices) - 1)

        test_indices.extend(shuffled_indices[:test_count])
        train_indices.extend(shuffled_indices[test_count:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def evaluate_track_holdout(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    top_k: int = 10,
    folds: int = 1,
    include_listened: bool = False,
) -> dict[str, float]:
    """Evaluate track similarity with repeated per-user holdout splits."""
    validate_ranking_parameters(top_k)
    if type(folds) is not int or folds < 1:
        raise ValueError("folds must be a positive integer.")
    if type(include_listened) is not bool:
        raise ValueError("include_listened must be a boolean.")

    fold_metrics: list[dict[str, float]] = []
    for fold in range(folds):
        train_df, test_df = train_test_split_tracks_by_user(df, random_state=42 + fold)
        if test_df.empty:
            raise ValueError(
                "No held-out track interactions. Each user needs at least "
                "two distinct tracks for track evaluation."
            )
        resources = build_track_serving_resources(train_df, metadata_df)
        catalog = set(resources.track_ids)
        recommended_lists: list[list[str]] = []
        relevant_lists: list[list[str]] = []
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        for user_id, user_test in test_df.groupby("user_id"):
            relevant = sorted({str(track_id) for track_id in user_test["track_id"]})
            recommendations = recommend_tracks_for_user(
                user_id=str(user_id),
                user_track_matrix=resources.user_track_matrix,
                track_similarity_matrix=resources.similarity_matrix,
                track_id_to_index=resources.track_id_to_index,
                top_k=top_k,
                include_listened=include_listened,
            )
            recommended = [rec["track_id"] for rec in recommendations]
            recommended_lists.append(recommended)
            relevant_lists.append(relevant)
            precisions.append(precision_at_k(recommended, relevant, top_k))
            recalls.append(recall_at_k(recommended, relevant, top_k))
            ndcgs.append(ndcg_at_k(recommended, relevant, top_k))
        fold_metrics.append(
            {
                "precision_at_k": float(np.mean(precisions)),
                "recall_at_k": float(np.mean(recalls)),
                "map_at_k": map_at_k(recommended_lists, relevant_lists, top_k),
                "ndcg_at_k": float(np.mean(ndcgs)),
                "catalog_coverage": catalog_coverage(recommended_lists, catalog),
                "average_popularity": average_popularity(
                    recommended_lists, resources.track_stats
                ),
                "novelty_at_k": novelty_at_k(recommended_lists, resources.track_stats),
            }
        )
    return {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in fold_metrics[0]
    }
