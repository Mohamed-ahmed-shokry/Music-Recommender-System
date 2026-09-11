"""Holdout evaluation for track similarity recommendations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from music_recommender.config import DEFAULT_CONTENT_WEIGHT
from music_recommender.content import validate_content_weight
from music_recommender.evaluate import (
    average_popularity,
    catalog_coverage,
    explanation_coverage,
    map_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
    serendipity_at_k,
)
from music_recommender.ranking import validate_ranking_parameters
from music_recommender.tracks import (
    TrackServingResources,
    artist_affinity_to_track_scores,
    artist_taste_scores_for_user,
    build_track_serving_resources,
    normalize_track_interactions,
    recommend_popular_tracks,
    recommend_tracks_for_user,
    train_track_artist_taste,
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


def _summarize_track_lists(
    recommended_lists: list[list[str]],
    relevant_lists: list[list[str]],
    top_k: int,
    catalog: set[str],
    resources: TrackServingResources,
    explanation_coverage_value: float = 0.0,
) -> dict[str, float]:
    """Average ranking metrics over per-user recommendation lists."""
    precisions = [
        precision_at_k(recommended, relevant, top_k)
        for recommended, relevant in zip(recommended_lists, relevant_lists, strict=True)
    ]
    recalls = [
        recall_at_k(recommended, relevant, top_k)
        for recommended, relevant in zip(recommended_lists, relevant_lists, strict=True)
    ]
    ndcgs = [
        ndcg_at_k(recommended, relevant, top_k)
        for recommended, relevant in zip(recommended_lists, relevant_lists, strict=True)
    ]
    serendipities = [
        serendipity_at_k(recommended, relevant, resources.track_stats, top_k)
        for recommended, relevant in zip(recommended_lists, relevant_lists, strict=True)
    ]
    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "map_at_k": map_at_k(recommended_lists, relevant_lists, top_k),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "catalog_coverage": catalog_coverage(recommended_lists, catalog),
        "average_popularity": average_popularity(
            recommended_lists, resources.track_stats
        ),
        "novelty_at_k": novelty_at_k(recommended_lists, resources.track_stats),
        "serendipity_at_k": float(np.mean(serendipities)),
        "explanation_coverage": explanation_coverage_value,
    }


def evaluate_track_holdout(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    top_k: int = 10,
    folds: int = 1,
    include_listened: bool = False,
    compare_baseline: bool = False,
    popularity_penalty: float = 0.0,
    diversity: float = 0.0,
    method: str = "similarity",
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Evaluate track recommendations with repeated per-user holdout splits.

    The evaluation arm runs the audio-feature ``similarity`` strategy by
    default; pass ``method="hybrid"`` to blend collaborative artist taste
    (trained on the same held-in interactions) with the audio features. When
    ``compare_baseline`` is set, a global-popularity arm is evaluated on the
    same holdouts, making the popularity-bias tradeoff explicit.
    """
    validate_ranking_parameters(top_k, diversity, popularity_penalty)
    if method not in ("similarity", "hybrid"):
        raise ValueError("method must be one of: similarity, hybrid.")
    validate_content_weight(content_weight)
    if type(folds) is not int or folds < 1:
        raise ValueError("folds must be a positive integer.")
    if type(include_listened) is not bool:
        raise ValueError("include_listened must be a boolean.")
    if type(compare_baseline) is not bool:
        raise ValueError("compare_baseline must be a boolean.")

    similarity_folds: list[dict[str, float]] = []
    popularity_folds: list[dict[str, float]] = []
    for fold in range(folds):
        train_df, test_df = train_test_split_tracks_by_user(df, random_state=42 + fold)
        if test_df.empty:
            raise ValueError(
                "No held-out track interactions. Each user needs at least "
                "two distinct tracks for track evaluation."
            )
        resources = build_track_serving_resources(train_df, metadata_df)
        catalog = set(resources.track_ids)
        taste_model = None
        user_id_to_index: dict[str, int] = {}
        artist_id_to_index: dict[str, int] = {}
        if method == "hybrid":
            (
                taste_model,
                user_id_to_index,
                artist_id_to_index,
            ) = train_track_artist_taste(train_df)
        track_artists = {
            track_id: str(entry["artist_id"])
            for track_id, entry in resources.track_lookup.items()
        }
        similarity_lists: list[list[str]] = []
        popularity_lists: list[list[str]] = []
        relevant_lists: list[list[str]] = []
        explained_recommendations: list[list[dict[str, Any]]] = []
        track_name_lookup = {
            str(row.track_id): str(row.track_name)
            for row in metadata_df.itertuples(index=False)
        }
        track_artist_lookup = {
            str(row.track_id): str(row.artist_name)
            for row in metadata_df.itertuples(index=False)
        }
        for user_id, user_test in test_df.groupby("user_id"):
            relevant = sorted({str(track_id) for track_id in user_test["track_id"]})
            artist_taste_per_track = None
            if method == "hybrid":
                artist_scores = artist_taste_scores_for_user(
                    taste_model, user_id_to_index, artist_id_to_index, str(user_id)
                )
                if artist_scores is not None:
                    artist_taste_per_track = artist_affinity_to_track_scores(
                        artist_scores=artist_scores,
                        artist_id_to_index=artist_id_to_index,
                        track_id_to_index=resources.track_id_to_index,
                        track_artists=track_artists,
                    )
            similarity_recommendations = recommend_tracks_for_user(
                user_id=str(user_id),
                user_track_matrix=resources.user_track_matrix,
                track_similarity_matrix=resources.similarity_matrix,
                track_id_to_index=resources.track_id_to_index,
                top_k=top_k,
                include_listened=include_listened,
                track_stats=resources.track_stats,
                feature_matrix=resources.feature_matrix,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                explain=True,
                track_name_lookup=track_name_lookup,
                track_artist_lookup=track_artist_lookup,
                artist_taste_per_track=artist_taste_per_track,
                content_weight=content_weight,
            )
            similarity_lists.append(
                [rec["track_id"] for rec in similarity_recommendations]
            )
            explained_recommendations.append(similarity_recommendations)
            relevant_lists.append(relevant)
            if compare_baseline:
                popularity_lists.append(
                    [
                        rec["track_id"]
                        for rec in recommend_popular_tracks(
                            user_id=str(user_id),
                            user_track_matrix=resources.user_track_matrix,
                            track_stats=resources.track_stats,
                            top_k=top_k,
                            include_listened=include_listened,
                        )
                    ]
                )
        similarity_folds.append(
            _summarize_track_lists(
                similarity_lists,
                relevant_lists,
                top_k,
                catalog,
                resources,
                explanation_coverage_value=explanation_coverage(
                    explained_recommendations
                ),
            )
        )
        if compare_baseline:
            popularity_folds.append(
                _summarize_track_lists(
                    popularity_lists, relevant_lists, top_k, catalog, resources
                )
            )
    similarity_metrics = {
        metric: float(np.mean([fold[metric] for fold in similarity_folds]))
        for metric in similarity_folds[0]
    }
    if not compare_baseline:
        return similarity_metrics
    return {
        "similarity": similarity_metrics,
        "popularity": {
            metric: float(np.mean([fold[metric] for fold in popularity_folds]))
            for metric in popularity_folds[0]
        },
    }


def write_track_report(
    metrics: dict[str, float] | dict[str, dict[str, float]],
    report_dir: str | Path,
    *,
    top_k: int,
    folds: int,
    report_name: str | None = None,
) -> Path:
    """Persist a track evaluation run as a JSON report.

    The report records the generated-at timestamp, the evaluation
    configuration, and the per-arm metrics under a stable schema so results
    from different runs can be compared side by side. The parent directory is
    created if needed.
    """
    report_path = Path(report_dir) / f"{report_name or 'track_evaluation'}.json"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "folds": folds,
        "metrics": metrics,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def load_track_report(report_path: Path) -> dict[str, Any]:
    """Load and validate a persisted track evaluation report."""
    if not report_path.exists():
        raise FileNotFoundError(f"Track report not found: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Failed to parse track report '{report_path}': {error}"
        ) from error
    if not isinstance(report, dict) or "metrics" not in report:
        raise ValueError(f"File '{report_path}' is not a track evaluation report.")
    return report
