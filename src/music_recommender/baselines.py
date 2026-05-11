"""Baseline recommenders used for cold-start and evaluation."""

from __future__ import annotations

from typing import Any

Recommendation = dict[str, str | float | int]


def popular_artists(
    artist_stats: dict[str, dict[str, Any]],
    top_k: int,
    exclude_artist_ids: set[str] | None = None,
) -> list[Recommendation]:
    """Return artists ranked by training-set popularity."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    excluded = exclude_artist_ids or set()
    ranked_stats = sorted(
        artist_stats.values(),
        key=lambda item: (
            int(item["popularity_rank"]),
            str(item["artist_id"]),
        ),
    )

    recommendations: list[Recommendation] = []
    for stats in ranked_stats:
        artist_id = str(stats["artist_id"])
        if artist_id in excluded:
            continue
        recommendations.append(
            {
                "artist_id": artist_id,
                "artist_name": str(stats["artist_name"]),
                "score": float(stats["total_plays"]),
                "popularity_rank": int(stats["popularity_rank"]),
            }
        )
        if len(recommendations) == top_k:
            break

    return recommendations
