"""Cross-surface evaluation comparison and parity reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from music_recommender.ranking import validate_ranking_parameters

SURFACE_QUALITY_METRICS: tuple[str, ...] = (
    "precision_at_k",
    "recall_at_k",
    "map_at_k",
    "ndcg_at_k",
    "catalog_coverage",
    "average_popularity",
    "novelty_at_k",
    "serendipity_at_k",
    "unexpectedness_at_k",
    "intra_list_diversity",
    "explanation_coverage",
)


def _extract_primary_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Extract primary metric dict from single-arm or multi-arm evaluation results."""
    if not metrics:
        raise ValueError("metrics must not be empty.")
    if "precision_at_k" in metrics:
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if "als" in metrics and isinstance(metrics["als"], dict):
        return {
            k: float(v)
            for k, v in metrics["als"].items()
            if isinstance(v, (int, float))
        }
    if "similarity" in metrics and isinstance(metrics["similarity"], dict):
        return {
            k: float(v)
            for k, v in metrics["similarity"].items()
            if isinstance(v, (int, float))
        }
    first_val = next(iter(metrics.values()))
    if isinstance(first_val, dict):
        return {
            k: float(v) for k, v in first_val.items() if isinstance(v, (int, float))
        }
    raise ValueError("Could not extract valid metric dictionary.")


def compare_surface_metrics(
    artist_metrics: Mapping[str, Any],
    track_metrics: Mapping[str, Any],
    *,
    top_k: int,
    folds: int = 1,
) -> dict[str, Any]:
    """Compare artist-side and track-side evaluation metrics side by side.

    Computes per-metric deltas (artist - track) across all ranking metrics,
    returning a structured summary suitable for reporting and persistence.
    """
    validate_ranking_parameters(top_k)
    if type(folds) is not int or folds < 1:
        raise ValueError("folds must be a positive integer.")

    artist_primary = _extract_primary_metrics(artist_metrics)
    track_primary = _extract_primary_metrics(track_metrics)

    comparison: dict[str, dict[str, float]] = {}
    for metric in SURFACE_QUALITY_METRICS:
        if metric in artist_primary and metric in track_primary:
            a_val = float(artist_primary[metric])
            t_val = float(track_primary[metric])
            comparison[metric] = {
                "artist": a_val,
                "track": t_val,
                "delta": round(a_val - t_val, 4),
            }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "folds": folds,
        "artist": dict(artist_metrics),
        "track": dict(track_metrics),
        "comparison": comparison,
    }


def write_surface_comparison_report(
    comparison: Mapping[str, Any],
    report_dir: Path | str,
    *,
    report_name: str | None = None,
) -> Path:
    """Persist a cross-surface evaluation comparison as a JSON report."""
    target_dir = Path(report_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / f"{report_name or 'surface_comparison'}.json"
    report_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def load_surface_comparison_report(report_path: Path | str) -> dict[str, Any]:
    """Load and validate a persisted cross-surface comparison report."""
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Surface comparison report not found: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Failed to parse surface comparison report '{path}': {error}"
        ) from error
    if (
        not isinstance(report, dict)
        or "comparison" not in report
        or "artist" not in report
        or "track" not in report
    ):
        raise ValueError(
            f"'{path}' is not a valid surface comparison report "
            "(missing 'comparison', 'artist', or 'track')."
        )
    return report
