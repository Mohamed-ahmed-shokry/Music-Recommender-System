"""Tests for cross-surface evaluation parity reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from music_recommender import cli
from music_recommender.surfaces import (
    compare_surface_metrics,
    load_surface_comparison_report,
    write_surface_comparison_report,
)

runner = CliRunner()


def _dummy_metrics() -> dict[str, float]:
    return {
        "precision_at_k": 0.15,
        "recall_at_k": 0.20,
        "map_at_k": 0.12,
        "ndcg_at_k": 0.18,
        "catalog_coverage": 0.40,
        "average_popularity": 500.0,
        "novelty_at_k": 0.70,
        "serendipity_at_k": 0.05,
        "unexpectedness_at_k": 0.30,
        "intra_list_diversity": 0.60,
        "explanation_coverage": 1.0,
    }


def test_compare_surface_metrics_calculates_deltas() -> None:
    artist_metrics = _dummy_metrics()
    track_metrics = {k: v * 0.8 for k, v in artist_metrics.items()}

    comparison = compare_surface_metrics(
        artist_metrics=artist_metrics,
        track_metrics=track_metrics,
        top_k=5,
        folds=1,
    )

    assert comparison["top_k"] == 5
    assert comparison["folds"] == 1
    assert "precision_at_k" in comparison["comparison"]

    prec_delta = comparison["comparison"]["precision_at_k"]["delta"]
    assert prec_delta == pytest.approx(0.15 - 0.12, abs=1e-4)


def test_compare_surface_metrics_handles_multi_arm_dicts() -> None:
    artist_comparison = {"als": _dummy_metrics(), "popularity": _dummy_metrics()}
    track_comparison = {
        "similarity": {k: v * 0.9 for k, v in _dummy_metrics().items()},
        "popularity": _dummy_metrics(),
    }

    comparison = compare_surface_metrics(
        artist_metrics=artist_comparison,
        track_metrics=track_comparison,
        top_k=10,
        folds=2,
    )

    assert comparison["top_k"] == 10
    assert comparison["folds"] == 2
    assert "ndcg_at_k" in comparison["comparison"]


def test_compare_surface_metrics_fallback_to_first_dict() -> None:
    artist_dict = {"custom_arm": _dummy_metrics()}
    track_dict = {"custom_track": _dummy_metrics()}

    comparison = compare_surface_metrics(
        artist_metrics=artist_dict,
        track_metrics=track_dict,
        top_k=5,
    )
    assert "map_at_k" in comparison["comparison"]


def test_compare_surface_metrics_validates_inputs() -> None:
    metrics = _dummy_metrics()

    with pytest.raises(ValueError, match="top_k"):
        compare_surface_metrics(metrics, metrics, top_k=0)

    with pytest.raises(ValueError, match="folds"):
        compare_surface_metrics(metrics, metrics, top_k=5, folds=0)

    with pytest.raises(ValueError, match="metrics must not be empty"):
        compare_surface_metrics({}, metrics, top_k=5)

    with pytest.raises(ValueError, match="metrics must not be empty"):
        compare_surface_metrics(metrics, {}, top_k=5)

    with pytest.raises(ValueError, match="Could not extract valid metric dictionary"):
        compare_surface_metrics({"key": "invalid"}, metrics, top_k=5)


def test_surface_comparison_report_roundtrip(tmp_path: Path) -> None:
    metrics = _dummy_metrics()
    comparison = compare_surface_metrics(metrics, metrics, top_k=5, folds=1)

    written = write_surface_comparison_report(
        comparison,
        tmp_path,
        report_name="parity_report",
    )
    assert written.exists()
    assert written.name == "parity_report.json"

    loaded = load_surface_comparison_report(written)
    assert loaded["top_k"] == 5
    assert "comparison" in loaded
    assert "artist" in loaded
    assert "track" in loaded


def test_load_surface_comparison_report_validates_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_surface_comparison_report(missing_path)

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse"):
        load_surface_comparison_report(corrupt_path)

    invalid_schema = tmp_path / "invalid.json"
    invalid_schema.write_text(json.dumps({"top_k": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid surface comparison report"):
        load_surface_comparison_report(invalid_schema)


def test_cli_evaluate_surfaces() -> None:
    result = runner.invoke(
        cli.app,
        ["evaluate-surfaces", "--top-k", "5", "--folds", "1"],
    )

    assert result.exit_code == 0
    assert "Cross-surface evaluation over 1 fold(s) (top_k=5):" in result.output
    assert "precision_at_k" in result.output
    assert "recall_at_k" in result.output
    assert "ndcg_at_k" in result.output
    assert "Surface comparison report written to:" in result.output


def test_cli_evaluate_surfaces_compare_all(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        [
            "evaluate-surfaces",
            "--top-k",
            "5",
            "--folds",
            "1",
            "--compare-all",
            "--report-dir",
            str(tmp_path),
            "--report-name",
            "custom_parity",
        ],
    )

    assert result.exit_code == 0
    assert "Detailed surface arm breakdowns:" in result.output
    assert "Artist surface arms:" in result.output
    assert "Track surface arms:" in result.output
    assert (tmp_path / "custom_parity.json").exists()


def test_cli_evaluate_surfaces_handles_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "RAW_DATA_PATH", Path("non_existent_data.csv"))
    result = runner.invoke(
        cli.app,
        ["evaluate-surfaces", "--top-k", "5", "--folds", "1"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
