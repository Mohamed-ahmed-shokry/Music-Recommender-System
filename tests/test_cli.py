from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from typer.testing import CliRunner

import music_recommender.cli as cli
from music_recommender import __version__
from music_recommender.tracking import ExperimentTrackingError

runner = CliRunner()


def test_cli_reports_installed_version() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


class RecordingRun:
    enabled = True
    run_id = "run-123"
    tracking_uri = "https://mlflow.example"

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.tags: dict[str, Any] = {}
        self.artifacts: list[tuple[Any, str | None]] = []
        self.dict_artifacts: list[tuple[dict[str, Any], str]] = []

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self.tags.update(tags)

    def log_artifact(
        self,
        path: Any,
        artifact_path: str | None = None,
    ) -> None:
        self.artifacts.append((path, artifact_path))

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.dict_artifacts.append((payload, artifact_file))


def tracking_context(
    run: RecordingRun,
    captured_config: dict[str, Any],
):
    @contextmanager
    def fake_tracking_run(**kwargs: Any) -> Iterator[RecordingRun]:
        captured_config.update(kwargs)
        yield run

    return fake_tracking_run


def metric_row() -> dict[str, float]:
    return {
        "precision_at_k": 0.25,
        "recall_at_k": 0.5,
        "map_at_k": 0.4,
        "ndcg_at_k": 0.45,
        "catalog_coverage": 0.8,
        "average_popularity": 42.0,
        "novelty_at_k": 0.6,
        "explanation_coverage": 1.0,
        "intra_list_diversity": 0.7,
    }


def test_train_command_logs_configuration_and_dataset_metrics(monkeypatch) -> None:
    recorded_run = RecordingRun()
    tracking_config: dict[str, Any] = {}
    matrix = csr_matrix([[10.0, 0.0], [0.0, 5.0]])
    mappings = {
        "user_id_to_index": {"user_1": 0, "user_2": 1},
        "artist_id_to_index": {"artist_1": 0, "artist_2": 1},
    }
    model = SimpleNamespace(training_device="cpu", gpu_fallback_reason=None)
    monkeypatch.setattr(
        cli,
        "tracking_run",
        tracking_context(recorded_run, tracking_config),
    )
    monkeypatch.setattr(
        cli,
        "train_and_save_model",
        lambda **_: (model, matrix, mappings),
    )

    result = runner.invoke(
        cli.app,
        [
            "train",
            "--factors",
            "8",
            "--iterations",
            "2",
            "--no-use-gpu",
            "--track",
            "--tracking-uri",
            "https://mlflow.example",
            "--experiment-name",
            "training-tests",
            "--run-name",
            "small-als",
            "--no-log-artifact",
        ],
    )

    assert result.exit_code == 0
    assert tracking_config["enabled"] is True
    assert tracking_config["experiment_name"] == "training-tests"
    assert tracking_config["run_name"] == "small-als"
    assert recorded_run.params["factors"] == 8
    assert recorded_run.params["iterations"] == 2
    assert recorded_run.params["use_gpu"] is False
    assert recorded_run.metrics == {
        "num_users": 2,
        "num_artists": 2,
        "num_interactions": 2,
        "matrix_density": 0.5,
    }
    assert recorded_run.tags["training_device"] == "cpu"
    assert not recorded_run.artifacts
    assert "MLflow run ID: run-123" in result.output


def test_evaluate_command_logs_all_strategy_metrics(monkeypatch) -> None:
    recorded_run = RecordingRun()
    tracking_config: dict[str, Any] = {}
    metrics = {
        strategy: metric_row()
        for strategy in ("als", "popularity", "content", "hybrid")
    }
    monkeypatch.setattr(
        cli,
        "tracking_run",
        tracking_context(recorded_run, tracking_config),
    )
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    monkeypatch.setattr(
        cli,
        "load_and_validate_artist_metadata",
        lambda *_: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_repeated_holdout",
        lambda *_args, **_kwargs: metrics,
    )

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--top-k",
            "5",
            "--folds",
            "2",
            "--compare-all",
            "--no-use-gpu",
            "--track",
            "--tracking-uri",
            "https://mlflow.example",
            "--experiment-name",
            "evaluation-tests",
        ],
    )

    assert result.exit_code == 0
    assert tracking_config["experiment_name"] == "evaluation-tests"
    assert recorded_run.params["top_k"] == 5
    assert recorded_run.params["folds"] == 2
    assert recorded_run.params["compare_all"] is True
    assert recorded_run.metrics == metrics
    assert recorded_run.tags["strategies"] == "als,popularity,content,hybrid"
    assert recorded_run.dict_artifacts == [(metrics, "evaluation/metrics.json")]
    assert "Evaluation over 2 fold(s):" in result.output
    assert "MLflow run ID: run-123" in result.output


def test_train_command_reports_tracking_configuration_error(monkeypatch) -> None:
    @contextmanager
    def failing_tracking_run(**_: Any) -> Iterator[RecordingRun]:
        raise ExperimentTrackingError("tracking server is required")
        yield RecordingRun()

    monkeypatch.setattr(cli, "tracking_run", failing_tracking_run)

    result = runner.invoke(cli.app, ["train", "--track"])

    assert result.exit_code == 1
    assert "Error: tracking server is required" in result.output


def test_train_command_reports_validation_error(monkeypatch) -> None:
    def fail_training(**_: Any) -> None:
        raise ValueError("factors must be a positive integer")

    monkeypatch.setattr(cli, "train_and_save_model", fail_training)

    result = runner.invoke(cli.app, ["train", "--no-use-gpu"])

    assert result.exit_code == 1
    assert "Error: factors must be a positive integer" in result.output
    assert result.exception is not None


def test_evaluate_command_reports_missing_data(monkeypatch) -> None:
    def fail_to_load_data(_path: Any) -> None:
        raise FileNotFoundError("interactions file is missing")

    monkeypatch.setattr(cli, "load_and_validate_interactions", fail_to_load_data)

    result = runner.invoke(cli.app, ["evaluate", "--no-use-gpu"])

    assert result.exit_code == 1
    assert "Error: interactions file is missing" in result.output
    assert result.exception is not None
