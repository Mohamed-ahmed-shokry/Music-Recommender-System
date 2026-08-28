from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from typer.testing import CliRunner

import music_recommender.cli as cli
from music_recommender import __version__
from music_recommender.cli import _format_artifact_age, _parse_csv_option
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


def test_format_artifact_age_returns_hours() -> None:
    result = _format_artifact_age("2026-01-01T00:00:00+00:00")

    assert result.endswith("h")


def test_format_artifact_age_treats_naive_timestamp_as_utc() -> None:
    result = _format_artifact_age("2026-01-01T00:00:00")

    assert result.endswith("h") or result.endswith("m") or result.endswith("s")


def test_format_artifact_age_returns_seconds_for_recent_timestamp() -> None:
    created = datetime.now(UTC).isoformat()

    result = _format_artifact_age(created)

    assert result.endswith("s")


def test_parse_csv_option_splits_comma_separated_values() -> None:
    result = _parse_csv_option("artist_1,artist_2,artist_3")

    assert result == ["artist_1", "artist_2", "artist_3"]


def test_parse_csv_option_strips_whitespace() -> None:
    result = _parse_csv_option(" a , b , c ")

    assert result == ["a", "b", "c"]


def test_parse_csv_option_returns_empty_list_for_empty_string() -> None:
    result = _parse_csv_option("")

    assert result == []


class FakeService:
    def __init__(self) -> None:
        self.user_response = {
            "strategy": "hybrid_personalized",
            "message": "welcome",
            "recommendations": [
                {
                    "artist_id": "artist_7",
                    "artist_name": "Taylor Swift",
                    "score": 0.42,
                    "popularity_rank": 5,
                    "reasons": ["Shares pop with The Weeknd"],
                }
            ],
        }
        self.profile_response = {
            "strategy": "content_profile",
            "recommendations": [
                {
                    "artist_id": "artist_7",
                    "artist_name": "Taylor Swift",
                    "score": 0.5,
                    "score_components": {"content_score": 0.5},
                }
            ],
        }
        self.session_response = {
            "strategy": "session_hybrid",
            "message": "session built",
            "recommendations": [
                {
                    "artist_id": "artist_7",
                    "artist_name": "Taylor Swift",
                    "score": 0.3,
                }
            ],
        }
        self.popular_response = {
            "strategy": "popular_fallback",
            "recommendations": [
                {
                    "artist_id": "artist_2",
                    "artist_name": "Drake",
                    "score": 174.0,
                    "popularity_rank": 1,
                }
            ],
        }
        self.similar_response = {
            "strategy": "hybrid_similarity",
            "similar_artists": [
                {
                    "artist_id": "artist_3",
                    "artist_name": "Kendrick Lamar",
                    "score": 0.6,
                }
            ],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "4",
            "metadata": {
                "created_at": "2026-08-28T12:00:00+00:00",
                "num_users": 50,
                "num_artists": 100,
                "num_interactions": 500,
                "training_device": "cpu",
                "gpu_fallback_reason": None,
                "dataset": {"sha256": "abc123"},
                "metadata_dataset": {"sha256": "def456"},
            },
            "training_config": {
                "factors": 8,
                "regularization": 0.1,
                "iterations": 2,
                "alpha": 40.0,
            },
            "hybrid_config": {"default_content_weight": 0.25},
            "content": {"num_features": 5, "feature_names": ["a", "b"]},
        }

    def recommend_user(self, **_: Any) -> dict[str, Any]:
        return self.user_response

    def recommend_profile(self, **_: Any) -> dict[str, Any]:
        return self.profile_response

    def recommend_session(self, **_: Any) -> dict[str, Any]:
        return self.session_response

    def popular_artists(self, **_: Any) -> dict[str, Any]:
        return self.popular_response

    def similar_artists(self, **_: Any) -> dict[str, Any]:
        return self.similar_response

    def content_similar_artists(self, **_: Any) -> dict[str, Any]:
        return self.similar_response


def install_fake_service(
    monkeypatch,
    service: FakeService | None = None,
) -> FakeService:
    fake = service or FakeService()
    monkeypatch.setattr(
        cli,
        "RecommenderService",
        SimpleNamespace(from_artifacts=lambda *_: fake),
    )
    return fake


def test_prepare_data_command_reports_counts(monkeypatch) -> None:
    df = pd.DataFrame({"user_id": ["u1"], "artist_id": ["a1"]})
    matrix = csr_matrix([[1.0]])
    mappings = {
        "user_id_to_index": {"u1": 0},
        "artist_id_to_index": {"a1": 0},
    }
    monkeypatch.setattr(
        cli,
        "prepare_training_data",
        lambda **_: (df, matrix, mappings),
    )

    result = runner.invoke(cli.app, ["prepare-data"])

    assert result.exit_code == 0
    assert "Data prepared successfully." in result.output
    assert "Users: 1" in result.output
    assert "Artists: 1" in result.output


def test_artifact_info_command_prints_metadata(monkeypatch) -> None:
    install_fake_service(monkeypatch)

    result = runner.invoke(cli.app, ["artifact-info"])

    assert result.exit_code == 0
    assert "Artifact version: 4" in result.output
    assert "Users: 50" in result.output
    assert "Artists: 100" in result.output
    assert "Factors: 8" in result.output


def test_recommend_user_command_prints_recommendations(monkeypatch) -> None:
    install_fake_service(monkeypatch)

    result = runner.invoke(
        cli.app,
        ["recommend-user", "--user-id", "user_1", "--top-k", "5"],
    )

    assert result.exit_code == 0
    assert "Recommendations for user_1:" in result.output
    assert "Strategy: hybrid_personalized" in result.output
    assert "Taylor Swift" in result.output
    assert "popularity rank: 5" in result.output


def test_recommend_user_command_reports_unknown_user(monkeypatch) -> None:
    fake = FakeService()
    fake.user_response = {
        "strategy": "popular_fallback",
        "message": "Unknown user_id 'new_user'. Returning popular artists.",
        "recommendations": [
            {
                "artist_id": "artist_2",
                "artist_name": "Drake",
                "score": 174.0,
            }
        ],
    }
    install_fake_service(monkeypatch, fake)

    result = runner.invoke(cli.app, ["recommend-user", "--user-id", "new_user"])

    assert result.exit_code == 0
    assert "Unknown user_id 'new_user'." in result.output


def test_recommend_profile_command_prints_recommendations(monkeypatch) -> None:
    install_fake_service(monkeypatch)

    result = runner.invoke(
        cli.app,
        ["recommend-profile", "--artist-ids", "artist_1,artist_6", "--top-k", "10"],
    )

    assert result.exit_code == 0
    assert "Profile recommendations:" in result.output
    assert "Strategy: content_profile" in result.output
    assert "content score: 0.5000" in result.output


def test_recommend_session_command_prints_recommendations_with_message(
    monkeypatch,
) -> None:
    install_fake_service(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "recommend-session",
            "--user-id",
            "user_1",
            "--artist-ids",
            "artist_1",
            "--exclude-artist-ids",
            "artist_2",
        ],
    )

    assert result.exit_code == 0
    assert "Session recommendations:" in result.output
    assert "Strategy: session_hybrid" in result.output
    assert "session built" in result.output


def test_popular_artists_command_prints_recommendations(monkeypatch) -> None:
    install_fake_service(monkeypatch)

    result = runner.invoke(cli.app, ["popular-artists", "--top-k", "5"])

    assert result.exit_code == 0
    assert "Popular artists:" in result.output
    assert "Strategy: popular_fallback" in result.output
    assert "popularity rank: 1" in result.output


def test_similar_artists_command_prints_similar_artists(monkeypatch) -> None:
    install_fake_service(monkeypatch)

    result = runner.invoke(
        cli.app,
        ["similar-artists", "--artist-id", "artist_2", "--method", "hybrid"],
    )

    assert result.exit_code == 0
    assert "Artists similar to artist_2:" in result.output
    assert "Strategy: hybrid_similarity" in result.output


def test_content_similar_artists_command_prints_similar_artists(monkeypatch) -> None:
    install_fake_service(monkeypatch)

    result = runner.invoke(
        cli.app,
        ["content-similar-artists", "--artist-id", "artist_2"],
    )

    assert result.exit_code == 0
    assert "Content-similar artists for artist_2:" in result.output


def test_demo_command_trains_and_prints_recommendations(monkeypatch) -> None:
    install_fake_service(monkeypatch)
    trained = []
    monkeypatch.setattr(
        cli,
        "ARTIFACT_BUNDLE_PATH",
        SimpleNamespace(exists=lambda: False),
    )
    monkeypatch.setattr(cli, "train_and_save_model", lambda **_: trained.append(True))

    result = runner.invoke(cli.app, ["demo"])

    assert result.exit_code == 0
    assert trained
    assert "Recommendations for user_1:" in result.output
    assert "Content-similar artists for artist_2:" in result.output


def test_command_reports_service_load_error(monkeypatch) -> None:
    def fail_load(*_: Any) -> None:
        raise FileNotFoundError("artifact bundle is missing")

    monkeypatch.setattr(
        cli,
        "RecommenderService",
        SimpleNamespace(from_artifacts=fail_load),
    )

    result = runner.invoke(cli.app, ["artifact-info"])

    assert result.exit_code == 1
    assert "Error: artifact bundle is missing" in result.output
    assert result.exception is not None


def test_train_command_catches_training_runtime_error(monkeypatch) -> None:
    def fail_training(**_: Any) -> None:
        raise RuntimeError("implicit training failed")

    monkeypatch.setattr(cli, "train_and_save_model", fail_training)

    result = runner.invoke(cli.app, ["train", "--no-use-gpu"])

    assert result.exit_code == 1
    assert "Error: training failed: implicit training failed" in result.output
    assert result.exception is not None
