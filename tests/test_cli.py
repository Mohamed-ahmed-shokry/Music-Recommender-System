from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
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
        "unexpectedness_at_k": 0.3,
        "serendipity_at_k": 0.2,
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


def test_train_command_logs_champion_ranking_settings(monkeypatch) -> None:
    recorded_run = RecordingRun()
    tracking_config: dict[str, Any] = {}
    matrix = csr_matrix([[10.0, 0.0], [0.0, 5.0]])
    mappings = {
        "user_id_to_index": {"user_1": 0, "user_2": 1},
        "artist_id_to_index": {"artist_1": 0, "artist_2": 1},
    }
    model = SimpleNamespace(training_device="cpu", gpu_fallback_reason=None)
    trained_settings: dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "tracking_run",
        tracking_context(recorded_run, tracking_config),
    )

    def record_training(**kwargs: Any) -> tuple[Any, Any, Any]:
        trained_settings.update(kwargs)
        return model, matrix, mappings

    monkeypatch.setattr(cli, "train_and_save_model", record_training)

    result = runner.invoke(
        cli.app,
        [
            "train",
            "--no-use-gpu",
            "--track",
            "--popularity-penalty",
            "0.2",
            "--diversity",
            "0.5",
            "--include-listened",
        ],
    )

    assert result.exit_code == 0
    assert trained_settings["popularity_penalty"] == 0.2
    assert trained_settings["diversity"] == 0.5
    assert trained_settings["include_listened"] is True
    assert recorded_run.params["popularity_penalty"] == 0.2
    assert recorded_run.params["diversity"] == 0.5
    assert recorded_run.params["include_listened"] is True
    assert (
        "Default ranking settings: penalty=0.2, diversity=0.5, include_listened=True"
        in result.output
    )


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


def test_evaluate_command_with_compare_settings_logs_and_prints_labels(
    monkeypatch,
) -> None:
    recorded_run = RecordingRun()
    tracking_config: dict[str, Any] = {}
    diverse_row = metric_row()
    diverse_row["ndcg_at_k"] = 0.9
    diverse_row["precision_at_k"] = 0.8
    comparison = {"control": metric_row(), "diverse": diverse_row}
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
        "compare_parameter_settings",
        lambda *_args, **_kwargs: comparison,
    )

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--top-k",
            "5",
            "--no-use-gpu",
            "--track",
            "--compare-settings",
            "control:;diverse:popularity_penalty=0.2,diversity=0.5",
        ],
    )

    assert result.exit_code == 0
    assert recorded_run.params["compare_settings"] == (
        "control:;diverse:popularity_penalty=0.2,diversity=0.5"
    )
    assert not recorded_run.metrics
    assert recorded_run.tags["strategies"] == "control,diverse"
    assert recorded_run.dict_artifacts == [(comparison, "evaluation/metrics.json")]
    assert "control:" in result.output
    assert "diverse:" in result.output
    assert "  precision_at_k: diverse" in result.output
    assert "Overall: control won 8 of 10 metrics." in result.output


def test_evaluate_command_rejects_compare_settings_with_all(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )

    result = runner.invoke(
        cli.app,
        ["evaluate", "--compare-all", "--compare-settings", "control:;diverse:"],
    )

    assert result.exit_code == 2


def test_evaluate_command_promotes_winning_setting(monkeypatch) -> None:
    trained: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    diverse_row = metric_row()
    diverse_row["ndcg_at_k"] = 0.9
    diverse_row["precision_at_k"] = 0.8
    comparison = {"control": metric_row(), "diverse": diverse_row}
    monkeypatch.setattr(
        cli,
        "compare_parameter_settings",
        lambda *_args, **_kwargs: comparison,
    )
    monkeypatch.setattr(
        cli,
        "train_and_save_model",
        lambda **kwargs: trained.append(kwargs),
    )

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--no-use-gpu",
            "--compare-settings",
            "control:;diverse:popularity_penalty=0.2,diversity=0.5",
            "--promote-winner",
        ],
    )

    assert result.exit_code == 0
    assert "Promoting the winning setting" in result.output
    assert "Promoted 'control' ranking settings" in result.output
    assert trained == [
        {
            "popularity_penalty": 0.0,
            "diversity": 0.0,
            "include_listened": False,
        }
    ]


def test_evaluate_command_promotes_diverse_winner_with_params(monkeypatch) -> None:
    trained: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    diverse_row = metric_row()
    for key, value in diverse_row.items():
        if isinstance(value, float):
            diverse_row[key] = value + 0.1
    comparison = {"control": metric_row(), "diverse": diverse_row}
    monkeypatch.setattr(
        cli,
        "compare_parameter_settings",
        lambda *_args, **_kwargs: comparison,
    )
    monkeypatch.setattr(
        cli,
        "train_and_save_model",
        lambda **kwargs: trained.append(kwargs),
    )

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--no-use-gpu",
            "--compare-settings",
            "control:;diverse:popularity_penalty=0.2,diversity=0.5",
            "--promote-winner",
        ],
    )

    assert result.exit_code == 0
    assert "Promoted 'diverse' ranking settings" in result.output
    assert trained == [
        {
            "popularity_penalty": 0.2,
            "diversity": 0.5,
            "include_listened": False,
        }
    ]


def test_evaluate_command_requires_compare_settings_for_promotion(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )

    result = runner.invoke(cli.app, ["evaluate", "--promote-winner"])

    assert result.exit_code == 2
    assert "--promote-winner requires --compare-settings" in result.output


def test_evaluate_command_promotion_reports_retrain_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    monkeypatch.setattr(
        cli,
        "compare_parameter_settings",
        lambda *_args, **_kwargs: {"diverse": metric_row()},
    )
    monkeypatch.setattr(
        cli,
        "train_and_save_model",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("retrain boom")),
    )

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--no-use-gpu",
            "--compare-settings",
            "diverse:popularity_penalty=0.2",
            "--promote-winner",
        ],
    )

    assert result.exit_code == 1
    assert "promotion failed: retrain boom" in result.output


def test_parse_parameter_settings_parses_labels_and_values() -> None:
    parameter_sets = cli._parse_parameter_settings(
        "control:;diverse:popularity_penalty=0.2,diversity=0.5;flag:include_listened=true,count=5;raw:name=xyz"
    )

    assert parameter_sets["control"] == {}
    assert parameter_sets["diverse"] == {
        "popularity_penalty": 0.2,
        "diversity": 0.5,
    }
    assert parameter_sets["flag"] == {"include_listened": True, "count": 5}
    assert parameter_sets["raw"] == {"name": "xyz"}


def test_parse_parameter_settings_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        cli._parse_parameter_settings("not-a-setting")
    with pytest.raises(ValueError):
        cli._parse_parameter_settings("label:key")
    with pytest.raises(ValueError):
        cli._parse_parameter_settings("dup:;dup:")


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


def test_format_artifact_age_returns_minutes() -> None:
    created = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()

    result = _format_artifact_age(created)

    assert result.endswith("m")


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
            "ranking_config": {
                "include_listened": True,
                "popularity_penalty": 0.2,
                "diversity": 0.4,
            },
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
    assert (
        "Default ranking settings: penalty=0.2, diversity=0.4, include_listened=True"
        in result.output
    )


def test_artifact_info_defaults_ranking_settings_without_config(monkeypatch) -> None:
    fake = FakeService()
    original_metadata = fake.metadata

    def metadata_without_ranking() -> dict[str, Any]:
        metadata = original_metadata()
        metadata.pop("ranking_config")
        return metadata

    fake.metadata = metadata_without_ranking
    install_fake_service(monkeypatch, service=fake)

    result = runner.invoke(cli.app, ["artifact-info"])

    assert result.exit_code == 0
    assert (
        "Default ranking settings: penalty=0.0, diversity=0.0, include_listened=False"
        in result.output
    )


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


def test_demo_command_catches_training_runtime_error(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "ARTIFACT_BUNDLE_PATH",
        SimpleNamespace(exists=lambda: False),
    )

    def fail_training(**_: Any) -> None:
        raise RuntimeError("implicit training failed")

    monkeypatch.setattr(cli, "train_and_save_model", fail_training)

    result = runner.invoke(cli.app, ["demo"])

    assert result.exit_code == 1
    assert "Error: demo failed: implicit training failed" in result.output
    assert result.exception is not None


def test_prepare_metadata_command_validates_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"user_id": ["u1"], "artist_id": ["a1"]}),
    )
    monkeypatch.setattr(
        cli,
        "load_and_validate_artist_metadata",
        lambda *_: pd.DataFrame({"artist_id": ["a1"]}),
    )

    result = runner.invoke(cli.app, ["prepare-metadata"])

    assert result.exit_code == 0
    assert "Metadata validated successfully." in result.output
    assert "Metadata rows: 1" in result.output


def test_prepare_metadata_command_reports_validation_error(monkeypatch) -> None:
    def fail_to_load(_: Any) -> None:
        raise ValueError("metadata validation failed")

    monkeypatch.setattr(cli, "load_and_validate_interactions", fail_to_load)

    result = runner.invoke(cli.app, ["prepare-metadata"])

    assert result.exit_code == 1
    assert "Error: metadata validation failed" in result.output


def test_train_command_logs_serving_artifact_when_enabled(monkeypatch) -> None:
    recorded_run = RecordingRun()
    tracking_config: dict[str, Any] = {}
    model = SimpleNamespace(training_device="cpu", gpu_fallback_reason=None)
    matrix = csr_matrix([[10.0, 0.0], [0.0, 5.0]])
    mappings = {
        "user_id_to_index": {"user_1": 0, "user_2": 1},
        "artist_id_to_index": {"artist_1": 0, "artist_2": 1},
    }
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
            "--no-use-gpu",
            "--track",
            "--tracking-uri",
            "https://mlflow.example",
        ],
    )

    assert result.exit_code == 0
    assert recorded_run.artifacts == [(cli.ARTIFACT_BUNDLE_PATH, "serving")]


def test_train_command_reports_gpu_fallback_reason(monkeypatch) -> None:
    model = SimpleNamespace(
        training_device="cpu",
        gpu_fallback_reason="cuda unavailable",
    )
    monkeypatch.setattr(
        cli,
        "train_and_save_model",
        lambda **_: (
            model,
            csr_matrix([[10.0]]),
            {"user_id_to_index": {"u1": 0}, "artist_id_to_index": {"a1": 0}},
        ),
    )

    result = runner.invoke(cli.app, ["train", "--no-use-gpu"])

    assert result.exit_code == 0
    assert "GPU fallback reason: cuda unavailable" in result.output


def test_artifact_info_reports_gpu_fallback_reason(monkeypatch) -> None:
    class FallbackFakeService(FakeService):
        def metadata(self) -> dict[str, Any]:
            metadata = super().metadata()
            metadata["metadata"]["gpu_fallback_reason"] = "cuda unavailable"
            return metadata

    install_fake_service(monkeypatch, FallbackFakeService())

    result = runner.invoke(cli.app, ["artifact-info"])

    assert result.exit_code == 0
    assert "GPU fallback reason: cuda unavailable" in result.output


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["recommend-user", "--user-id", "user_1"], "service unavailable"),
        (["recommend-profile", "--artist-ids", "artist_1"], "service unavailable"),
        (["recommend-session"], "service unavailable"),
        (["popular-artists"], "service unavailable"),
        (["similar-artists", "--artist-id", "artist_2"], "service unavailable"),
        (
            ["content-similar-artists", "--artist-id", "artist_2"],
            "service unavailable",
        ),
    ],
)
def test_recommendation_commands_report_service_errors(
    monkeypatch,
    arguments: list[str],
    message: str,
) -> None:
    def fail_load(*_: Any) -> None:
        raise ValueError(message)

    monkeypatch.setattr(
        cli,
        "RecommenderService",
        SimpleNamespace(from_artifacts=fail_load),
    )

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert result.exception is not None


def test_evaluate_command_compare_baseline_prints_both_rows(monkeypatch) -> None:
    metrics = dict.fromkeys(("als", "popularity"), metric_row())
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_repeated_holdout",
        lambda *_args, **_kwargs: metrics,
    )

    result = runner.invoke(cli.app, ["evaluate", "--compare-baseline"])

    assert result.exit_code == 0
    assert "ALS:" in result.output
    assert "Popularity:" in result.output


def test_evaluate_command_als_only_prints_single_row(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_repeated_holdout",
        lambda *_args, **_kwargs: metric_row(),
    )

    result = runner.invoke(cli.app, ["evaluate"])

    assert result.exit_code == 0
    assert "ALS:" in result.output
    assert "Popularity:" not in result.output


def test_evaluate_command_learn_to_rank_forwards_flag_and_prints_ltr(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    metrics = {"als": metric_row(), "ltr": metric_row()}

    def fake_holdout(*args, **kwargs) -> str:
        captured.update(kwargs)
        return metrics

    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )
    monkeypatch.setattr(cli, "evaluate_repeated_holdout", fake_holdout)

    result = runner.invoke(
        cli.app,
        ["evaluate", "--no-use-gpu", "--learn-to-rank"],
    )

    assert result.exit_code == 0
    assert captured["learn_to_rank"] is True
    assert "ALS:" in result.output
    assert "LTR:" in result.output


def test_evaluate_command_learn_to_rank_tags_ltr_strategy(monkeypatch) -> None:
    recorded_run = RecordingRun()
    tracking_config: dict[str, Any] = {}
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

    def fake_holdout(*args, **kwargs) -> str:
        return {"als": metric_row(), "ltr": metric_row()}

    monkeypatch.setattr(cli, "evaluate_repeated_holdout", fake_holdout)

    result = runner.invoke(
        cli.app,
        ["evaluate", "--no-use-gpu", "--track", "--learn-to-rank"],
    )

    assert result.exit_code == 0
    assert recorded_run.tags["strategies"] == "als,ltr"
    assert recorded_run.params["learn_to_rank"] is True


def test_evaluate_command_learn_to_rank_with_compare_all_prints_ltr(
    monkeypatch,
) -> None:
    metrics = dict.fromkeys(
        ("als", "popularity", "content", "hybrid", "ltr"), metric_row()
    )
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )

    def fake_holdout(*args, **kwargs) -> str:
        return metrics

    monkeypatch.setattr(cli, "evaluate_repeated_holdout", fake_holdout)

    result = runner.invoke(
        cli.app,
        ["evaluate", "--no-use-gpu", "--compare-all", "--learn-to-rank"],
    )

    assert result.exit_code == 0
    assert "ALS:" in result.output
    assert "Hybrid:" in result.output
    assert "LTR:" in result.output


def test_evaluate_command_learn_to_rank_with_compare_baseline_prints_ltr(
    monkeypatch,
) -> None:
    metrics = dict.fromkeys(("als", "popularity", "ltr"), metric_row())
    monkeypatch.setattr(
        cli,
        "load_and_validate_interactions",
        lambda _: pd.DataFrame({"artist_id": ["artist_1"]}),
    )

    def fake_holdout(*args, **kwargs) -> str:
        return metrics

    monkeypatch.setattr(cli, "evaluate_repeated_holdout", fake_holdout)

    result = runner.invoke(
        cli.app,
        ["evaluate", "--no-use-gpu", "--compare-baseline", "--learn-to-rank"],
    )

    assert result.exit_code == 0
    assert "ALS:" in result.output
    assert "Popularity:" in result.output
    assert "LTR:" in result.output


def test_demo_command_reports_service_error(monkeypatch) -> None:
    def fail_load(*_: Any) -> None:
        raise ValueError("artifact is invalid")

    monkeypatch.setattr(
        cli,
        "RecommenderService",
        SimpleNamespace(from_artifacts=fail_load),
    )

    result = runner.invoke(cli.app, ["demo"])

    assert result.exit_code == 1
    assert "Error: artifact is invalid" in result.output
    assert result.exception is not None
