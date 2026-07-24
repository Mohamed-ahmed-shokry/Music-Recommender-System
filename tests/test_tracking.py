from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import music_recommender.tracking as tracking
from music_recommender.tracking import (
    ExperimentTrackingError,
    TrackedRun,
    flatten_metrics,
    resolve_tracking_uri,
    tracking_run,
)


class FakeActiveRun:
    def __init__(self, run_id: str = "run-123") -> None:
        self.info = SimpleNamespace(run_id=run_id)
        self.exit_args: tuple[Any, ...] | None = None

    def __enter__(self) -> FakeActiveRun:
        return self

    def __exit__(self, *args: Any) -> None:
        self.exit_args = args


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.start_kwargs: dict[str, Any] = {}
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str | None]] = []
        self.dict_artifacts: list[tuple[dict[str, Any], str]] = []
        self.active_run = FakeActiveRun()

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, **kwargs: Any) -> FakeActiveRun:
        self.start_kwargs = kwargs
        return self.active_run

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_artifact(
        self,
        path: str,
        artifact_path: str | None = None,
    ) -> None:
        self.artifacts.append((path, artifact_path))

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.dict_artifacts.append((payload, artifact_file))


def test_flatten_metrics_flattens_strategies_and_ignores_invalid_values() -> None:
    metrics = {
        "als": {"recall_at_k": 0.5, "precision_at_k": 0.25},
        "folds": 2,
        "enabled": True,
        "missing": None,
        "infinite": float("inf"),
    }

    assert flatten_metrics(metrics) == {
        "als.recall_at_k": 0.5,
        "als.precision_at_k": 0.25,
        "folds": 2.0,
    }


def test_resolve_tracking_uri_accepts_argument_and_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://environment.example")

    assert resolve_tracking_uri("https://argument.example") == (
        "https://argument.example"
    )
    assert resolve_tracking_uri() == "https://environment.example"


@pytest.mark.parametrize(
    "tracking_uri",
    ["file:///tmp/mlruns", "sqlite:///mlflow.db"],
)
def test_resolve_tracking_uri_rejects_local_stores(tracking_uri: str) -> None:
    with pytest.raises(ExperimentTrackingError, match="remote MLflow server"):
        resolve_tracking_uri(tracking_uri)


def test_resolve_tracking_uri_requires_configuration(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ExperimentTrackingError, match="requires --tracking-uri"):
        resolve_tracking_uri()


def test_tracked_run_logs_supported_payloads(tmp_path: Path) -> None:
    fake_mlflow = FakeMlflow()
    run = TrackedRun(enabled=True, run_id="run-123", _mlflow=fake_mlflow)
    artifact = tmp_path / "artifact.joblib"
    artifact.write_bytes(b"artifact")

    run.log_params({"factors": 32, "ignored": None})
    run.log_metrics({"als": {"recall": 0.5}, "invalid": float("nan")})
    run.set_tags({"device": "cpu", "ignored": None})
    run.log_artifact(artifact, artifact_path="serving")
    run.log_dict({"recall": 0.5}, "evaluation/metrics.json")

    assert fake_mlflow.params == {"factors": 32}
    assert fake_mlflow.metrics == {"als.recall": 0.5}
    assert fake_mlflow.tags == {"device": "cpu"}
    assert fake_mlflow.artifacts == [(str(artifact), "serving")]
    assert fake_mlflow.dict_artifacts == [({"recall": 0.5}, "evaluation/metrics.json")]


def test_tracked_run_rejects_missing_artifact(tmp_path: Path) -> None:
    run = TrackedRun(enabled=True, _mlflow=FakeMlflow())

    with pytest.raises(ExperimentTrackingError, match="does not exist"):
        run.log_artifact(tmp_path / "missing.joblib")


def test_disabled_run_is_a_no_op() -> None:
    with tracking_run(
        enabled=False,
        experiment_name="ignored",
    ) as run:
        run.log_params({"factors": 32})
        run.log_metrics({"recall": 0.5})

    assert not run.enabled
    assert run.run_id is None


def test_tracking_run_manages_mlflow_lifecycle(monkeypatch) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(tracking, "_load_mlflow", lambda: fake_mlflow)

    with tracking_run(
        enabled=True,
        tracking_uri="https://mlflow.example",
        experiment_name="training",
        run_name="als-32",
        tags={"workflow": "training", "attempt": 1},
    ) as run:
        run.log_metrics({"recall": 0.5})

    assert run.run_id == "run-123"
    assert run.tracking_uri == "https://mlflow.example"
    assert fake_mlflow.tracking_uri == "https://mlflow.example"
    assert fake_mlflow.experiment_name == "training"
    assert fake_mlflow.start_kwargs == {
        "run_name": "als-32",
        "tags": {"workflow": "training", "attempt": "1"},
    }
    assert fake_mlflow.metrics == {"recall": 0.5}
    assert fake_mlflow.active_run.exit_args == (None, None, None)


def test_tracking_run_marks_failed_body(monkeypatch) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(tracking, "_load_mlflow", lambda: fake_mlflow)

    with pytest.raises(RuntimeError, match="training failed"):
        with tracking_run(
            enabled=True,
            tracking_uri="https://mlflow.example",
            experiment_name="training",
        ):
            raise RuntimeError("training failed")

    assert fake_mlflow.active_run.exit_args is not None
    assert fake_mlflow.active_run.exit_args[0] is RuntimeError


def test_missing_mlflow_dependency_has_install_guidance(monkeypatch) -> None:
    def missing_import(_: str) -> Any:
        raise ModuleNotFoundError

    monkeypatch.setattr(tracking.importlib, "import_module", missing_import)

    with pytest.raises(ExperimentTrackingError, match="--extra tracking"):
        tracking._load_mlflow()
