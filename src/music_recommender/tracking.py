"""Optional MLflow experiment tracking helpers."""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MLFLOW_TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"
DEFAULT_TRAINING_EXPERIMENT = "music-recommender-training"
DEFAULT_EVALUATION_EXPERIMENT = "music-recommender-evaluation"


class ExperimentTrackingError(RuntimeError):
    """Raised when an explicitly requested tracking operation fails."""


def flatten_metrics(
    metrics: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, float]:
    """Flatten nested metric dictionaries into MLflow-compatible names."""
    flattened: dict[str, float] = {}
    for name, value in metrics.items():
        metric_name = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, metric_name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                flattened[metric_name] = numeric_value
    return flattened


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Resolve an explicit remote MLflow tracking URI."""
    candidate = tracking_uri.strip() if isinstance(tracking_uri, str) else tracking_uri
    env_uri = os.getenv(MLFLOW_TRACKING_URI_ENV_VAR)
    if isinstance(env_uri, str):
        env_uri = env_uri.strip()
    resolved_uri = candidate or env_uri
    if not resolved_uri:
        raise ExperimentTrackingError(
            "Experiment tracking requires --tracking-uri or the "
            "MLFLOW_TRACKING_URI environment variable."
        )
    if resolved_uri.startswith(("file:", "sqlite:")):
        raise ExperimentTrackingError(
            "The lightweight tracking client requires a remote MLflow server URI, "
            "for example http://127.0.0.1:5000."
        )
    return resolved_uri


def _load_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as error:
        raise ExperimentTrackingError(
            "MLflow tracking is not installed. Run 'uv sync --extra tracking' first."
        ) from error


def _tracking_call(description: str, action: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return action(*args, **kwargs)
    except Exception as error:
        raise ExperimentTrackingError(
            f"MLflow could not {description}: {error}"
        ) from error


@dataclass
class TrackedRun:
    """Small typed facade over an optional active MLflow run."""

    enabled: bool
    run_id: str | None = None
    tracking_uri: str | None = None
    experiment_name: str | None = None
    _mlflow: Any | None = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log non-null run parameters."""
        if not self.enabled or self._mlflow is None:
            return
        normalized = {
            str(key): value for key, value in params.items() if value is not None
        }
        _tracking_call("log parameters", self._mlflow.log_params, normalized)

    def log_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Flatten and log finite numeric run metrics."""
        if not self.enabled or self._mlflow is None:
            return
        normalized = flatten_metrics(metrics)
        if normalized:
            _tracking_call("log metrics", self._mlflow.log_metrics, normalized)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        """Log non-null tags as strings."""
        if not self.enabled or self._mlflow is None:
            return
        normalized = {
            str(key): str(value) for key, value in tags.items() if value is not None
        }
        if normalized:
            _tracking_call("set tags", self._mlflow.set_tags, normalized)

    def log_artifact(
        self,
        path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Upload an existing file to the active run."""
        if not self.enabled or self._mlflow is None:
            return
        artifact = Path(path)
        if not artifact.is_file():
            raise ExperimentTrackingError(
                f"Tracking artifact does not exist: {artifact}"
            )
        _tracking_call(
            "log an artifact",
            self._mlflow.log_artifact,
            str(artifact),
            artifact_path=artifact_path,
        )

    def log_dict(self, payload: Mapping[str, Any], artifact_file: str) -> None:
        """Log a serializable dictionary as a run artifact."""
        if not self.enabled or self._mlflow is None:
            return
        _tracking_call(
            "log a dictionary artifact",
            self._mlflow.log_dict,
            dict(payload),
            artifact_file,
        )


@contextmanager
def tracking_run(
    *,
    enabled: bool,
    tracking_uri: str | None = None,
    experiment_name: str,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[TrackedRun]:
    """Create an MLflow run only when tracking is explicitly enabled."""
    if not enabled:
        yield TrackedRun(enabled=False)
        return

    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow = _load_mlflow()
    _tracking_call("set the tracking URI", mlflow.set_tracking_uri, resolved_uri)
    _tracking_call("select the experiment", mlflow.set_experiment, experiment_name)
    run_context = _tracking_call(
        "start a run",
        mlflow.start_run,
        run_name=run_name,
        tags={
            str(key): str(value)
            for key, value in (tags or {}).items()
            if value is not None
        },
    )
    active_run = _tracking_call("enter a run", run_context.__enter__)

    tracked_run = TrackedRun(
        enabled=True,
        run_id=str(active_run.info.run_id),
        tracking_uri=resolved_uri,
        experiment_name=experiment_name,
        _mlflow=mlflow,
    )
    try:
        yield tracked_run
    except BaseException:
        run_context.__exit__(*sys.exc_info())
        raise
    else:
        _tracking_call("finish the run", run_context.__exit__, None, None, None)
