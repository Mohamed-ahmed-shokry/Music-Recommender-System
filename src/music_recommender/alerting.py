"""Threshold alert rules over Prometheus metric families.

Rules are declared once and reused for in-process evaluation, webhook
notification dispatch, and generation of a Prometheus rule file.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_METRIC_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_WINDOW_RE = re.compile(r"^[1-9][0-9]*[smhd]$")

COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

PROMETHEUS_EVALUATION_INTERVAL_SECONDS = 15


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    WARNING = "warning"
    CRITICAL = "critical"


class AlertMode(StrEnum):
    """How a metric value is derived before the threshold comparison."""

    VALUE = "value"
    INCREASE = "increase"


@dataclass(frozen=True)
class AlertRule:
    """A single threshold rule evaluated against one metric.

    ``mode=INCREASE`` compares the growth of a counter between consecutive
    evaluations (``increase(metric[window])`` in PromQL) rather than its
    absolute value.
    """

    name: str
    metric: str
    comparator: str
    threshold: float
    severity: AlertSeverity | str
    summary: str
    labels: dict[str, str] = field(default_factory=dict)
    mode: AlertMode | str = AlertMode.VALUE
    for_evaluations: int = 1
    window: str = "5m"

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"Invalid alert rule name: '{self.name}'")
        if not _METRIC_RE.match(self.metric):
            raise ValueError(
                f"Invalid metric name '{self.metric}' in alert rule '{self.name}'"
            )
        if self.comparator not in COMPARATORS:
            raise ValueError(
                f"Unsupported comparator '{self.comparator}' in alert rule "
                f"'{self.name}'; expected one of {sorted(COMPARATORS)}"
            )
        try:
            object.__setattr__(self, "severity", AlertSeverity(self.severity))
        except ValueError:
            raise ValueError(
                f"Invalid severity '{self.severity}' in alert rule '{self.name}'"
            ) from None
        try:
            object.__setattr__(self, "mode", AlertMode(self.mode))
        except ValueError:
            raise ValueError(
                f"Invalid mode '{self.mode}' in alert rule '{self.name}'"
            ) from None
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, int | float
        ):
            raise ValueError(f"Threshold must be numeric in alert rule '{self.name}'")
        if self.for_evaluations < 1:
            raise ValueError(
                f"for_evaluations must be >= 1 in alert rule '{self.name}'"
            )
        if not _WINDOW_RE.match(self.window):
            raise ValueError(
                f"Invalid window '{self.window}' in alert rule '{self.name}'"
            )
        if not self.summary.strip():
            raise ValueError(f"Alert rule '{self.name}' requires a summary")
        for label_name in self.labels:
            if not _LABEL_RE.match(label_name):
                raise ValueError(
                    f"Invalid label name '{label_name}' in alert rule '{self.name}'"
                )

    def is_breached(self, value: float) -> bool:
        """Return True when ``value`` violates this rule's threshold."""
        return COMPARATORS[self.comparator](value, self.threshold)


DEFAULT_ALERT_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        name="ServiceDown",
        metric="music_recommender_up",
        comparator="==",
        threshold=0,
        severity=AlertSeverity.CRITICAL,
        summary="Recommender service is unavailable (artifacts not loaded).",
    ),
    AlertRule(
        name="StreamingHealthDegraded",
        metric="music_recommender_streaming_health",
        labels={"status": "degraded"},
        comparator="==",
        threshold=1,
        severity=AlertSeverity.WARNING,
        summary="Streaming feedback pipeline health is degraded.",
        for_evaluations=2,
    ),
    AlertRule(
        name="StreamingHealthUnhealthy",
        metric="music_recommender_streaming_health",
        labels={"status": "unhealthy"},
        comparator="==",
        threshold=1,
        severity=AlertSeverity.CRITICAL,
        summary="Streaming feedback pipeline health is unhealthy.",
    ),
    AlertRule(
        name="StreamingQueueHighUtilization",
        metric="music_recommender_streaming_queue_utilization_ratio",
        comparator=">=",
        threshold=0.8,
        severity=AlertSeverity.WARNING,
        summary="Streaming feedback queue is above 80% utilization.",
        for_evaluations=3,
    ),
    AlertRule(
        name="StreamingRecordsDropped",
        metric="music_recommender_streaming_records_dropped_total",
        comparator=">",
        threshold=0,
        severity=AlertSeverity.CRITICAL,
        summary="Feedback records are being dropped by queue backpressure.",
        mode=AlertMode.INCREASE,
    ),
    AlertRule(
        name="StreamingFlushErrors",
        metric="music_recommender_streaming_flush_errors_total",
        comparator=">",
        threshold=0,
        severity=AlertSeverity.CRITICAL,
        summary="Streaming feedback batch flushes are failing.",
        mode=AlertMode.INCREASE,
    ),
    AlertRule(
        name="MaintenanceWorkerStopped",
        metric="music_recommender_maintenance_running",
        comparator="==",
        threshold=0,
        severity=AlertSeverity.WARNING,
        summary="Bandit maintenance worker is configured but not running.",
        for_evaluations=2,
    ),
    AlertRule(
        name="MaintenanceSweepErrors",
        metric="music_recommender_maintenance_errors_total",
        comparator=">",
        threshold=0,
        severity=AlertSeverity.WARNING,
        summary="Bandit maintenance sweeps are raising errors.",
        mode=AlertMode.INCREASE,
    ),
    AlertRule(
        name="BanditDriftViolation",
        metric="music_recommender_maintenance_drift_violations_total",
        comparator=">",
        threshold=0,
        severity=AlertSeverity.WARNING,
        summary="Bandit policy drift guardrail breach detected.",
        mode=AlertMode.INCREASE,
    ),
    AlertRule(
        name="BanditAutoRollback",
        metric="music_recommender_maintenance_rollbacks_total",
        comparator=">",
        threshold=0,
        severity=AlertSeverity.CRITICAL,
        summary="Bandit policy was automatically rolled back to the champion.",
        mode=AlertMode.INCREASE,
    ),
)
