"""Threshold alert rules over Prometheus metric families.

Rules are declared once and reused for in-process evaluation, webhook
notification dispatch, and generation of a Prometheus rule file.
"""

from __future__ import annotations

import operator
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from music_recommender.telemetry import MetricFamily

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


class AlertStatus(StrEnum):
    """Lifecycle status of a rule."""

    INACTIVE = "inactive"
    PENDING = "pending"
    FIRING = "firing"


@dataclass(frozen=True)
class AlertTransition:
    """A notification-worthy state change (firing or resolved)."""

    rule: AlertRule
    event: str
    value: float | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable notification payload."""
        return {
            "alert": self.rule.name,
            "event": self.event,
            "severity": str(self.rule.severity),
            "summary": self.rule.summary,
            "metric": self.rule.metric,
            "labels": dict(self.rule.labels),
            "threshold": self.rule.threshold,
            "comparator": self.rule.comparator,
            "value": self.value,
            "timestamp": self.timestamp,
        }


@dataclass
class AlertState:
    """Mutable evaluation state for one rule."""

    rule: AlertRule
    status: AlertStatus = AlertStatus.INACTIVE
    value: float | None = None
    consecutive_breaches: int = 0
    active_since: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the rule and its state."""
        return {
            "alert": self.rule.name,
            "status": str(self.status),
            "severity": str(self.rule.severity),
            "summary": self.rule.summary,
            "metric": self.rule.metric,
            "labels": dict(self.rule.labels),
            "comparator": self.rule.comparator,
            "threshold": self.rule.threshold,
            "mode": str(self.rule.mode),
            "for_evaluations": self.rule.for_evaluations,
            "value": self.value,
            "consecutive_breaches": self.consecutive_breaches,
            "active_since": self.active_since,
        }


def lookup_metric_value(
    families: Sequence[MetricFamily],
    metric: str,
    labels: Mapping[str, str] | None = None,
) -> float | None:
    """Sum the samples named ``metric`` whose labels include ``labels``.

    Returns ``None`` when no sample matches.
    """
    wanted = {k: str(v) for k, v in (labels or {}).items()}
    total: float | None = None
    for family in families:
        for sample in family.samples:
            if sample.name != metric:
                continue
            if any(str(sample.labels.get(k)) != v for k, v in wanted.items()):
                continue
            total = (total or 0.0) + float(sample.value)
    return total


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AlertEvaluator:
    """Thread-safe rule evaluator tracking pending/firing/resolved state."""

    def __init__(
        self,
        rules: Sequence[AlertRule] = DEFAULT_ALERT_RULES,
        *,
        clock: Callable[[], str] = _utc_now_iso,
    ) -> None:
        names = [r.name for r in rules]
        if len(names) != len(set(names)):
            raise ValueError("Alert rule names must be unique")
        self.rules = tuple(rules)
        self._clock = clock
        self._lock = threading.Lock()
        self._states = {r.name: AlertState(rule=r) for r in self.rules}
        self._previous: dict[str, float] = {}

    def _observe(self, rule: AlertRule, raw: float | None) -> float | None:
        """Derive the comparison value for ``rule`` from the raw sample."""
        if rule.mode is not AlertMode.INCREASE:
            return raw
        if raw is None:
            return None
        previous = self._previous.get(rule.name)
        self._previous[rule.name] = raw
        if previous is None:
            return None  # first observation is only a baseline
        return raw if raw < previous else raw - previous  # counter reset

    def evaluate(self, families: Sequence[MetricFamily]) -> list[AlertTransition]:
        """Evaluate every rule and return firing/resolved transitions."""
        transitions: list[AlertTransition] = []
        with self._lock:
            now = self._clock()
            for rule in self.rules:
                state = self._states[rule.name]
                raw = lookup_metric_value(families, rule.metric, rule.labels)
                value = self._observe(rule, raw)
                state.value = value
                if value is not None and rule.is_breached(value):
                    state.consecutive_breaches += 1
                    if state.consecutive_breaches >= rule.for_evaluations:
                        if state.status is not AlertStatus.FIRING:
                            state.status = AlertStatus.FIRING
                            state.active_since = now
                            transitions.append(
                                AlertTransition(rule, "firing", value, now)
                            )
                    else:
                        state.status = AlertStatus.PENDING
                else:
                    if state.status is AlertStatus.FIRING:
                        transitions.append(
                            AlertTransition(rule, "resolved", value, now)
                        )
                    state.status = AlertStatus.INACTIVE
                    state.consecutive_breaches = 0
                    state.active_since = None
        return transitions

    def states(self) -> list[dict[str, Any]]:
        """Return the current state of all rules."""
        with self._lock:
            return [self._states[r.name].to_dict() for r in self.rules]

    def firing(self) -> list[dict[str, Any]]:
        """Return only the states currently firing."""
        return [s for s in self.states() if s["status"] == AlertStatus.FIRING]
