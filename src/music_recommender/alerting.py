"""Threshold alert rules over Prometheus metric families.

Rules are declared once and reused for in-process evaluation, webhook
notification dispatch, and generation of a Prometheus rule file.
"""

from __future__ import annotations

import json
import logging
import operator
import re
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from music_recommender.telemetry import MetricFamily

logger = logging.getLogger("music_recommender.alerting")

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

    @property
    def is_firing(self) -> bool:
        """Return True if this transition represents a firing alert."""
        return self.event == "firing"

    @property
    def is_resolved(self) -> bool:
        """Return True if this transition represents a resolved alert."""
        return self.event == "resolved"

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


@dataclass(frozen=True)
class ReceivedAlertEvent:
    """An alert notification received by the webhook receiver."""

    alertname: str
    status: str
    severity: str
    summary: str
    channel: str | None = None
    metric: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    value: float | None = None
    starts_at: str | None = None
    ends_at: str | None = None
    received_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "alertname": self.alertname,
            "status": self.status,
            "severity": self.severity,
            "summary": self.summary,
            "channel": self.channel,
            "metric": self.metric,
            "labels": dict(self.labels),
            "value": self.value,
            "starts_at": self.starts_at,
            "ends_at": self.ends_at,
            "received_at": self.received_at,
        }


def parse_alert_payload(
    payload: Mapping[str, Any],
    channel: str | None = None,
) -> list[ReceivedAlertEvent]:
    """Parse Alertmanager v4 JSON or single AlertTransition payloads."""
    events: list[ReceivedAlertEvent] = []
    raw_alerts = payload.get("alerts")
    if isinstance(raw_alerts, list) and raw_alerts:
        common_labels = payload.get("commonLabels")
        common_annotations = payload.get("commonAnnotations")
        c_labels = common_labels if isinstance(common_labels, dict) else {}
        c_annos = common_annotations if isinstance(common_annotations, dict) else {}
        for item in raw_alerts:
            if not isinstance(item, dict):
                continue
            labels = item.get("labels")
            annos = item.get("annotations")
            i_labels = labels if isinstance(labels, dict) else {}
            i_annos = annos if isinstance(annos, dict) else {}
            merged_labels: dict[str, str] = {
                str(k): str(v) for k, v in {**c_labels, **i_labels}.items()
            }
            alertname = (
                i_labels.get("alertname")
                or item.get("alert")
                or c_labels.get("alertname")
                or "UnknownAlert"
            )
            status = str(item.get("status") or payload.get("status") or "firing")
            severity = str(
                merged_labels.get("severity")
                or (channel if channel in ("critical", "warning") else "warning")
            )
            summary = str(
                i_annos.get("summary")
                or c_annos.get("summary")
                or item.get("summary")
                or ""
            )
            metric = merged_labels.get("metric") or item.get("metric")
            raw_val = item.get("value")
            val: float | None = None
            if isinstance(raw_val, (int, float)):
                val = float(raw_val)
            starts_at = item.get("startsAt")
            ends_at = item.get("endsAt")
            events.append(
                ReceivedAlertEvent(
                    alertname=str(alertname),
                    status=status,
                    severity=severity,
                    summary=summary,
                    channel=channel,
                    metric=str(metric) if metric else None,
                    labels=merged_labels,
                    value=val,
                    starts_at=str(starts_at) if starts_at else None,
                    ends_at=str(ends_at) if ends_at else None,
                )
            )
    else:
        alertname = str(
            payload.get("alert") or payload.get("alertname") or "UnknownAlert"
        )
        status = str(payload.get("event") or payload.get("status") or "firing")
        raw_sev = payload.get("severity")
        severity = str(
            raw_sev or (channel if channel in ("critical", "warning") else "warning")
        )
        summary = str(payload.get("summary") or "")
        metric = payload.get("metric")
        raw_labels = payload.get("labels")
        single_labels: dict[str, str] = (
            {str(k): str(v) for k, v in raw_labels.items()}
            if isinstance(raw_labels, dict)
            else {}
        )
        raw_val = payload.get("value")
        val = float(raw_val) if isinstance(raw_val, (int, float)) else None
        starts_at = payload.get("timestamp")
        events.append(
            ReceivedAlertEvent(
                alertname=alertname,
                status=status,
                severity=severity,
                summary=summary,
                channel=channel,
                metric=str(metric) if metric else None,
                labels=single_labels,
                value=val,
                starts_at=str(starts_at) if starts_at else None,
            )
        )
    return events


@dataclass
class DeadLetterAlertRecord:
    """Dropped or unparseable alert payload captured for auditing."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    reason: str = "unparseable_payload"
    raw_payload: str = ""
    channel: str = "default"
    error_message: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    source: str | None = None
    retry_count: int = 0
    last_retry_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "reason": self.reason,
            "raw_payload": self.raw_payload,
            "channel": self.channel,
            "error_message": self.error_message,
            "headers": self.headers,
            "source": self.source,
            "retry_count": self.retry_count,
            "last_retry_at": self.last_retry_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeadLetterAlertRecord:
        return cls(
            timestamp=str(data.get("timestamp") or datetime.now(UTC).isoformat()),
            reason=str(data.get("reason") or "unspecified"),
            raw_payload=str(data.get("raw_payload") or ""),
            channel=str(data.get("channel") or "default"),
            error_message=str(data["error_message"])
            if data.get("error_message")
            else None,
            headers=dict(data.get("headers") or {}),
            source=str(data["source"]) if data.get("source") else None,
            retry_count=int(data.get("retry_count") or 0),
            last_retry_at=str(data["last_retry_at"])
            if data.get("last_retry_at")
            else None,
        )


class ThrottleDecision(StrEnum):
    """Decision made by the alert notification throttler."""

    ALLOWED = "allowed"
    SUPPRESSED_DUPLICATE = "suppressed_duplicate"
    RATE_LIMITED = "rate_limited"
    REPEAT_SUPPRESSED = "repeat_suppressed"


@dataclass
class AlertThrottleConfig:
    """Configuration for sliding-window rate limiting and deduplication."""

    dedup_window_seconds: float = 300.0
    max_alerts_per_window: int = 10
    window_seconds: float = 60.0
    repeat_interval_seconds: float = 1800.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dedup_window_seconds": self.dedup_window_seconds,
            "max_alerts_per_window": self.max_alerts_per_window,
            "window_seconds": self.window_seconds,
            "repeat_interval_seconds": self.repeat_interval_seconds,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlertThrottleConfig:
        return cls(
            dedup_window_seconds=float(data.get("dedup_window_seconds", 300.0)),
            max_alerts_per_window=int(data.get("max_alerts_per_window", 10)),
            window_seconds=float(data.get("window_seconds", 60.0)),
            repeat_interval_seconds=float(data.get("repeat_interval_seconds", 1800.0)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class AlertEscalationTier:
    """A single tier within an alert escalation policy."""

    level: int = 1
    delay_seconds: float = 0.0
    channels: list[str] = field(default_factory=lambda: ["default"])
    repeat: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "delay_seconds": self.delay_seconds,
            "channels": list(self.channels),
            "repeat": self.repeat,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlertEscalationTier:
        return cls(
            level=int(data.get("level", 1)),
            delay_seconds=float(data.get("delay_seconds", 0.0)),
            channels=list(data.get("channels", ["default"])),
            repeat=bool(data.get("repeat", False)),
        )


@dataclass
class AlertEscalationPolicy:
    """Escalation policy defining ordered tiers for alert notifications."""

    name: str = "default"
    tiers: list[AlertEscalationTier] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.tiers:
            self.tiers = [
                AlertEscalationTier(level=1, delay_seconds=0.0, channels=["default"])
            ]
        self.tiers.sort(key=lambda t: t.delay_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tiers": [t.to_dict() for t in self.tiers],
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlertEscalationPolicy:
        raw_tiers = data.get("tiers", [])
        tiers = [AlertEscalationTier.from_dict(t) for t in raw_tiers]
        return cls(
            name=str(data.get("name", "default")),
            tiers=tiers,
            enabled=bool(data.get("enabled", True)),
        )


class RemediationActionType(StrEnum):
    """Supported automated self-healing action types."""

    FLUSH_FEEDBACK_QUEUE = "flush_feedback_queue"
    RESTART_MAINTENANCE_WORKER = "restart_maintenance_worker"
    ROTATE_ALERT_STORAGE = "rotate_alert_storage"
    PURGE_DLQ = "purge_dlq"
    TRIGGER_ONLINE_FOLD = "trigger_online_fold"
    CUSTOM_CALLBACK = "custom_callback"


class RemediationExecutionStatus(StrEnum):
    """Result status of a remediation execution attempt."""

    EXECUTED = "executed"
    COOLDOWN_SUPPRESSED = "cooldown_suppressed"
    MAX_ATTEMPTS_EXCEEDED = "max_attempts_exceeded"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass
class RemediationRule:
    """Configuration mapping an alert rule to an automated remediation action."""

    rule_name: str
    action_type: RemediationActionType | str
    cooldown_seconds: float = 300.0
    max_attempts: int = 3
    enabled: bool = True
    trigger_on_severity: AlertSeverity | str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "action_type": str(self.action_type),
            "cooldown_seconds": self.cooldown_seconds,
            "max_attempts": self.max_attempts,
            "enabled": self.enabled,
            "trigger_on_severity": (
                str(self.trigger_on_severity) if self.trigger_on_severity else None
            ),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemediationRule:
        action = data.get("action_type", RemediationActionType.CUSTOM_CALLBACK)
        return cls(
            rule_name=str(data["rule_name"]),
            action_type=str(action),
            cooldown_seconds=float(data.get("cooldown_seconds", 300.0)),
            max_attempts=int(data.get("max_attempts", 3)),
            enabled=bool(data.get("enabled", True)),
            trigger_on_severity=data.get("trigger_on_severity"),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class RemediationActionResult:
    """Record of a single remediation execution attempt."""

    rule_name: str
    action_type: str
    status: RemediationExecutionStatus
    message: str
    timestamp: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "action_type": self.action_type,
            "status": str(self.status),
            "message": self.message,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "details": dict(self.details),
        }


class CircuitBreakerState(StrEnum):
    """Lifecycle state of an outbound circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for outbound notification circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_success_threshold: int = 2
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "half_open_success_threshold": self.half_open_success_threshold,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CircuitBreakerConfig:
        return cls(
            failure_threshold=int(data.get("failure_threshold", 5)),
            recovery_timeout_seconds=float(data.get("recovery_timeout_seconds", 30.0)),
            half_open_success_threshold=int(data.get("half_open_success_threshold", 2)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class AlertLogRotationConfig:
    """Configuration for size-based rotation and age retention of alert logs."""

    max_file_size_bytes: int = 10 * 1024 * 1024
    max_backups: int = 5
    retention_days: float = 30.0
    auto_rotate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_file_size_bytes": self.max_file_size_bytes,
            "max_backups": self.max_backups,
            "retention_days": self.retention_days,
            "auto_rotate": self.auto_rotate,
        }


@dataclass
class DLQRetryPolicy:
    """Exponential backoff retry policy for dead-letter alert queue records."""

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 30.0

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate exponential delay for given attempt count."""
        raw = self.base_delay_seconds * (self.backoff_multiplier**retry_count)
        return min(raw, self.max_delay_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "base_delay_seconds": self.base_delay_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "max_delay_seconds": self.max_delay_seconds,
        }


@dataclass
class DLQRetryResult:
    """Outcome of attempting to re-deliver or re-process a single DLQ record."""

    index: int
    channel: str
    reason: str
    success: bool
    retry_count: int
    error: str | None = None
    status_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "channel": self.channel,
            "reason": self.reason,
            "success": self.success,
            "retry_count": self.retry_count,
            "error": self.error,
            "status_code": self.status_code,
        }


@dataclass
class DLQRetryReport:
    """Aggregated report of DLQ retry operation."""

    total_scanned: int
    retried: int
    succeeded: int
    failed: int
    exhausted: int
    dry_run: bool
    results: list[DLQRetryResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_scanned": self.total_scanned,
            "retried": self.retried,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "exhausted": self.exhausted,
            "dry_run": self.dry_run,
            "results": [r.to_dict() for r in self.results],
        }


class AlertSpilloverStorage:
    """Thread-safe persistent JSONL storage for evicted alerts and DLQ records."""

    def __init__(
        self,
        storage_dir: Path | str,
        spillover_filename: str = "alerts_spillover.jsonl",
        dlq_filename: str = "alerts_dlq.jsonl",
        max_file_size_bytes: int = 10 * 1024 * 1024,
        rotation_config: AlertLogRotationConfig | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.spillover_path = self.storage_dir / spillover_filename
        self.dlq_path = self.storage_dir / dlq_filename
        self.rotation_config = rotation_config or AlertLogRotationConfig(
            max_file_size_bytes=max_file_size_bytes
        )
        self.max_file_size_bytes = self.rotation_config.max_file_size_bytes
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Could not create alert storage directory %s: %s",
                self.storage_dir,
                exc,
            )

    def _rotate_file_if_needed(self, target_path: Path) -> bool:
        """Rotate file if it exceeds size limit. Caller must hold self._lock."""
        if not self.rotation_config.auto_rotate or not target_path.exists():
            return False
        try:
            if target_path.stat().st_size >= self.rotation_config.max_file_size_bytes:
                return self._rotate_file_locked(
                    target_path, self.rotation_config.max_backups
                )
        except OSError:
            pass
        return False

    def _rotate_file_locked(self, target_path: Path, max_backups: int) -> bool:
        """Rotate target_path -> target_path.1, etc. Caller must hold self._lock."""
        if not target_path.exists() or max_backups <= 0:
            return False
        try:
            oldest = target_path.with_name(f"{target_path.name}.{max_backups}")
            if oldest.exists():
                oldest.unlink(missing_ok=True)
            for i in range(max_backups - 1, 0, -1):
                src = target_path.with_name(f"{target_path.name}.{i}")
                dst = target_path.with_name(f"{target_path.name}.{i + 1}")
                if src.exists():
                    src.replace(dst)
            first_backup = target_path.with_name(f"{target_path.name}.1")
            target_path.replace(first_backup)
            return True
        except OSError as exc:
            logger.warning("Failed rotating %s: %s", target_path, exc)
            return False

    def rotate_storage(self, max_backups: int | None = None) -> dict[str, bool]:
        """Manually rotate both spillover and DLQ files."""
        limit = (
            max_backups if max_backups is not None else self.rotation_config.max_backups
        )
        with self._lock:
            spill_rotated = self._rotate_file_locked(self.spillover_path, limit)
            dlq_rotated = self._rotate_file_locked(self.dlq_path, limit)
        return {
            "spillover_rotated": spill_rotated,
            "dlq_rotated": dlq_rotated,
        }

    def record_spillover(self, event: ReceivedAlertEvent) -> bool:
        """Append an evicted alert event to the spillover JSONL file."""
        return self._append_jsonl(self.spillover_path, event.to_dict())

    def record_dead_letter(self, record: DeadLetterAlertRecord) -> bool:
        """Append an unparseable or rejected alert to the DLQ JSONL file."""
        return self._append_jsonl(self.dlq_path, record.to_dict())

    def _append_jsonl(self, target_path: Path, item: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                self._rotate_file_if_needed(target_path)
                line = json.dumps(item, ensure_ascii=False) + "\n"
                with target_path.open("a", encoding="utf-8") as f:
                    f.write(line)
                return True
            except OSError as exc:
                logger.warning(
                    "Failed writing to alert storage %s: %s", target_path, exc
                )
                return False

    def read_spillover(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        status: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read recent spilled alert events from disk in reverse chronological order."""
        return self._read_jsonl(
            self.spillover_path,
            limit=limit,
            channel=channel,
            status=status,
            severity=severity,
        )

    def read_dead_letters(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read recent dead-letter records from disk in reverse chronological order."""
        return self._read_jsonl(self.dlq_path, limit=limit)

    def _read_jsonl(
        self,
        target_path: Path,
        limit: int = 50,
        channel: str | None = None,
        status: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        if not target_path.exists():
            return []
        with self._lock:
            try:
                with target_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError as exc:
                logger.warning(
                    "Failed reading from alert storage %s: %s", target_path, exc
                )
                return []

        results: list[dict[str, Any]] = []
        for line in reversed(lines):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                doc = json.loads(line_str)
            except Exception:
                continue
            if channel and doc.get("channel") != channel:
                continue
            if status and doc.get("status") != status:
                continue
            if severity and doc.get("severity") != severity:
                continue
            results.append(doc)
            if len(results) >= limit:
                break
        return results

    def purge_dead_letters(self) -> int:
        """Purge all records in the dead-letter queue, returning count purged."""
        with self._lock:
            if not self.dlq_path.exists():
                return 0
            try:
                with self.dlq_path.open("r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
                self.dlq_path.write_text("", encoding="utf-8")
                return count
            except OSError as exc:
                logger.warning("Failed purging DLQ at %s: %s", self.dlq_path, exc)
                return 0

    def retry_dead_letters(
        self,
        policy: DLQRetryPolicy | None = None,
        target_url: str | None = None,
        channel: str | None = None,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> DLQRetryReport:
        """Retry delivery or reprocessing of dead-letter alert queue records."""
        pol = policy or DLQRetryPolicy()
        with self._lock:
            if not self.dlq_path.exists():
                return DLQRetryReport(
                    total_scanned=0,
                    retried=0,
                    succeeded=0,
                    failed=0,
                    exhausted=0,
                    dry_run=dry_run,
                )

            try:
                with self.dlq_path.open("r", encoding="utf-8") as f:
                    raw_lines = f.readlines()
            except OSError as exc:
                logger.warning(
                    "Failed reading DLQ for retry %s: %s", self.dlq_path, exc
                )
                return DLQRetryReport(
                    total_scanned=0,
                    retried=0,
                    succeeded=0,
                    failed=0,
                    exhausted=0,
                    dry_run=dry_run,
                )

            records: list[DeadLetterAlertRecord] = []
            for line in raw_lines:
                s = line.strip()
                if not s:
                    continue
                try:
                    data = json.loads(s)
                    records.append(DeadLetterAlertRecord.from_dict(data))
                except Exception:
                    continue

            results: list[DLQRetryResult] = []
            retained_records: list[DeadLetterAlertRecord] = []
            retried = 0
            succeeded = 0
            failed = 0
            exhausted = 0

            for idx, rec in enumerate(records):
                if channel and rec.channel != channel:
                    retained_records.append(rec)
                    continue
                if reason and rec.reason != reason:
                    retained_records.append(rec)
                    continue
                if rec.retry_count >= pol.max_retries:
                    exhausted += 1
                    retained_records.append(rec)
                    results.append(
                        DLQRetryResult(
                            index=idx,
                            channel=rec.channel,
                            reason=rec.reason,
                            success=False,
                            retry_count=rec.retry_count,
                            error=f"Max retries ({pol.max_retries}) exhausted",
                        )
                    )
                    continue

                retried += 1
                if dry_run:
                    succeeded += 1
                    results.append(
                        DLQRetryResult(
                            index=idx,
                            channel=rec.channel,
                            reason=rec.reason,
                            success=True,
                            retry_count=rec.retry_count + 1,
                            status_code=200,
                        )
                    )
                    retained_records.append(rec)
                else:
                    url = target_url or "http://localhost:8000/alerts/webhook"
                    success, code, err = self._dispatch_retry_record(url, rec)
                    if success:
                        succeeded += 1
                        results.append(
                            DLQRetryResult(
                                index=idx,
                                channel=rec.channel,
                                reason=rec.reason,
                                success=True,
                                retry_count=rec.retry_count + 1,
                                status_code=code,
                            )
                        )
                    else:
                        failed += 1
                        rec.retry_count += 1
                        rec.last_retry_at = datetime.now(UTC).isoformat()
                        rec.error_message = err
                        retained_records.append(rec)
                        results.append(
                            DLQRetryResult(
                                index=idx,
                                channel=rec.channel,
                                reason=rec.reason,
                                success=False,
                                retry_count=rec.retry_count,
                                status_code=code,
                                error=err,
                            )
                        )

            if not dry_run:
                try:
                    with self.dlq_path.open("w", encoding="utf-8") as f:
                        for r in retained_records:
                            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
                except OSError as exc:
                    logger.warning(
                        "Failed writing updated DLQ %s: %s", self.dlq_path, exc
                    )

            return DLQRetryReport(
                total_scanned=len(records),
                retried=retried,
                succeeded=succeeded,
                failed=failed,
                exhausted=exhausted,
                dry_run=dry_run,
                results=results,
            )

    def _dispatch_retry_record(
        self, url: str, rec: DeadLetterAlertRecord
    ) -> tuple[bool, int | None, str | None]:
        """Dispatch a single dead-letter record to a target webhook URL."""
        payload_str = rec.raw_payload.strip() if rec.raw_payload else ""
        if payload_str.startswith("{") or payload_str.startswith("["):
            payload_bytes = payload_str.encode("utf-8")
        else:
            payload_bytes = json.dumps(rec.to_dict()).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "MusicRecommender-DLQRetry/1.0",
                **rec.headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                status_code = getattr(resp, "status", 200)
                return True, int(status_code), None
        except urllib.error.HTTPError as exc:
            return False, exc.code, f"HTTP {exc.code}: {exc.reason}"
        except Exception as exc:
            return False, None, str(exc)

    def purge_spillover(self) -> int:
        """Purge all records in the spillover file, returning count purged."""
        with self._lock:
            if not self.spillover_path.exists():
                return 0
            try:
                with self.spillover_path.open("r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
                self.spillover_path.write_text("", encoding="utf-8")
                return count
            except OSError as exc:
                logger.warning(
                    "Failed purging spillover at %s: %s", self.spillover_path, exc
                )
                return 0

    def enforce_retention(
        self,
        max_age_days: float | None = None,
        max_backups: int | None = None,
    ) -> dict[str, int]:
        """Prune outdated rotated files and events older than retention cutoff."""
        age_limit = (
            max_age_days
            if max_age_days is not None
            else self.rotation_config.retention_days
        )
        backups_limit = (
            max_backups if max_backups is not None else self.rotation_config.max_backups
        )
        cutoff_epoch = time.time() - (age_limit * 86400.0)

        pruned_files = 0
        with self._lock:
            for base_path in (self.spillover_path, self.dlq_path):
                parent = base_path.parent
                prefix = base_path.name + "."
                if parent.exists():
                    for entry in parent.iterdir():
                        if entry.name.startswith(prefix):
                            suffix = entry.name[len(prefix) :]
                            if suffix.isdigit():
                                idx = int(suffix)
                                should_delete = idx > backups_limit
                                if (
                                    not should_delete
                                    and entry.stat().st_mtime < cutoff_epoch
                                ):
                                    should_delete = True
                                if should_delete:
                                    try:
                                        entry.unlink(missing_ok=True)
                                        pruned_files += 1
                                    except OSError as exc:
                                        logger.warning(
                                            "Failed pruning rotated file %s: %s",
                                            entry,
                                            exc,
                                        )
        return {
            "pruned_files": pruned_files,
        }

    def create_archive_bundle(
        self,
        output_path: Path | str,
        format: str = "tar.gz",
        include_spillover: bool = True,
        include_dlq: bool = True,
    ) -> Path:
        """Create a compressed snapshot archive of alert spillover and DLQ files."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fmt = format.lower().strip()
        if fmt not in ("tar.gz", "tgz", "zip"):
            raise ValueError(
                f"Unsupported archive format: {format}. Must be 'tar.gz' or 'zip'."
            )

        with self._lock:
            files_to_archive: list[Path] = []
            if include_spillover:
                if self.spillover_path.exists():
                    files_to_archive.append(self.spillover_path)
                files_to_archive.extend(
                    sorted(self.storage_dir.glob(f"{self.spillover_path.name}.*"))
                )
            if include_dlq:
                if self.dlq_path.exists():
                    files_to_archive.append(self.dlq_path)
                files_to_archive.extend(
                    sorted(self.storage_dir.glob(f"{self.dlq_path.name}.*"))
                )

            manifest = {
                "version": "1.0",
                "created_at": datetime.now(UTC).isoformat(),
                "storage_dir": str(self.storage_dir),
                "file_count": len(files_to_archive),
                "files": [f.name for f in files_to_archive],
            }
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

            if fmt in ("tar.gz", "tgz"):
                import io

                with tarfile.open(out, "w:gz") as tar:
                    ti = tarfile.TarInfo(name="manifest.json")
                    ti.size = len(manifest_bytes)
                    ti.mtime = int(time.time())
                    tar.addfile(ti, io.BytesIO(manifest_bytes))
                    for f in files_to_archive:
                        tar.add(f, arcname=f.name)
            else:
                with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("manifest.json", manifest_bytes)
                    for f in files_to_archive:
                        zf.write(f, arcname=f.name)
        return out

    def restore_archive_bundle(
        self,
        archive_path: Path | str,
        target_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Extract an alert archive bundle into the target directory."""
        arch = Path(archive_path)
        if not arch.exists():
            raise FileNotFoundError(f"Archive bundle not found: {arch}")
        dest = Path(target_dir) if target_dir is not None else self.storage_dir
        dest.mkdir(parents=True, exist_ok=True)

        extracted_files: list[str] = []
        manifest: dict[str, Any] = {}

        if tarfile.is_tarfile(arch):
            with tarfile.open(arch, "r:*") as tar:
                manifest_member = (
                    tar.getmember("manifest.json")
                    if "manifest.json" in tar.getnames()
                    else None
                )
                if manifest_member is not None:
                    mf = tar.extractfile(manifest_member)
                    if mf is not None:
                        manifest = json.loads(mf.read().decode("utf-8"))
                for member in tar.getmembers():
                    if member.name != "manifest.json":
                        tar.extract(member, path=dest)
                        extracted_files.append(member.name)
        elif zipfile.is_zipfile(arch):
            with zipfile.ZipFile(arch, "r") as zf:
                if "manifest.json" in zf.namelist():
                    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                for name in zf.namelist():
                    if name != "manifest.json":
                        zf.extract(name, path=dest)
                        extracted_files.append(name)
        else:
            raise ValueError(f"Unknown archive format: {arch}")

        return {
            "status": "extracted",
            "target_dir": str(dest),
            "manifest": manifest,
            "extracted_files": extracted_files,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return counts and file sizes for spillover and DLQ."""
        with self._lock:
            spill_size = (
                self.spillover_path.stat().st_size
                if self.spillover_path.exists()
                else 0
            )
            dlq_size = self.dlq_path.stat().st_size if self.dlq_path.exists() else 0
            spill_count = 0
            if self.spillover_path.exists():
                try:
                    with self.spillover_path.open("r", encoding="utf-8") as f:
                        spill_count = sum(1 for line in f if line.strip())
                except OSError:
                    pass
            dlq_count = 0
            if self.dlq_path.exists():
                try:
                    with self.dlq_path.open("r", encoding="utf-8") as f:
                        dlq_count = sum(1 for line in f if line.strip())
                except OSError:
                    pass
            spill_rotations = (
                len(list(self.storage_dir.glob(f"{self.spillover_path.name}.*")))
                if self.storage_dir.exists()
                else 0
            )
            dlq_rotations = (
                len(list(self.storage_dir.glob(f"{self.dlq_path.name}.*")))
                if self.storage_dir.exists()
                else 0
            )
        return {
            "spillover_count": spill_count,
            "spillover_bytes": spill_size,
            "spillover_rotations": spill_rotations,
            "dlq_count": dlq_count,
            "dlq_bytes": dlq_size,
            "dlq_rotations": dlq_rotations,
            "storage_dir": str(self.storage_dir),
            "max_file_size_bytes": self.rotation_config.max_file_size_bytes,
            "max_backups": self.rotation_config.max_backups,
            "retention_days": self.rotation_config.retention_days,
        }


@dataclass
class ReplayItemResult:
    """Result of replaying an individual alert notification."""

    index: int
    alertname: str
    status: str
    success: bool
    status_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "alertname": self.alertname,
            "status": self.status,
            "success": self.success,
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass
class ReplayReport:
    """Aggregated outcome of an alert replay sequence."""

    total_events: int
    dispatched: int
    failed: int
    dry_run: bool
    target_url: str | None
    results: list[ReplayItemResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "dispatched": self.dispatched,
            "failed": self.failed,
            "dry_run": self.dry_run,
            "target_url": self.target_url,
            "results": [r.to_dict() for r in self.results],
            "duration_seconds": round(self.duration_seconds, 4),
        }


class AlertReplayEngine:
    """Loads past alert events or DLQ records and dispatches them sequentially."""

    def __init__(
        self,
        default_target_url: str = "http://localhost:8000/alerts/webhook",
        timeout_seconds: float = 5.0,
    ) -> None:
        self.default_target_url = default_target_url
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def load_events_from_file(file_path: Path | str) -> list[dict[str, Any]]:
        """Load alert records from a JSON or JSONL file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Alert replay file not found: {p}")

        content = p.read_text(encoding="utf-8").strip()
        if not content:
            return []

        if content.startswith("["):
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
            except Exception:
                pass

        events: list[dict[str, Any]] = []
        for line in content.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                doc = json.loads(s)
                if isinstance(doc, dict):
                    events.append(doc)
            except Exception:
                continue
        return events

    def replay(
        self,
        events: Sequence[dict[str, Any]],
        *,
        target_url: str | None = None,
        dry_run: bool = False,
        interval_seconds: float = 0.0,
    ) -> ReplayReport:
        """Replay a sequence of alert dictionaries with optional delays."""
        url = target_url or self.default_target_url
        results: list[ReplayItemResult] = []
        dispatched = 0
        failed = 0
        start_time = time.monotonic()

        for idx, item in enumerate(events):
            alertname = str(
                item.get("alertname") or item.get("alert") or f"Alert_{idx}"
            )
            status = str(item.get("status") or item.get("event") or "firing")

            if dry_run:
                results.append(
                    ReplayItemResult(
                        index=idx,
                        alertname=alertname,
                        status=status,
                        success=True,
                        status_code=200,
                        error=None,
                    )
                )
                dispatched += 1
            else:
                success, code, err = self._dispatch_one(url, item)
                if success:
                    dispatched += 1
                else:
                    failed += 1
                results.append(
                    ReplayItemResult(
                        index=idx,
                        alertname=alertname,
                        status=status,
                        success=success,
                        status_code=code,
                        error=err,
                    )
                )

            if interval_seconds > 0 and idx < len(events) - 1:
                time.sleep(interval_seconds)

        elapsed = time.monotonic() - start_time
        return ReplayReport(
            total_events=len(events),
            dispatched=dispatched,
            failed=failed,
            dry_run=dry_run,
            target_url=url,
            results=results,
            duration_seconds=elapsed,
        )

    def _dispatch_one(
        self, url: str, item: dict[str, Any]
    ) -> tuple[bool, int | None, str | None]:
        payload_bytes = json.dumps(item).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "MusicRecommender-AlertReplayEngine/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                code = int(resp.status)
                return (200 <= code < 300, code, None)
        except urllib.error.HTTPError as exc:
            return (False, exc.code, str(exc.reason))
        except Exception as exc:
            return (False, None, str(exc))


class AlertHistoryBuffer:
    """Thread-safe, bounded in-memory ring buffer of received alert events."""

    def __init__(
        self,
        capacity: int = 100,
        *,
        spillover_storage: AlertSpilloverStorage | None = None,
        spill_on_evict: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.spillover_storage = spillover_storage
        self.spill_on_evict = spill_on_evict
        self._events: list[ReceivedAlertEvent] = []
        self._lock = threading.Lock()

    def record(self, event: ReceivedAlertEvent) -> None:
        """Record a single alert event, evicting the oldest if at capacity."""
        evicted_event: ReceivedAlertEvent | None = None
        with self._lock:
            if len(self._events) >= self.capacity:
                evicted_event = self._events.pop(0)
            self._events.append(event)

        if (
            evicted_event is not None
            and self.spill_on_evict
            and self.spillover_storage is not None
        ):
            self.spillover_storage.record_spillover(evicted_event)

    def record_many(self, events: Sequence[ReceivedAlertEvent]) -> int:
        """Record multiple alert events, returning the number recorded."""
        count = 0
        for event in events:
            self.record(event)
            count += 1
        return count

    def get_events(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        status: str | None = None,
        severity: str | None = None,
    ) -> list[ReceivedAlertEvent]:
        """Return recorded events matching criteria, newest first."""
        with self._lock:
            matching = list(self._events)

        if channel:
            matching = [e for e in matching if e.channel == channel]
        if status:
            matching = [e for e in matching if e.status == status.lower()]
        if severity:
            matching = [e for e in matching if e.severity == severity.lower()]

        matching.reverse()
        return matching[: max(1, limit)]

    def clear(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()

    def get_combined_events(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        status: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return combined events from in-memory buffer and persistent spillover."""
        memory_events = [
            e.to_dict()
            for e in self.get_events(
                limit=limit,
                channel=channel,
                status=status,
                severity=severity,
            )
        ]
        if len(memory_events) >= limit or self.spillover_storage is None:
            return memory_events[:limit]

        remaining = limit - len(memory_events)
        spillover_events = self.spillover_storage.read_spillover(
            limit=remaining,
            channel=channel,
            status=status,
            severity=severity,
        )
        return memory_events + spillover_events

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


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


class AlertNotificationThrottler:
    """Thread-safe rate-limiter and deduplicator for alert notifications."""

    def __init__(self, config: AlertThrottleConfig | None = None) -> None:
        self.config = config or AlertThrottleConfig()
        self._lock = threading.Lock()
        self._last_transitions: dict[tuple[str, str], float] = {}
        self._sliding_timestamps: dict[str, list[float]] = {}
        self._last_firing_dispatch: dict[str, float] = {}

        self._total_evaluated: int = 0
        self._allowed_count: int = 0
        self._suppressed_duplicate_count: int = 0
        self._rate_limited_count: int = 0
        self._repeat_suppressed_count: int = 0

    def should_dispatch(
        self,
        transition: AlertTransition,
        now: float | None = None,
    ) -> tuple[bool, ThrottleDecision, str]:
        """Determine whether an alert transition should be dispatched or throttled."""
        if not self.config.enabled:
            return True, ThrottleDecision.ALLOWED, "Throttling disabled"

        ts = now if now is not None else time.time()
        rule_name = transition.rule.name
        state_key = (rule_name, transition.event)

        with self._lock:
            self._total_evaluated += 1

            # 1. Exact duplicate transition check
            last_ts = self._last_transitions.get(state_key)
            if (
                last_ts is not None
                and (ts - last_ts) < self.config.dedup_window_seconds
            ):
                self._suppressed_duplicate_count += 1
                return (
                    False,
                    ThrottleDecision.SUPPRESSED_DUPLICATE,
                    (
                        f"Duplicate transition for {rule_name} within "
                        f"{self.config.dedup_window_seconds}s window"
                    ),
                )

            # 2. Ongoing firing repeat interval check
            if transition.is_firing:
                last_firing = self._last_firing_dispatch.get(rule_name)
                if (
                    last_firing is not None
                    and (ts - last_firing) < self.config.repeat_interval_seconds
                ):
                    self._repeat_suppressed_count += 1
                    return (
                        False,
                        ThrottleDecision.REPEAT_SUPPRESSED,
                        (
                            f"Ongoing firing alert {rule_name} suppressed "
                            f"until {self.config.repeat_interval_seconds}s interval"
                        ),
                    )

            # 3. Sliding-window rate limit
            cutoff = ts - self.config.window_seconds
            timestamps = self._sliding_timestamps.setdefault(rule_name, [])
            self._sliding_timestamps[rule_name] = [t for t in timestamps if t >= cutoff]
            if (
                len(self._sliding_timestamps[rule_name])
                >= self.config.max_alerts_per_window
            ):
                self._rate_limited_count += 1
                return (
                    False,
                    ThrottleDecision.RATE_LIMITED,
                    (
                        f"Rate limit exceeded: {self.config.max_alerts_per_window} "
                        f"in {self.config.window_seconds}s"
                    ),
                )

            # Allowed - record timestamp
            self._last_transitions[state_key] = ts
            self._sliding_timestamps[rule_name].append(ts)
            if transition.is_firing:
                self._last_firing_dispatch[rule_name] = ts
            self._allowed_count += 1
            return True, ThrottleDecision.ALLOWED, "Allowed"

    def get_stats(self) -> dict[str, Any]:
        """Return throttling and suppression metrics."""
        with self._lock:
            return {
                "total_evaluated": self._total_evaluated,
                "allowed_count": self._allowed_count,
                "suppressed_duplicate_count": self._suppressed_duplicate_count,
                "rate_limited_count": self._rate_limited_count,
                "repeat_suppressed_count": self._repeat_suppressed_count,
                "tracked_rules": len(self._sliding_timestamps),
                "config": self.config.to_dict(),
            }

    def reset(self) -> None:
        """Clear all in-memory throttling state."""
        with self._lock:
            self._last_transitions.clear()
            self._sliding_timestamps.clear()
            self._last_firing_dispatch.clear()
            self._total_evaluated = 0
            self._allowed_count = 0
            self._suppressed_duplicate_count = 0
            self._rate_limited_count = 0
            self._repeat_suppressed_count = 0


@dataclass(frozen=True)
class EscalationEvent:
    """An escalation event triggered when an alert reaches a new tier."""

    rule_name: str
    tier_level: int
    delay_seconds: float
    channels: list[str]
    elapsed_seconds: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "tier_level": self.tier_level,
            "delay_seconds": self.delay_seconds,
            "channels": list(self.channels),
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp": self.timestamp,
        }


class AlertEscalationEngine:
    """Tracks firing duration and triggers escalation tiers across channels."""

    def __init__(
        self,
        policies: dict[str, AlertEscalationPolicy] | None = None,
        default_policy: AlertEscalationPolicy | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._policies: dict[str, AlertEscalationPolicy] = dict(policies or {})
        self._default_policy: AlertEscalationPolicy = (
            default_policy or AlertEscalationPolicy(name="default")
        )
        self._firing_start_times: dict[str, float] = {}
        self._notified_tiers: dict[str, set[int]] = {}
        self._escalation_history: list[EscalationEvent] = []

    def register_policy(self, rule_name: str, policy: AlertEscalationPolicy) -> None:
        """Assign an escalation policy to an alert rule name."""
        with self._lock:
            self._policies[rule_name] = policy

    def get_policy(self, rule_name: str) -> AlertEscalationPolicy:
        """Return the escalation policy for a rule, falling back to default."""
        with self._lock:
            return self._policies.get(rule_name, self._default_policy)

    def evaluate_transition(
        self,
        transition: AlertTransition,
        now: float | None = None,
    ) -> list[EscalationEvent]:
        """Evaluate an alert transition and return newly triggered escalation events."""
        ts = now if now is not None else time.time()
        rule_name = transition.rule.name
        events: list[EscalationEvent] = []

        with self._lock:
            if transition.is_resolved:
                self._firing_start_times.pop(rule_name, None)
                self._notified_tiers.pop(rule_name, None)
                return events

            if not transition.is_firing:
                return events

            policy = self._policies.get(rule_name, self._default_policy)
            if not policy.enabled:
                return events

            if rule_name not in self._firing_start_times:
                self._firing_start_times[rule_name] = ts
                self._notified_tiers[rule_name] = set()

            start_ts = self._firing_start_times[rule_name]
            elapsed = ts - start_ts
            notified = self._notified_tiers.setdefault(rule_name, set())

            iso_now = datetime.fromtimestamp(ts, tz=UTC).isoformat()
            for tier in policy.tiers:
                if elapsed >= tier.delay_seconds and (
                    tier.level not in notified or tier.repeat
                ):
                    notified.add(tier.level)
                    ev = EscalationEvent(
                        rule_name=rule_name,
                        tier_level=tier.level,
                        delay_seconds=tier.delay_seconds,
                        channels=list(tier.channels),
                        elapsed_seconds=elapsed,
                        timestamp=iso_now,
                    )
                    events.append(ev)
                    self._escalation_history.append(ev)

            return events

    def get_active_firing_durations(self, now: float | None = None) -> dict[str, float]:
        """Return rule name -> active firing duration in seconds."""
        ts = now if now is not None else time.time()
        with self._lock:
            return {
                name: max(0.0, ts - start)
                for name, start in self._firing_start_times.items()
            }

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent escalation event history."""
        with self._lock:
            return [ev.to_dict() for ev in reversed(self._escalation_history[-limit:])]

    def reset(self) -> None:
        """Clear all active firing state and history."""
        with self._lock:
            self._firing_start_times.clear()
            self._notified_tiers.clear()
            self._escalation_history.clear()


RemediationHandler = Callable[
    [AlertTransition | None, dict[str, Any]],
    tuple[bool, str, dict[str, Any]],
]


class IncidentRemediationRunner:
    """Automated self-healing runner executing targeted actions on alert triggers."""

    def __init__(
        self,
        rules: list[RemediationRule] | None = None,
        handlers: dict[str, RemediationHandler] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._rules: dict[str, list[RemediationRule]] = {}
        for r in rules or []:
            self.register_rule(r)
        self._handlers: dict[str, RemediationHandler] = dict(handlers or {})
        self._last_executed: dict[tuple[str, str], float] = {}
        self._attempt_counts: dict[tuple[str, str], int] = {}
        self._history: list[RemediationActionResult] = []

    def register_rule(self, rule: RemediationRule) -> None:
        """Register a remediation rule for an alert rule name."""
        with self._lock:
            self._rules.setdefault(rule.rule_name, []).append(rule)

    def register_handler(
        self,
        action_type: RemediationActionType | str,
        handler: RemediationHandler,
    ) -> None:
        """Register an execution callback handler for an action type."""
        with self._lock:
            self._handlers[str(action_type)] = handler

    def handle_transition(
        self,
        transition: AlertTransition,
        now: float | None = None,
        dry_run: bool = False,
    ) -> list[RemediationActionResult]:
        """Evaluate an alert transition and execute any matching remediation rules."""
        ts = now if now is not None else time.time()
        rule_name = transition.rule.name
        results: list[RemediationActionResult] = []

        with self._lock:
            if transition.is_resolved:
                for key in list(self._attempt_counts.keys()):
                    if key[0] == rule_name:
                        self._attempt_counts[key] = 0
                return results

            if not transition.is_firing:
                return results

            matching_rules = self._rules.get(rule_name, [])
            for rule in matching_rules:
                if not rule.enabled:
                    continue
                if (
                    rule.trigger_on_severity
                    and str(rule.trigger_on_severity).lower()
                    != str(transition.rule.severity).lower()
                ):
                    continue

                act_type = str(rule.action_type)
                key = (rule_name, act_type)
                last_run = self._last_executed.get(key)
                if last_run is not None and (ts - last_run) < rule.cooldown_seconds:
                    res = RemediationActionResult(
                        rule_name=rule_name,
                        action_type=act_type,
                        status=RemediationExecutionStatus.COOLDOWN_SUPPRESSED,
                        message=(
                            f"Action '{act_type}' suppressed by cooldown "
                            f"({rule.cooldown_seconds}s)"
                        ),
                        timestamp=datetime.fromtimestamp(ts, tz=UTC).isoformat(),
                    )
                    results.append(res)
                    self._history.append(res)
                    continue

                attempts = self._attempt_counts.get(key, 0)
                if attempts >= rule.max_attempts:
                    res = RemediationActionResult(
                        rule_name=rule_name,
                        action_type=act_type,
                        status=RemediationExecutionStatus.MAX_ATTEMPTS_EXCEEDED,
                        message=(
                            f"Action '{act_type}' exceeded max attempts "
                            f"({rule.max_attempts})"
                        ),
                        timestamp=datetime.fromtimestamp(ts, tz=UTC).isoformat(),
                    )
                    results.append(res)
                    self._history.append(res)
                    continue

                handler = self._handlers.get(act_type)
                if handler is None:
                    res = RemediationActionResult(
                        rule_name=rule_name,
                        action_type=act_type,
                        status=RemediationExecutionStatus.FAILED,
                        message=f"No handler registered for action '{act_type}'",
                        timestamp=datetime.fromtimestamp(ts, tz=UTC).isoformat(),
                    )
                    results.append(res)
                    self._history.append(res)
                    continue

                if dry_run:
                    res = RemediationActionResult(
                        rule_name=rule_name,
                        action_type=act_type,
                        status=RemediationExecutionStatus.DRY_RUN,
                        message=(f"Dry run: action '{act_type}' would be executed"),
                        timestamp=datetime.fromtimestamp(ts, tz=UTC).isoformat(),
                        details=dict(rule.parameters),
                    )
                    results.append(res)
                    self._history.append(res)
                    continue

                t0 = time.perf_counter()
                try:
                    success, msg, details = handler(transition, rule.parameters)
                    dur_ms = (time.perf_counter() - t0) * 1000.0
                    status = (
                        RemediationExecutionStatus.EXECUTED
                        if success
                        else RemediationExecutionStatus.FAILED
                    )
                    res = RemediationActionResult(
                        rule_name=rule_name,
                        action_type=act_type,
                        status=status,
                        message=msg,
                        timestamp=datetime.fromtimestamp(ts, tz=UTC).isoformat(),
                        duration_ms=dur_ms,
                        details=details,
                    )
                except Exception as exc:
                    dur_ms = (time.perf_counter() - t0) * 1000.0
                    res = RemediationActionResult(
                        rule_name=rule_name,
                        action_type=act_type,
                        status=RemediationExecutionStatus.FAILED,
                        message=f"Handler exception: {exc}",
                        timestamp=datetime.fromtimestamp(ts, tz=UTC).isoformat(),
                        duration_ms=dur_ms,
                    )

                self._last_executed[key] = ts
                self._attempt_counts[key] = attempts + 1
                results.append(res)
                self._history.append(res)

        return results

    def execute_action(
        self,
        rule_name: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> RemediationActionResult:
        """Trigger a remediation action manually on demand."""
        ts = time.time()
        iso_ts = datetime.fromtimestamp(ts, tz=UTC).isoformat()
        params = parameters or {}

        with self._lock:
            handler = self._handlers.get(action_type)
            if handler is None:
                res = RemediationActionResult(
                    rule_name=rule_name,
                    action_type=action_type,
                    status=RemediationExecutionStatus.FAILED,
                    message=f"No handler registered for action '{action_type}'",
                    timestamp=iso_ts,
                )
                self._history.append(res)
                return res

            if dry_run:
                res = RemediationActionResult(
                    rule_name=rule_name,
                    action_type=action_type,
                    status=RemediationExecutionStatus.DRY_RUN,
                    message=f"Dry run: action '{action_type}' would be executed",
                    timestamp=iso_ts,
                    details=params,
                )
                self._history.append(res)
                return res

            t0 = time.perf_counter()
            try:
                success, msg, details = handler(None, params)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                status = (
                    RemediationExecutionStatus.EXECUTED
                    if success
                    else RemediationExecutionStatus.FAILED
                )
                res = RemediationActionResult(
                    rule_name=rule_name,
                    action_type=action_type,
                    status=status,
                    message=msg,
                    timestamp=iso_ts,
                    duration_ms=dur_ms,
                    details=details,
                )
            except Exception as exc:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                res = RemediationActionResult(
                    rule_name=rule_name,
                    action_type=action_type,
                    status=RemediationExecutionStatus.FAILED,
                    message=f"Handler exception: {exc}",
                    timestamp=iso_ts,
                    duration_ms=dur_ms,
                )

            key = (rule_name, action_type)
            self._last_executed[key] = ts
            self._attempt_counts[key] = self._attempt_counts.get(key, 0) + 1
            self._history.append(res)
            return res

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent remediation execution history."""
        with self._lock:
            return [res.to_dict() for res in reversed(self._history[-limit:])]

    def get_rules(self) -> list[dict[str, Any]]:
        """Return all registered remediation rules."""
        with self._lock:
            flat_rules: list[dict[str, Any]] = []
            for rule_list in self._rules.values():
                for r in rule_list:
                    flat_rules.append(r.to_dict())
            return flat_rules

    def get_stats(self) -> dict[str, Any]:
        """Return remediation metrics and counts."""
        with self._lock:
            executed = sum(
                1
                for r in self._history
                if r.status == RemediationExecutionStatus.EXECUTED
            )
            failed = sum(
                1
                for r in self._history
                if r.status == RemediationExecutionStatus.FAILED
            )
            suppressed = sum(
                1
                for r in self._history
                if r.status == RemediationExecutionStatus.COOLDOWN_SUPPRESSED
            )
            return {
                "total_runs": len(self._history),
                "executed_count": executed,
                "failed_count": failed,
                "suppressed_count": suppressed,
                "registered_rules": sum(len(rl) for rl in self._rules.values()),
                "registered_handlers": sorted(self._handlers.keys()),
            }

    def reset(self) -> None:
        """Clear all execution tracking, cooldowns, and history."""
        with self._lock:
            self._last_executed.clear()
            self._attempt_counts.clear()
            self._history.clear()


class CircuitBreaker:
    """Thread-safe circuit breaker protecting outbound notification dispatchers."""

    def __init__(
        self,
        name: str = "default",
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.Lock()
        self._state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0
        self._last_state_change: float = time.time()
        self._total_trips: int = 0
        self._total_rejected: int = 0
        self._total_calls: int = 0

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._state

    def allow_request(self, now: float | None = None) -> tuple[bool, str]:
        """Check whether an outbound call is allowed through the circuit breaker."""
        if not self.config.enabled:
            return True, "Circuit breaker disabled"

        ts = now if now is not None else time.time()
        with self._lock:
            self._total_calls += 1

            if self._state == CircuitBreakerState.CLOSED:
                return True, "Circuit closed (healthy)"

            if self._state == CircuitBreakerState.OPEN:
                elapsed = ts - self._last_state_change
                if elapsed >= self.config.recovery_timeout_seconds:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._consecutive_successes = 0
                    self._last_state_change = ts
                    logger.info(
                        "Circuit breaker '%s' entering HALF_OPEN "
                        "probe state after %.1fs",
                        self.name,
                        elapsed,
                    )
                    return True, "Circuit half-open: probing downstream recovery"

                self._total_rejected += 1
                remaining = max(0.0, self.config.recovery_timeout_seconds - elapsed)
                return (
                    False,
                    (
                        f"Circuit open: failing fast "
                        f"({remaining:.1f}s cooldown remaining)"
                    ),
                )

            # HALF_OPEN
            return True, "Circuit half-open: probe request in flight"

    def record_success(self) -> None:
        """Record a successful outbound delivery."""
        if not self.config.enabled:
            return

        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._consecutive_successes += 1
                if (
                    self._consecutive_successes
                    >= self.config.half_open_success_threshold
                ):
                    self._state = CircuitBreakerState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
                    self._last_state_change = time.time()
                    logger.info(
                        "Circuit breaker '%s' recovered and CLOSED "
                        "after %d consecutive successes",
                        self.name,
                        self.config.half_open_success_threshold,
                    )
            elif self._state == CircuitBreakerState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self, reason: str = "") -> None:
        """Record an outbound delivery failure."""
        if not self.config.enabled:
            return

        ts = time.time()
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                self._last_state_change = ts
                self._total_trips += 1
                self._consecutive_successes = 0
                logger.warning(
                    "Circuit breaker '%s' probe failed: %s. Tripping back to OPEN.",
                    self.name,
                    reason,
                )
            elif self._state == CircuitBreakerState.CLOSED:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    self._last_state_change = ts
                    self._total_trips += 1
                    logger.warning(
                        "Circuit breaker '%s' tripped OPEN after "
                        "%d consecutive failures (%s)",
                        self.name,
                        self._consecutive_failures,
                        reason,
                    )

    def get_stats(self) -> dict[str, Any]:
        """Return circuit breaker metrics and health status."""
        with self._lock:
            return {
                "name": self.name,
                "state": str(self._state),
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "total_trips": self._total_trips,
                "total_rejected": self._total_rejected,
                "total_calls": self._total_calls,
                "config": self.config.to_dict(),
            }

    def reset(self) -> None:
        """Reset the circuit breaker back to initial CLOSED state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._total_trips = 0
            self._total_rejected = 0
            self._total_calls = 0
            self._last_state_change = time.time()


class WebhookDispatcher:
    """Deliver alert transitions to an HTTP(S) webhook as JSON.

    Delivery failures are retried and then logged; they never propagate to
    the caller, so alerting cannot break request serving.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        sender: Callable[[str, bytes, float], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if urlsplit(url).scheme not in ("http", "https"):
            raise ValueError(f"Alert webhook URL must be http(s): '{url}'")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sender = sender or _post_json
        self._sleep = sleep

    def dispatch(self, transitions: Sequence[AlertTransition]) -> int:
        """POST one payload per transition; return the delivered count."""
        delivered = 0
        for transition in transitions:
            body = json.dumps(
                {"source": "music-recommender", **transition.to_dict()}
            ).encode("utf-8")
            if self._deliver(transition.rule.name, body):
                delivered += 1
        return delivered

    def _deliver(self, alert_name: str, body: bytes) -> bool:
        for attempt in range(1, self.max_attempts + 1):
            try:
                self._sender(self.url, body, self.timeout_seconds)
                return True
            except Exception as exc:
                logger.warning(
                    "Alert webhook delivery failed for %s (attempt %d/%d): %s",
                    alert_name,
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if attempt < self.max_attempts:
                    self._sleep(self.backoff_seconds * 2 ** (attempt - 1))
        return False


def _post_json(url: str, body: bytes, timeout: float) -> None:
    request = urllib.request.Request(  # noqa: S310 - scheme validated in __init__
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        status = getattr(response, "status", 200)
        if status >= 400:
            raise OSError(f"webhook responded with HTTP {status}")


def _send_http_request(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
    method: str = "POST",
) -> tuple[int, str]:
    request = urllib.request.Request(  # noqa: S310
        url,
        data=body if body else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = getattr(response, "status", 200)
            res_body = response.read().decode("utf-8", errors="replace")
            return int(status), res_body
    except urllib.error.HTTPError as exc:
        err_body = (
            exc.read().decode("utf-8", errors="replace")
            if hasattr(exc, "read")
            else str(exc)
        )
        return int(exc.code), err_body
    except Exception as exc:
        return 0, str(exc)


class PagerDutyGateway:
    """Delivers alert transitions to PagerDuty Events API v2."""

    def __init__(
        self,
        routing_key: str,
        endpoint_url: str = "https://events.pagerduty.com/v2/enqueue",
        timeout_seconds: float = 5.0,
        sender: Callable[[str, bytes, dict[str, str], float, str], tuple[int, str]]
        | None = None,
    ) -> None:
        self.routing_key = routing_key
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self._sender = sender or _send_http_request

    def send(self, transition: AlertTransition) -> tuple[bool, int, str]:
        """Dispatch an alert transition to PagerDuty."""
        action = "trigger" if transition.is_firing else "resolve"
        sev = str(transition.rule.severity).lower()
        pd_severity = "critical" if sev == "critical" else "warning"

        payload = {
            "routing_key": self.routing_key,
            "event_action": action,
            "dedup_key": f"music-recommender-{transition.rule.name}",
            "payload": {
                "summary": (
                    f"[{sev.upper()}] {transition.rule.name}: {transition.rule.summary}"
                ),
                "source": "music-recommender",
                "severity": pd_severity,
                "timestamp": transition.timestamp,
                "custom_details": {
                    "metric": transition.rule.metric,
                    "value": transition.value,
                    "comparator": transition.rule.comparator,
                    "threshold": transition.rule.threshold,
                },
            },
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        status, response = self._sender(
            self.endpoint_url, body, headers, self.timeout_seconds, "POST"
        )
        success = 200 <= status < 300
        return success, status, response


class OpsgenieGateway:
    """Delivers alert transitions to Opsgenie Alert API v2."""

    def __init__(
        self,
        api_key: str,
        endpoint_url: str = "https://api.opsgenie.com/v2/alerts",
        timeout_seconds: float = 5.0,
        sender: Callable[[str, bytes, dict[str, str], float, str], tuple[int, str]]
        | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self._sender = sender or _send_http_request

    def send(self, transition: AlertTransition) -> tuple[bool, int, str]:
        """Dispatch an alert transition to Opsgenie."""
        alias = f"music-recommender-{transition.rule.name}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"GenieKey {self.api_key}",
        }
        if transition.is_firing:
            sev = str(transition.rule.severity).lower()
            priority = "P1" if sev == "critical" else "P3"
            payload = {
                "message": f"[{sev.upper()}] {transition.rule.name}",
                "alias": alias,
                "description": transition.rule.summary,
                "priority": priority,
                "details": {
                    "metric": transition.rule.metric,
                    "value": str(transition.value),
                    "comparator": transition.rule.comparator,
                    "threshold": str(transition.rule.threshold),
                },
            }
            body = json.dumps(payload).encode("utf-8")
            status, response = self._sender(
                self.endpoint_url, body, headers, self.timeout_seconds, "POST"
            )
        else:
            close_url = (
                f"{self.endpoint_url.rstrip('/')}/{alias}/close?identifierType=alias"
            )
            body = json.dumps({"note": "Alert condition resolved"}).encode("utf-8")
            status, response = self._sender(
                close_url, body, headers, self.timeout_seconds, "POST"
            )
        success = 200 <= status < 300
        return success, status, response


class MultiChannelDispatcher:
    """Coordinates dispatching alert transitions across multiple gateways
    and channels.
    """

    def __init__(
        self,
        channels: dict[str, Callable[[AlertTransition], tuple[bool, int, str]]]
        | None = None,
        throttler: AlertNotificationThrottler | None = None,
        escalation_engine: AlertEscalationEngine | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._channels: dict[
            str, Callable[[AlertTransition], tuple[bool, int, str]]
        ] = dict(channels or {})
        self.throttler = throttler or AlertNotificationThrottler()
        self.escalation_engine = escalation_engine or AlertEscalationEngine()
        self._dispatch_count: int = 0
        self._error_count: int = 0

    def register_channel(
        self,
        name: str,
        handler: Callable[[AlertTransition], tuple[bool, int, str]],
    ) -> None:
        """Register a notification handler for a channel name."""
        with self._lock:
            self._channels[name] = handler

    def dispatch(
        self,
        transition: AlertTransition,
        channels: Sequence[str] | None = None,
        now: float | None = None,
    ) -> dict[str, tuple[bool, int, str]]:
        """Dispatch transition across channels with throttling and escalation checks."""
        allowed, decision, reason = self.throttler.should_dispatch(transition, now=now)
        if not allowed:
            return {
                "throttled": (
                    False,
                    429,
                    f"Throttled ({decision.value}): {reason}",
                )
            }

        escalation_events = self.escalation_engine.evaluate_transition(
            transition, now=now
        )

        target_channels: set[str] = set()
        if channels is not None:
            target_channels.update(channels)
        elif escalation_events:
            for ev in escalation_events:
                for ch in ev.channels:
                    if ch == "default":
                        target_channels.update(self._channels.keys())
                    else:
                        target_channels.add(ch)
        else:
            target_channels.update(self._channels.keys())

        if not target_channels and self._channels:
            target_channels = set(self._channels.keys())

        results: dict[str, tuple[bool, int, str]] = {}
        with self._lock:
            self._dispatch_count += 1

        for ch in target_channels:
            handler = self._channels.get(ch)
            if handler is None:
                results[ch] = (False, 404, f"Unknown channel '{ch}'")
                continue
            try:
                ok, code, msg = handler(transition)
                results[ch] = (ok, code, msg)
                if not ok:
                    with self._lock:
                        self._error_count += 1
            except Exception as exc:
                with self._lock:
                    self._error_count += 1
                results[ch] = (False, 500, str(exc))

        return results

    def get_stats(self) -> dict[str, Any]:
        """Return multi-channel dispatch metrics."""
        with self._lock:
            return {
                "dispatch_count": self._dispatch_count,
                "error_count": self._error_count,
                "registered_channels": sorted(self._channels.keys()),
                "throttler": self.throttler.get_stats(),
                "active_firing_durations": (
                    self.escalation_engine.get_active_firing_durations()
                ),
            }

    def reset(self) -> None:
        """Clear dispatcher counters, throttler, and escalation engine."""
        with self._lock:
            self._dispatch_count = 0
            self._error_count = 0
            self.throttler.reset()
            self.escalation_engine.reset()


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _promql_expression(rule: AlertRule) -> str:
    selector = rule.metric
    if rule.labels:
        matchers = ",".join(
            f"{k}={json.dumps(v)}" for k, v in sorted(rule.labels.items())
        )
        selector = f"{selector}{{{matchers}}}"
    if rule.mode is AlertMode.INCREASE:
        selector = f"increase({selector}[{rule.window}])"
    return f"{selector} {rule.comparator} {_format_number(rule.threshold)}"


DEFAULT_ALERTMANAGER_WEBHOOK_URL = "http://api:8000/alerts/webhook"


@dataclass(frozen=True)
class AlertmanagerConfig:
    """Configuration for Prometheus Alertmanager routing and receivers."""

    default_webhook_url: str = DEFAULT_ALERTMANAGER_WEBHOOK_URL
    critical_webhook_url: str | None = None
    warning_webhook_url: str | None = None
    resolve_timeout: str = "5m"
    group_wait: str = "10s"
    group_interval: str = "30s"
    critical_group_wait: str = "5s"
    critical_repeat_interval: str = "1h"
    warning_group_wait: str = "30s"
    warning_repeat_interval: str = "4h"
    inhibit_warning_on_service_down: bool = True


def render_prometheus_rules(
    rules: Sequence[AlertRule] = DEFAULT_ALERT_RULES,
    *,
    group_name: str = "music_recommender_alerts",
) -> str:
    """Render ``rules`` as a Prometheus alerting rule file (YAML)."""
    interval = PROMETHEUS_EVALUATION_INTERVAL_SECONDS
    lines = [
        "# Prometheus alerting rules for the Music Recommender System.",
        "# GENERATED from music_recommender.alerting.DEFAULT_ALERT_RULES.",
        "# Regenerate with:",
        "#   uv run python -m music_recommender.alerting configs/alert-rules.yml",
        "groups:",
        f"  - name: {group_name}",
        f"    interval: {interval}s",
        "    rules:",
    ]
    for rule in rules:
        lines.append(f"      - alert: {rule.name}")
        lines.append(f"        expr: {json.dumps(_promql_expression(rule))}")
        hold = (rule.for_evaluations - 1) * interval
        if hold:
            lines.append(f"        for: {hold}s")
        lines.append("        labels:")
        lines.append(f"          severity: {rule.severity}")
        lines.append("        annotations:")
        lines.append(f"          summary: {json.dumps(rule.summary)}")
    return "\n".join(lines) + "\n"


def render_alertmanager_config(
    config: AlertmanagerConfig | None = None,
) -> str:
    """Render Prometheus Alertmanager configuration YAML."""
    cfg = config or AlertmanagerConfig()
    crit_url = cfg.critical_webhook_url or f"{cfg.default_webhook_url}?channel=critical"
    warn_url = cfg.warning_webhook_url or f"{cfg.default_webhook_url}?channel=warning"

    lines = [
        "# Prometheus Alertmanager configuration for the Music Recommender System.",
        "# GENERATED from music_recommender.alerting.render_alertmanager_config.",
        "# Regenerate with:",
        "#   uv run python -m music_recommender.alerting \\",
        "#     --alertmanager configs/alertmanager.yml",
        "global:",
        f"  resolve_timeout: {cfg.resolve_timeout}",
        "",
        "route:",
        "  group_by:",
        "    - alertname",
        "    - severity",
        f"  group_wait: {cfg.group_wait}",
        f"  group_interval: {cfg.group_interval}",
        f"  repeat_interval: {cfg.warning_repeat_interval}",
        "  receiver: default-webhook",
        "  routes:",
        "    - match:",
        "        severity: critical",
        "      receiver: critical-webhook",
        f"      group_wait: {cfg.critical_group_wait}",
        f"      repeat_interval: {cfg.critical_repeat_interval}",
        "    - match:",
        "        severity: warning",
        "      receiver: warning-webhook",
        f"      group_wait: {cfg.warning_group_wait}",
        f"      repeat_interval: {cfg.warning_repeat_interval}",
        "",
        "inhibit_rules:",
    ]

    if cfg.inhibit_warning_on_service_down:
        lines.extend(
            [
                "  - source_match:",
                "      alertname: ServiceDown",
                "    target_match:",
                "      severity: warning",
                "    equal: []",
            ]
        )
    else:
        lines.append("  []")

    lines.extend(
        [
            "",
            "receivers:",
            "  - name: default-webhook",
            "    webhook_configs:",
            f"      - url: {cfg.default_webhook_url}",
            "        send_resolved: true",
            "  - name: critical-webhook",
            "    webhook_configs:",
            f"      - url: {crit_url}",
            "        send_resolved: true",
            "  - name: warning-webhook",
            "    webhook_configs:",
            f"      - url: {warn_url}",
            "        send_resolved: true",
        ]
    )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class AlertCausalRule:
    """Directed causal dependency between an upstream cause and downstream alert."""

    parent_alert: str
    child_alert: str
    description: str = ""
    max_time_lag_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.parent_alert or not self.child_alert:
            raise ValueError("parent_alert and child_alert must not be empty.")
        if self.parent_alert == self.child_alert:
            raise ValueError(f"Self-referential causal rule: {self.parent_alert}")


DEFAULT_ALERT_CORRELATION_RULES: tuple[AlertCausalRule, ...] = (
    AlertCausalRule(
        parent_alert="MaintenanceSweepErrors",
        child_alert="MaintenanceWorkerStopped",
        description="Repeated sweep errors cause maintenance worker daemon stoppage.",
    ),
    AlertCausalRule(
        parent_alert="MaintenanceWorkerStopped",
        child_alert="StreamingQueueHighUtilization",
        description="Stopped worker causes queue saturation.",
    ),
    AlertCausalRule(
        parent_alert="StreamingQueueHighUtilization",
        child_alert="StreamingRecordsDropped",
        description="Queue capacity exhaustion triggers backpressure drop.",
    ),
    AlertCausalRule(
        parent_alert="StreamingRecordsDropped",
        child_alert="StreamingHealthUnhealthy",
        description="Dropped feedback records degrade streaming health.",
    ),
    AlertCausalRule(
        parent_alert="StreamingFlushErrors",
        child_alert="StreamingHealthDegraded",
        description="Transient batch flush errors degrade streaming pipeline health.",
    ),
    AlertCausalRule(
        parent_alert="StreamingFlushErrors",
        child_alert="StreamingHealthUnhealthy",
        description="Persistent flush failures render streaming pipeline unhealthy.",
    ),
    AlertCausalRule(
        parent_alert="BanditDriftViolation",
        child_alert="BanditAutoRollback",
        description="Drift safety breaches trigger automated champion policy rollback.",
    ),
    AlertCausalRule(
        parent_alert="ServiceDown",
        child_alert="StreamingHealthUnhealthy",
        description="Service outage knocks out dependent streaming services.",
    ),
)


@dataclass(frozen=True)
class AlertCorrelationReport:
    """Incident root-cause correlation analysis of active alert notifications."""

    root_causes: list[str]
    symptoms: list[str]
    unlinked_alerts: list[str]
    causal_chains: list[list[str]]
    total_active_alerts: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "root_causes": list(self.root_causes),
            "symptoms": list(self.symptoms),
            "unlinked_alerts": list(self.unlinked_alerts),
            "causal_chains": [list(chain) for chain in self.causal_chains],
            "total_active_alerts": self.total_active_alerts,
            "summary": self.summary,
        }


class AlertCorrelationGraph:
    """Directed Acyclic Graph (DAG) for incident root-cause attribution."""

    def __init__(
        self,
        rules: Sequence[AlertCausalRule] = DEFAULT_ALERT_CORRELATION_RULES,
    ) -> None:
        self.rules = tuple(rules)
        self._children: dict[str, list[AlertCausalRule]] = {}
        self._parents: dict[str, list[AlertCausalRule]] = {}

        for rule in self.rules:
            self._children.setdefault(rule.parent_alert, []).append(rule)
            self._parents.setdefault(rule.child_alert, []).append(rule)

    def correlate(
        self,
        events: Sequence[ReceivedAlertEvent | dict[str, Any] | str],
        window_seconds: float = 300.0,
    ) -> AlertCorrelationReport:
        """Analyze a collection of alerts and attribute root causes and symptoms."""
        parsed_alerts: list[dict[str, Any]] = []

        for item in events:
            if isinstance(item, str):
                parsed_alerts.append(
                    {
                        "alertname": item,
                        "status": "firing",
                        "timestamp": time.time(),
                    }
                )
            elif isinstance(item, dict):
                alertname = str(item.get("alertname") or item.get("name") or "")
                status = str(item.get("status", "firing")).lower()
                ts_val = item.get("timestamp") or item.get("starts_at")
                ts = time.time()
                if isinstance(ts_val, (int, float)):
                    ts = float(ts_val)
                elif isinstance(ts_val, str):
                    try:
                        ts = datetime.fromisoformat(
                            ts_val.replace("Z", "+00:00")
                        ).timestamp()
                    except ValueError:
                        ts = time.time()
                if alertname:
                    parsed_alerts.append(
                        {"alertname": alertname, "status": status, "timestamp": ts}
                    )
            elif hasattr(item, "alertname"):
                # ReceivedAlertEvent instance
                status = getattr(item, "status", "firing").lower()
                ts_str = getattr(item, "starts_at", None)
                ts = time.time()
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(
                            ts_str.replace("Z", "+00:00")
                        ).timestamp()
                    except ValueError:
                        ts = time.time()
                parsed_alerts.append(
                    {
                        "alertname": item.alertname,
                        "status": status,
                        "timestamp": ts,
                    }
                )

        firing_alerts = [
            a for a in parsed_alerts if a.get("status") in ("firing", "active")
        ]
        if not firing_alerts:
            return AlertCorrelationReport(
                root_causes=[],
                symptoms=[],
                unlinked_alerts=[],
                causal_chains=[],
                total_active_alerts=0,
                summary="No active firing alerts to correlate.",
            )

        # Map alertname to earliest occurrence timestamp
        alert_times: dict[str, float] = {}
        for a in firing_alerts:
            name = a["alertname"]
            ts = a["timestamp"]
            if name not in alert_times or ts < alert_times[name]:
                alert_times[name] = ts

        firing_names = set(alert_times.keys())
        symptom_set: set[str] = set()
        parent_map: dict[str, list[str]] = {}

        for child_name in firing_names:
            child_ts = alert_times[child_name]
            possible_rules = self._parents.get(child_name, [])
            for rule in possible_rules:
                if rule.parent_alert in firing_names:
                    parent_ts = alert_times[rule.parent_alert]
                    time_diff = child_ts - parent_ts
                    effective_max = min(rule.max_time_lag_seconds, window_seconds)
                    # Causal parent fired before child or within allowed window
                    if -5.0 <= time_diff <= effective_max:
                        symptom_set.add(child_name)
                        parent_map.setdefault(child_name, []).append(rule.parent_alert)

        root_causes: list[str] = []
        unlinked: list[str] = []

        for name in sorted(firing_names):
            if name in symptom_set:
                continue
            # Check if this alert caused any other firing alert
            has_children = False
            for rule in self._children.get(name, []):
                if rule.child_alert in firing_names:
                    has_children = True
                    break
            if has_children:
                root_causes.append(name)
            else:
                unlinked.append(name)

        symptoms = sorted(symptom_set)

        # Build causal chains from root causes down
        causal_chains: list[list[str]] = []

        def _build_chains(current: str, path: list[str]) -> None:
            child_rules = self._children.get(current, [])
            active_children = [
                r.child_alert
                for r in child_rules
                if r.child_alert in firing_names and r.child_alert not in path
            ]
            if not active_children:
                if len(path) > 1:
                    causal_chains.append(list(path))
                return
            for child in active_children:
                _build_chains(child, path + [child])

        for root in root_causes:
            _build_chains(root, [root])

        if root_causes:
            summary = (
                f"Identified {len(root_causes)} root cause(s) "
                f"({', '.join(root_causes)}) linking {len(symptoms)} symptom(s)."
            )
        elif unlinked:
            summary = f"No causal links detected; {len(unlinked)} independent alert(s)."
        else:
            summary = "No root causes isolated."

        return AlertCorrelationReport(
            root_causes=root_causes,
            symptoms=symptoms,
            unlinked_alerts=unlinked,
            causal_chains=causal_chains,
            total_active_alerts=len(firing_names),
            summary=summary,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Write generated Prometheus rules or Alertmanager config to PATH or stdout."""
    import sys
    from pathlib import Path

    args = list(sys.argv[1:] if argv is None else argv)
    is_alertmanager = False
    if "--alertmanager" in args:
        is_alertmanager = True
        args.remove("--alertmanager")

    if len(args) > 1:
        sys.stderr.write(
            "usage: python -m music_recommender.alerting [--alertmanager] [PATH]\n"
        )
        return 2

    text = (
        render_alertmanager_config() if is_alertmanager else render_prometheus_rules()
    )
    if args:
        Path(args[0]).write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
