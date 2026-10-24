"""Threshold alert rules over Prometheus metric families.

Rules are declared once and reused for in-process evaluation, webhook
notification dispatch, and generation of a Prometheus rule file.
"""

from __future__ import annotations

import json
import logging
import operator
import re
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
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
