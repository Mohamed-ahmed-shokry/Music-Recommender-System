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
import urllib.error
import urllib.request
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "reason": self.reason,
            "raw_payload": self.raw_payload,
            "channel": self.channel,
            "error_message": self.error_message,
            "headers": self.headers,
            "source": self.source,
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
        )


class AlertSpilloverStorage:
    """Thread-safe persistent JSONL storage for evicted alerts and DLQ records."""

    def __init__(
        self,
        storage_dir: Path | str,
        spillover_filename: str = "alerts_spillover.jsonl",
        dlq_filename: str = "alerts_dlq.jsonl",
        max_file_size_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.spillover_path = self.storage_dir / spillover_filename
        self.dlq_path = self.storage_dir / dlq_filename
        self.max_file_size_bytes = max_file_size_bytes
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
        return {
            "spillover_count": spill_count,
            "spillover_bytes": spill_size,
            "dlq_count": dlq_count,
            "dlq_bytes": dlq_size,
            "storage_dir": str(self.storage_dir),
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
