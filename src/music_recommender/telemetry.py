"""Prometheus text exposition format metric models, formatter, and telemetry."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

_VALID_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_VALID_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class MetricType(StrEnum):
    """Supported Prometheus metric types."""

    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    UNTYPED = "untyped"


def escape_help_string(help_text: str) -> str:
    """Escape backslashes and newlines in Prometheus HELP documentation."""
    return help_text.replace("\\", "\\\\").replace("\n", "\\n")


def escape_label_value(label_value: Any) -> str:
    """Escape backslashes, double quotes, and newlines in Prometheus label values."""
    text = str(label_value)
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def format_metric_value(value: float | int | bool) -> str:
    """Format numeric values according to Prometheus text specification."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    # Format float cleanly without scientific notation for standard magnitudes
    # where possible, but maintain precision
    return f"{value:.6g}" if abs(value) < 1e-4 or abs(value) >= 1e6 else f"{value}"


@dataclass(frozen=True)
class MetricSample:
    """A single metric value with associated labels and optional timestamp."""

    name: str
    value: float | int | bool
    labels: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        if not _VALID_NAME_RE.match(self.name):
            raise ValueError(f"Invalid Prometheus metric sample name: '{self.name}'")
        for label_name in self.labels:
            if not _VALID_LABEL_NAME_RE.match(label_name):
                raise ValueError(
                    f"Invalid Prometheus label name: '{label_name}' "
                    f"in metric '{self.name}'"
                )

    def format_line(self) -> str:
        """Render a single sample line for Prometheus text format."""
        val_str = format_metric_value(self.value)
        if not self.labels:
            line = f"{self.name} {val_str}"
        else:
            sorted_labels = sorted(self.labels.items())
            label_parts = [f'{k}="{escape_label_value(v)}"' for k, v in sorted_labels]
            line = f"{self.name}{{{','.join(label_parts)}}} {val_str}"

        if self.timestamp_ms is not None:
            line = f"{line} {self.timestamp_ms}"
        return line

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to dictionary representation."""
        data: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "labels": dict(self.labels),
        }
        if self.timestamp_ms is not None:
            data["timestamp_ms"] = self.timestamp_ms
        return data


@dataclass
class MetricFamily:
    """A family of metric samples sharing a common name, help text, and type."""

    name: str
    help_text: str
    metric_type: MetricType | str
    samples: list[MetricSample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not _VALID_NAME_RE.match(self.name):
            raise ValueError(f"Invalid Prometheus metric family name: '{self.name}'")
        if isinstance(self.metric_type, str):
            self.metric_type = MetricType(self.metric_type.lower())

    def add_sample(
        self,
        value: float | int | bool,
        labels: dict[str, Any] | None = None,
        name: str | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        """Append a sample to the metric family."""
        sample_name = name or self.name
        self.samples.append(
            MetricSample(
                name=sample_name,
                value=value,
                labels=labels or {},
                timestamp_ms=timestamp_ms,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metric family to dictionary representation."""
        type_str = (
            self.metric_type.value
            if isinstance(self.metric_type, MetricType)
            else str(self.metric_type)
        )
        return {
            "name": self.name,
            "help": self.help_text,
            "type": type_str,
            "samples": [s.to_dict() for s in self.samples],
        }

    def format_exposition(self) -> str:
        """Render the metric family including # HELP and # TYPE headers."""
        type_str = (
            self.metric_type.value
            if isinstance(self.metric_type, MetricType)
            else str(self.metric_type)
        )
        lines: list[str] = [
            f"# HELP {self.name} {escape_help_string(self.help_text)}",
            f"# TYPE {self.name} {type_str}",
        ]
        for sample in self.samples:
            lines.append(sample.format_line())
        return "\n".join(lines)


def render_prometheus_exposition(families: list[MetricFamily]) -> str:
    """Render a list of metric families into Prometheus 0.0.4 text exposition format.

    The specification requires lines to end with a final newline.
    """
    if not families:
        return ""
    rendered_blocks = [f.format_exposition() for f in families if f.samples]
    if not rendered_blocks:
        return ""
    return "\n".join(rendered_blocks) + "\n"


def escape_influxdb_key(text: str) -> str:
    """Escape commas, equal signs, and spaces for InfluxDB measurements/tag keys."""
    return (
        text.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace("=", "\\=")
        .replace(" ", "\\ ")
    )


def escape_influxdb_tag_value(value: Any) -> str:
    """Escape commas, equal signs, and spaces for InfluxDB tag values."""
    text = str(value)
    return (
        text.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace("=", "\\=")
        .replace(" ", "\\ ")
    )


def format_influxdb_field_value(value: float | int | bool | str) -> str | None:
    """Format a scalar value for InfluxDB Line Protocol field representation."""
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, int):
        return f"{value}i"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return f"{value:.6g}" if abs(value) < 1e-4 or abs(value) >= 1e6 else f"{value}"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return None


def render_influxdb_line_protocol(
    families: list[MetricFamily],
    default_measurement: str = "music_recommender",
    fallback_timestamp_ns: int | None = None,
) -> str:
    """Render metric families into InfluxDB Line Protocol 2.0 text format."""
    lines: list[str] = []
    measurement_str = escape_influxdb_key(default_measurement)

    for fam in families:
        for sample in fam.samples:
            field_val = format_influxdb_field_value(sample.value)
            if field_val is None:
                continue

            field_str = f"{escape_influxdb_key(sample.name)}={field_val}"

            if sample.labels:
                sorted_tags = sorted(sample.labels.items())
                tag_parts = [
                    f"{escape_influxdb_key(k)}={escape_influxdb_tag_value(v)}"
                    for k, v in sorted_tags
                ]
                tags_str = ",".join(tag_parts)
                line_prefix = f"{measurement_str},{tags_str}"
            else:
                line_prefix = measurement_str

            ts_ns = (
                sample.timestamp_ms * 1_000_000
                if sample.timestamp_ms is not None
                else fallback_timestamp_ns
            )

            if ts_ns is not None:
                lines.append(f"{line_prefix} {field_str} {ts_ns}")
            else:
                lines.append(f"{line_prefix} {field_str}")

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def collect_service_metric_families(
    service: Any | None,
    *,
    custom_labels: dict[str, str] | None = None,
) -> list[MetricFamily]:
    """Extract Prometheus metric families from a RecommenderService instance.

    Handles instances with or without streaming feedback queues, maintenance
    workers, cold-start policies, or loaded artifacts, gracefully reporting
    inactive/zero values where appropriate.
    """
    from music_recommender import __version__

    families: list[MetricFamily] = []

    # 1. Service Info & Availability
    info_labels: dict[str, Any] = {
        "version": __version__,
        "python_version": sys.version.split()[0],
    }
    if custom_labels:
        info_labels.update(custom_labels)

    info_fam = MetricFamily(
        name="music_recommender_info",
        help_text="Recommender system version and environment metadata.",
        metric_type=MetricType.GAUGE,
    )
    info_fam.add_sample(value=1, labels=info_labels)
    families.append(info_fam)

    up_fam = MetricFamily(
        name="music_recommender_up",
        help_text="Operational availability of the recommender service (1=up, 0=down).",
        metric_type=MetricType.GAUGE,
    )
    up_fam.add_sample(value=1 if service is not None else 0)
    families.append(up_fam)

    if service is None:
        return families

    # 2. Artifact Inventory
    artifact = getattr(service, "artifact", None)
    if artifact is not None:
        metadata = getattr(artifact, "metadata", {}) or {}
        num_users = int(metadata.get("num_users", 0))
        num_artists = int(metadata.get("num_items", 0))
        num_interactions = int(metadata.get("num_interactions", 0))

        users_fam = MetricFamily(
            name="music_recommender_artifact_users_count",
            help_text="Total number of users indexed in active model artifacts.",
            metric_type=MetricType.GAUGE,
        )
        users_fam.add_sample(value=num_users)
        families.append(users_fam)

        artists_fam = MetricFamily(
            name="music_recommender_artifact_artists_count",
            help_text="Total artists/items indexed in active model artifacts.",
            metric_type=MetricType.GAUGE,
        )
        artists_fam.add_sample(value=num_artists)
        families.append(artists_fam)

        interactions_fam = MetricFamily(
            name="music_recommender_artifact_interactions_count",
            help_text="Total historical interactions in active model artifacts.",
            metric_type=MetricType.GAUGE,
        )
        interactions_fam.add_sample(value=num_interactions)
        families.append(interactions_fam)

    # 3. Bandit Policy Arms & Weights
    policy = getattr(service, "cold_start_policy", None)
    arms_count = len(policy) if isinstance(policy, dict) else 0
    arms_fam = MetricFamily(
        name="music_recommender_bandit_arms_count",
        help_text="Total number of configured cold-start bandit arms.",
        metric_type=MetricType.GAUGE,
    )
    arms_fam.add_sample(value=arms_count)
    families.append(arms_fam)

    if isinstance(policy, dict) and policy:
        weights_fam = MetricFamily(
            name="music_recommender_bandit_arm_weight",
            help_text="Current serving probability weight per cold-start bandit arm.",
            metric_type=MetricType.GAUGE,
        )
        for arm_name, weight in sorted(policy.items()):
            weights_fam.add_sample(value=float(weight), labels={"arm": str(arm_name)})
        families.append(weights_fam)

    # Champion snapshot info
    if hasattr(service, "get_champion_snapshot"):
        try:
            champ = service.get_champion_snapshot()
            if champ and isinstance(champ, dict):
                champ_fam = MetricFamily(
                    name="music_recommender_bandit_champion_info",
                    help_text="Metadata of the current champion bandit snapshot.",
                    metric_type=MetricType.GAUGE,
                )
                snap_name = str(
                    champ.get("snapshot_name")
                    or champ.get("snapshot_path")
                    or "champion"
                )
                champ_fam.add_sample(value=1, labels={"snapshot": snap_name})
                families.append(champ_fam)
        except Exception:
            pass

    # 4. Streaming Feedback Queue Telemetry
    q = getattr(service, "feedback_queue", None)
    q_enabled = q is not None
    q_enabled_fam = MetricFamily(
        name="music_recommender_streaming_queue_enabled",
        help_text="Whether the streaming feedback queue is enabled (1=yes, 0=no).",
        metric_type=MetricType.GAUGE,
    )
    q_enabled_fam.add_sample(value=1 if q_enabled else 0)
    families.append(q_enabled_fam)

    if q is not None and hasattr(q, "get_metrics"):
        try:
            qm = q.get_metrics()
            q_size_fam = MetricFamily(
                name="music_recommender_streaming_queue_size",
                help_text="Current buffered record count in streaming feedback queue.",
                metric_type=MetricType.GAUGE,
            )
            q_size_fam.add_sample(value=qm.queue_size)
            families.append(q_size_fam)

            q_cap_fam = MetricFamily(
                name="music_recommender_streaming_queue_capacity",
                help_text="Maximum capacity of streaming feedback queue.",
                metric_type=MetricType.GAUGE,
            )
            q_cap_fam.add_sample(value=qm.max_queue_size)
            families.append(q_cap_fam)

            q_util_fam = MetricFamily(
                name="music_recommender_streaming_queue_utilization_ratio",
                help_text="Current buffer utilization ratio (0.0 to 1.0).",
                metric_type=MetricType.GAUGE,
            )
            q_util_fam.add_sample(value=round(qm.utilization_pct / 100.0, 4))
            families.append(q_util_fam)

            enq_fam = MetricFamily(
                name="music_recommender_streaming_records_enqueued_total",
                help_text="Total feedback records enqueued into streaming queue.",
                metric_type=MetricType.COUNTER,
            )
            enq_fam.add_sample(value=qm.enqueued_count)
            families.append(enq_fam)

            flush_rec_fam = MetricFamily(
                name="music_recommender_streaming_records_flushed_total",
                help_text="Total feedback records flushed to disk by streaming queue.",
                metric_type=MetricType.COUNTER,
            )
            flush_rec_fam.add_sample(value=qm.flushed_count)
            families.append(flush_rec_fam)

            drop_fam = MetricFamily(
                name="music_recommender_streaming_records_dropped_total",
                help_text="Total feedback records dropped due to queue backpressure.",
                metric_type=MetricType.COUNTER,
            )
            drop_fam.add_sample(value=qm.dropped_count)
            families.append(drop_fam)

            flush_err_fam = MetricFamily(
                name="music_recommender_streaming_flush_errors_total",
                help_text="Total batch flush failures encountered by streaming queue.",
                metric_type=MetricType.COUNTER,
            )
            flush_err_fam.add_sample(value=qm.flush_errors)
            families.append(flush_err_fam)

            flushes_fam = MetricFamily(
                name="music_recommender_streaming_flushes_total",
                help_text="Total flush executions executed by streaming queue.",
                metric_type=MetricType.COUNTER,
            )
            flushes_fam.add_sample(value=qm.flushes_count)
            families.append(flushes_fam)
        except Exception:
            pass

    # 5. Bandit Maintenance Worker Telemetry
    worker = getattr(service, "maintenance_worker", None)
    worker_enabled = worker is not None
    w_enabled_fam = MetricFamily(
        name="music_recommender_maintenance_enabled",
        help_text="Whether async maintenance daemon is enabled (1=yes, 0=no).",
        metric_type=MetricType.GAUGE,
    )
    w_enabled_fam.add_sample(value=1 if worker_enabled else 0)
    families.append(w_enabled_fam)

    if worker is not None and hasattr(worker, "get_metrics"):
        try:
            wm = worker.get_metrics()
            running_fam = MetricFamily(
                name="music_recommender_maintenance_running",
                help_text="Liveness of maintenance daemon (1=running, 0=stopped).",
                metric_type=MetricType.GAUGE,
            )
            running_fam.add_sample(value=1 if wm.is_running and not wm.is_closed else 0)
            families.append(running_fam)

            cycles_fam = MetricFamily(
                name="music_recommender_maintenance_cycles_total",
                help_text="Total background poll cycles executed by worker.",
                metric_type=MetricType.COUNTER,
            )
            cycles_fam.add_sample(value=wm.cycles_count)
            families.append(cycles_fam)

            sweeps_fam = MetricFamily(
                name="music_recommender_maintenance_sweeps_total",
                help_text="Total policy maintenance feedback sweeps performed.",
                metric_type=MetricType.COUNTER,
            )
            sweeps_fam.add_sample(value=wm.sweeps_count)
            families.append(sweeps_fam)

            folded_fam = MetricFamily(
                name="music_recommender_maintenance_records_folded_total",
                help_text="Total feedback records folded into bandit state by sweeps.",
                metric_type=MetricType.COUNTER,
            )
            folded_fam.add_sample(value=wm.records_folded_count)
            families.append(folded_fam)

            errors_fam = MetricFamily(
                name="music_recommender_maintenance_errors_total",
                help_text="Total sweep errors encountered by maintenance worker.",
                metric_type=MetricType.COUNTER,
            )
            errors_fam.add_sample(value=wm.errors_count)
            families.append(errors_fam)

            dur_fam = MetricFamily(
                name="music_recommender_maintenance_sweep_duration_seconds",
                help_text="Sweep execution latency statistics in seconds.",
                metric_type=MetricType.GAUGE,
            )
            dur_fam.add_sample(
                value=wm.last_sweep_duration_seconds, labels={"stat": "last"}
            )
            dur_fam.add_sample(
                value=wm.min_sweep_duration_seconds, labels={"stat": "min"}
            )
            dur_fam.add_sample(
                value=wm.max_sweep_duration_seconds, labels={"stat": "max"}
            )
            dur_fam.add_sample(
                value=wm.avg_sweep_duration_seconds, labels={"stat": "avg"}
            )
            dur_fam.add_sample(
                value=wm.total_sweep_duration_seconds, labels={"stat": "total"}
            )
            families.append(dur_fam)

            drift_evals_fam = MetricFamily(
                name="music_recommender_maintenance_drift_evaluations_total",
                help_text="Total automated drift safety evaluations performed.",
                metric_type=MetricType.COUNTER,
            )
            drift_evals_fam.add_sample(value=wm.drift_evaluations_count)
            families.append(drift_evals_fam)

            drift_viols_fam = MetricFamily(
                name="music_recommender_maintenance_drift_violations_total",
                help_text="Total drift guardrail safety breaches detected.",
                metric_type=MetricType.COUNTER,
            )
            drift_viols_fam.add_sample(value=wm.drift_violations_count)
            families.append(drift_viols_fam)

            rollbacks_fam = MetricFamily(
                name="music_recommender_maintenance_rollbacks_total",
                help_text="Total automated rollbacks to champion snapshots executed.",
                metric_type=MetricType.COUNTER,
            )
            rollbacks_fam.add_sample(value=wm.rollbacks_count)
            families.append(rollbacks_fam)
        except Exception:
            pass

    # 6. Streaming Health Status
    if hasattr(service, "streaming_health"):
        try:
            h = service.streaming_health()
            curr_status = str(h.get("status", "unknown")).lower()
            health_fam = MetricFamily(
                name="music_recommender_streaming_health",
                help_text="Streaming health flag (1 for current status, 0 otherwise).",
                metric_type=MetricType.GAUGE,
            )
            for st in ("healthy", "degraded", "unhealthy", "disabled"):
                health_fam.add_sample(
                    value=1 if st == curr_status else 0, labels={"status": st}
                )
            families.append(health_fam)
        except Exception:
            pass

    return families


DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    float("inf"),
)


class RequestTelemetryTracker:
    """Thread-safe collector for HTTP and service request metrics."""

    def __init__(
        self,
        *,
        latency_buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS,
    ) -> None:
        self._lock = threading.Lock()
        self.latency_buckets = tuple(sorted(latency_buckets))
        self._requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._duration_sums: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._duration_buckets: dict[tuple[str, str, float], int] = defaultdict(int)

    def record_request(
        self,
        endpoint: str,
        method: str = "GET",
        status_code: int = 200,
        duration_seconds: float = 0.0,
    ) -> None:
        """Record an observed request and its execution latency in seconds."""
        norm_endpoint = endpoint.strip() or "/"
        norm_method = method.strip().upper() or "GET"
        dur = max(0.0, float(duration_seconds))

        with self._lock:
            key = (norm_endpoint, norm_method, int(status_code))
            self._requests[key] += 1

            dur_key = (norm_endpoint, norm_method)
            self._duration_sums[dur_key] += dur
            self._duration_counts[dur_key] += 1
            for b in self.latency_buckets:
                if dur <= b:
                    self._duration_buckets[(norm_endpoint, norm_method, b)] += 1

    def to_metric_families(self) -> list[MetricFamily]:
        """Convert recorded request telemetry into Prometheus metric families."""
        with self._lock:
            req_items = list(self._requests.items())
            dur_sums = dict(self._duration_sums)
            dur_counts = dict(self._duration_counts)
            dur_buckets = dict(self._duration_buckets)

        families: list[MetricFamily] = []

        # 1. Total Requests Counter
        req_fam = MetricFamily(
            name="music_recommender_http_requests_total",
            help_text="Total HTTP and service requests processed.",
            metric_type=MetricType.COUNTER,
        )
        for (ep, m, sc), count in sorted(req_items):
            req_fam.add_sample(
                value=count,
                labels={"endpoint": ep, "method": m, "status": str(sc)},
            )
        families.append(req_fam)

        # 2. Latency Histogram
        hist_fam = MetricFamily(
            name="music_recommender_http_request_duration_seconds",
            help_text="HTTP request duration distribution in seconds.",
            metric_type=MetricType.HISTOGRAM,
        )
        for ep, m in sorted(dur_counts.keys()):
            for b in self.latency_buckets:
                b_count = dur_buckets.get((ep, m, b), 0)
                le_str = "+Inf" if math.isinf(b) else str(b)
                hist_fam.add_sample(
                    name="music_recommender_http_request_duration_seconds_bucket",
                    value=b_count,
                    labels={"endpoint": ep, "method": m, "le": le_str},
                )
            hist_fam.add_sample(
                name="music_recommender_http_request_duration_seconds_sum",
                value=round(dur_sums.get((ep, m), 0.0), 6),
                labels={"endpoint": ep, "method": m},
            )
            hist_fam.add_sample(
                name="music_recommender_http_request_duration_seconds_count",
                value=dur_counts.get((ep, m), 0),
                labels={"endpoint": ep, "method": m},
            )
        families.append(hist_fam)

        return families

    def reset(self) -> None:
        """Reset internal metrics storage."""
        with self._lock:
            self._requests.clear()
            self._duration_sums.clear()
            self._duration_counts.clear()
            self._duration_buckets.clear()


_GLOBAL_TRACKER = RequestTelemetryTracker()


def get_global_request_tracker() -> RequestTelemetryTracker:
    """Return the application-wide singleton request telemetry tracker."""
    return _GLOBAL_TRACKER


def _apply_timestamps(
    families: list[MetricFamily],
    timestamp_ms: int,
) -> list[MetricFamily]:
    """Return new MetricFamily list with timestamp_ms applied to all samples."""
    new_families: list[MetricFamily] = []
    for fam in families:
        new_fam = MetricFamily(
            name=fam.name,
            help_text=fam.help_text,
            metric_type=fam.metric_type,
        )
        for s in fam.samples:
            new_fam.samples.append(
                MetricSample(
                    name=s.name,
                    value=s.value,
                    labels=s.labels,
                    timestamp_ms=timestamp_ms,
                )
            )
        new_families.append(new_fam)
    return new_families


def export_prometheus_metrics(
    service: Any | None,
    *,
    request_tracker: RequestTelemetryTracker | None = None,
    custom_labels: dict[str, str] | None = None,
    include_timestamps: bool = False,
) -> str:
    """Export complete Prometheus text exposition document for the service."""
    families = collect_service_metric_families(service, custom_labels=custom_labels)
    tracker = request_tracker or get_global_request_tracker()
    families.extend(tracker.to_metric_families())
    if include_timestamps:
        now_ms = int(time.time() * 1000)
        families = _apply_timestamps(families, now_ms)
    return render_prometheus_exposition(families)


def export_metrics_json(
    service: Any | None,
    *,
    request_tracker: RequestTelemetryTracker | None = None,
    custom_labels: dict[str, str] | None = None,
    include_timestamps: bool = False,
    indent: int | None = 2,
) -> str:
    """Export complete metrics in structured JSON format for the service."""
    families = collect_service_metric_families(service, custom_labels=custom_labels)
    tracker = request_tracker or get_global_request_tracker()
    families.extend(tracker.to_metric_families())
    if include_timestamps:
        now_ms = int(time.time() * 1000)
        families = _apply_timestamps(families, now_ms)
    metrics_data = {"metrics": [f.to_dict() for f in families if f.samples]}
    return json.dumps(metrics_data, indent=indent) + "\n"


def export_influxdb_metrics(
    service: Any | None,
    *,
    request_tracker: RequestTelemetryTracker | None = None,
    custom_labels: dict[str, str] | None = None,
    default_measurement: str = "music_recommender",
    include_timestamps: bool = True,
) -> str:
    """Export complete metrics in InfluxDB Line Protocol 2.0 format for the service."""
    families = collect_service_metric_families(service, custom_labels=custom_labels)
    tracker = request_tracker or get_global_request_tracker()
    families.extend(tracker.to_metric_families())
    fallback_ts_ns = int(time.time() * 1_000_000_000) if include_timestamps else None
    return render_influxdb_line_protocol(
        families,
        default_measurement=default_measurement,
        fallback_timestamp_ns=fallback_ts_ns,
    )


class TelemetryPushProtocol(StrEnum):
    """Supported push protocols for remote telemetry collection."""

    INFLUXDB = "influxdb"
    PROMETHEUS_TEXT = "prometheus_text"
    PROMETHEUS_JSON = "prometheus_json"


@dataclass(frozen=True)
class TelemetryPushConfig:
    """Configuration for dispatching telemetry metrics to a remote collector."""

    endpoint_url: str
    protocol: TelemetryPushProtocol | str = TelemetryPushProtocol.INFLUXDB
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    default_measurement: str = "music_recommender"
    dry_run: bool = False
    include_timestamps: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol",
            (
                self.protocol
                if isinstance(self.protocol, TelemetryPushProtocol)
                else TelemetryPushProtocol(str(self.protocol).lower())
            ),
        )
        if not self.endpoint_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid telemetry push endpoint URL '{self.endpoint_url}'. "
                "Must start with http:// or https://"
            )


@dataclass(frozen=True)
class TelemetryPushResult:
    """Report outcome of pushing metrics to a remote telemetry collector."""

    success: bool
    samples_pushed: int
    protocol: str
    endpoint_url: str
    status_code: int | None = None
    duration_ms: float = 0.0
    error: str | None = None
    payload_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "success": self.success,
            "samples_pushed": self.samples_pushed,
            "protocol": self.protocol,
            "endpoint_url": self.endpoint_url,
            "status_code": self.status_code,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "payload_bytes": self.payload_bytes,
        }


class TelemetryPushAdapter:
    """HTTP client adapter for pushing metric samples to remote collectors."""

    def serialize_payload(
        self,
        families: list[MetricFamily],
        config: TelemetryPushConfig,
    ) -> tuple[bytes, str]:
        """Serialize metric families into protocol-specific bytes and content type."""
        proto = (
            config.protocol.value
            if isinstance(config.protocol, TelemetryPushProtocol)
            else str(config.protocol).lower()
        )

        if proto == TelemetryPushProtocol.INFLUXDB:
            fb_ts = (
                int(time.time() * 1_000_000_000) if config.include_timestamps else None
            )
            body = render_influxdb_line_protocol(
                families,
                default_measurement=config.default_measurement,
                fallback_timestamp_ns=fb_ts,
            )
            return body.encode("utf-8"), "text/plain; charset=utf-8"

        if proto == TelemetryPushProtocol.PROMETHEUS_TEXT:
            body = render_prometheus_exposition(families)
            return (
                body.encode("utf-8"),
                "text/plain; version=0.0.4; charset=utf-8",
            )

        if proto == TelemetryPushProtocol.PROMETHEUS_JSON:
            metrics_data = {"metrics": [f.to_dict() for f in families if f.samples]}
            body = json.dumps(metrics_data, indent=2) + "\n"
            return body.encode("utf-8"), "application/json; charset=utf-8"

        raise ValueError(f"Unsupported telemetry push protocol: '{proto}'")

    def push(
        self,
        families: list[MetricFamily],
        config: TelemetryPushConfig,
    ) -> TelemetryPushResult:
        """Dispatch serialized metric families to remote endpoint."""
        total_samples = sum(len(f.samples) for f in families)
        proto_str = (
            config.protocol.value
            if isinstance(config.protocol, TelemetryPushProtocol)
            else str(config.protocol)
        )

        try:
            payload_bytes, content_type = self.serialize_payload(families, config)
        except Exception as exc:
            return TelemetryPushResult(
                success=False,
                samples_pushed=0,
                protocol=proto_str,
                endpoint_url=config.endpoint_url,
                error=f"Serialization failed: {exc}",
            )

        if config.dry_run:
            return TelemetryPushResult(
                success=True,
                samples_pushed=total_samples,
                protocol=proto_str,
                endpoint_url=config.endpoint_url,
                status_code=200,
                duration_ms=0.0,
                payload_bytes=len(payload_bytes),
            )

        headers = {
            "Content-Type": content_type,
            "User-Agent": "MusicRecommender-TelemetryPush/1.0",
        }
        headers.update(config.headers)

        req = urllib.request.Request(
            config.endpoint_url,
            data=payload_bytes,
            headers=headers,
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=config.timeout_seconds) as resp:
                raw_status = getattr(resp, "status", getattr(resp, "code", 200))
                try:
                    status = int(raw_status)
                except (TypeError, ValueError):
                    status = 200
                duration_ms = (time.perf_counter() - start) * 1000
                return TelemetryPushResult(
                    success=200 <= status < 300,
                    samples_pushed=total_samples,
                    protocol=proto_str,
                    endpoint_url=config.endpoint_url,
                    status_code=status,
                    duration_ms=duration_ms,
                    payload_bytes=len(payload_bytes),
                )
        except urllib.error.HTTPError as http_err:
            duration_ms = (time.perf_counter() - start) * 1000
            return TelemetryPushResult(
                success=False,
                samples_pushed=0,
                protocol=proto_str,
                endpoint_url=config.endpoint_url,
                status_code=http_err.code,
                duration_ms=duration_ms,
                error=f"HTTP {http_err.code}: {http_err.reason}",
                payload_bytes=len(payload_bytes),
            )
        except Exception as err:
            duration_ms = (time.perf_counter() - start) * 1000
            return TelemetryPushResult(
                success=False,
                samples_pushed=0,
                protocol=proto_str,
                endpoint_url=config.endpoint_url,
                duration_ms=duration_ms,
                error=str(err),
                payload_bytes=len(payload_bytes),
            )


KNOWN_PROMETHEUS_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "music_recommender_info",
        "music_recommender_up",
        "music_recommender_artifact_users_count",
        "music_recommender_artifact_artists_count",
        "music_recommender_artifact_interactions_count",
        "music_recommender_bandit_arms_count",
        "music_recommender_bandit_arm_weight",
        "music_recommender_bandit_champion_info",
        "music_recommender_streaming_queue_enabled",
        "music_recommender_streaming_queue_size",
        "music_recommender_streaming_queue_capacity",
        "music_recommender_streaming_queue_utilization_ratio",
        "music_recommender_streaming_records_enqueued_total",
        "music_recommender_streaming_records_flushed_total",
        "music_recommender_streaming_records_dropped_total",
        "music_recommender_streaming_flush_errors_total",
        "music_recommender_streaming_flushes_total",
        "music_recommender_maintenance_enabled",
        "music_recommender_maintenance_running",
        "music_recommender_maintenance_cycles_total",
        "music_recommender_maintenance_sweeps_total",
        "music_recommender_maintenance_records_folded_total",
        "music_recommender_maintenance_errors_total",
        "music_recommender_maintenance_sweep_duration_seconds",
        "music_recommender_maintenance_drift_evaluations_total",
        "music_recommender_maintenance_drift_violations_total",
        "music_recommender_maintenance_rollbacks_total",
        "music_recommender_streaming_health",
        "music_recommender_http_requests_total",
        "music_recommender_http_request_duration_seconds",
        "music_recommender_http_request_duration_seconds_bucket",
        "music_recommender_http_request_duration_seconds_sum",
        "music_recommender_http_request_duration_seconds_count",
    }
)


@dataclass(frozen=True)
class DashboardValidationResult:
    """Validation report for Grafana dashboard Prometheus metric bindings."""

    is_valid: bool
    dashboard_title: str
    total_panels: int
    total_queries: int
    referenced_metrics: set[str] = field(default_factory=set)
    unknown_metrics: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation summary into JSON-serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "dashboard_title": self.dashboard_title,
            "total_panels": self.total_panels,
            "total_queries": self.total_queries,
            "referenced_metrics": sorted(self.referenced_metrics),
            "unknown_metrics": sorted(self.unknown_metrics),
            "errors": list(self.errors),
        }


_PROMETHEUS_METRIC_REF_RE = re.compile(r"\b(music_recommender_[a-zA-Z0-9_]+)\b")


def validate_dashboard_metrics(
    dashboard_data: dict[str, Any] | str | Path,
    *,
    allowed_metrics: frozenset[str] | set[str] | None = None,
) -> DashboardValidationResult:
    """Validate that Grafana dashboard panels bind to valid Prometheus metric names."""
    valid_names = (
        allowed_metrics
        if allowed_metrics is not None
        else KNOWN_PROMETHEUS_METRIC_NAMES
    )
    errors: list[str] = []

    if isinstance(dashboard_data, (str, Path)):
        p = Path(dashboard_data)
        if not p.is_file():
            return DashboardValidationResult(
                is_valid=False,
                dashboard_title=str(dashboard_data),
                total_panels=0,
                total_queries=0,
                errors=[f"Dashboard file does not exist: {p}"],
            )
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as err:
            return DashboardValidationResult(
                is_valid=False,
                dashboard_title=str(dashboard_data),
                total_panels=0,
                total_queries=0,
                errors=[f"Failed to parse dashboard JSON: {err}"],
            )
    elif isinstance(dashboard_data, dict):
        data = dashboard_data
    else:
        return DashboardValidationResult(
            is_valid=False,
            dashboard_title="Unknown",
            total_panels=0,
            total_queries=0,
            errors=[f"Invalid dashboard data type: {type(dashboard_data).__name__}"],
        )

    title = str(data.get("title", "Untitled Dashboard"))
    panels: list[dict[str, Any]] = []

    def _collect_panels(panel_list: list[Any]) -> None:
        for p_item in panel_list:
            if isinstance(p_item, dict):
                panels.append(p_item)
                if "panels" in p_item and isinstance(p_item["panels"], list):
                    _collect_panels(p_item["panels"])

    raw_panels = data.get("panels", [])
    if isinstance(raw_panels, list):
        _collect_panels(raw_panels)

    total_queries = 0
    referenced_metrics: set[str] = set()
    unknown_metrics: set[str] = set()

    for p_idx, panel in enumerate(panels):
        panel_title = panel.get("title", f"Panel #{p_idx}")
        targets = panel.get("targets", [])
        if not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, dict):
                continue
            expr = target.get("expr")
            if not expr or not isinstance(expr, str):
                continue
            total_queries += 1
            matches = _PROMETHEUS_METRIC_REF_RE.findall(expr)
            for m in matches:
                referenced_metrics.add(m)
                if m not in valid_names:
                    unknown_metrics.add(m)
                    errors.append(
                        f"Panel '{panel_title}' references unknown metric '{m}' "
                        f"in expression: {expr}"
                    )

    if not panels:
        errors.append("Dashboard contains no panels.")

    is_valid = len(errors) == 0 and len(referenced_metrics) > 0
    if not errors and not referenced_metrics:
        errors.append("Dashboard contains no Prometheus metric references.")
        is_valid = False

    return DashboardValidationResult(
        is_valid=is_valid,
        dashboard_title=title,
        total_panels=len(panels),
        total_queries=total_queries,
        referenced_metrics=referenced_metrics,
        unknown_metrics=unknown_metrics,
        errors=errors,
    )
