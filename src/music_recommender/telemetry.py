"""Prometheus text exposition format metric models, formatter, and telemetry."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from enum import StrEnum
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
            label_parts = [
                f'{k}="{escape_label_value(v)}"' for k, v in sorted_labels
            ]
            line = f"{self.name}{{{','.join(label_parts)}}} {val_str}"

        if self.timestamp_ms is not None:
            line = f"{line} {self.timestamp_ms}"
        return line


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
    if policy is not None and isinstance(policy, dict):
        arms_fam = MetricFamily(
            name="music_recommender_bandit_arms_count",
            help_text="Total number of configured cold-start bandit arms.",
            metric_type=MetricType.GAUGE,
        )
        arms_fam.add_sample(value=len(policy))
        families.append(arms_fam)

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

