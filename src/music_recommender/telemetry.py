"""Prometheus text exposition format metric models, formatter, and telemetry."""

from __future__ import annotations

import math
import re
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
