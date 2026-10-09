"""Tests for Prometheus exposition models, formatting, and telemetry."""

from __future__ import annotations

import pytest

from music_recommender.telemetry import (
    MetricFamily,
    MetricSample,
    MetricType,
    escape_help_string,
    escape_label_value,
    format_metric_value,
    render_prometheus_exposition,
)


def test_escape_help_string() -> None:
    assert escape_help_string("Simple docstring") == "Simple docstring"
    assert escape_help_string("Line 1\nLine 2\\path") == "Line 1\\nLine 2\\\\path"


def test_escape_label_value() -> None:
    assert escape_label_value("simple") == "simple"
    assert escape_label_value('val"with"quotes') == 'val\\"with\\"quotes'
    assert escape_label_value("line1\nline2\\dir") == "line1\\nline2\\\\dir"
    assert escape_label_value(123) == "123"


def test_format_metric_value() -> None:
    assert format_metric_value(True) == "1"
    assert format_metric_value(False) == "0"
    assert format_metric_value(42) == "42"
    assert format_metric_value(3.14) == "3.14"
    assert format_metric_value(float("nan")) == "NaN"
    assert format_metric_value(float("inf")) == "+Inf"
    assert format_metric_value(float("-inf")) == "-Inf"


def test_metric_sample_validation() -> None:
    # Valid sample
    sample = MetricSample(
        name="test_metric",
        value=1.5,
        labels={"env": "prod", "worker_id": "w1"},
    )
    assert sample.format_line() == 'test_metric{env="prod",worker_id="w1"} 1.5'

    # Sample with timestamp
    sample_ts = MetricSample(
        name="test_metric",
        value=10,
        labels={},
        timestamp_ms=1600000000000,
    )
    assert sample_ts.format_line() == "test_metric 10 1600000000000"

    # Invalid metric name
    with pytest.raises(ValueError, match="Invalid Prometheus metric sample name"):
        MetricSample(name="invalid-name-with-hyphens", value=1.0)

    # Invalid label name
    with pytest.raises(ValueError, match="Invalid Prometheus label name"):
        MetricSample(name="valid_name", value=1.0, labels={"invalid-label": "v"})


def test_metric_family_rendering() -> None:
    family = MetricFamily(
        name="http_requests_total",
        help_text="Total HTTP requests handled.",
        metric_type=MetricType.COUNTER,
    )
    family.add_sample(value=100, labels={"method": "GET", "handler": "/metrics"})
    family.add_sample(value=5, labels={"method": "POST", "handler": "/recommend"})

    exposition = family.format_exposition()
    expected = (
        "# HELP http_requests_total Total HTTP requests handled.\n"
        "# TYPE http_requests_total counter\n"
        'http_requests_total{handler="/metrics",method="GET"} 100\n'
        'http_requests_total{handler="/recommend",method="POST"} 5'
    )
    assert exposition == expected


def test_render_prometheus_exposition() -> None:
    f1 = MetricFamily(
        name="service_up",
        help_text="Service operational status.",
        metric_type="gauge",
    )
    f1.add_sample(1)

    f2 = MetricFamily(
        name="empty_family",
        help_text="Family without samples.",
        metric_type="counter",
    )

    rendered = render_prometheus_exposition([f1, f2])
    assert rendered.endswith("\n")
    assert "# HELP service_up Service operational status." in rendered
    assert "# TYPE service_up gauge" in rendered
    assert "service_up 1" in rendered
    assert "empty_family" not in rendered

    assert render_prometheus_exposition([]) == ""
    assert render_prometheus_exposition([f2]) == ""
