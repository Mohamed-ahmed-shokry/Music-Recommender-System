"""Tests for Prometheus exposition models, formatting, and telemetry."""

from __future__ import annotations

import pytest

from music_recommender.telemetry import (
    MetricFamily,
    MetricSample,
    MetricType,
    collect_service_metric_families,
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


def test_collect_service_metric_families_none_service() -> None:
    families = collect_service_metric_families(
        None, custom_labels={"cluster": "prod-1"}
    )
    names = {f.name: f for f in families}
    assert "music_recommender_info" in names
    assert "music_recommender_up" in names

    info_sample = names["music_recommender_info"].samples[0]
    assert info_sample.value == 1
    assert info_sample.labels["cluster"] == "prod-1"
    assert "version" in info_sample.labels
    assert "python_version" in info_sample.labels

    up_sample = names["music_recommender_up"].samples[0]
    assert up_sample.value == 0


def test_collect_service_metric_families_mock_service() -> None:
    class MockArtifact:
        metadata = {
            "num_users": 1500,
            "num_items": 400,
            "num_interactions": 35000,
        }

    class MockQueueMetrics:
        queue_size = 42
        max_queue_size = 1000
        utilization_pct = 4.2
        enqueued_count = 100
        flushed_count = 58
        dropped_count = 0
        flush_errors = 0
        flushes_count = 3

    class MockQueue:
        def get_metrics(self) -> MockQueueMetrics:
            return MockQueueMetrics()

    class MockWorkerMetrics:
        is_running = True
        is_closed = False
        cycles_count = 12
        sweeps_count = 4
        records_folded_count = 58
        errors_count = 0
        last_sweep_duration_seconds = 0.15
        min_sweep_duration_seconds = 0.12
        max_sweep_duration_seconds = 0.20
        avg_sweep_duration_seconds = 0.16
        total_sweep_duration_seconds = 0.64
        drift_evaluations_count = 4
        drift_violations_count = 0
        rollbacks_count = 0

    class MockWorker:
        def get_metrics(self) -> MockWorkerMetrics:
            return MockWorkerMetrics()

    class MockService:
        artifact = MockArtifact()
        cold_start_policy = {
            "artist_sim": 0.4,
            "content_sim": 0.6,
        }
        feedback_queue = MockQueue()
        maintenance_worker = MockWorker()

        def get_champion_snapshot(self) -> dict[str, str]:
            return {"snapshot_name": "bandit_snap_champion_v1.json"}

        def streaming_health(self) -> dict[str, str]:
            return {"status": "healthy"}

    service = MockService()
    families = collect_service_metric_families(
        service, custom_labels={"env": "staging"}
    )
    names = {f.name: f for f in families}

    assert names["music_recommender_up"].samples[0].value == 1
    assert names["music_recommender_artifact_users_count"].samples[0].value == 1500
    assert names["music_recommender_artifact_artists_count"].samples[0].value == 400
    assert (
        names["music_recommender_artifact_interactions_count"].samples[0].value
        == 35000
    )

    # Bandit arms
    assert names["music_recommender_bandit_arms_count"].samples[0].value == 2
    arm_weights = {
        s.labels["arm"]: s.value
        for s in names["music_recommender_bandit_arm_weight"].samples
    }
    assert arm_weights["artist_sim"] == 0.4
    assert arm_weights["content_sim"] == 0.6

    # Champion
    champ_sample = names["music_recommender_bandit_champion_info"].samples[0]
    assert champ_sample.labels["snapshot"] == "bandit_snap_champion_v1.json"

    # Streaming queue
    assert names["music_recommender_streaming_queue_enabled"].samples[0].value == 1
    assert names["music_recommender_streaming_queue_size"].samples[0].value == 42
    assert (
        names["music_recommender_streaming_queue_capacity"].samples[0].value == 1000
    )
    assert (
        names["music_recommender_streaming_queue_utilization_ratio"].samples[0].value
        == 0.042
    )
    assert (
        names["music_recommender_streaming_records_enqueued_total"].samples[0].value
        == 100
    )
    assert (
        names["music_recommender_streaming_records_flushed_total"].samples[0].value
        == 58
    )

    # Maintenance worker
    assert names["music_recommender_maintenance_enabled"].samples[0].value == 1
    assert names["music_recommender_maintenance_running"].samples[0].value == 1
    assert (
        names["music_recommender_maintenance_sweeps_total"].samples[0].value == 4
    )
    assert (
        names["music_recommender_maintenance_records_folded_total"].samples[0].value
        == 58
    )
    sweep_dur_samples = {
        s.labels["stat"]: s.value
        for s in names["music_recommender_maintenance_sweep_duration_seconds"].samples
    }
    assert sweep_dur_samples["last"] == 0.15
    assert sweep_dur_samples["min"] == 0.12
    assert sweep_dur_samples["max"] == 0.20

    # Streaming health
    health_samples = {
        s.labels["status"]: s.value
        for s in names["music_recommender_streaming_health"].samples
    }
    assert health_samples["healthy"] == 1
    assert health_samples["degraded"] == 0
    assert health_samples["unhealthy"] == 0


def test_collect_service_metric_families_disabled_components() -> None:
    class MinimalService:
        artifact = None
        cold_start_policy = None
        feedback_queue = None
        maintenance_worker = None

        def streaming_health(self) -> dict[str, str]:
            return {"status": "disabled"}

    families = collect_service_metric_families(MinimalService())
    names = {f.name: f for f in families}

    assert names["music_recommender_up"].samples[0].value == 1
    assert names["music_recommender_streaming_queue_enabled"].samples[0].value == 0
    assert names["music_recommender_maintenance_enabled"].samples[0].value == 0
    assert "music_recommender_streaming_queue_size" not in names
    assert "music_recommender_maintenance_running" not in names

    health_samples = {
        s.labels["status"]: s.value
        for s in names["music_recommender_streaming_health"].samples
    }
    assert health_samples["disabled"] == 1

