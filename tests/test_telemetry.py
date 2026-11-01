"""Tests for Prometheus exposition models, formatting, and telemetry."""

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from music_recommender.telemetry import (
    MetricFamily,
    MetricSample,
    MetricType,
    RequestTelemetryTracker,
    TelemetryPushAdapter,
    TelemetryPushConfig,
    TelemetryPushProtocol,
    collect_service_metric_families,
    escape_help_string,
    escape_influxdb_key,
    escape_influxdb_tag_value,
    escape_label_value,
    export_influxdb_metrics,
    export_metrics_json,
    export_prometheus_metrics,
    format_influxdb_field_value,
    format_metric_value,
    render_influxdb_line_protocol,
    render_prometheus_exposition,
    validate_dashboard_metrics,
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
        names["music_recommender_artifact_interactions_count"].samples[0].value == 35000
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
    assert names["music_recommender_streaming_queue_capacity"].samples[0].value == 1000
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
    assert names["music_recommender_maintenance_sweeps_total"].samples[0].value == 4
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


def test_request_telemetry_tracker() -> None:
    tracker = RequestTelemetryTracker(latency_buckets=(0.01, 0.05, 0.1, float("inf")))

    # Record 2 requests
    tracker.record_request(
        "/recommend/user", method="GET", status_code=200, duration_seconds=0.02
    )
    tracker.record_request(
        "/recommend/user", method="GET", status_code=200, duration_seconds=0.08
    )
    tracker.record_request(
        "/recommend/user", method="GET", status_code=500, duration_seconds=0.005
    )

    families = tracker.to_metric_families()
    names = {f.name: f for f in families}

    assert "music_recommender_http_requests_total" in names
    req_samples = names["music_recommender_http_requests_total"].samples
    assert len(req_samples) == 2

    sc_counts = {s.labels["status"]: s.value for s in req_samples}
    assert sc_counts["200"] == 2
    assert sc_counts["500"] == 1

    assert "music_recommender_http_request_duration_seconds" in names
    hist_fam = names["music_recommender_http_request_duration_seconds"]

    # Check sum and count
    sum_samples = [s for s in hist_fam.samples if s.name.endswith("_sum")]
    count_samples = [s for s in hist_fam.samples if s.name.endswith("_count")]
    assert len(sum_samples) == 1
    assert len(count_samples) == 1
    assert sum_samples[0].value == 0.105
    assert count_samples[0].value == 3

    # Check bucket samples
    bucket_samples = {
        s.labels["le"]: s.value for s in hist_fam.samples if s.name.endswith("_bucket")
    }
    assert bucket_samples["0.01"] == 1  # 0.005 <= 0.01
    assert bucket_samples["0.05"] == 2  # 0.005 and 0.02 <= 0.05
    assert bucket_samples["0.1"] == 3  # 0.005, 0.02, 0.08 <= 0.1
    assert bucket_samples["+Inf"] == 3

    # Test reset
    tracker.reset()
    assert len(tracker.to_metric_families()[0].samples) == 0


def test_export_prometheus_metrics() -> None:
    tracker = RequestTelemetryTracker()
    tracker.record_request(
        "/health", method="GET", status_code=200, duration_seconds=0.001
    )

    text = export_prometheus_metrics(
        service=None,
        request_tracker=tracker,
        custom_labels={"app": "recommender"},
    )
    assert text.endswith("\n")
    assert "# HELP music_recommender_info" in text
    assert 'music_recommender_info{app="recommender"' in text
    assert "music_recommender_up 0" in text
    assert "# HELP music_recommender_http_requests_total" in text
    assert (
        'music_recommender_http_requests_total{endpoint="/health",'
        'method="GET",status="200"} 1' in text
    )
    assert "# TYPE music_recommender_http_request_duration_seconds histogram" in text


def test_export_metrics_json() -> None:
    import json

    tracker = RequestTelemetryTracker()
    tracker.record_request(
        "/alerts", method="GET", status_code=200, duration_seconds=0.005
    )

    json_text = export_metrics_json(
        service=None,
        request_tracker=tracker,
        custom_labels={"region": "eu-west-1"},
    )
    assert json_text.endswith("\n")
    data = json.loads(json_text)
    assert "metrics" in data
    assert isinstance(data["metrics"], list)

    metrics_by_name = {m["name"]: m for m in data["metrics"]}
    assert "music_recommender_info" in metrics_by_name
    info_m = metrics_by_name["music_recommender_info"]
    assert info_m["type"] == "gauge"
    assert info_m["samples"][0]["labels"]["region"] == "eu-west-1"

    assert "music_recommender_up" in metrics_by_name
    assert metrics_by_name["music_recommender_up"]["samples"][0]["value"] == 0

    assert "music_recommender_http_requests_total" in metrics_by_name
    http_m = metrics_by_name["music_recommender_http_requests_total"]
    assert http_m["type"] == "counter"
    sample = http_m["samples"][0]
    assert sample["labels"]["endpoint"] == "/alerts"
    assert sample["value"] == 1


def test_telemetry_configs_valid_yaml() -> None:
    root_dir = Path(__file__).resolve().parent.parent

    # 1. Prometheus config validation
    prom_path = root_dir / "configs" / "prometheus.yml"
    assert prom_path.exists(), f"Prometheus config missing at {prom_path}"

    with open(prom_path, encoding="utf-8") as f:
        prom_data = yaml.safe_load(f)

    assert isinstance(prom_data, dict)
    assert "global" in prom_data
    assert "scrape_configs" in prom_data
    scrape_jobs = {job["job_name"]: job for job in prom_data["scrape_configs"]}
    assert "music_recommender" in scrape_jobs
    mr_job = scrape_jobs["music_recommender"]
    assert mr_job["metrics_path"] == "/metrics"
    assert "localhost:8000" in mr_job["static_configs"][0]["targets"]

    # 2. OpenTelemetry Collector config validation
    otel_path = root_dir / "configs" / "otel-collector-config.yaml"
    assert otel_path.exists(), f"OpenTelemetry config missing at {otel_path}"

    with open(otel_path, encoding="utf-8") as f:
        otel_data = yaml.safe_load(f)

    assert isinstance(otel_data, dict)
    assert "receivers" in otel_data
    assert "prometheus" in otel_data["receivers"]
    assert "processors" in otel_data
    assert "batch" in otel_data["processors"]
    assert "exporters" in otel_data
    assert "otlp" in otel_data["exporters"]
    assert "prometheus" in otel_data["exporters"]
    assert "service" in otel_data
    assert "pipelines" in otel_data["service"]
    assert "metrics" in otel_data["service"]["pipelines"]


def test_validate_dashboard_metrics_valid_overview() -> None:
    overview_path = (
        Path(__file__).parent.parent
        / "configs"
        / "grafana"
        / "dashboards"
        / "music-recommender-overview.json"
    )
    result = validate_dashboard_metrics(overview_path)
    assert result.is_valid
    assert result.dashboard_title == "Music Recommender System - Overview"
    assert result.total_panels == 23
    assert result.total_queries >= 20
    assert len(result.referenced_metrics) >= 15
    assert len(result.unknown_metrics) == 0
    assert len(result.errors) == 0

    as_dict = result.to_dict()
    assert as_dict["is_valid"] is True
    assert "music_recommender_up" in as_dict["referenced_metrics"]


def test_validate_dashboard_metrics_invalid_and_edge_cases(tmp_path: Path) -> None:
    # 1. Non-existent file
    missing = tmp_path / "nonexistent.json"
    res_missing = validate_dashboard_metrics(missing)
    assert not res_missing.is_valid
    assert any("does not exist" in e for e in res_missing.errors)

    # 2. Malformed JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    res_bad = validate_dashboard_metrics(bad_json)
    assert not res_bad.is_valid
    assert any("Failed to parse" in e for e in res_bad.errors)

    # 3. Invalid data type
    res_type = validate_dashboard_metrics(12345)  # type: ignore[arg-type]
    assert not res_type.is_valid
    assert any("Invalid dashboard data type" in e for e in res_type.errors)

    # 4. Unknown metrics detection
    bad_metric_dash = {
        "title": "Bad Metric Dashboard",
        "panels": [
            {
                "title": "Panel 1",
                "targets": [
                    {"expr": "rate(music_recommender_fictional_metric_total[5m])"}
                ],
            }
        ],
    }
    res_unknown = validate_dashboard_metrics(bad_metric_dash)
    assert not res_unknown.is_valid
    assert "music_recommender_fictional_metric_total" in res_unknown.unknown_metrics
    assert any("references unknown metric" in e for e in res_unknown.errors)

    # 5. Empty panels
    empty_dash = {"title": "Empty Dashboard", "panels": []}
    res_empty = validate_dashboard_metrics(empty_dash)
    assert not res_empty.is_valid
    assert any("contains no panels" in e for e in res_empty.errors)


def test_escape_influxdb_key_and_tag_value() -> None:
    assert escape_influxdb_key("simple_key") == "simple_key"
    assert escape_influxdb_key("key,with=special chars") == r"key\,with\=special\ chars"
    assert escape_influxdb_tag_value("tag value") == r"tag\ value"
    assert escape_influxdb_tag_value("foo=bar,baz") == r"foo\=bar\,baz"


def test_format_influxdb_field_value() -> None:
    assert format_influxdb_field_value(True) == "t"
    assert format_influxdb_field_value(False) == "f"
    assert format_influxdb_field_value(42) == "42i"
    assert format_influxdb_field_value(3.14159) == "3.14159"
    assert format_influxdb_field_value('quoted "text"') == '"quoted \\"text\\""'
    assert format_influxdb_field_value(float("nan")) is None
    assert format_influxdb_field_value(float("inf")) is None


def test_render_influxdb_line_protocol() -> None:
    fam = MetricFamily(
        name="music_recommender_queue_size",
        help_text="Queue size",
        metric_type=MetricType.GAUGE,
    )
    fam.add_sample(
        value=10, labels={"env": "prod", "host": "srv 01"}, timestamp_ms=1700000000000
    )

    lines = render_influxdb_line_protocol([fam], default_measurement="recommender")
    expected = (
        "recommender,env=prod,host=srv\\ 01 "
        "music_recommender_queue_size=10i 1700000000000000000\n"
    )
    assert lines == expected

    # Empty families
    assert render_influxdb_line_protocol([]) == ""


def test_export_influxdb_metrics_with_service() -> None:
    exported = export_influxdb_metrics(
        service=None,
        custom_labels={"tier": "streaming"},
        default_measurement="test_music",
        include_timestamps=True,
    )
    assert "test_music," in exported
    assert "tier=streaming" in exported
    assert "music_recommender_up=" in exported


def test_telemetry_push_config_validation() -> None:
    cfg = TelemetryPushConfig(
        endpoint_url="http://localhost:8086/api/v2/write",
        protocol="influxdb",
    )
    assert cfg.protocol == TelemetryPushProtocol.INFLUXDB

    with pytest.raises(ValueError, match="Invalid telemetry push endpoint URL"):
        TelemetryPushConfig(endpoint_url="invalid_url")

    with pytest.raises(ValueError, match="is not a valid TelemetryPushProtocol"):
        TelemetryPushConfig(
            endpoint_url="http://localhost:8086", protocol="unsupported_proto"
        )


def test_telemetry_push_adapter_dry_run() -> None:
    fam = MetricFamily(
        name="music_recommender_up",
        help_text="Service status",
        metric_type=MetricType.GAUGE,
    )
    fam.add_sample(value=1)

    adapter = TelemetryPushAdapter()

    for proto in (
        TelemetryPushProtocol.INFLUXDB,
        TelemetryPushProtocol.PROMETHEUS_TEXT,
        TelemetryPushProtocol.PROMETHEUS_JSON,
    ):
        cfg = TelemetryPushConfig(
            endpoint_url="http://localhost:8086",
            protocol=proto,
            dry_run=True,
        )
        res = adapter.push([fam], cfg)
        assert res.success is True
        assert res.samples_pushed == 1
        assert res.status_code == 200
        assert res.payload_bytes > 0
        as_dict = res.to_dict()
        assert as_dict["success"] is True


def test_telemetry_push_adapter_mock_http() -> None:
    fam = MetricFamily(
        name="music_recommender_up",
        help_text="Service status",
        metric_type=MetricType.GAUGE,
    )
    fam.add_sample(value=1)
    adapter = TelemetryPushAdapter()
    cfg = TelemetryPushConfig(
        endpoint_url="http://collector.example.com/write",
        protocol=TelemetryPushProtocol.INFLUXDB,
    )

    # 1. Success 204
    mock_resp = MagicMock()
    mock_resp.status = 204
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = adapter.push([fam], cfg)
        assert res.success is True
        assert res.status_code == 204

    # 2. HTTP Error 401
    http_err = urllib.error.HTTPError(
        url=cfg.endpoint_url,
        code=401,
        msg="Unauthorized",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        res_err = adapter.push([fam], cfg)
        assert res_err.success is False
        assert res_err.status_code == 401
        assert "Unauthorized" in (res_err.error or "")

    # 3. Connection Error
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        res_conn = adapter.push([fam], cfg)
        assert res_conn.success is False
        assert "Connection refused" in (res_conn.error or "")
