"""Integration tests for Prometheus metrics, alert rules, and dashboards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

import api.main as api_main
from music_recommender.alerting import (
    DEFAULT_ALERT_RULES,
    AlertEvaluator,
    render_prometheus_rules,
)
from music_recommender.telemetry import (
    KNOWN_PROMETHEUS_METRIC_NAMES,
    collect_service_metric_families,
    export_metrics_json,
    export_prometheus_metrics,
    validate_dashboard_metrics,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).parent.parent


class MockIntegratedService:
    """Mock service providing complete telemetry metrics for integration test."""

    cold_start_policy = {"popular": 0.5, "balanced": 0.3, "long_tail": 0.2}

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "service": "music-recommender"}

    def streaming_health(self) -> dict[str, str]:
        return {"status": "healthy"}

    def get_champion_snapshot(self) -> dict[str, str]:
        return {"snapshot_name": "champion_v1.json"}


def test_grafana_overview_dashboard_end_to_end(repo_root: Path) -> None:
    """Verify that the shipped Grafana overview dashboard is valid."""
    dashboard_path = (
        repo_root
        / "configs"
        / "grafana"
        / "dashboards"
        / "music-recommender-overview.json"
    )
    assert dashboard_path.is_file(), f"Dashboard missing at: {dashboard_path}"

    result = validate_dashboard_metrics(dashboard_path)
    assert result.is_valid, f"Dashboard validation failed: {result.errors}"
    assert result.total_panels == 23
    assert result.total_queries >= 20
    assert len(result.errors) == 0
    assert len(result.unknown_metrics) == 0

    subsystem_checks = {
        "service_liveness": "music_recommender_up",
        "version_info": "music_recommender_info",
        "artifacts": "music_recommender_artifact_users_count",
        "bandit_arms": "music_recommender_bandit_arm_weight",
        "streaming_queue": "music_recommender_streaming_queue_utilization_ratio",
        "maintenance_worker": "music_recommender_maintenance_running",
        "http_traffic": "music_recommender_http_requests_total",
    }
    for subsystem, metric_name in subsystem_checks.items():
        assert metric_name in result.referenced_metrics, (
            f"Subsystem '{subsystem}' metric '{metric_name}' missing from dashboard"
        )


def test_grafana_provisioning_configurations(repo_root: Path) -> None:
    """Verify that Grafana provisioning files are valid YAML."""
    dashboards_prov = (
        repo_root
        / "configs"
        / "grafana"
        / "provisioning"
        / "dashboards"
        / "dashboards.yaml"
    )
    assert dashboards_prov.is_file()
    with open(dashboards_prov, encoding="utf-8") as f:
        dash_data = yaml.safe_load(f)
    assert dash_data["apiVersion"] == 1
    assert len(dash_data["providers"]) > 0
    assert dash_data["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"

    datasources_prov = (
        repo_root
        / "configs"
        / "grafana"
        / "provisioning"
        / "datasources"
        / "datasources.yaml"
    )
    assert datasources_prov.is_file()
    with open(datasources_prov, encoding="utf-8") as f:
        ds_data = yaml.safe_load(f)
    assert ds_data["apiVersion"] == 1
    assert len(ds_data["datasources"]) > 0
    assert ds_data["datasources"][0]["type"] == "prometheus"
    assert ds_data["datasources"][0]["url"] == "http://prometheus:9090"


def test_alert_rules_metric_references_align_with_telemetry(repo_root: Path) -> None:
    """Verify that all default alert rules reference known metrics."""
    for rule in DEFAULT_ALERT_RULES:
        assert rule.metric in KNOWN_PROMETHEUS_METRIC_NAMES, (
            f"AlertRule '{rule.name}' references unknown metric '{rule.metric}'"
        )

    alert_rules_file = repo_root / "configs" / "alert-rules.yml"
    assert alert_rules_file.is_file()
    generated_yaml = render_prometheus_rules(DEFAULT_ALERT_RULES)
    assert alert_rules_file.read_text(encoding="utf-8") == generated_yaml


def test_observability_pipeline_api_and_service_integration() -> None:
    """Verify end-to-end integration between FastAPI and RecommenderService."""
    with TestClient(api_main.app) as client:
        orig_service = api_main.service
        orig_err = api_main.service_load_error
        try:
            api_main.service = MockIntegratedService()
            api_main.service_load_error = None

            health_resp = client.get("/health")
            assert health_resp.status_code == 200

            metrics_resp = client.get("/metrics")
            assert metrics_resp.status_code == 200
            assert "text/plain" in metrics_resp.headers["content-type"]
            text = metrics_resp.text

            assert "music_recommender_info" in text
            assert "music_recommender_up 1" in text
            assert "music_recommender_http_requests_total" in text

            json_metrics_text = export_metrics_json(api_main.service)
            parsed = json.loads(json_metrics_text)
            assert "metrics" in parsed
            assert isinstance(parsed["metrics"], list)

            metric_names = {item["name"] for item in parsed["metrics"]}
            assert "music_recommender_info" in metric_names
            assert "music_recommender_up" in metric_names

            prom_text = export_prometheus_metrics(api_main.service)
            assert "music_recommender_info" in prom_text
            assert "music_recommender_up 1" in prom_text
        finally:
            api_main.service = orig_service
            api_main.service_load_error = orig_err


def test_alert_evaluator_against_live_telemetry() -> None:
    """Verify AlertEvaluator evaluates against extracted metric families."""
    service = MockIntegratedService()
    families = collect_service_metric_families(service)

    evaluator = AlertEvaluator(DEFAULT_ALERT_RULES)
    transitions = evaluator.evaluate(families)

    assert isinstance(transitions, list)
    states = evaluator.states()
    assert len(states) == len(DEFAULT_ALERT_RULES)
    for s in states:
        assert s["status"] in ("inactive", "pending", "firing", "resolved")
