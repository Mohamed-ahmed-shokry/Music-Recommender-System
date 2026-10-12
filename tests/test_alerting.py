"""Tests for alert rules, evaluation, dispatch, and rule-file generation."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from music_recommender.alerting import (
    DEFAULT_ALERT_RULES,
    AlertEvaluator,
    AlertMode,
    AlertRule,
    AlertSeverity,
    AlertTransition,
    WebhookDispatcher,
    lookup_metric_value,
    render_prometheus_rules,
)
from music_recommender.telemetry import MetricFamily, collect_service_metric_families


def _rule(**overrides: object) -> AlertRule:
    kwargs: dict[str, object] = {
        "name": "HighThing",
        "metric": "some_metric",
        "comparator": ">",
        "threshold": 1,
        "severity": "warning",
        "summary": "Thing is high.",
    }
    kwargs.update(overrides)
    return AlertRule(**kwargs)  # type: ignore[arg-type]


def test_rule_normalizes_severity_and_mode() -> None:
    rule = _rule(severity="critical", mode="increase")
    assert rule.severity is AlertSeverity.CRITICAL
    assert rule.mode is AlertMode.INCREASE


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "bad name"},
        {"name": "1Bad"},
        {"metric": "bad-metric"},
        {"comparator": "=>"},
        {"severity": "fatal"},
        {"mode": "rate"},
        {"threshold": "high"},
        {"threshold": True},
        {"for_evaluations": 0},
        {"window": "five"},
        {"summary": "  "},
        {"labels": {"bad-label": "x"}},
    ],
)
def test_rule_rejects_invalid_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _rule(**overrides)


@pytest.mark.parametrize(
    ("comparator", "value", "expected"),
    [
        (">", 2, True),
        (">", 1, False),
        (">=", 1, True),
        ("<", 0, True),
        ("<=", 2, False),
        ("==", 1, True),
        ("!=", 1, False),
    ],
)
def test_rule_is_breached(comparator: str, value: float, expected: bool) -> None:
    assert _rule(comparator=comparator).is_breached(value) is expected


def test_default_rules_have_unique_names() -> None:
    names = [r.name for r in DEFAULT_ALERT_RULES]
    assert len(names) == len(set(names))
    assert {
        "ServiceDown",
        "StreamingQueueHighUtilization",
        "StreamingRecordsDropped",
        "StreamingFlushErrors",
        "MaintenanceWorkerStopped",
        "BanditDriftViolation",
    } <= set(names)


# --- evaluator -------------------------------------------------------------


def _families(**metrics: float) -> list[MetricFamily]:
    out = []
    for name, value in metrics.items():
        fam = MetricFamily(name=name, help_text="h", metric_type="gauge")
        fam.add_sample(value)
        out.append(fam)
    return out


def _fixed_clock() -> str:
    return "2026-01-01T00:00:00+00:00"


def test_lookup_metric_value_filters_labels_and_sums() -> None:
    fam = MetricFamily(name="m", help_text="h", metric_type="gauge")
    fam.add_sample(1, {"status": "a"})
    fam.add_sample(2, {"status": "b"})
    fam.add_sample(4, {"status": "b"})
    assert lookup_metric_value([fam], "m", {"status": "b"}) == 6
    assert lookup_metric_value([fam], "m") == 7
    assert lookup_metric_value([fam], "m", {"status": "z"}) is None
    assert lookup_metric_value([fam], "other") is None


def test_evaluator_rejects_duplicate_rule_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        AlertEvaluator([_rule(), _rule()])


def test_evaluator_fires_immediately_and_resolves() -> None:
    ev = AlertEvaluator([_rule(metric="m")], clock=_fixed_clock)
    assert ev.evaluate(_families(m=0)) == []
    fired = ev.evaluate(_families(m=5))
    assert [(t.rule.name, t.event, t.value) for t in fired] == [
        ("HighThing", "firing", 5)
    ]
    assert ev.evaluate(_families(m=6)) == []  # still firing: no re-notification
    assert ev.firing()[0]["active_since"] == _fixed_clock()
    resolved = ev.evaluate(_families(m=0))
    assert [t.event for t in resolved] == ["resolved"]
    assert ev.states()[0]["status"] == "inactive"


def test_evaluator_debounces_with_for_evaluations() -> None:
    ev = AlertEvaluator([_rule(metric="m", for_evaluations=3)])
    assert ev.evaluate(_families(m=5)) == []
    assert ev.states()[0]["status"] == "pending"
    assert ev.evaluate(_families(m=0)) == []  # recovery resets the streak
    assert ev.states()[0]["consecutive_breaches"] == 0
    assert ev.evaluate(_families(m=5)) == []
    assert ev.evaluate(_families(m=5)) == []
    assert [t.event for t in ev.evaluate(_families(m=5))] == ["firing"]


def test_evaluator_ignores_missing_metric_and_resolves_firing() -> None:
    ev = AlertEvaluator([_rule(metric="m")])
    assert ev.evaluate([]) == []
    assert ev.states()[0]["value"] is None
    ev.evaluate(_families(m=5))
    assert [t.event for t in ev.evaluate([])] == ["resolved"]


def test_evaluator_increase_mode_uses_deltas_and_handles_resets() -> None:
    ev = AlertEvaluator([_rule(metric="c", threshold=0, mode="increase")])
    assert ev.evaluate(_families(c=10)) == []  # baseline only
    assert ev.evaluate(_families(c=10)) == []  # no growth
    fired = ev.evaluate(_families(c=13))
    assert [(t.event, t.value) for t in fired] == [("firing", 3)]
    assert [t.event for t in ev.evaluate(_families(c=13))] == ["resolved"]
    reset = ev.evaluate(_families(c=2))  # counter reset after restart
    assert [(t.event, t.value) for t in reset] == [("firing", 2)]


def test_transition_payload_is_json_ready() -> None:
    ev = AlertEvaluator([_rule(metric="m", labels={"a": "b"})], clock=_fixed_clock)
    fam = MetricFamily(name="m", help_text="h", metric_type="gauge")
    fam.add_sample(5, {"a": "b"})
    payload = ev.evaluate([fam])[0].to_dict()
    assert json.loads(json.dumps(payload)) == {
        "alert": "HighThing",
        "event": "firing",
        "severity": "warning",
        "summary": "Thing is high.",
        "metric": "m",
        "labels": {"a": "b"},
        "threshold": 1,
        "comparator": ">",
        "value": 5,
        "timestamp": _fixed_clock(),
    }


# --- webhook dispatcher ----------------------------------------------------


def _transition(event: str = "firing") -> AlertTransition:
    return AlertTransition(_rule(), event, 5.0, _fixed_clock())


def test_dispatcher_rejects_bad_url_and_attempts() -> None:
    with pytest.raises(ValueError, match="http"):
        WebhookDispatcher("file:///etc/passwd")
    with pytest.raises(ValueError, match="max_attempts"):
        WebhookDispatcher("http://x", max_attempts=0)


def test_dispatcher_posts_one_payload_per_transition() -> None:
    sent: list[tuple[str, dict[str, object], float]] = []

    def sender(url: str, body: bytes, timeout: float) -> None:
        sent.append((url, json.loads(body), timeout))

    d = WebhookDispatcher("http://hook", timeout_seconds=2.0, sender=sender)
    assert d.dispatch([_transition(), _transition("resolved")]) == 2
    assert [s[1]["event"] for s in sent] == ["firing", "resolved"]
    assert sent[0][0] == "http://hook" and sent[0][2] == 2.0
    assert sent[0][1]["source"] == "music-recommender"


def test_dispatcher_retries_with_backoff_then_succeeds() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky(url: str, body: bytes, timeout: float) -> None:
        calls.append(1)
        if len(calls) < 3:
            raise OSError("boom")

    d = WebhookDispatcher(
        "http://hook", backoff_seconds=1.0, sender=flaky, sleep=sleeps.append
    )
    assert d.dispatch([_transition()]) == 1
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_dispatcher_swallows_final_failure(caplog: pytest.LogCaptureFixture) -> None:
    def broken(url: str, body: bytes, timeout: float) -> None:
        raise OSError("down")

    d = WebhookDispatcher(
        "http://hook", max_attempts=2, sender=broken, sleep=lambda s: None
    )
    with caplog.at_level("WARNING", logger="music_recommender.alerting"):
        assert d.dispatch([_transition()]) == 0
    assert "attempt 2/2" in caplog.text


def test_dispatcher_default_sender_posts_json_over_http() -> None:
    received: list[tuple[str | None, dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            received.append(
                (self.headers["Content-Type"], json.loads(self.rfile.read(length)))
            )
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/hook"
        assert WebhookDispatcher(url).dispatch([_transition()]) == 1
    finally:
        server.shutdown()
        server.server_close()
    assert received[0][0] == "application/json"
    assert received[0][1]["alert"] == "HighThing"


def test_dispatcher_default_sender_reports_http_errors() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(500)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/hook"
        d = WebhookDispatcher(url, max_attempts=1)
        assert d.dispatch([_transition()]) == 0
    finally:
        server.shutdown()
        server.server_close()


# --- Prometheus rule file --------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_render_prometheus_rules_structure() -> None:
    doc = yaml.safe_load(render_prometheus_rules(DEFAULT_ALERT_RULES))
    (group,) = doc["groups"]
    assert [r["alert"] for r in group["rules"]] == [r.name for r in DEFAULT_ALERT_RULES]
    by_name = {r["alert"]: r for r in group["rules"]}
    assert by_name["ServiceDown"]["expr"] == "music_recommender_up == 0"
    assert by_name["ServiceDown"]["labels"] == {"severity": "critical"}
    assert "for" not in by_name["ServiceDown"]
    assert by_name["StreamingHealthDegraded"]["expr"] == (
        'music_recommender_streaming_health{status="degraded"} == 1'
    )
    assert by_name["StreamingQueueHighUtilization"]["for"] == "30s"
    assert by_name["StreamingRecordsDropped"]["expr"] == (
        "increase(music_recommender_streaming_records_dropped_total[5m]) > 0"
    )


def test_checked_in_rule_file_matches_generator() -> None:
    on_disk = (CONFIGS_DIR / "alert-rules.yml").read_text(encoding="utf-8")
    assert on_disk == render_prometheus_rules(DEFAULT_ALERT_RULES)


def test_prometheus_config_references_rule_file() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "prometheus.yml").read_text(encoding="utf-8"))
    assert "alert-rules.yml" in " ".join(cfg["rule_files"])


# --- integration with telemetry -------------------------------------------


class _QueueMetrics:
    queue_size = 900
    max_queue_size = 1000
    utilization_pct = 90.0
    enqueued_count = 100
    flushed_count = 10
    dropped_count = 0
    flush_errors = 0
    flushes_count = 1


class _WorkerMetrics:
    is_running = True
    is_closed = False
    cycles_count = 1
    sweeps_count = 1
    records_folded_count = 1
    errors_count = 0
    last_sweep_duration_seconds = 0.1
    min_sweep_duration_seconds = 0.1
    max_sweep_duration_seconds = 0.1
    avg_sweep_duration_seconds = 0.1
    total_sweep_duration_seconds = 0.1
    drift_evaluations_count = 1
    drift_violations_count = 0
    rollbacks_count = 0


class _Holder:
    def __init__(self, metrics: object) -> None:
        self._metrics = metrics

    def get_metrics(self) -> object:
        return self._metrics


class _FakeService:
    artifact = None
    cold_start_policy = None

    def __init__(self) -> None:
        self.queue_metrics = _QueueMetrics()
        self.worker_metrics = _WorkerMetrics()
        self.feedback_queue = _Holder(self.queue_metrics)
        self.maintenance_worker = _Holder(self.worker_metrics)
        self.health = "healthy"

    def streaming_health(self) -> dict[str, str]:
        return {"status": self.health}


def test_default_rules_reference_emitted_metrics() -> None:
    families = collect_service_metric_families(_FakeService())
    for rule in DEFAULT_ALERT_RULES:
        assert lookup_metric_value(families, rule.metric, rule.labels) is not None, (
            rule.name
        )


def test_default_rules_fire_on_live_service_metrics() -> None:
    service = _FakeService()
    ev = AlertEvaluator()

    def run() -> dict[str, str]:
        ev.evaluate(collect_service_metric_families(service))
        return {s["alert"]: s["status"] for s in ev.states()}

    baseline = run()
    assert baseline["ServiceDown"] == "inactive"
    assert baseline["StreamingQueueHighUtilization"] == "pending"

    service.queue_metrics.dropped_count = 5
    service.worker_metrics.is_running = False
    service.worker_metrics.drift_violations_count = 2
    service.health = "unhealthy"
    run()
    states = run()
    assert states["StreamingHealthUnhealthy"] == "firing"
    assert states["MaintenanceWorkerStopped"] == "firing"
    assert states["StreamingQueueHighUtilization"] == "firing"
    # increase-mode alerts fired on the first delta and resolved on the next
    assert states["StreamingRecordsDropped"] == "inactive"
    assert states["BanditDriftViolation"] == "inactive"


def test_service_down_fires_without_service() -> None:
    ev = AlertEvaluator()
    fired = ev.evaluate(collect_service_metric_families(None))
    assert [t.rule.name for t in fired] == ["ServiceDown"]
