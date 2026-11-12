"""Tests for alert rules, evaluation, dispatch, and rule-file generation."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from music_recommender.alerting import (
    DEFAULT_ALERT_RULES,
    AlertEvaluator,
    AlertHistoryBuffer,
    AlertmanagerConfig,
    AlertMode,
    AlertRule,
    AlertSeverity,
    AlertSilenceRule,
    AlertSilencingEngine,
    AlertTransition,
    MaintenanceWindow,
    MultiChannelDispatcher,
    ReceivedAlertEvent,
    SilenceDecision,
    SilenceMatcher,
    SlackWebhookAdapter,
    TeamsWebhookAdapter,
    WebhookDispatcher,
    lookup_metric_value,
    parse_alert_payload,
    render_alertmanager_config,
    render_prometheus_rules,
)
from music_recommender.alerting import (
    main as alerting_main,
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
        thread.join(timeout=2.0)
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
        thread.join(timeout=2.0)


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


def test_main_writes_rule_file_and_validates_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "rules.yml"
    assert alerting_main([str(target)]) == 0
    assert target.read_text(encoding="utf-8") == render_prometheus_rules()
    assert b"\r\n" not in target.read_bytes()

    assert alerting_main([]) == 0
    assert capsys.readouterr().out == render_prometheus_rules()

    assert alerting_main(["a", "b"]) == 2
    assert "usage" in capsys.readouterr().err


def test_render_alertmanager_config() -> None:
    text = render_alertmanager_config()
    parsed = yaml.safe_load(text)
    assert parsed["global"]["resolve_timeout"] == "5m"
    assert parsed["route"]["receiver"] == "default-webhook"
    routes = parsed["route"]["routes"]
    assert len(routes) == 2
    assert routes[0]["match"]["severity"] == "critical"
    assert routes[0]["receiver"] == "critical-webhook"
    assert routes[1]["match"]["severity"] == "warning"
    assert routes[1]["receiver"] == "warning-webhook"

    inhibit = parsed["inhibit_rules"]
    assert len(inhibit) == 1
    assert inhibit[0]["source_match"]["alertname"] == "ServiceDown"
    assert inhibit[0]["target_match"]["severity"] == "warning"

    receivers = {r["name"]: r["webhook_configs"][0]["url"] for r in parsed["receivers"]}
    assert receivers["default-webhook"] == "http://api:8000/alerts/webhook"
    assert (
        receivers["critical-webhook"]
        == "http://api:8000/alerts/webhook?channel=critical"
    )
    assert (
        receivers["warning-webhook"] == "http://api:8000/alerts/webhook?channel=warning"
    )


def test_render_alertmanager_config_custom() -> None:
    cfg = AlertmanagerConfig(
        default_webhook_url="https://alerts.example.com/hook",
        critical_webhook_url="https://pager.example.com/critical",
        warning_webhook_url="https://chat.example.com/warning",
        resolve_timeout="10m",
        critical_repeat_interval="30m",
        inhibit_warning_on_service_down=False,
    )
    text = render_alertmanager_config(cfg)
    parsed = yaml.safe_load(text)
    assert parsed["global"]["resolve_timeout"] == "10m"
    assert parsed["route"]["routes"][0]["repeat_interval"] == "30m"
    assert parsed["inhibit_rules"] == []

    receivers = {r["name"]: r["webhook_configs"][0]["url"] for r in parsed["receivers"]}
    assert receivers["default-webhook"] == "https://alerts.example.com/hook"
    assert receivers["critical-webhook"] == "https://pager.example.com/critical"
    assert receivers["warning-webhook"] == "https://chat.example.com/warning"


def test_main_alertmanager_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "alertmanager.yml"
    assert alerting_main(["--alertmanager", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == render_alertmanager_config()

    assert alerting_main(["--alertmanager"]) == 0
    assert capsys.readouterr().out == render_alertmanager_config()

    assert alerting_main(["--alertmanager", "extra1", "extra2"]) == 2
    assert "usage" in capsys.readouterr().err


def test_parse_alert_payload_alertmanager_v4() -> None:
    payload = {
        "version": "4",
        "status": "firing",
        "commonLabels": {"environment": "production"},
        "commonAnnotations": {"source": "prometheus"},
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "ServiceDown",
                    "severity": "critical",
                    "metric": "music_recommender_up",
                },
                "annotations": {
                    "summary": "Service is unavailable.",
                },
                "startsAt": "2026-10-24T12:00:00Z",
                "endsAt": None,
                "value": 0.0,
            },
            {
                "status": "resolved",
                "labels": {
                    "alertname": "StreamingHealthDegraded",
                    "severity": "warning",
                },
                "annotations": {
                    "summary": "Streaming pipeline recovered.",
                },
                "startsAt": "2026-10-24T12:00:00Z",
                "endsAt": "2026-10-24T12:05:00Z",
            },
        ],
    }

    events = parse_alert_payload(payload, channel="critical")
    assert len(events) == 2

    e1 = events[0]
    assert e1.alertname == "ServiceDown"
    assert e1.status == "firing"
    assert e1.severity == "critical"
    assert e1.summary == "Service is unavailable."
    assert e1.channel == "critical"
    assert e1.metric == "music_recommender_up"
    assert e1.labels["environment"] == "production"
    assert e1.labels["severity"] == "critical"
    assert e1.value == 0.0
    assert e1.starts_at == "2026-10-24T12:00:00Z"
    assert e1.ends_at is None
    d1 = e1.to_dict()
    assert d1["alertname"] == "ServiceDown"

    e2 = events[1]
    assert e2.alertname == "StreamingHealthDegraded"
    assert e2.status == "resolved"
    assert e2.severity == "warning"
    assert e2.channel == "critical"


def test_parse_alert_payload_single_transition() -> None:
    payload = {
        "alert": "MaintenanceWorkerStopped",
        "event": "firing",
        "severity": "warning",
        "summary": "Maintenance worker stopped.",
        "metric": "music_recommender_maintenance_running",
        "labels": {"component": "worker"},
        "value": 0.0,
        "timestamp": "2026-10-24T12:10:00Z",
    }
    events = parse_alert_payload(payload, channel="warning")
    assert len(events) == 1
    e = events[0]
    assert e.alertname == "MaintenanceWorkerStopped"
    assert e.status == "firing"
    assert e.severity == "warning"
    assert e.summary == "Maintenance worker stopped."
    assert e.channel == "warning"
    assert e.labels["component"] == "worker"
    assert e.value == 0.0
    assert e.starts_at == "2026-10-24T12:10:00Z"


def test_alert_history_buffer() -> None:
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        AlertHistoryBuffer(capacity=0)

    buf = AlertHistoryBuffer(capacity=3)
    assert len(buf) == 0
    assert buf.get_events() == []

    e1 = ReceivedAlertEvent(
        "Alert1", "firing", "warning", "Summary 1", channel="warning"
    )
    e2 = ReceivedAlertEvent(
        "Alert2", "firing", "critical", "Summary 2", channel="critical"
    )
    e3 = ReceivedAlertEvent(
        "Alert3", "resolved", "warning", "Summary 3", channel="warning"
    )
    e4 = ReceivedAlertEvent(
        "Alert4", "firing", "critical", "Summary 4", channel="critical"
    )

    assert buf.record_many([e1, e2]) == 2
    assert len(buf) == 2

    buf.record(e3)
    assert len(buf) == 3

    # Adding 4th exceeds capacity 3, evicts e1
    buf.record(e4)
    assert len(buf) == 3

    all_events = buf.get_events()
    # Newest first
    assert [e.alertname for e in all_events] == ["Alert4", "Alert3", "Alert2"]

    # Filter by channel
    crit_events = buf.get_events(channel="critical")
    assert [e.alertname for e in crit_events] == ["Alert4", "Alert2"]

    # Filter by status
    resolved_events = buf.get_events(status="resolved")
    assert [e.alertname for e in resolved_events] == ["Alert3"]

    # Filter by severity
    warn_events = buf.get_events(severity="warning")
    assert [e.alertname for e in warn_events] == ["Alert3"]

    # Limit
    limited = buf.get_events(limit=1)
    assert len(limited) == 1
    assert limited[0].alertname == "Alert4"

    buf.clear()
    assert len(buf) == 0


def test_dead_letter_alert_record_roundtrip() -> None:
    from music_recommender.alerting import DeadLetterAlertRecord

    rec = DeadLetterAlertRecord(
        reason="malformed_json",
        raw_payload="invalid {json",
        channel="critical",
        error_message="Expecting property name",
        headers={"content-type": "application/json"},
        source="test_client",
    )
    d = rec.to_dict()
    assert d["reason"] == "malformed_json"
    assert d["raw_payload"] == "invalid {json"
    assert d["channel"] == "critical"
    assert d["headers"]["content-type"] == "application/json"

    restored = DeadLetterAlertRecord.from_dict(d)
    assert restored.reason == rec.reason
    assert restored.raw_payload == rec.raw_payload
    assert restored.channel == rec.channel
    assert restored.headers == rec.headers


def test_alert_spillover_storage(tmp_path: Path) -> None:
    from music_recommender.alerting import (
        AlertSpilloverStorage,
        DeadLetterAlertRecord,
        ReceivedAlertEvent,
    )

    storage = AlertSpilloverStorage(storage_dir=tmp_path / "alerts")
    assert storage.get_stats()["spillover_count"] == 0
    assert storage.get_stats()["dlq_count"] == 0

    ev1 = ReceivedAlertEvent("Spill1", "firing", "warning", "Sum1", channel="warning")
    ev2 = ReceivedAlertEvent("Spill2", "firing", "critical", "Sum2", channel="critical")
    assert storage.record_spillover(ev1)
    assert storage.record_spillover(ev2)

    dlq1 = DeadLetterAlertRecord(reason="unparseable", raw_payload="bad-data")
    assert storage.record_dead_letter(dlq1)

    stats = storage.get_stats()
    assert stats["spillover_count"] == 2
    assert stats["dlq_count"] == 1

    # Read spillover (newest first)
    spills = storage.read_spillover(limit=10)
    assert len(spills) == 2
    assert spills[0]["alertname"] == "Spill2"
    assert spills[1]["alertname"] == "Spill1"

    # Filter spillover
    crit_spills = storage.read_spillover(severity="critical")
    assert len(crit_spills) == 1
    assert crit_spills[0]["alertname"] == "Spill2"

    # Read DLQ
    dlq_items = storage.read_dead_letters(limit=10)
    assert len(dlq_items) == 1
    assert dlq_items[0]["reason"] == "unparseable"

    # Purge DLQ and spillover
    assert storage.purge_dead_letters() == 1
    assert storage.get_stats()["dlq_count"] == 0
    assert storage.read_dead_letters() == []

    assert storage.purge_spillover() == 2
    assert storage.get_stats()["spillover_count"] == 0


def test_alert_spillover_storage_corrupted_lines(tmp_path: Path) -> None:
    from music_recommender.alerting import AlertSpilloverStorage

    storage_dir = tmp_path / "alerts"
    storage_dir.mkdir(parents=True, exist_ok=True)
    # Write corrupted lines alongside valid line
    spill_file = storage_dir / "alerts_spillover.jsonl"
    spill_file.write_text(
        "invalid line 1\n"
        + '{"alertname": "ValidAlert", "status": "firing", '
        + '"severity": "warning", "channel": "warning"}\n'
        + "invalid line 2\n",
        encoding="utf-8",
    )

    storage = AlertSpilloverStorage(storage_dir=storage_dir)
    items = storage.read_spillover(limit=10)
    assert len(items) == 1
    assert items[0]["alertname"] == "ValidAlert"


def test_alert_history_buffer_eviction_spillover(tmp_path: Path) -> None:
    from music_recommender.alerting import (
        AlertHistoryBuffer,
        AlertSpilloverStorage,
        ReceivedAlertEvent,
    )

    storage = AlertSpilloverStorage(storage_dir=tmp_path / "alerts")
    buf = AlertHistoryBuffer(capacity=2, spillover_storage=storage, spill_on_evict=True)

    e1 = ReceivedAlertEvent("E1", "firing", "warning", "Sum1", channel="warning")
    e2 = ReceivedAlertEvent("E2", "firing", "critical", "Sum2", channel="critical")
    e3 = ReceivedAlertEvent("E3", "resolved", "warning", "Sum3", channel="warning")

    buf.record(e1)
    buf.record(e2)
    assert storage.get_stats()["spillover_count"] == 0

    # Evicts e1
    buf.record(e3)
    assert len(buf) == 2
    assert storage.get_stats()["spillover_count"] == 1
    spills = storage.read_spillover()
    assert len(spills) == 1
    assert spills[0]["alertname"] == "E1"

    # Combined events query: memory (E3, E2) + spillover (E1)
    combined = buf.get_combined_events(limit=10)
    assert len(combined) == 3
    assert [x["alertname"] for x in combined] == ["E3", "E2", "E1"]


def test_alert_replay_engine_file_loading(tmp_path: Path) -> None:
    from music_recommender.alerting import AlertReplayEngine

    engine = AlertReplayEngine()
    with pytest.raises(FileNotFoundError):
        engine.load_events_from_file(tmp_path / "nonexistent.json")

    # JSON array
    json_path = tmp_path / "events.json"
    json_path.write_text('[{"alertname": "A1"}, {"alertname": "A2"}]', encoding="utf-8")
    loaded_json = engine.load_events_from_file(json_path)
    assert len(loaded_json) == 2
    assert loaded_json[0]["alertname"] == "A1"

    # JSON Lines
    jsonl_path = tmp_path / "events.jsonl"
    jsonl_path.write_text(
        '{"alertname": "L1"}\n\n{"alertname": "L2"}\n', encoding="utf-8"
    )
    loaded_jsonl = engine.load_events_from_file(jsonl_path)
    assert len(loaded_jsonl) == 2
    assert loaded_jsonl[0]["alertname"] == "L1"


def test_alert_replay_engine_dry_run_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from music_recommender.alerting import AlertReplayEngine

    engine = AlertReplayEngine()
    events = [
        {"alertname": "AlertA", "status": "firing"},
        {"alertname": "AlertB", "status": "resolved"},
    ]

    # Dry-run test
    report_dry = engine.replay(events, dry_run=True)
    assert report_dry.dry_run is True
    assert report_dry.total_events == 2
    assert report_dry.dispatched == 2
    assert report_dry.failed == 0
    assert len(report_dry.results) == 2
    assert report_dry.results[0].alertname == "AlertA"
    assert report_dry.results[0].success is True

    # Live dispatch mocked
    def mock_dispatch(
        self: object, url: str, item: dict[str, object]
    ) -> tuple[bool, int | None, str | None]:
        if item.get("alertname") == "AlertA":
            return (True, 200, None)
        return (False, 500, "Internal Server Error")

    monkeypatch.setattr(AlertReplayEngine, "_dispatch_one", mock_dispatch)
    report_live = engine.replay(events, dry_run=False, interval_seconds=0.01)
    assert report_live.dry_run is False
    assert report_live.total_events == 2
    assert report_live.dispatched == 1
    assert report_live.failed == 1
    assert report_live.results[0].success is True
    assert report_live.results[1].success is False


def test_alert_causal_rule_validation() -> None:
    from music_recommender.alerting import AlertCausalRule

    rule = AlertCausalRule("Parent", "Child", description="Test rule")
    assert rule.parent_alert == "Parent"
    assert rule.child_alert == "Child"

    with pytest.raises(ValueError, match="must not be empty"):
        AlertCausalRule("", "Child")

    with pytest.raises(ValueError, match="Self-referential causal rule"):
        AlertCausalRule("SameAlert", "SameAlert")


def test_alert_correlation_graph_empty_and_resolved() -> None:
    from music_recommender.alerting import AlertCorrelationGraph

    graph = AlertCorrelationGraph()
    res_empty = graph.correlate([])
    assert res_empty.total_active_alerts == 0
    assert res_empty.root_causes == []
    assert res_empty.symptoms == []

    # All resolved
    res_resolved = graph.correlate(
        [
            {"alertname": "StreamingQueueHighUtilization", "status": "resolved"},
            {"alertname": "StreamingRecordsDropped", "status": "resolved"},
        ]
    )
    assert res_resolved.total_active_alerts == 0


def test_alert_correlation_graph_root_cause_and_symptoms() -> None:
    from music_recommender.alerting import AlertCorrelationGraph

    graph = AlertCorrelationGraph()
    now = time.time()
    events = [
        {"alertname": "MaintenanceWorkerStopped", "status": "firing", "timestamp": now},
        {
            "alertname": "StreamingQueueHighUtilization",
            "status": "firing",
            "timestamp": now + 10,
        },
        {
            "alertname": "StreamingRecordsDropped",
            "status": "firing",
            "timestamp": now + 20,
        },
        {"alertname": "UnrelatedAlert", "status": "firing", "timestamp": now + 5},
    ]

    report = graph.correlate(events, window_seconds=300.0)
    assert report.total_active_alerts == 4
    assert "MaintenanceWorkerStopped" in report.root_causes
    assert "StreamingQueueHighUtilization" in report.symptoms
    assert "StreamingRecordsDropped" in report.symptoms
    assert "UnrelatedAlert" in report.unlinked_alerts
    assert len(report.causal_chains) >= 1
    assert report.causal_chains[0] == [
        "MaintenanceWorkerStopped",
        "StreamingQueueHighUtilization",
        "StreamingRecordsDropped",
    ]

    d = report.to_dict()
    assert d["total_active_alerts"] == 4
    assert "MaintenanceWorkerStopped" in d["root_causes"]


def test_alert_correlation_graph_time_window_expiry() -> None:
    from music_recommender.alerting import AlertCorrelationGraph

    graph = AlertCorrelationGraph()
    t0 = 1000.0
    events = [
        {
            "alertname": "StreamingQueueHighUtilization",
            "status": "firing",
            "timestamp": t0,
        },
        # Child alert fires 500s later (window is 100s)
        {
            "alertname": "StreamingRecordsDropped",
            "status": "firing",
            "timestamp": t0 + 500.0,
        },
    ]
    report = graph.correlate(events, window_seconds=100.0)
    # Child alert should not be linked because lag exceeds window
    assert "StreamingRecordsDropped" not in report.symptoms
    assert "StreamingRecordsDropped" in report.unlinked_alerts


def test_alert_correlation_graph_received_alert_event_objects() -> None:
    from music_recommender.alerting import (
        AlertCorrelationGraph,
        AlertSeverity,
        ReceivedAlertEvent,
    )

    graph = AlertCorrelationGraph()
    e1 = ReceivedAlertEvent(
        alertname="BanditDriftViolation",
        status="firing",
        severity=AlertSeverity.WARNING,
        summary="Policy drift detected",
        starts_at="2026-10-31T01:00:00Z",
    )
    e2 = ReceivedAlertEvent(
        alertname="BanditAutoRollback",
        status="firing",
        severity=AlertSeverity.CRITICAL,
        summary="Auto rollback executed",
        starts_at="2026-10-31T01:01:00Z",
    )
    report = graph.correlate([e1, e2], window_seconds=300.0)
    assert report.root_causes == ["BanditDriftViolation"]
    assert report.symptoms == ["BanditAutoRollback"]
    assert report.causal_chains == [["BanditDriftViolation", "BanditAutoRollback"]]


def test_alert_log_rotation_config() -> None:
    from music_recommender.alerting import AlertLogRotationConfig

    cfg = AlertLogRotationConfig(
        max_file_size_bytes=1024,
        max_backups=3,
        retention_days=7.0,
        auto_rotate=True,
    )
    d = cfg.to_dict()
    assert d["max_file_size_bytes"] == 1024
    assert d["max_backups"] == 3
    assert d["retention_days"] == 7.0
    assert d["auto_rotate"] is True


def test_alert_spillover_storage_rotation(tmp_path: Path) -> None:
    from music_recommender.alerting import (
        AlertLogRotationConfig,
        AlertSpilloverStorage,
        DeadLetterAlertRecord,
        ReceivedAlertEvent,
    )

    storage_dir = tmp_path / "alerts"
    cfg = AlertLogRotationConfig(
        max_file_size_bytes=150,  # low threshold to trigger auto-rotation
        max_backups=3,
        auto_rotate=True,
    )
    storage = AlertSpilloverStorage(storage_dir=storage_dir, rotation_config=cfg)

    # 1. Write several events to trigger size-capped rotation
    for i in range(5):
        ev = ReceivedAlertEvent(
            alertname=f"Alert_{i}",
            status="firing",
            severity="warning",
            summary=f"Event summary description {i}",
        )
        assert storage.record_spillover(ev)

    # Verify that rotated files exist
    rotations = list(storage_dir.glob("alerts_spillover.jsonl.*"))
    assert len(rotations) >= 1

    # 2. Test manual rotation
    dlq_rec = DeadLetterAlertRecord(reason="unparseable", raw_payload="bad-data")
    assert storage.record_dead_letter(dlq_rec)
    res = storage.rotate_storage(max_backups=2)
    assert res["spillover_rotated"] is True
    assert res["dlq_rotated"] is True
    assert (storage_dir / "alerts_dlq.jsonl.1").exists()

    stats = storage.get_stats()
    assert stats["spillover_rotations"] >= 1
    assert stats["dlq_rotations"] == 1
    assert stats["max_backups"] == 3


def test_alert_spillover_storage_retention_pruning(tmp_path: Path) -> None:
    import os
    import time

    from music_recommender.alerting import (
        AlertLogRotationConfig,
        AlertSpilloverStorage,
    )

    storage_dir = tmp_path / "alerts_retention"
    cfg = AlertLogRotationConfig(max_backups=2, retention_days=1.0)
    storage = AlertSpilloverStorage(storage_dir=storage_dir, rotation_config=cfg)

    # Create dummy rotated files: .1, .2, .3, .4
    f1 = storage_dir / "alerts_spillover.jsonl.1"
    f2 = storage_dir / "alerts_spillover.jsonl.2"
    f3 = storage_dir / "alerts_spillover.jsonl.3"
    f4 = storage_dir / "alerts_spillover.jsonl.4"
    for f in (f1, f2, f3, f4):
        f.write_text('{"dummy": true}\n', encoding="utf-8")

    # Enforce retention with max_backups=2 (should prune .3 and .4)
    res = storage.enforce_retention(max_backups=2)
    assert res["pruned_files"] == 2
    assert f1.exists()
    assert f2.exists()
    assert not f3.exists()
    assert not f4.exists()

    # Old age pruning: set f2 mtime to 3 days ago
    old_time = time.time() - (3 * 86400)
    os.utime(f2, (old_time, old_time))
    res2 = storage.enforce_retention(max_age_days=1.0)
    assert res2["pruned_files"] == 1
    assert not f2.exists()
    assert f1.exists()


def test_alert_spillover_storage_archive_tar_and_zip(tmp_path: Path) -> None:
    from music_recommender.alerting import (
        AlertSpilloverStorage,
        DeadLetterAlertRecord,
        ReceivedAlertEvent,
    )

    src_dir = tmp_path / "alerts_src"
    storage = AlertSpilloverStorage(storage_dir=src_dir)
    ev = ReceivedAlertEvent(
        alertname="ArchiveTestAlert",
        status="firing",
        severity="critical",
        summary="Testing archive bundling",
    )
    storage.record_spillover(ev)
    dlq = DeadLetterAlertRecord(reason="unparseable", raw_payload="malformed_json_here")
    storage.record_dead_letter(dlq)

    # 1. Test .tar.gz bundle
    tar_path = tmp_path / "bundle.tar.gz"
    res_path = storage.create_archive_bundle(tar_path, format="tar.gz")
    assert res_path.exists()
    assert res_path.stat().st_size > 0

    restore_dir = tmp_path / "alerts_restored_tar"
    restore_info = storage.restore_archive_bundle(tar_path, target_dir=restore_dir)
    assert restore_info["status"] == "extracted"
    assert (restore_dir / "alerts_spillover.jsonl").exists()
    assert (restore_dir / "alerts_dlq.jsonl").exists()
    assert restore_info["manifest"]["file_count"] == 2

    # 2. Test .zip bundle
    zip_path = tmp_path / "bundle.zip"
    res_zip = storage.create_archive_bundle(zip_path, format="zip")
    assert res_zip.exists()
    assert res_zip.stat().st_size > 0

    restore_zip_dir = tmp_path / "alerts_restored_zip"
    restore_zip_info = storage.restore_archive_bundle(
        zip_path, target_dir=restore_zip_dir
    )
    assert restore_zip_info["status"] == "extracted"
    assert (restore_zip_dir / "alerts_spillover.jsonl").exists()
    assert (restore_zip_dir / "alerts_dlq.jsonl").exists()


def test_dlq_retry_policy() -> None:
    from music_recommender.alerting import DLQRetryPolicy

    pol = DLQRetryPolicy(
        max_retries=4,
        base_delay_seconds=2.0,
        backoff_multiplier=2.0,
        max_delay_seconds=10.0,
    )
    assert pol.calculate_delay(0) == 2.0
    assert pol.calculate_delay(1) == 4.0
    assert pol.calculate_delay(2) == 8.0
    assert pol.calculate_delay(3) == 10.0  # capped at max_delay_seconds
    d = pol.to_dict()
    assert d["max_retries"] == 4


def test_alert_spillover_storage_dlq_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from music_recommender.alerting import (
        AlertSpilloverStorage,
        DeadLetterAlertRecord,
        DLQRetryPolicy,
    )

    storage_dir = tmp_path / "alerts_dlq_retry"
    storage = AlertSpilloverStorage(storage_dir=storage_dir)

    r1 = DeadLetterAlertRecord(
        reason="network_timeout",
        raw_payload='{"alertname": "Test1"}',
        channel="default",
    )
    r2 = DeadLetterAlertRecord(
        reason="invalid_signature",
        raw_payload='{"alertname": "Test2"}',
        channel="critical",
    )
    assert storage.record_dead_letter(r1)
    assert storage.record_dead_letter(r2)

    # 1. Dry run
    report_dry = storage.retry_dead_letters(dry_run=True)
    assert report_dry.dry_run is True
    assert report_dry.total_scanned == 2
    assert report_dry.succeeded == 2
    assert len(storage.read_dead_letters()) == 2

    # 2. Channel filter with simulated failure
    def mock_fail(url: str, rec: DeadLetterAlertRecord) -> tuple[bool, int, str]:
        return False, 500, "Internal Server Error"

    monkeypatch.setattr(storage, "_dispatch_retry_record", mock_fail)
    report_fail = storage.retry_dead_letters(
        policy=DLQRetryPolicy(max_retries=1), channel="default"
    )
    assert report_fail.retried == 1
    assert report_fail.failed == 1

    records_after_fail = storage.read_dead_letters()
    assert len(records_after_fail) == 2
    r_default = [r for r in records_after_fail if r["channel"] == "default"][0]
    assert r_default["retry_count"] == 1

    # 3. Retry again: max_retries exhausted
    report_exhausted = storage.retry_dead_letters(
        policy=DLQRetryPolicy(max_retries=1), channel="default"
    )
    assert report_exhausted.exhausted == 1

    # 4. Success case: resolves and removes from DLQ
    def mock_success(url: str, rec: DeadLetterAlertRecord) -> tuple[bool, int, None]:
        return True, 200, None

    monkeypatch.setattr(storage, "_dispatch_retry_record", mock_success)
    report_success = storage.retry_dead_letters(channel="critical")
    assert report_success.succeeded == 1
    # Critical record is resolved and removed; only default remains
    remaining = storage.read_dead_letters()
    assert len(remaining) == 1
    assert remaining[0]["channel"] == "default"


def test_throttler_disabled() -> None:
    from music_recommender.alerting import (
        AlertNotificationThrottler,
        AlertThrottleConfig,
        AlertTransition,
        ThrottleDecision,
    )

    config = AlertThrottleConfig(enabled=False)
    throttler = AlertNotificationThrottler(config)
    rule = _rule(name="HighLatency")
    transition = AlertTransition(
        rule=rule,
        event="firing",
        value=250.0,
        timestamp="2026-11-09T10:00:00Z",
    )

    allowed, decision, reason = throttler.should_dispatch(transition, now=100.0)
    assert allowed is True
    assert decision == ThrottleDecision.ALLOWED
    assert "disabled" in reason


def test_throttler_deduplication() -> None:
    from music_recommender.alerting import (
        AlertNotificationThrottler,
        AlertThrottleConfig,
        AlertTransition,
        ThrottleDecision,
    )

    config = AlertThrottleConfig(
        dedup_window_seconds=30.0,
        repeat_interval_seconds=300.0,
        max_alerts_per_window=10,
    )
    throttler = AlertNotificationThrottler(config)
    rule = _rule(name="HighLatency")
    t1 = AlertTransition(
        rule=rule,
        event="resolved",
        value=150.0,
        timestamp="2026-11-09T10:00:00Z",
    )

    # First transition allowed
    allowed, decision, _ = throttler.should_dispatch(t1, now=100.0)
    assert allowed is True
    assert decision == ThrottleDecision.ALLOWED

    # Duplicate transition within 30s suppressed
    allowed, decision, reason = throttler.should_dispatch(t1, now=110.0)
    assert allowed is False
    assert decision == ThrottleDecision.SUPPRESSED_DUPLICATE
    assert "Duplicate transition" in reason

    # After dedup window passes, allowed again
    allowed, decision, _ = throttler.should_dispatch(t1, now=135.0)
    assert allowed is True
    assert decision == ThrottleDecision.ALLOWED


def test_throttler_repeat_interval() -> None:
    from music_recommender.alerting import (
        AlertNotificationThrottler,
        AlertThrottleConfig,
        AlertTransition,
        ThrottleDecision,
    )

    config = AlertThrottleConfig(
        dedup_window_seconds=10.0,
        repeat_interval_seconds=60.0,
        max_alerts_per_window=10,
    )
    throttler = AlertNotificationThrottler(config)
    rule = _rule(name="HighLatency")
    t_firing = AlertTransition(
        rule=rule,
        event="firing",
        value=250.0,
        timestamp="2026-11-09T10:00:00Z",
    )

    # Initial firing is allowed
    allowed, decision, _ = throttler.should_dispatch(t_firing, now=100.0)
    assert allowed is True
    assert decision == ThrottleDecision.ALLOWED

    # 15 seconds later: dedup window has passed, but repeat interval hasn't
    allowed, decision, reason = throttler.should_dispatch(t_firing, now=115.0)
    assert allowed is False
    assert decision == ThrottleDecision.REPEAT_SUPPRESSED
    assert "Ongoing firing alert" in reason

    # 65 seconds later: repeat interval has passed
    allowed, decision, _ = throttler.should_dispatch(t_firing, now=165.0)
    assert allowed is True
    assert decision == ThrottleDecision.ALLOWED


def test_throttler_rate_limiting_and_sliding_window() -> None:
    from music_recommender.alerting import (
        AlertNotificationThrottler,
        AlertThrottleConfig,
        AlertTransition,
        ThrottleDecision,
    )

    config = AlertThrottleConfig(
        dedup_window_seconds=1.0,
        repeat_interval_seconds=1.0,
        window_seconds=60.0,
        max_alerts_per_window=2,
    )
    throttler = AlertNotificationThrottler(config)
    rule = _rule(name="CpuSpike")

    t1 = AlertTransition(rule=rule, event="firing", value=95.0, timestamp="T1")
    t2 = AlertTransition(rule=rule, event="resolved", value=80.0, timestamp="T2")
    t3 = AlertTransition(rule=rule, event="firing", value=96.0, timestamp="T3")

    # t1 at ts=10
    allowed1, d1, _ = throttler.should_dispatch(t1, now=10.0)
    assert allowed1 is True
    assert d1 == ThrottleDecision.ALLOWED

    # t2 at ts=20
    allowed2, d2, _ = throttler.should_dispatch(t2, now=20.0)
    assert allowed2 is True
    assert d2 == ThrottleDecision.ALLOWED

    # t3 at ts=30: limit of 2 in 60s window reached!
    allowed3, d3, reason = throttler.should_dispatch(t3, now=30.0)
    assert allowed3 is False
    assert d3 == ThrottleDecision.RATE_LIMITED
    assert "Rate limit exceeded" in reason

    # at ts=75: t1 at ts=10 is outside the 60s sliding window (75 - 60 = 15 > 10)
    allowed4, d4, _ = throttler.should_dispatch(t3, now=75.0)
    assert allowed4 is True
    assert d4 == ThrottleDecision.ALLOWED


def test_throttler_stats_and_reset() -> None:
    from music_recommender.alerting import (
        AlertNotificationThrottler,
        AlertThrottleConfig,
        AlertTransition,
    )

    throttler = AlertNotificationThrottler(
        AlertThrottleConfig(
            dedup_window_seconds=10.0,
            repeat_interval_seconds=60.0,
            max_alerts_per_window=1,
            window_seconds=60.0,
        )
    )
    rule = _rule(name="ErrorRate")
    t = AlertTransition(rule=rule, event="firing", value=10.0, timestamp="T")

    throttler.should_dispatch(t, now=10.0)  # allowed
    throttler.should_dispatch(t, now=12.0)  # duplicate suppressed
    throttler.should_dispatch(t, now=25.0)  # repeat suppressed / rate limited

    stats = throttler.get_stats()
    assert stats["total_evaluated"] == 3
    assert stats["allowed_count"] == 1
    assert stats["suppressed_duplicate_count"] == 1
    assert stats["tracked_rules"] == 1

    throttler.reset()
    clean_stats = throttler.get_stats()
    assert clean_stats["total_evaluated"] == 0
    assert clean_stats["allowed_count"] == 0
    assert clean_stats["tracked_rules"] == 0


def test_escalation_tier_and_policy_serialization() -> None:
    from music_recommender.alerting import (
        AlertEscalationPolicy,
        AlertEscalationTier,
    )

    t1 = AlertEscalationTier(level=1, delay_seconds=0.0, channels=["slack"])
    t2 = AlertEscalationTier(
        level=2, delay_seconds=300.0, channels=["pagerduty", "email"], repeat=True
    )
    policy = AlertEscalationPolicy(name="database_p1", tiers=[t2, t1], enabled=True)

    # Post init sorts tiers by delay
    assert policy.tiers[0].level == 1
    assert policy.tiers[1].level == 2

    p_dict = policy.to_dict()
    assert p_dict["name"] == "database_p1"
    assert len(p_dict["tiers"]) == 2
    assert p_dict["tiers"][0]["channels"] == ["slack"]

    restored = AlertEscalationPolicy.from_dict(p_dict)
    assert restored.name == policy.name
    assert len(restored.tiers) == 2
    assert restored.tiers[1].repeat is True


def test_escalation_multi_tier_lifecycle() -> None:
    from music_recommender.alerting import (
        AlertEscalationEngine,
        AlertEscalationPolicy,
        AlertEscalationTier,
        AlertTransition,
    )

    policy = AlertEscalationPolicy(
        name="api_critical",
        tiers=[
            AlertEscalationTier(level=1, delay_seconds=0.0, channels=["slack"]),
            AlertEscalationTier(level=2, delay_seconds=60.0, channels=["pagerduty"]),
            AlertEscalationTier(
                level=3, delay_seconds=180.0, channels=["opsgenie"], repeat=True
            ),
        ],
    )
    engine = AlertEscalationEngine()
    engine.register_policy("ServiceDown", policy)

    rule = _rule(name="ServiceDown")
    firing_t = AlertTransition(rule=rule, event="firing", value=1.0, timestamp="T0")

    # 1. At t=1000: Tier 1 triggers immediately
    events = engine.evaluate_transition(firing_t, now=1000.0)
    assert len(events) == 1
    assert events[0].tier_level == 1
    assert events[0].channels == ["slack"]

    # 2. At t=1030 (30s elapsed): neither tier 2 (60s) nor tier 3 (180s) has triggered
    events_30s = engine.evaluate_transition(firing_t, now=1030.0)
    assert len(events_30s) == 0

    # Active duration is 30s
    durations = engine.get_active_firing_durations(now=1030.0)
    assert durations["ServiceDown"] == 30.0

    # 3. At t=1065 (65s elapsed): Tier 2 triggers
    events_65s = engine.evaluate_transition(firing_t, now=1065.0)
    assert len(events_65s) == 1
    assert events_65s[0].tier_level == 2
    assert events_65s[0].channels == ["pagerduty"]

    # 4. At t=1080: Tier 2 already triggered, no new tier reached
    assert len(engine.evaluate_transition(firing_t, now=1080.0)) == 0

    # 5. At t=1200 (200s elapsed): Tier 3 triggers
    events_200s = engine.evaluate_transition(firing_t, now=1200.0)
    assert len(events_200s) == 1
    assert events_200s[0].tier_level == 3
    assert events_200s[0].channels == ["opsgenie"]

    # Tier 3 has repeat=True: at t=1250, Tier 3 triggers again
    events_repeat = engine.evaluate_transition(firing_t, now=1250.0)
    assert len(events_repeat) == 1
    assert events_repeat[0].tier_level == 3

    # Escalation history records events
    history = engine.get_history(limit=10)
    assert len(history) == 4

    # 6. Alert resolves
    resolved_t = AlertTransition(
        rule=rule, event="resolved", value=0.0, timestamp="T_res"
    )
    res_events = engine.evaluate_transition(resolved_t, now=1300.0)
    assert len(res_events) == 0
    assert engine.get_active_firing_durations(now=1300.0) == {}


def test_escalation_disabled_and_reset() -> None:
    from music_recommender.alerting import (
        AlertEscalationEngine,
        AlertEscalationPolicy,
        AlertTransition,
    )

    policy = AlertEscalationPolicy(name="disabled_p", enabled=False)
    engine = AlertEscalationEngine()
    engine.register_policy("IgnoredRule", policy)

    rule = _rule(name="IgnoredRule")
    t = AlertTransition(rule=rule, event="firing", value=1.0, timestamp="T")
    assert len(engine.evaluate_transition(t, now=100.0)) == 0

    # Reset clears policies and history
    engine.reset()
    assert len(engine.get_history()) == 0
    assert engine.get_active_firing_durations() == {}


def test_pagerduty_gateway_firing_and_resolve() -> None:
    from music_recommender.alerting import AlertTransition, PagerDutyGateway

    sent_requests: list[dict[str, Any]] = []

    def mock_sender(
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
        method: str,
    ) -> tuple[int, str]:
        sent_requests.append(
            {
                "url": url,
                "body": json.loads(body.decode("utf-8")),
                "headers": headers,
                "method": method,
            }
        )
        return 202, '{"status": "success", "message": "Event processed"}'

    gateway = PagerDutyGateway(
        routing_key="test-key-123",
        endpoint_url="https://events.pagerduty.com/v2/enqueue",
        sender=mock_sender,
    )

    rule = _rule(name="HighErrorRate", severity="critical")
    t_firing = AlertTransition(
        rule=rule, event="firing", value=15.0, timestamp="2026-11-09T10:00:00Z"
    )

    ok, status, resp = gateway.send(t_firing)
    assert ok is True
    assert status == 202
    assert len(sent_requests) == 1
    req = sent_requests[0]
    assert req["body"]["event_action"] == "trigger"
    assert req["body"]["routing_key"] == "test-key-123"
    assert req["body"]["payload"]["severity"] == "critical"

    # Now test resolve
    t_resolved = AlertTransition(
        rule=rule, event="resolved", value=0.0, timestamp="2026-11-09T10:05:00Z"
    )
    ok_res, status_res, _ = gateway.send(t_resolved)
    assert ok_res is True
    assert status_res == 202
    assert sent_requests[1]["body"]["event_action"] == "resolve"


def test_opsgenie_gateway_firing_and_close() -> None:
    from music_recommender.alerting import AlertTransition, OpsgenieGateway

    sent_requests: list[dict[str, Any]] = []

    def mock_sender(
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
        method: str,
    ) -> tuple[int, str]:
        sent_requests.append(
            {
                "url": url,
                "body": json.loads(body.decode("utf-8")),
                "headers": headers,
                "method": method,
            }
        )
        return 202, '{"result": "Request Will Be Processed"}'

    gateway = OpsgenieGateway(
        api_key="opsgenie-secret-key",
        endpoint_url="https://api.opsgenie.com/v2/alerts",
        sender=mock_sender,
    )

    rule = _rule(name="HighMemoryUsage", severity="critical")
    t_firing = AlertTransition(
        rule=rule, event="firing", value=92.5, timestamp="2026-11-09T10:00:00Z"
    )

    ok, status, _ = gateway.send(t_firing)
    assert ok is True
    assert status == 202
    assert len(sent_requests) == 1
    assert (
        sent_requests[0]["headers"]["Authorization"] == "GenieKey opsgenie-secret-key"
    )
    assert sent_requests[0]["body"]["priority"] == "P1"

    # Resolved transition sends close request
    t_resolved = AlertTransition(
        rule=rule, event="resolved", value=50.0, timestamp="2026-11-09T10:10:00Z"
    )
    ok_close, status_close, _ = gateway.send(t_resolved)
    assert ok_close is True
    assert status_close == 202
    assert "close?identifierType=alias" in sent_requests[1]["url"]


def test_multichannel_dispatcher_routing_and_throttling() -> None:
    from music_recommender.alerting import (
        AlertNotificationThrottler,
        AlertThrottleConfig,
        AlertTransition,
        MultiChannelDispatcher,
    )

    calls: list[tuple[str, str]] = []

    def fake_channel(name: str) -> Any:
        return lambda t: (calls.append((name, t.event)) or True, 200, "OK")

    throttler = AlertNotificationThrottler(
        AlertThrottleConfig(dedup_window_seconds=10.0, repeat_interval_seconds=15.0)
    )
    dispatcher = MultiChannelDispatcher(throttler=throttler)
    dispatcher.register_channel("slack", fake_channel("slack"))
    dispatcher.register_channel("pagerduty", fake_channel("pagerduty"))

    rule = _rule(name="DiskFull")
    t = AlertTransition(
        rule=rule, event="firing", value=99.0, timestamp="2026-11-09T10:00:00Z"
    )

    # 1. Initial dispatch to all registered channels
    res1 = dispatcher.dispatch(t, now=100.0)
    assert res1["slack"][0] is True
    assert res1["pagerduty"][0] is True
    assert len(calls) == 2

    # 2. Duplicate dispatch within dedup window is throttled
    res2 = dispatcher.dispatch(t, now=105.0)
    assert "throttled" in res2
    assert res2["throttled"][0] is False
    assert len(calls) == 2  # No extra calls made

    # 3. Explicit channel targeting
    calls.clear()
    res3 = dispatcher.dispatch(t, channels=["slack"], now=120.0)
    assert "slack" in res3
    assert "pagerduty" not in res3
    assert len(calls) == 1

    # 4. Stats and reset
    stats = dispatcher.get_stats()
    assert stats["dispatch_count"] == 2
    assert "slack" in stats["registered_channels"]

    dispatcher.reset()
    assert dispatcher.get_stats()["dispatch_count"] == 0


def test_remediation_rule_serialization() -> None:
    from music_recommender.alerting import (
        RemediationActionType,
        RemediationRule,
    )

    rule = RemediationRule(
        rule_name="StreamingQueueHighUtilization",
        action_type=RemediationActionType.FLUSH_FEEDBACK_QUEUE,
        cooldown_seconds=120.0,
        max_attempts=2,
        trigger_on_severity="critical",
        parameters={"force": True},
    )

    data = rule.to_dict()
    assert data["rule_name"] == "StreamingQueueHighUtilization"
    assert data["action_type"] == "flush_feedback_queue"
    assert data["cooldown_seconds"] == 120.0
    assert data["trigger_on_severity"] == "critical"
    assert data["parameters"] == {"force": True}

    restored = RemediationRule.from_dict(data)
    assert restored.rule_name == rule.rule_name
    assert restored.action_type == rule.action_type
    assert restored.cooldown_seconds == 120.0
    assert restored.max_attempts == 2
    assert restored.parameters == {"force": True}


def test_remediation_runner_lifecycle_and_cooldown() -> None:
    from music_recommender.alerting import (
        AlertTransition,
        IncidentRemediationRunner,
        RemediationActionType,
        RemediationExecutionStatus,
        RemediationRule,
    )

    executed_calls: list[str] = []

    def mock_flush(
        transition: AlertTransition | None,
        params: dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any]]:
        name = transition.rule.name if transition else "manual"
        executed_calls.append(name)
        return True, "Queue flushed successfully", {"flushed_records": 42}

    runner = IncidentRemediationRunner()
    runner.register_handler(RemediationActionType.FLUSH_FEEDBACK_QUEUE, mock_flush)

    rule = RemediationRule(
        rule_name="StreamingQueueFull",
        action_type=RemediationActionType.FLUSH_FEEDBACK_QUEUE,
        cooldown_seconds=60.0,
        max_attempts=2,
    )
    runner.register_rule(rule)

    alert_rule = _rule(name="StreamingQueueFull", severity="critical")
    t_firing = AlertTransition(
        rule=alert_rule,
        event="firing",
        value=98.0,
        timestamp="2026-11-10T10:00:00Z",
    )

    # 1. First trigger at t=100 -> executes successfully
    results_1 = runner.handle_transition(t_firing, now=100.0)
    assert len(results_1) == 1
    assert results_1[0].status == RemediationExecutionStatus.EXECUTED
    assert results_1[0].details["flushed_records"] == 42
    assert len(executed_calls) == 1

    # 2. Immediate second trigger at t=120 (< 60s cooldown) -> suppressed
    results_2 = runner.handle_transition(t_firing, now=120.0)
    assert len(results_2) == 1
    assert results_2[0].status == RemediationExecutionStatus.COOLDOWN_SUPPRESSED
    assert len(executed_calls) == 1  # No extra handler call

    # 3. Third trigger at t=170 (> 60s cooldown) -> executes second time
    results_3 = runner.handle_transition(t_firing, now=170.0)
    assert len(results_3) == 1
    assert results_3[0].status == RemediationExecutionStatus.EXECUTED
    assert len(executed_calls) == 2

    # 4. Fourth trigger at t=240 (> 60s cooldown, but max_attempts=2 exceeded)
    results_4 = runner.handle_transition(t_firing, now=240.0)
    assert len(results_4) == 1
    assert results_4[0].status == RemediationExecutionStatus.MAX_ATTEMPTS_EXCEEDED
    assert len(executed_calls) == 2

    # 5. Alert resolves -> resets attempt counters for this rule
    t_resolved = AlertTransition(
        rule=alert_rule,
        event="resolved",
        value=10.0,
        timestamp="2026-11-10T10:05:00Z",
    )
    res_resolved = runner.handle_transition(t_resolved, now=300.0)
    assert len(res_resolved) == 0

    # 6. Alert fires again at t=370 (> cooldown and attempts reset) -> executes
    results_5 = runner.handle_transition(t_firing, now=370.0)
    assert len(results_5) == 1
    assert results_5[0].status == RemediationExecutionStatus.EXECUTED
    assert len(executed_calls) == 3


def test_remediation_runner_dry_run_and_manual_execute() -> None:
    from music_recommender.alerting import (
        AlertTransition,
        IncidentRemediationRunner,
        RemediationActionType,
        RemediationExecutionStatus,
        RemediationRule,
    )

    runner = IncidentRemediationRunner()
    rule = RemediationRule(
        rule_name="ServiceRestartReq",
        action_type=RemediationActionType.RESTART_MAINTENANCE_WORKER,
        cooldown_seconds=10.0,
    )
    runner.register_rule(rule)

    alert_rule = _rule(name="ServiceRestartReq")
    t = AlertTransition(rule=alert_rule, event="firing", value=1.0, timestamp="T")

    # Missing handler -> FAILED
    fail_res = runner.handle_transition(t, now=10.0)
    assert fail_res[0].status == RemediationExecutionStatus.FAILED

    # Register handler
    called = []
    runner.register_handler(
        RemediationActionType.RESTART_MAINTENANCE_WORKER,
        lambda tr, p: (called.append(p) or True, "Restarted", {"worker_id": 1}),
    )

    # Dry-run
    dry_res = runner.handle_transition(t, now=25.0, dry_run=True)
    assert dry_res[0].status == RemediationExecutionStatus.DRY_RUN
    assert len(called) == 0

    # Manual execute
    man_res = runner.execute_action(
        "ServiceRestartReq",
        RemediationActionType.RESTART_MAINTENANCE_WORKER,
        parameters={"worker": "test"},
    )
    assert man_res.status == RemediationExecutionStatus.EXECUTED
    assert len(called) == 1

    # Check stats and history
    stats = runner.get_stats()
    assert stats["total_runs"] == 3
    assert stats["executed_count"] == 1
    assert len(runner.get_rules()) == 1

    runner.reset()
    assert runner.get_stats()["total_runs"] == 0


def test_circuit_breaker_config_serialization() -> None:
    from music_recommender.alerting import CircuitBreakerConfig

    cfg = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=45.0,
        half_open_success_threshold=1,
        enabled=True,
    )
    data = cfg.to_dict()
    assert data["failure_threshold"] == 3
    assert data["recovery_timeout_seconds"] == 45.0

    restored = CircuitBreakerConfig.from_dict(data)
    assert restored.failure_threshold == 3
    assert restored.recovery_timeout_seconds == 45.0
    assert restored.half_open_success_threshold == 1


def test_circuit_breaker_state_transitions() -> None:
    from music_recommender.alerting import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
    )

    cb = CircuitBreaker(
        name="test_cb",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=20.0,
            half_open_success_threshold=2,
        ),
    )

    # 1. Initially CLOSED
    assert cb.state == CircuitBreakerState.CLOSED
    allowed, msg = cb.allow_request(now=100.0)
    assert allowed is True
    assert "healthy" in msg

    # 2. Record 2 failures (< threshold 3) -> still CLOSED
    cb.record_failure("error 1", now=100.0)
    cb.record_failure("error 2", now=100.0)
    assert cb.state == CircuitBreakerState.CLOSED

    # 3. Third failure -> trips OPEN
    cb.record_failure("error 3", now=100.0)
    assert cb.state == CircuitBreakerState.OPEN
    stats = cb.get_stats()
    assert stats["total_trips"] == 1
    assert stats["state"] == "open"

    # 4. Request within recovery timeout (10s elapsed < 20s) -> rejected
    allowed_blocked, block_msg = cb.allow_request(now=110.0)
    assert allowed_blocked is False
    assert "Circuit open" in block_msg
    assert cb.get_stats()["total_rejected"] == 1

    # 5. Request after recovery timeout (25s elapsed >= 20s) -> probe
    allowed_probe, probe_msg = cb.allow_request(now=125.0)
    assert allowed_probe is True
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert "half-open" in probe_msg.lower()

    # 6. Probe failure -> trips back to OPEN
    cb.record_failure("probe failed", now=125.0)
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.get_stats()["total_trips"] == 2

    # 7. Advance time again -> enters HALF_OPEN
    allowed_probe2, _ = cb.allow_request(now=150.0)
    assert allowed_probe2 is True
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # 8. Record 2 consecutive successes -> closes circuit!
    cb.record_success(now=150.0)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb.record_success(now=151.0)
    assert cb.state == CircuitBreakerState.CLOSED

    # Healthy again
    allowed_recovered, _ = cb.allow_request(now=155.0)
    assert allowed_recovered is True


def test_circuit_breaker_disabled_and_reset() -> None:
    from music_recommender.alerting import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
    )

    cb = CircuitBreaker(config=CircuitBreakerConfig(enabled=False))
    cb.record_failure("fail")
    cb.record_failure("fail")
    cb.record_failure("fail")
    cb.record_failure("fail")
    cb.record_failure("fail")
    cb.record_failure("fail")
    allowed, _ = cb.allow_request()
    assert allowed is True

    # Test reset
    cb_active = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=1))
    cb_active.record_failure("fail")
    assert cb_active.state == CircuitBreakerState.OPEN
    cb_active.reset()
    assert cb_active.state == CircuitBreakerState.CLOSED
    assert cb_active.get_stats()["total_trips"] == 0


def test_webhook_dispatcher_with_circuit_breaker() -> None:
    from music_recommender.alerting import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerState,
        WebhookDispatcher,
    )

    cb = CircuitBreaker(
        name="test-webhook",
        config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=30.0),
    )
    calls = []

    def failing_sender(url: str, body: bytes, timeout: float) -> None:
        calls.append(url)
        raise ConnectionRefusedError("connection refused")

    dispatcher = WebhookDispatcher(
        url="http://test-webhook.internal/hook",
        max_attempts=1,
        circuit_breaker=cb,
        sender=failing_sender,
    )

    # First dispatch fails and records failure
    res1 = dispatcher.dispatch([_transition()])
    assert res1 == 0
    assert len(calls) == 1
    assert cb.state == CircuitBreakerState.CLOSED

    # Second dispatch fails and trips breaker
    res2 = dispatcher.dispatch([_transition()])
    assert res2 == 0
    assert len(calls) == 2
    assert cb.state == CircuitBreakerState.OPEN

    # Third dispatch is shed fast by circuit breaker without calling failing_sender
    res3 = dispatcher.dispatch([_transition()])
    assert res3 == 0
    assert len(calls) == 2  # sender was NOT called
    assert cb.get_stats()["total_trips"] == 1


def test_multichannel_dispatcher_circuit_breaker_isolation() -> None:
    from music_recommender.alerting import (
        AlertNotificationThrottler,
        AlertThrottleConfig,
        CircuitBreakerConfig,
        CircuitBreakerState,
        MultiChannelDispatcher,
    )

    pd_calls = 0
    slack_calls = 0

    def pd_handler(t: AlertTransition) -> tuple[bool, int, str]:
        nonlocal pd_calls
        pd_calls += 1
        return False, 500, "PD upstream error"

    def slack_handler(t: AlertTransition) -> tuple[bool, int, str]:
        nonlocal slack_calls
        slack_calls += 1
        return True, 200, "Slack ok"

    dispatcher = MultiChannelDispatcher(
        channels={"pagerduty": pd_handler, "slack": slack_handler},
        throttler=AlertNotificationThrottler(AlertThrottleConfig(enabled=False)),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout_seconds=60.0
        ),
    )

    t = _transition()

    # Dispatch 1: PD fails, Slack succeeds
    res1 = dispatcher.dispatch(t, channels=["pagerduty", "slack"], now=10.0)
    assert res1["pagerduty"] == (False, 500, "PD upstream error")
    assert res1["slack"] == (True, 200, "Slack ok")
    assert pd_calls == 1
    assert slack_calls == 1

    # Dispatch 2: PD fails again -> trips PD circuit breaker
    res2 = dispatcher.dispatch(t, channels=["pagerduty", "slack"], now=11.0)
    assert res2["pagerduty"] == (False, 500, "PD upstream error")
    assert res2["slack"] == (True, 200, "Slack ok")
    assert pd_calls == 2
    assert slack_calls == 2

    pd_cb = dispatcher.get_circuit_breaker("pagerduty")
    slack_cb = dispatcher.get_circuit_breaker("slack")
    assert pd_cb.state == CircuitBreakerState.OPEN
    assert slack_cb.state == CircuitBreakerState.CLOSED

    # Dispatch 3: PD fails fast (503) due to open circuit, Slack still succeeds
    res3 = dispatcher.dispatch(t, channels=["pagerduty", "slack"], now=12.0)
    assert res3["pagerduty"][0] is False
    assert res3["pagerduty"][1] == 503
    assert "Circuit open" in res3["pagerduty"][2]
    assert res3["slack"] == (True, 200, "Slack ok")
    assert pd_calls == 2  # PD handler not called!
    assert slack_calls == 3

    # Check stats include circuit breakers
    stats = dispatcher.get_stats()
    assert "circuit_breakers" in stats
    assert "pagerduty" in stats["circuit_breakers"]
    assert stats["circuit_breakers"]["pagerduty"]["state"] == "open"
    assert stats["circuit_breakers"]["slack"]["state"] == "closed"

    # Reset resets circuit breakers
    dispatcher.reset()
    assert pd_cb.state == CircuitBreakerState.CLOSED


def test_silence_matcher_matching() -> None:
    matcher_name = SilenceMatcher(name="alertname", value="HighErrorRate")
    assert matcher_name.matches("HighErrorRate", "critical")
    assert not matcher_name.matches("LowMemory", "critical")

    matcher_sev = SilenceMatcher(name="severity", value="warning")
    assert matcher_sev.matches("AnyAlert", "warning")
    assert not matcher_sev.matches("AnyAlert", "critical")

    matcher_lbl = SilenceMatcher(name="env", value="prod")
    assert matcher_lbl.matches("AnyAlert", "critical", {"env": "prod"})
    assert not matcher_lbl.matches("AnyAlert", "critical", {"env": "staging"})
    assert not matcher_lbl.matches("AnyAlert", "critical", {})

    matcher_regex = SilenceMatcher(
        name="cluster", value=r"^eu-(west|central)-\d$", is_regex=True
    )
    assert matcher_regex.matches("AnyAlert", "critical", {"cluster": "eu-west-1"})
    assert not matcher_regex.matches("AnyAlert", "critical", {"cluster": "us-east-1"})

    bad_regex = SilenceMatcher(name="cluster", value=r"[invalid", is_regex=True)
    assert not bad_regex.matches("AnyAlert", "critical", {"cluster": "eu-west-1"})

    data = matcher_regex.to_dict()
    restored = SilenceMatcher.from_dict(data)
    assert restored.name == "cluster"
    assert restored.value == r"^eu-(west|central)-\d$"
    assert restored.is_regex is True


def test_alert_silence_rule_lifecycle() -> None:
    now = 1000.0
    rule = AlertSilenceRule(
        silence_id="sil-1",
        matchers=[
            SilenceMatcher(name="alertname", value="HighMemory"),
            SilenceMatcher(name="severity", value="critical"),
        ],
        starts_at=900.0,
        ends_at=1100.0,
        author="sre-team",
        comment="Planned load test",
    )
    assert rule.is_active(now=now)
    assert rule.matches("HighMemory", "critical")
    assert not rule.matches("HighMemory", "warning")
    assert not rule.matches("LowDisk", "critical")

    # Time boundaries
    assert not rule.is_active(now=850.0)
    assert not rule.is_active(now=1150.0)

    # Disabled toggle
    rule.enabled = False
    assert not rule.is_active(now=now)

    # Roundtrip serialization
    data = rule.to_dict()
    restored = AlertSilenceRule.from_dict(data)
    assert restored.silence_id == "sil-1"
    assert len(restored.matchers) == 2
    assert restored.author == "sre-team"
    assert restored.comment == "Planned load test"


def test_maintenance_window_suppression() -> None:
    now = 2000.0
    mw_all = MaintenanceWindow(
        window_id="mw-1",
        name="Cluster Upgrade",
        starts_at=1900.0,
        ends_at=2100.0,
        suppress_all=True,
    )
    assert mw_all.is_active(now=now)
    assert mw_all.suppresses("AnyAlert", now=now)
    assert not mw_all.suppresses("AnyAlert", now=2200.0)

    mw_target = MaintenanceWindow(
        window_id="mw-2",
        name="Database Reindex",
        starts_at=1900.0,
        ends_at=2100.0,
        suppress_all=False,
        affected_rules=["DatabaseLatencyHigh", "LockContention"],
    )
    assert mw_target.suppresses("DatabaseLatencyHigh", now=now)
    assert not mw_target.suppresses("HighMemoryUsage", now=now)

    # Disabled toggle
    mw_target.enabled = False
    assert not mw_target.suppresses("DatabaseLatencyHigh", now=now)

    # Roundtrip serialization
    data = mw_target.to_dict()
    restored = MaintenanceWindow.from_dict(data)
    assert restored.window_id == "mw-2"
    assert restored.name == "Database Reindex"
    assert "DatabaseLatencyHigh" in restored.affected_rules


def test_alert_silencing_engine_evaluation_and_persistence(tmp_path: Path) -> None:
    now = 3000.0
    storage = tmp_path / "silences.json"
    engine = AlertSilencingEngine(storage_path=storage)

    # Initially empty
    assert engine.list_silences() == []
    assert engine.list_maintenance_windows() == []
    dec, target = engine.evaluate("HighErrorRate", "critical", now=now)
    assert dec == SilenceDecision.ACTIVE
    assert target is None

    # Add silence rule
    silence = AlertSilenceRule(
        silence_id="sil-test",
        matchers=[SilenceMatcher(name="alertname", value="HighErrorRate")],
        starts_at=2900.0,
        ends_at=3100.0,
    )
    engine.add_silence(silence)
    assert engine.get_silence("sil-test") is not None
    assert len(engine.list_silences(active_only=True, now=now)) == 1

    # Matching alert is suppressed
    dec, target = engine.evaluate("HighErrorRate", "critical", now=now)
    assert dec == SilenceDecision.SUPPRESSED_SILENCE_RULE
    assert target == "sil-test"

    # Non-matching alert remains active
    dec, target = engine.evaluate("MemoryExhausted", "critical", now=now)
    assert dec == SilenceDecision.ACTIVE

    # Maintenance window takes precedence
    mw = MaintenanceWindow(
        window_id="mw-global",
        starts_at=2900.0,
        ends_at=3100.0,
        suppress_all=True,
    )
    engine.add_maintenance_window(mw)
    dec, target = engine.evaluate("HighErrorRate", "critical", now=now)
    assert dec == SilenceDecision.SUPPRESSED_MAINTENANCE_WINDOW
    assert target == "mw-global"

    # Persistence verification: reload into fresh engine
    assert storage.exists()
    engine2 = AlertSilencingEngine(storage_path=storage)
    assert engine2.get_silence("sil-test") is not None
    assert engine2.get_maintenance_window("mw-global") is not None

    # Removal
    assert engine.remove_silence("sil-test")
    assert not engine.remove_silence("nonexistent")
    assert engine.remove_maintenance_window("mw-global")
    dec, target = engine.evaluate("HighErrorRate", "critical", now=now)
    assert dec == SilenceDecision.ACTIVE


def test_slack_webhook_adapter_payload_and_send() -> None:
    sent_requests: list[tuple[str, bytes, dict[str, str], float, str]] = []

    def mock_sender(
        url: str, body: bytes, headers: dict[str, str], timeout: float, method: str
    ) -> tuple[int, str]:
        sent_requests.append((url, body, headers, timeout, method))
        return 200, "ok"

    adapter = SlackWebhookAdapter(
        webhook_url="https://hooks.slack.com/services/T00/B00/X00",
        channel="#incidents",
        dashboard_url="https://dash.example.com",
        sender=mock_sender,
    )

    crit_rule = _rule(
        name="DiskFull",
        metric="node_disk_avail",
        threshold=10,
        comparator="<",
        severity="critical",
        summary="Disk space low",
    )
    t_crit = AlertTransition(
        rule=crit_rule,
        event="firing",
        value=5.0,
        timestamp="2026-11-11T12:00:00Z",
    )

    payload_crit = adapter.build_payload(t_crit)
    assert payload_crit["channel"] == "#incidents"
    assert payload_crit["attachments"][0]["color"] == "#E01E5A"
    blocks = payload_crit["attachments"][0]["blocks"]
    assert blocks[0]["type"] == "header"
    assert "CRITICAL" in blocks[0]["text"]["text"]
    assert blocks[1]["type"] == "section"
    assert "DiskFull" in payload_crit["text"]
    assert blocks[2]["type"] == "actions"
    assert blocks[2]["elements"][0]["url"] == "https://dash.example.com"

    # Send critical
    ok, code, msg = adapter.send(t_crit)
    assert ok is True
    assert code == 200
    assert len(sent_requests) == 1
    assert sent_requests[0][0] == "https://hooks.slack.com/services/T00/B00/X00"
    assert sent_requests[0][4] == "POST"

    # Warning severity
    warn_rule = _rule(name="HighLatency", severity="warning", summary="Latency warning")
    t_warn = AlertTransition(
        rule=warn_rule,
        event="firing",
        value=1.5,
        timestamp="2026-11-11T12:05:00Z",
    )
    payload_warn = adapter.build_payload(t_warn)
    assert payload_warn["attachments"][0]["color"] == "#ECB22E"

    # Resolved state
    t_res = AlertTransition(
        rule=crit_rule,
        event="resolved",
        value=50.0,
        timestamp="2026-11-11T12:10:00Z",
    )
    payload_res = adapter.build_payload(t_res)
    assert payload_res["attachments"][0]["color"] == "#2EB886"
    assert "RESOLVED" in payload_res["attachments"][0]["blocks"][0]["text"]["text"]


def test_teams_webhook_adapter_payload_and_send() -> None:
    sent_requests: list[tuple[str, bytes, dict[str, str], float, str]] = []

    def mock_sender(
        url: str, body: bytes, headers: dict[str, str], timeout: float, method: str
    ) -> tuple[int, str]:
        sent_requests.append((url, body, headers, timeout, method))
        return 200, "ok"

    adapter = TeamsWebhookAdapter(
        webhook_url="https://outlook.office.com/webhook/xxx",
        dashboard_url="https://dash.example.com",
        sender=mock_sender,
    )

    crit_rule = _rule(
        name="ServiceDown",
        metric="up",
        threshold=1,
        comparator="==",
        severity="critical",
        summary="Service is unreachable",
    )
    t_crit = AlertTransition(
        rule=crit_rule,
        event="firing",
        value=0.0,
        timestamp="2026-11-11T12:00:00Z",
    )

    payload_crit = adapter.build_payload(t_crit)
    assert payload_crit["@type"] == "MessageCard"
    assert payload_crit["themeColor"] == "E01E5A"
    assert "CRITICAL" in payload_crit["title"]
    facts = payload_crit["sections"][0]["facts"]
    assert any(f["name"] == "Rule" and f["value"] == "ServiceDown" for f in facts)
    assert any(f["name"] == "Current Value" and f["value"] == "0.0000" for f in facts)
    assert payload_crit["potentialAction"][0]["targets"][0]["uri"] == (
        "https://dash.example.com"
    )

    # Send critical
    ok, code, msg = adapter.send(t_crit)
    assert ok is True
    assert code == 200
    assert len(sent_requests) == 1

    # Resolved transition
    t_res = AlertTransition(
        rule=crit_rule,
        event="resolved",
        value=1.0,
        timestamp="2026-11-11T12:15:00Z",
    )
    payload_res = adapter.build_payload(t_res)
    assert payload_res["themeColor"] == "2EB886"
    assert "RESOLVED" in payload_res["title"]


def test_multichannel_dispatcher_silencing_suppression() -> None:
    now = 5000.0
    called_channels: list[str] = []

    def fake_handler(ch: str) -> Any:
        def _h(t: AlertTransition) -> tuple[bool, int, str]:
            called_channels.append(ch)
            return True, 200, "ok"

        return _h

    silencing = AlertSilencingEngine()
    silence = AlertSilenceRule(
        silence_id="sil-lat",
        matchers=[SilenceMatcher(name="alertname", value="HighLatency")],
        starts_at=4900.0,
        ends_at=5100.0,
    )
    silencing.add_silence(silence)

    dispatcher = MultiChannelDispatcher(
        channels={"slack": fake_handler("slack"), "teams": fake_handler("teams")},
        silencing_engine=silencing,
    )

    # HighLatency is silenced
    lat_rule = _rule(name="HighLatency", summary="High Latency")
    t_lat = AlertTransition(lat_rule, "firing", 2.0, "2026-11-11T12:00:00Z")
    res_lat = dispatcher.dispatch(t_lat, now=now)
    assert "silenced" in res_lat
    assert res_lat["silenced"][0] is False
    assert res_lat["silenced"][1] == 204
    assert "suppressed_silence_rule" in res_lat["silenced"][2]
    assert called_channels == []

    # DiskFull is NOT silenced -> sent to both channels
    disk_rule = _rule(name="DiskFull", summary="Disk full")
    t_disk = AlertTransition(disk_rule, "firing", 99.0, "2026-11-11T12:00:00Z")
    res_disk = dispatcher.dispatch(t_disk, now=now)
    assert "slack" in res_disk
    assert "teams" in res_disk
    assert res_disk["slack"] == (True, 200, "ok")
    assert sorted(called_channels) == ["slack", "teams"]

    # Now add maintenance window that suppresses all alerts
    mw = MaintenanceWindow(
        window_id="mw-outage",
        starts_at=4900.0,
        ends_at=5100.0,
        suppress_all=True,
    )
    silencing.add_maintenance_window(mw)

    res_disk2 = dispatcher.dispatch(t_disk, now=now)
    assert "silenced" in res_disk2
    assert res_disk2["silenced"][1] == 204
    assert "suppressed_maintenance_window" in res_disk2["silenced"][2]


def test_multichannel_dispatcher_chat_registration_and_stats() -> None:
    silencing = AlertSilencingEngine()
    silencing.add_silence(
        AlertSilenceRule(silence_id="s1", starts_at=10.0, ends_at=100.0)
    )
    silencing.add_maintenance_window(
        MaintenanceWindow(window_id="m1", starts_at=10.0, ends_at=100.0)
    )

    dispatcher = MultiChannelDispatcher(silencing_engine=silencing)
    slack_adapter = dispatcher.register_slack(webhook_url="https://slack.example.com")
    teams_adapter = dispatcher.register_teams(webhook_url="https://teams.example.com")

    assert isinstance(slack_adapter, SlackWebhookAdapter)
    assert isinstance(teams_adapter, TeamsWebhookAdapter)

    stats = dispatcher.get_stats()
    assert "slack" in stats["registered_channels"]
    assert "teams" in stats["registered_channels"]
    assert "silencing" in stats
    assert stats["silencing"]["total_silences"] == 1
    assert stats["silencing"]["total_maintenance_windows"] == 1


def test_incident_lifecycle_record_lifecycle_and_serialization() -> None:
    from music_recommender.alerting import IncidentLifecycleRecord

    rec = IncidentLifecycleRecord(
        incident_id="inc-1001",
        rule_name="StreamingRecordsDropped",
        severity="critical",
        started_at=1000.0,
        detection_latency_seconds=12.5,
        remediations_triggered=1,
        silenced=False,
        root_causes=["StreamingQueueHighUtilization"],
        labels={"tier": "streaming"},
    )
    assert rec.is_active is True
    assert rec.get_duration(now=1050.0) == 50.0

    # Resolve incident
    rec.resolve(timestamp=1120.0)
    assert rec.is_active is False
    assert rec.duration_seconds == 120.0
    assert rec.get_duration(now=2000.0) == 120.0

    # Roundtrip serialization
    payload = rec.to_dict()
    assert payload["incident_id"] == "inc-1001"
    assert payload["rule_name"] == "StreamingRecordsDropped"
    assert payload["severity"] == "critical"
    assert payload["duration_seconds"] == 120.0
    assert payload["detection_latency_seconds"] == 12.5
    assert payload["remediations_triggered"] == 1
    assert payload["root_causes"] == ["StreamingQueueHighUtilization"]
    assert payload["labels"] == {"tier": "streaming"}
    assert payload["is_active"] is False

    restored = IncidentLifecycleRecord.from_dict(payload)
    assert restored.incident_id == "inc-1001"
    assert restored.rule_name == "StreamingRecordsDropped"
    assert restored.started_at == 1000.0
    assert restored.resolved_at == 1120.0
    assert restored.duration_seconds == 120.0
    assert restored.detection_latency_seconds == 12.5
    assert restored.is_active is False


def test_rule_incident_stats_serialization() -> None:
    from music_recommender.alerting import RuleIncidentStats

    stats = RuleIncidentStats(
        rule_name="DiskFull",
        total_incidents=3,
        active_incidents=1,
        resolved_incidents=2,
        mttd_seconds=15.2,
        mttr_seconds=120.5,
        mtbf_seconds=3600.0,
        total_downtime_seconds=241.0,
        longest_incident_seconds=150.0,
        slo_target_pct=99.9,
        slo_compliance_pct=99.72,
    )
    d = stats.to_dict()
    assert d["rule_name"] == "DiskFull"
    assert d["total_incidents"] == 3
    assert d["mttd_seconds"] == 15.2
    assert d["mttr_seconds"] == 120.5
    assert d["slo_compliance_pct"] == 99.72

    restored = RuleIncidentStats.from_dict(d)
    assert restored.rule_name == "DiskFull"
    assert restored.total_incidents == 3
    assert restored.active_incidents == 1


def test_alert_incident_analytics_calculations() -> None:
    from music_recommender.alerting import (
        AlertIncidentAnalytics,
        IncidentLifecycleRecord,
    )

    # Empty case
    empty_stats = AlertIncidentAnalytics.compute_rule_stats(
        rule_name="NonExistentRule",
        incidents=[],
        window_seconds=1000.0,
    )
    assert empty_stats.total_incidents == 0
    assert empty_stats.active_incidents == 0
    assert empty_stats.slo_compliance_pct == 100.0
    assert empty_stats.mtbf_seconds == 1000.0

    # Populated incidents for StreamingRecordsDropped:
    # Incident 1: started 100.0, resolved 160.0 (duration 60s, latency 10s)
    # Incident 2: started 300.0, resolved 340.0 (duration 40s, latency 20s)
    inc1 = IncidentLifecycleRecord(
        incident_id="i1",
        rule_name="StreamingRecordsDropped",
        started_at=100.0,
        resolved_at=160.0,
        duration_seconds=60.0,
        detection_latency_seconds=10.0,
    )
    inc2 = IncidentLifecycleRecord(
        incident_id="i2",
        rule_name="StreamingRecordsDropped",
        started_at=300.0,
        resolved_at=340.0,
        duration_seconds=40.0,
        detection_latency_seconds=20.0,
    )

    stats = AlertIncidentAnalytics.compute_rule_stats(
        rule_name="StreamingRecordsDropped",
        incidents=[inc1, inc2],
        window_seconds=1000.0,
    )
    assert stats.total_incidents == 2
    assert stats.resolved_incidents == 2
    assert stats.active_incidents == 0
    assert stats.mttd_seconds == 15.0  # (10 + 20) / 2
    assert stats.mttr_seconds == 50.0  # (60 + 40) / 2
    assert stats.total_downtime_seconds == 100.0
    assert stats.longest_incident_seconds == 60.0
    assert stats.mtbf_seconds == 450.0  # (1000 - 100) / 2
    assert stats.slo_compliance_pct == 90.0  # ((1000 - 100) / 1000) * 100

    # Add active incident for another rule: ServiceDown
    # (started 800.0, active at now=900.0)
    inc3 = IncidentLifecycleRecord(
        incident_id="i3",
        rule_name="ServiceDown",
        started_at=800.0,
        resolved_at=None,
        detection_latency_seconds=5.0,
    )

    all_stats = AlertIncidentAnalytics.compute_all_stats(
        incidents=[inc1, inc2, inc3],
        window_seconds=1000.0,
        now=900.0,
    )
    assert "StreamingRecordsDropped" in all_stats
    assert "ServiceDown" in all_stats
    assert all_stats["ServiceDown"].active_incidents == 1
    assert all_stats["ServiceDown"].total_downtime_seconds == 100.0  # 900 - 800

    summary = AlertIncidentAnalytics.compute_summary(
        incidents=[inc1, inc2, inc3],
        window_seconds=1000.0,
        now=900.0,
    )
    assert summary["total_incidents"] == 3
    assert summary["active_incidents"] == 1
    assert summary["resolved_incidents"] == 2
    assert "StreamingRecordsDropped" in summary["rules"]
    assert "ServiceDown" in summary["rules"]

