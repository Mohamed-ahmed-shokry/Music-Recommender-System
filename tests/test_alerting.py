"""Tests for alert rules, evaluation, dispatch, and rule-file generation."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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
    AlertTransition,
    ReceivedAlertEvent,
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
