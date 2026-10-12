"""Tests for alert rules, evaluation, dispatch, and rule-file generation."""

from __future__ import annotations

import json

import pytest

from music_recommender.alerting import (
    DEFAULT_ALERT_RULES,
    AlertEvaluator,
    AlertMode,
    AlertRule,
    AlertSeverity,
    lookup_metric_value,
)
from music_recommender.telemetry import MetricFamily


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
