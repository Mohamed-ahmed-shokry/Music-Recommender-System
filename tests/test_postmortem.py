"""Tests for automated incident postmortem generator, runbooks, and SLA reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.postmortem import (
    IncidentPostmortem,
    IncidentSeverity,
    PostmortemActionItem,
    PostmortemEngine,
    RunbookEntry,
    SLAComplianceReport,
    SLATarget,
    get_global_postmortem_engine,
    set_global_postmortem_engine,
)


def test_sla_target_validation_and_roundtrip() -> None:
    target = SLATarget(
        service_name="recommender-api",
        target_availability_pct=99.9,
        window_days=30,
    )
    assert target.window_minutes == pytest.approx(43200.0)
    assert target.allowed_downtime_minutes == pytest.approx(43.2)
    restored = SLATarget.from_dict(target.to_dict())
    assert restored == target

    with pytest.raises(ValueError, match="service_name"):
        SLATarget(service_name="  ")
    with pytest.raises(ValueError, match="target_availability_pct"):
        SLATarget(target_availability_pct=0.0)
    with pytest.raises(ValueError, match="window_days"):
        SLATarget(window_days=0)


def test_runbook_and_postmortem_models_roundtrip_and_markdown() -> None:
    runbook = RunbookEntry(
        runbook_id="rb-latency",
        title="High p99 Latency",
        root_cause_category="latency",
        symptom_tags=("latency", "p99"),
        remediation_steps=("Scale replicas", "Inspect cache hit rate"),
    )
    assert RunbookEntry.from_dict(runbook.to_dict()) == runbook

    with pytest.raises(ValueError, match="symptom_tags"):
        RunbookEntry(
            runbook_id="rb-empty",
            title="Empty",
            root_cause_category="other",
            symptom_tags=(),
            remediation_steps=("Step 1",),
        )

    action = PostmortemActionItem(
        action_id="act-1",
        description="Add p99 autoscaling rule",
        owner="sre",
    )
    pm = IncidentPostmortem(
        incident_id="inc-001",
        title="Recommendation Latency Spike",
        severity=IncidentSeverity.SEV2,
        duration_minutes=12.5,
        symptoms=("latency", "p99"),
        root_cause_summary="Cache eviction burst increased vector lookup time.",
        matched_runbook_ids=("rb-latency",),
        timeline=("00:00 Alert fired", "00:12 Mitigated"),
        action_items=(action,),
    )
    md = pm.to_markdown()
    assert "# Incident Postmortem: Recommendation Latency Spike (inc-001)" in md
    assert "`SEV2`" in md
    assert "`rb-latency`" in md
    assert IncidentPostmortem.from_dict(pm.to_dict()) == pm

    report = SLAComplianceReport(
        service_name="music-recommender",
        target_availability_pct=99.9,
        achieved_availability_pct=99.97,
        window_days=30,
        total_downtime_minutes=12.5,
        allowed_downtime_minutes=43.2,
        error_budget_consumed_pct=28.9352,
        error_budget_remaining_pct=71.0648,
        sla_breached=False,
        incident_count=1,
    )
    assert report.to_dict()["sla_breached"] is False


def test_postmortem_engine_runbook_matching_and_generation() -> None:
    engine = PostmortemEngine()
    assert len(engine.list_runbooks()) >= 5

    matched = engine.match_runbooks(["p99", "latency"])
    assert matched
    assert matched[0].runbook_id == "rb-latency-spike"

    pm = engine.generate_postmortem(
        incident_id="inc-101",
        title="p99 Latency Spike in US-East",
        severity=IncidentSeverity.SEV1,
        duration_minutes=15.0,
        symptoms=["latency", "p99", "timeout"],
    )
    assert pm.incident_id == "inc-101"
    assert "rb-latency-spike" in pm.matched_runbook_ids
    assert pm.action_items[0].priority == "P0"
    assert engine.get_postmortem("inc-101") == pm
    assert len(engine.list_postmortems()) == 1

    # Unmatched symptom still produces a fallback action item
    pm_unknown = engine.generate_postmortem(
        incident_id="inc-102",
        title="Unclassified Glitch",
        severity="sev3",
        duration_minutes=2.0,
        symptoms=["cosmic-ray"],
    )
    assert pm_unknown.matched_runbook_ids == ()
    assert pm_unknown.action_items[0].priority == "P1"


def test_postmortem_engine_sla_report_and_persistence(tmp_path: Path) -> None:
    engine = PostmortemEngine(
        sla_target=SLATarget(
            service_name="music-recommender",
            target_availability_pct=99.9,
            window_days=30,
        )
    )
    engine.generate_postmortem(
        incident_id="inc-201",
        title="Drift Outage",
        severity=IncidentSeverity.SEV2,
        duration_minutes=30.0,
        symptoms=["drift", "stale"],
    )
    engine.record_downtime_event(20.0)
    sla = engine.compute_sla_report()
    assert sla.total_downtime_minutes == pytest.approx(50.0)
    assert sla.allowed_downtime_minutes == pytest.approx(43.2)
    assert sla.sla_breached is True
    assert sla.error_budget_remaining_pct == pytest.approx(0.0)
    assert sla.incident_count == 2

    state_file = tmp_path / "postmortem_state.json"
    engine.save_state(state_file)
    restored = PostmortemEngine()
    restored.load_state(state_file)
    assert restored.get_postmortem("inc-201") is not None
    assert restored.compute_sla_report().total_downtime_minutes == pytest.approx(50.0)
    status = restored.get_status()
    assert status["postmortem_count"] == 1
    assert status["sla_report"]["sla_breached"] is True

    with pytest.raises(ValueError, match="duration_minutes"):
        engine.record_downtime_event(-1.0)


def test_global_postmortem_engine_accessor() -> None:
    original = get_global_postmortem_engine()
    custom = PostmortemEngine()
    try:
        set_global_postmortem_engine(custom)
        assert get_global_postmortem_engine() is custom
    finally:
        set_global_postmortem_engine(original)
