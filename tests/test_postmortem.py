"""Tests for automated incident postmortem generator, runbooks, and SLA reporting."""

from __future__ import annotations

import pytest

from music_recommender.postmortem import (
    IncidentPostmortem,
    IncidentSeverity,
    PostmortemActionItem,
    RunbookEntry,
    SLAComplianceReport,
    SLATarget,
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
