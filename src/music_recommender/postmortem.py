"""Automated incident postmortem generator, root-cause runbooks, and SLA reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class IncidentSeverity(StrEnum):
    """Severity classification for production incidents."""

    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"
    SEV4 = "sev4"


@dataclass(frozen=True)
class SLATarget:
    """Service Level Agreement target for a rolling observation window."""

    service_name: str = "music-recommender"
    target_availability_pct: float = 99.9
    window_days: int = 30

    def __post_init__(self) -> None:
        if not self.service_name.strip():
            raise ValueError("service_name must not be empty")
        if not (0.0 < self.target_availability_pct <= 100.0):
            raise ValueError("target_availability_pct must be in (0, 100]")
        if self.window_days <= 0:
            raise ValueError("window_days must be > 0")

    @property
    def window_minutes(self) -> float:
        return float(self.window_days * 24 * 60)

    @property
    def allowed_downtime_minutes(self) -> float:
        return self.window_minutes * (1.0 - self.target_availability_pct / 100.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "target_availability_pct": self.target_availability_pct,
            "window_days": self.window_days,
            "allowed_downtime_minutes": round(self.allowed_downtime_minutes, 4),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SLATarget:
        return cls(
            service_name=str(payload.get("service_name", "music-recommender")),
            target_availability_pct=float(payload.get("target_availability_pct", 99.9)),
            window_days=int(payload.get("window_days", 30)),
        )


@dataclass(frozen=True)
class RunbookEntry:
    """Actionable root-cause remediation runbook."""

    runbook_id: str
    title: str
    root_cause_category: str
    symptom_tags: tuple[str, ...]
    remediation_steps: tuple[str, ...]
    owner_team: str = "ml-platform"

    def __post_init__(self) -> None:
        if not self.runbook_id.strip():
            raise ValueError("runbook_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.symptom_tags:
            raise ValueError("symptom_tags must not be empty")
        if not self.remediation_steps:
            raise ValueError("remediation_steps must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "runbook_id": self.runbook_id,
            "title": self.title,
            "root_cause_category": self.root_cause_category,
            "symptom_tags": list(self.symptom_tags),
            "remediation_steps": list(self.remediation_steps),
            "owner_team": self.owner_team,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunbookEntry:
        return cls(
            runbook_id=str(payload["runbook_id"]),
            title=str(payload["title"]),
            root_cause_category=str(payload["root_cause_category"]),
            symptom_tags=tuple(str(t) for t in payload.get("symptom_tags", ())),
            remediation_steps=tuple(
                str(s) for s in payload.get("remediation_steps", ())
            ),
            owner_team=str(payload.get("owner_team", "ml-platform")),
        )


@dataclass(frozen=True)
class PostmortemActionItem:
    """Follow-up action item derived from an incident postmortem."""

    action_id: str
    description: str
    owner: str
    priority: str = "P1"
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "description": self.description,
            "owner": self.owner,
            "priority": self.priority,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PostmortemActionItem:
        return cls(
            action_id=str(payload["action_id"]),
            description=str(payload["description"]),
            owner=str(payload["owner"]),
            priority=str(payload.get("priority", "P1")),
            status=str(payload.get("status", "open")),
        )


@dataclass(frozen=True)
class IncidentPostmortem:
    """Structured incident postmortem report with Markdown export."""

    incident_id: str
    title: str
    severity: IncidentSeverity
    duration_minutes: float
    symptoms: tuple[str, ...]
    root_cause_summary: str
    matched_runbook_ids: tuple[str, ...] = ()
    timeline: tuple[str, ...] = ()
    action_items: tuple[PostmortemActionItem, ...] = ()
    generated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not self.incident_id.strip():
            raise ValueError("incident_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if self.duration_minutes < 0.0:
            raise ValueError("duration_minutes must be >= 0")

    def to_markdown(self) -> str:
        lines = [
            f"# Incident Postmortem: {self.title} ({self.incident_id})",
            "",
            f"- **Severity**: `{self.severity.value.upper()}`",
            f"- **Duration**: {self.duration_minutes:.2f} minutes",
            f"- **Generated At**: {self.generated_at}",
            f"- **Symptoms**: {', '.join(self.symptoms) if self.symptoms else 'none'}",
            "",
            "## Root Cause Summary",
            self.root_cause_summary,
            "",
            "## Matched Runbooks",
        ]
        if self.matched_runbook_ids:
            for rb_id in self.matched_runbook_ids:
                lines.append(f"- `{rb_id}`")
        else:
            lines.append("- None matched")
        lines.extend(["", "## Timeline"])
        for entry in self.timeline:
            lines.append(f"- {entry}")
        lines.extend(["", "## Action Items"])
        for item in self.action_items:
            lines.append(
                f"- [{item.priority}] ({item.owner}) {item.description} [{item.status}]"
            )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "severity": self.severity.value,
            "duration_minutes": round(self.duration_minutes, 4),
            "symptoms": list(self.symptoms),
            "root_cause_summary": self.root_cause_summary,
            "matched_runbook_ids": list(self.matched_runbook_ids),
            "timeline": list(self.timeline),
            "action_items": [item.to_dict() for item in self.action_items],
            "generated_at": self.generated_at,
            "markdown": self.to_markdown(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IncidentPostmortem:
        return cls(
            incident_id=str(payload["incident_id"]),
            title=str(payload["title"]),
            severity=IncidentSeverity(str(payload["severity"])),
            duration_minutes=float(payload["duration_minutes"]),
            symptoms=tuple(str(s) for s in payload.get("symptoms", ())),
            root_cause_summary=str(payload["root_cause_summary"]),
            matched_runbook_ids=tuple(
                str(r) for r in payload.get("matched_runbook_ids", ())
            ),
            timeline=tuple(str(t) for t in payload.get("timeline", ())),
            action_items=tuple(
                PostmortemActionItem.from_dict(item)
                for item in payload.get("action_items", ())
            ),
            generated_at=str(payload.get("generated_at", _utc_now_iso())),
        )


@dataclass(frozen=True)
class SLAComplianceReport:
    """Rolling SLA compliance and error-budget burn report."""

    service_name: str
    target_availability_pct: float
    achieved_availability_pct: float
    window_days: int
    total_downtime_minutes: float
    allowed_downtime_minutes: float
    error_budget_consumed_pct: float
    error_budget_remaining_pct: float
    sla_breached: bool
    incident_count: int
    generated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "target_availability_pct": self.target_availability_pct,
            "achieved_availability_pct": round(self.achieved_availability_pct, 6),
            "window_days": self.window_days,
            "total_downtime_minutes": round(self.total_downtime_minutes, 4),
            "allowed_downtime_minutes": round(self.allowed_downtime_minutes, 4),
            "error_budget_consumed_pct": round(self.error_budget_consumed_pct, 4),
            "error_budget_remaining_pct": round(self.error_budget_remaining_pct, 4),
            "sla_breached": self.sla_breached,
            "incident_count": self.incident_count,
            "generated_at": self.generated_at,
        }
