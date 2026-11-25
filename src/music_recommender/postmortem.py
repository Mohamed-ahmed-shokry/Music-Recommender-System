"""Automated incident postmortem generator, root-cause runbooks, and SLA reporting."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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


def _default_runbooks() -> list[RunbookEntry]:
    return [
        RunbookEntry(
            runbook_id="rb-latency-spike",
            title="Recommendation Latency Spike Remediation",
            root_cause_category="latency",
            symptom_tags=("latency", "p99", "timeout", "slow"),
            remediation_steps=(
                "Inspect p95/p99 latency breakdown across ANN index and feature store.",
                "Verify recommendation cache hit ratio and warm hot-user cache keys.",
                "Scale serving replicas or shed non-critical reranking stages.",
            ),
            owner_team="serving-sre",
        ),
        RunbookEntry(
            runbook_id="rb-feature-drift",
            title="Feature Drift & Stale Embedding Recovery",
            root_cause_category="data-drift",
            symptom_tags=("drift", "psi", "stale", "embedding", "quality"),
            remediation_steps=(
                "Run drift-status and inspect PSI/KS divergence per feature.",
                "Trigger automated retraining or roll back to a known-good snapshot.",
                "Validate offline NDCG/recall before promoting refreshed embeddings.",
            ),
            owner_team="ml-platform",
        ),
        RunbookEntry(
            runbook_id="rb-rate-limit-surge",
            title="Traffic Surge & Rate-Limit Saturation",
            root_cause_category="capacity",
            symptom_tags=("rate-limit", "429", "quota", "traffic", "surge"),
            remediation_steps=(
                "Inspect ratelimit-stats to identify noisy-neighbor tenants.",
                "Adjust tenant token-bucket limits or enable circuit breaker shedding.",
                "Verify downstream API dependency headroom.",
            ),
            owner_team="platform-sre",
        ),
        RunbookEntry(
            runbook_id="rb-region-failover",
            title="Multi-Region Artifact Sync & Geo-Failover",
            root_cause_category="availability",
            symptom_tags=("region", "failover", "replication", "outage", "5xx"),
            remediation_steps=(
                "Check replication-status and verify SHA-256 manifests per region.",
                "Shift traffic to healthy secondary region via geo-failover router.",
                "Re-sync degraded region edge cache before restoring primary traffic.",
            ),
            owner_team="infra-sre",
        ),
        RunbookEntry(
            runbook_id="rb-privacy-budget",
            title="Differential Privacy Budget Exhaustion",
            root_cause_category="privacy",
            symptom_tags=("privacy", "epsilon", "dp-budget", "consent"),
            remediation_steps=(
                "Audit privacy-status for high epsilon burn or consent revocations.",
                "Throttle federated rounds or increase Gaussian noise multiplier.",
                "Confirm consent-gated fallback recommendations remain active.",
            ),
            owner_team="privacy-engineering",
        ),
    ]


class PostmortemEngine:
    """Thread-safe incident postmortem generator, runbook matcher, and SLA tracker."""

    def __init__(
        self,
        sla_target: SLATarget | None = None,
        runbooks: list[RunbookEntry] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._sla_target = sla_target or SLATarget()
        self._runbooks: dict[str, RunbookEntry] = {}
        for rb in runbooks if runbooks is not None else _default_runbooks():
            self._runbooks[rb.runbook_id] = rb
        self._postmortems: dict[str, IncidentPostmortem] = {}
        self._extra_downtime_minutes: float = 0.0
        self._extra_downtime_events: int = 0

    @property
    def sla_target(self) -> SLATarget:
        with self._lock:
            return self._sla_target

    def set_sla_target(self, target: SLATarget) -> SLATarget:
        with self._lock:
            self._sla_target = target
            return self._sla_target

    def register_runbook(self, runbook: RunbookEntry) -> RunbookEntry:
        with self._lock:
            self._runbooks[runbook.runbook_id] = runbook
            return runbook

    def list_runbooks(self) -> list[RunbookEntry]:
        with self._lock:
            return list(self._runbooks.values())

    def match_runbooks(
        self,
        symptoms: list[str] | tuple[str, ...],
        *,
        top_k: int = 3,
    ) -> list[RunbookEntry]:
        """Rank root-cause runbooks by normalized symptom/tag overlap."""
        normalized = {s.strip().lower() for s in symptoms if s.strip()}
        if not normalized:
            return []
        with self._lock:
            scored: list[tuple[int, str, RunbookEntry]] = []
            for rb in self._runbooks.values():
                tags = {t.strip().lower() for t in rb.symptom_tags}
                score = len(normalized & tags)
                if score == 0:
                    # Partial substring match fallback for descriptive symptoms
                    score = sum(
                        1
                        for sym in normalized
                        for tag in tags
                        if tag in sym or sym in tag
                    )
                if score > 0:
                    scored.append((score, rb.runbook_id, rb))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [entry for _, _, entry in scored[: max(1, top_k)]]

    def record_downtime_event(self, duration_minutes: float) -> float:
        """Record standalone downtime minutes outside a generated postmortem."""
        if duration_minutes < 0.0:
            raise ValueError("duration_minutes must be >= 0")
        with self._lock:
            self._extra_downtime_minutes += float(duration_minutes)
            self._extra_downtime_events += 1
            return self._extra_downtime_minutes

    def generate_postmortem(
        self,
        incident_id: str,
        title: str,
        severity: IncidentSeverity | str,
        duration_minutes: float,
        symptoms: list[str] | tuple[str, ...],
        *,
        root_cause_summary: str | None = None,
        timeline: list[str] | tuple[str, ...] | None = None,
    ) -> IncidentPostmortem:
        """Synthesize a structured postmortem with matched runbooks and action items."""
        sev = (
            severity
            if isinstance(severity, IncidentSeverity)
            else IncidentSeverity(str(severity).lower())
        )
        clean_symptoms = tuple(s.strip() for s in symptoms if s.strip())
        matched = self.match_runbooks(clean_symptoms, top_k=3)
        matched_ids = tuple(rb.runbook_id for rb in matched)

        if root_cause_summary and root_cause_summary.strip():
            summary = root_cause_summary.strip()
        elif matched:
            categories = ", ".join(
                dict.fromkeys(rb.root_cause_category for rb in matched)
            )
            summary = (
                f"Automated root-cause triage matched {categories} "
                f"based on symptoms ({', '.join(clean_symptoms) or 'unspecified'})."
            )
        else:
            summary = (
                "Root cause requires manual investigation; no built-in runbook "
                f"matched symptoms ({', '.join(clean_symptoms) or 'unspecified'})."
            )

        if timeline:
            event_timeline = tuple(str(t) for t in timeline)
        else:
            event_timeline = (
                f"T+0m: Incident {incident_id} ({sev.value.upper()}) detected.",
                f"T+1m: Symptoms observed: {', '.join(clean_symptoms) or 'none'}.",
                f"T+{duration_minutes:.1f}m: Service stabilized and incident resolved.",
            )

        priority = (
            "P0" if sev in (IncidentSeverity.SEV1, IncidentSeverity.SEV2) else "P1"
        )
        action_items: list[PostmortemActionItem] = []
        if matched:
            for idx, rb in enumerate(matched, start=1):
                action_items.append(
                    PostmortemActionItem(
                        action_id=f"{incident_id}-act-{idx}",
                        description=(
                            f"Execute & automate {rb.title}: {rb.remediation_steps[0]}"
                        ),
                        owner=rb.owner_team,
                        priority=priority,
                    )
                )
        else:
            action_items.append(
                PostmortemActionItem(
                    action_id=f"{incident_id}-act-1",
                    description=(
                        "Author a new root-cause runbook and alert threshold "
                        "for this failure mode."
                    ),
                    owner="ml-platform",
                    priority=priority,
                )
            )

        postmortem = IncidentPostmortem(
            incident_id=incident_id,
            title=title,
            severity=sev,
            duration_minutes=float(duration_minutes),
            symptoms=clean_symptoms,
            root_cause_summary=summary,
            matched_runbook_ids=matched_ids,
            timeline=event_timeline,
            action_items=tuple(action_items),
        )
        with self._lock:
            self._postmortems[postmortem.incident_id] = postmortem
        return postmortem

    def get_postmortem(self, incident_id: str) -> IncidentPostmortem | None:
        with self._lock:
            return self._postmortems.get(incident_id)

    def list_postmortems(self) -> list[IncidentPostmortem]:
        with self._lock:
            return list(self._postmortems.values())

    def compute_sla_report(self) -> SLAComplianceReport:
        """Compute rolling SLA availability and error-budget burn metrics."""
        with self._lock:
            target = self._sla_target
            pm_downtime = sum(pm.duration_minutes for pm in self._postmortems.values())
            total_downtime = pm_downtime + self._extra_downtime_minutes
            window_mins = target.window_minutes
            allowed = target.allowed_downtime_minutes
            achieved = max(
                0.0,
                min(100.0, ((window_mins - total_downtime) / window_mins) * 100.0),
            )
            if allowed <= 0.0:
                consumed_pct = 100.0 if total_downtime > 0.0 else 0.0
            else:
                consumed_pct = (total_downtime / allowed) * 100.0
            remaining_pct = max(0.0, 100.0 - consumed_pct)
            breached = total_downtime > allowed
            incident_count = len(self._postmortems) + self._extra_downtime_events
            return SLAComplianceReport(
                service_name=target.service_name,
                target_availability_pct=target.target_availability_pct,
                achieved_availability_pct=achieved,
                window_days=target.window_days,
                total_downtime_minutes=total_downtime,
                allowed_downtime_minutes=allowed,
                error_budget_consumed_pct=consumed_pct,
                error_budget_remaining_pct=remaining_pct,
                sla_breached=breached,
                incident_count=incident_count,
            )

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            sla = self.compute_sla_report()
            return {
                "runbook_count": len(self._runbooks),
                "postmortem_count": len(self._postmortems),
                "sla_target": self._sla_target.to_dict(),
                "sla_report": sla.to_dict(),
                "recent_incident_ids": list(self._postmortems.keys())[-10:],
            }

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sla_target": self._sla_target.to_dict(),
                "runbooks": [rb.to_dict() for rb in self._runbooks.values()],
                "postmortems": [pm.to_dict() for pm in self._postmortems.values()],
                "extra_downtime_minutes": self._extra_downtime_minutes,
                "extra_downtime_events": self._extra_downtime_events,
            }

    def import_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            if "sla_target" in state:
                self._sla_target = SLATarget.from_dict(state["sla_target"])
            if "runbooks" in state:
                self._runbooks = {
                    rb.runbook_id: rb
                    for rb in (RunbookEntry.from_dict(r) for r in state["runbooks"])
                }
            if "postmortems" in state:
                self._postmortems = {
                    pm.incident_id: pm
                    for pm in (
                        IncidentPostmortem.from_dict(p) for p in state["postmortems"]
                    )
                }
            self._extra_downtime_minutes = float(
                state.get("extra_downtime_minutes", 0.0)
            )
            self._extra_downtime_events = int(state.get("extra_downtime_events", 0))

    def save_state(self, path: str | Path) -> Path:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(self.export_state(), indent=2),
            encoding="utf-8",
        )
        return target_path

    def load_state(self, path: str | Path) -> None:
        source_path = Path(path)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        self.import_state(payload)


_GLOBAL_POSTMORTEM_ENGINE: PostmortemEngine | None = None
_GLOBAL_LOCK = threading.Lock()


def get_global_postmortem_engine() -> PostmortemEngine:
    global _GLOBAL_POSTMORTEM_ENGINE
    with _GLOBAL_LOCK:
        if _GLOBAL_POSTMORTEM_ENGINE is None:
            _GLOBAL_POSTMORTEM_ENGINE = PostmortemEngine()
        return _GLOBAL_POSTMORTEM_ENGINE


def set_global_postmortem_engine(engine: PostmortemEngine | None) -> None:
    global _GLOBAL_POSTMORTEM_ENGINE
    with _GLOBAL_LOCK:
        _GLOBAL_POSTMORTEM_ENGINE = engine
