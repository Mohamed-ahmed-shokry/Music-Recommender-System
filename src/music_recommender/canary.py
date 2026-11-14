"""Automated model shadow deployment, canary traffic splitting, and A/B guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal


class DeploymentMode(StrEnum):
    """Active traffic-splitting and rollout lifecycle state."""

    CHAMPION_ONLY = "champion_only"
    SHADOW = "shadow"
    CANARY = "canary"
    ROLLED_BACK = "rolled_back"
    PROMOTED = "promoted"


@dataclass(frozen=True, slots=True)
class CanaryPolicy:
    """Configuration controlling canary traffic split and shadow mirroring."""

    canary_weight: float = 0.0
    shadow_enabled: bool = False
    shadow_sample_rate: float = 0.0
    sticky_bucketing: bool = True
    champion_version: str = "champion"
    candidate_version: str = "candidate"
    mode: DeploymentMode = DeploymentMode.CHAMPION_ONLY
    salt: str = "music-recommender-canary-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.canary_weight <= 1.0:
            raise ValueError("canary_weight must be between 0.0 and 1.0")
        if not 0.0 <= self.shadow_sample_rate <= 1.0:
            raise ValueError("shadow_sample_rate must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize canary policy to a JSON-compatible dictionary."""
        return {
            "canary_weight": self.canary_weight,
            "shadow_enabled": self.shadow_enabled,
            "shadow_sample_rate": self.shadow_sample_rate,
            "sticky_bucketing": self.sticky_bucketing,
            "champion_version": self.champion_version,
            "candidate_version": self.candidate_version,
            "mode": self.mode.value,
            "salt": self.salt,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CanaryPolicy:
        """Deserialize a CanaryPolicy from a dictionary."""
        return cls(
            canary_weight=float(payload.get("canary_weight", 0.0)),
            shadow_enabled=bool(payload.get("shadow_enabled", False)),
            shadow_sample_rate=float(payload.get("shadow_sample_rate", 0.0)),
            sticky_bucketing=bool(payload.get("sticky_bucketing", True)),
            champion_version=str(payload.get("champion_version", "champion")),
            candidate_version=str(payload.get("candidate_version", "candidate")),
            mode=DeploymentMode(
                payload.get("mode", DeploymentMode.CHAMPION_ONLY.value)
            ),
            salt=str(payload.get("salt", "music-recommender-canary-v1")),
        )


@dataclass(frozen=True, slots=True)
class ABGuardrailConfig:
    """Statistical and reliability thresholds for automated canary rollback."""

    min_samples_per_arm: int = 30
    max_error_rate_delta: float = 0.05
    max_latency_p95_ratio: float = 1.5
    max_latency_ms: float = 500.0
    min_ctr_lift_z: float = -1.96

    def __post_init__(self) -> None:
        if self.min_samples_per_arm < 1:
            raise ValueError("min_samples_per_arm must be >= 1")
        if self.max_error_rate_delta < 0.0:
            raise ValueError("max_error_rate_delta must be >= 0.0")
        if self.max_latency_p95_ratio < 1.0:
            raise ValueError("max_latency_p95_ratio must be >= 1.0")
        if self.max_latency_ms <= 0.0:
            raise ValueError("max_latency_ms must be > 0.0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize guardrail configuration to a JSON-compatible dictionary."""
        return {
            "min_samples_per_arm": self.min_samples_per_arm,
            "max_error_rate_delta": self.max_error_rate_delta,
            "max_latency_p95_ratio": self.max_latency_p95_ratio,
            "max_latency_ms": self.max_latency_ms,
            "min_ctr_lift_z": self.min_ctr_lift_z,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ABGuardrailConfig:
        """Deserialize an ABGuardrailConfig from a dictionary."""
        return cls(
            min_samples_per_arm=int(payload.get("min_samples_per_arm", 30)),
            max_error_rate_delta=float(payload.get("max_error_rate_delta", 0.05)),
            max_latency_p95_ratio=float(payload.get("max_latency_p95_ratio", 1.5)),
            max_latency_ms=float(payload.get("max_latency_ms", 500.0)),
            min_ctr_lift_z=float(payload.get("min_ctr_lift_z", -1.96)),
        )


@dataclass(slots=True)
class RoutingDecision:
    """Outcome of routing an entity across champion, canary, and shadow arms."""

    entity_id: str
    arm: Literal["champion", "canary"]
    shadow_arm: str | None
    bucket_score: float
    mode: DeploymentMode
    champion_version: str
    candidate_version: str
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing decision to a JSON-compatible dictionary."""
        return {
            "entity_id": self.entity_id,
            "arm": self.arm,
            "shadow_arm": self.shadow_arm,
            "bucket_score": round(self.bucket_score, 6),
            "mode": self.mode.value,
            "champion_version": self.champion_version,
            "candidate_version": self.candidate_version,
            "decided_at": self.decided_at,
        }
