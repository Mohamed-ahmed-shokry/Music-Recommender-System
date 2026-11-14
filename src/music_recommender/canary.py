"""Automated model shadow deployment, canary traffic splitting, and A/B guardrails."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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


@dataclass(slots=True)
class ArmMetrics:
    """Telemetry counters and bounded latency buffer for a serving arm."""

    requests: int = 0
    errors: int = 0
    conversions: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record(
        self,
        *,
        latency_ms: float,
        success: bool = True,
        converted: bool = False,
    ) -> None:
        """Record a single request outcome for this arm."""
        self.requests += 1
        if not success:
            self.errors += 1
        if converted:
            self.conversions += 1
        self.latencies_ms.append(max(0.0, float(latency_ms)))

    def error_rate(self) -> float:
        """Return proportion of failed requests."""
        return (self.errors / self.requests) if self.requests > 0 else 0.0

    def conversion_rate(self) -> float:
        """Return proportion of converted requests (CTR)."""
        return (self.conversions / self.requests) if self.requests > 0 else 0.0

    def latency_p95_ms(self) -> float:
        """Return 95th percentile latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        return ordered[idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialize arm metrics summary to a JSON-compatible dictionary."""
        return {
            "requests": self.requests,
            "errors": self.errors,
            "conversions": self.conversions,
            "error_rate": round(self.error_rate(), 6),
            "conversion_rate": round(self.conversion_rate(), 6),
            "latency_p95_ms": round(self.latency_p95_ms(), 3),
            "latencies_ms": list(self.latencies_ms),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArmMetrics:
        """Deserialize ArmMetrics from a dictionary."""
        latencies = deque(
            (float(x) for x in payload.get("latencies_ms", [])),
            maxlen=1000,
        )
        return cls(
            requests=int(payload.get("requests", 0)),
            errors=int(payload.get("errors", 0)),
            conversions=int(payload.get("conversions", 0)),
            latencies_ms=latencies,
        )


@dataclass(slots=True)
class GuardrailEvaluation:
    """Result of evaluating online A/B statistical and reliability guardrails."""

    status: Literal[
        "healthy",
        "insufficient_data",
        "rollback_triggered",
        "promote_ready",
    ]
    should_rollback: bool
    reasons: list[str]
    z_score: float
    p_value: float
    champion_ctr: float
    canary_ctr: float
    champion_error_rate: float
    canary_error_rate: float
    champion_p95_ms: float
    canary_p95_ms: float
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize guardrail evaluation to a JSON-compatible dictionary."""
        return {
            "status": self.status,
            "should_rollback": self.should_rollback,
            "reasons": list(self.reasons),
            "z_score": round(self.z_score, 4),
            "p_value": round(self.p_value, 6),
            "champion_ctr": round(self.champion_ctr, 6),
            "canary_ctr": round(self.canary_ctr, 6),
            "champion_error_rate": round(self.champion_error_rate, 6),
            "canary_error_rate": round(self.canary_error_rate, 6),
            "champion_p95_ms": round(self.champion_p95_ms, 3),
            "canary_p95_ms": round(self.canary_p95_ms, 3),
            "evaluated_at": self.evaluated_at,
        }


class ShadowCanaryRouter:
    """Thread-safe shadow mirroring, canary splitting, and A/B guardrail engine."""

    def __init__(
        self,
        *,
        policy: CanaryPolicy | None = None,
        guardrails: ABGuardrailConfig | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.policy = policy or CanaryPolicy()
        self.guardrails = guardrails or ABGuardrailConfig()
        self._champion_metrics = ArmMetrics()
        self._canary_metrics = ArmMetrics()
        self._shadow_comparisons: deque[dict[str, Any]] = deque(maxlen=200)
        self._rollback_reason: str | None = None
        self._last_evaluation: GuardrailEvaluation | None = None
        self._non_sticky_counter: int = 0

    def _bucket_score(self, entity_id: str) -> float:
        with self._lock:
            if not self.policy.sticky_bucketing:
                self._non_sticky_counter += 1
                key = f"{self.policy.salt}:{entity_id}:{self._non_sticky_counter}"
            else:
                key = f"{self.policy.salt}:{entity_id}"
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            raw = int.from_bytes(digest[:8], "big")
            return raw / float(1 << 64)

    def route(self, entity_id: str) -> RoutingDecision:
        """Route an entity to champion or canary (+ optional shadow)."""
        with self._lock:
            score = self._bucket_score(entity_id)
            mode = self.policy.mode
            arm: Literal["champion", "canary"] = "champion"
            shadow_arm: str | None = None

            if mode == DeploymentMode.PROMOTED or (
                mode == DeploymentMode.CANARY and score < self.policy.canary_weight
            ):
                arm = "canary"

            if (
                arm == "champion"
                and self.policy.shadow_enabled
                and mode in (DeploymentMode.SHADOW, DeploymentMode.CANARY)
                and self.policy.shadow_sample_rate > 0.0
            ):
                shadow_key = f"{self.policy.salt}:shadow:{entity_id}"
                shadow_digest = hashlib.sha256(shadow_key.encode("utf-8")).digest()
                shadow_score = int.from_bytes(shadow_digest[:8], "big") / float(1 << 64)
                if shadow_score < self.policy.shadow_sample_rate:
                    shadow_arm = self.policy.candidate_version

            return RoutingDecision(
                entity_id=entity_id,
                arm=arm,
                shadow_arm=shadow_arm,
                bucket_score=score,
                mode=mode,
                champion_version=self.policy.champion_version,
                candidate_version=self.policy.candidate_version,
            )

    def record_outcome(
        self,
        arm: Literal["champion", "canary"],
        *,
        latency_ms: float,
        success: bool = True,
        converted: bool = False,
    ) -> None:
        """Record request latency, error, and conversion outcome for an arm."""
        with self._lock:
            target = self._canary_metrics if arm == "canary" else self._champion_metrics
            target.record(
                latency_ms=latency_ms,
                success=success,
                converted=converted,
            )

    def record_shadow_comparison(
        self,
        *,
        entity_id: str,
        champion_items: Sequence[str],
        candidate_items: Sequence[str],
        champion_latency_ms: float,
        candidate_latency_ms: float,
    ) -> dict[str, Any]:
        """Record shadow candidate vs. champion output agreement and latency delta."""
        champ_set = set(champion_items)
        cand_set = set(candidate_items)
        union = champ_set | cand_set
        jaccard = (len(champ_set & cand_set) / len(union)) if union else 1.0
        entry = {
            "entity_id": entity_id,
            "jaccard_overlap": round(jaccard, 4),
            "champion_latency_ms": round(float(champion_latency_ms), 3),
            "candidate_latency_ms": round(float(candidate_latency_ms), 3),
            "latency_delta_ms": round(
                float(candidate_latency_ms) - float(champion_latency_ms), 3
            ),
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            self._shadow_comparisons.append(entry)
        return entry

    def update_policy(
        self,
        *,
        canary_weight: float | None = None,
        shadow_enabled: bool | None = None,
        shadow_sample_rate: float | None = None,
        sticky_bucketing: bool | None = None,
        champion_version: str | None = None,
        candidate_version: str | None = None,
        mode: DeploymentMode | str | None = None,
    ) -> CanaryPolicy:
        """Update canary/shadow routing policy fields atomically."""
        with self._lock:
            resolved_mode = self.policy.mode
            if mode is not None:
                resolved_mode = (
                    mode if isinstance(mode, DeploymentMode) else DeploymentMode(mode)
                )
            elif canary_weight is not None and canary_weight > 0.0:
                resolved_mode = DeploymentMode.CANARY
            elif shadow_enabled is True:
                resolved_mode = DeploymentMode.SHADOW

            new_policy = CanaryPolicy(
                canary_weight=(
                    self.policy.canary_weight
                    if canary_weight is None
                    else float(canary_weight)
                ),
                shadow_enabled=(
                    self.policy.shadow_enabled
                    if shadow_enabled is None
                    else bool(shadow_enabled)
                ),
                shadow_sample_rate=(
                    self.policy.shadow_sample_rate
                    if shadow_sample_rate is None
                    else float(shadow_sample_rate)
                ),
                sticky_bucketing=(
                    self.policy.sticky_bucketing
                    if sticky_bucketing is None
                    else bool(sticky_bucketing)
                ),
                champion_version=(
                    self.policy.champion_version
                    if champion_version is None
                    else str(champion_version)
                ),
                candidate_version=(
                    self.policy.candidate_version
                    if candidate_version is None
                    else str(candidate_version)
                ),
                mode=resolved_mode,
                salt=self.policy.salt,
            )
            self.policy = new_policy
            if resolved_mode != DeploymentMode.ROLLED_BACK:
                self._rollback_reason = None
            return new_policy

    def rollback(self, reason: str = "manual_rollback") -> CanaryPolicy:
        """Immediately halt canary/shadow traffic and revert to champion."""
        with self._lock:
            self._rollback_reason = reason
            self.policy = CanaryPolicy(
                canary_weight=0.0,
                shadow_enabled=False,
                shadow_sample_rate=0.0,
                sticky_bucketing=self.policy.sticky_bucketing,
                champion_version=self.policy.champion_version,
                candidate_version=self.policy.candidate_version,
                mode=DeploymentMode.ROLLED_BACK,
                salt=self.policy.salt,
            )
            return self.policy

    def promote(self) -> CanaryPolicy:
        """Promote candidate to full traffic."""
        with self._lock:
            self._rollback_reason = None
            self.policy = CanaryPolicy(
                canary_weight=1.0,
                shadow_enabled=False,
                shadow_sample_rate=0.0,
                sticky_bucketing=self.policy.sticky_bucketing,
                champion_version=self.policy.champion_version,
                candidate_version=self.policy.candidate_version,
                mode=DeploymentMode.PROMOTED,
                salt=self.policy.salt,
            )
            return self.policy

    def evaluate_guardrails(
        self,
        *,
        auto_rollback: bool = True,
    ) -> GuardrailEvaluation:
        """Evaluate online A/B statistical significance and reliability guardrails."""
        with self._lock:
            champ = self._champion_metrics
            cand = self._canary_metrics
            champ_ctr = champ.conversion_rate()
            cand_ctr = cand.conversion_rate()
            champ_err = champ.error_rate()
            cand_err = cand.error_rate()
            champ_p95 = champ.latency_p95_ms()
            cand_p95 = cand.latency_p95_ms()

            if (
                champ.requests < self.guardrails.min_samples_per_arm
                or cand.requests < self.guardrails.min_samples_per_arm
            ):
                eval_insuff = GuardrailEvaluation(
                    status="insufficient_data",
                    should_rollback=False,
                    reasons=[
                        f"Need >= {self.guardrails.min_samples_per_arm} samples "
                        f"per arm (champion={champ.requests}, canary={cand.requests})."
                    ],
                    z_score=0.0,
                    p_value=1.0,
                    champion_ctr=champ_ctr,
                    canary_ctr=cand_ctr,
                    champion_error_rate=champ_err,
                    canary_error_rate=cand_err,
                    champion_p95_ms=champ_p95,
                    canary_p95_ms=cand_p95,
                )
                self._last_evaluation = eval_insuff
                return eval_insuff

            p_pool = (champ.conversions + cand.conversions) / (
                champ.requests + cand.requests
            )
            se = math.sqrt(
                max(0.0, p_pool * (1.0 - p_pool))
                * ((1.0 / champ.requests) + (1.0 / cand.requests))
            )
            if se > 0.0:
                z_score = (cand_ctr - champ_ctr) / se
                p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
            else:
                z_score = 0.0
                p_value = 1.0

            reasons: list[str] = []
            if cand_err - champ_err > self.guardrails.max_error_rate_delta:
                reasons.append(
                    f"Canary error rate delta {cand_err - champ_err:.4f} exceeds "
                    f"{self.guardrails.max_error_rate_delta:.4f}"
                )
            if cand_p95 > self.guardrails.max_latency_ms:
                reasons.append(
                    f"Canary p95 latency {cand_p95:.1f}ms exceeds "
                    f"{self.guardrails.max_latency_ms:.1f}ms"
                )
            if (
                champ_p95 > 0.0
                and (cand_p95 / champ_p95) > self.guardrails.max_latency_p95_ratio
            ):
                reasons.append(
                    f"Canary p95 ratio {cand_p95 / champ_p95:.2f}x exceeds "
                    f"{self.guardrails.max_latency_p95_ratio:.2f}x"
                )
            if z_score < self.guardrails.min_ctr_lift_z:
                reasons.append(
                    f"CTR regression detected (z={z_score:.2f} < "
                    f"{self.guardrails.min_ctr_lift_z:.2f}, p={p_value:.4f})"
                )

            should_rollback = len(reasons) > 0
            if should_rollback:
                status: Literal[
                    "healthy",
                    "insufficient_data",
                    "rollback_triggered",
                    "promote_ready",
                ] = "rollback_triggered"
                if auto_rollback:
                    self.rollback(reason="; ".join(reasons))
            elif z_score >= abs(self.guardrails.min_ctr_lift_z):
                status = "promote_ready"
            else:
                status = "healthy"

            result = GuardrailEvaluation(
                status=status,
                should_rollback=should_rollback,
                reasons=reasons,
                z_score=z_score,
                p_value=p_value,
                champion_ctr=champ_ctr,
                canary_ctr=cand_ctr,
                champion_error_rate=champ_err,
                canary_error_rate=cand_err,
                champion_p95_ms=champ_p95,
                canary_p95_ms=cand_p95,
            )
            self._last_evaluation = result
            return result

    def get_status(self) -> dict[str, Any]:
        """Return canary policy, arm metrics, shadow summary, and evaluation."""
        with self._lock:
            comparisons = list(self._shadow_comparisons)
            avg_overlap = (
                sum(c["jaccard_overlap"] for c in comparisons) / len(comparisons)
                if comparisons
                else 1.0
            )
            return {
                "policy": self.policy.to_dict(),
                "guardrails": self.guardrails.to_dict(),
                "rollback_reason": self._rollback_reason,
                "arms": {
                    "champion": self._champion_metrics.to_dict(),
                    "canary": self._canary_metrics.to_dict(),
                },
                "shadow_summary": {
                    "sample_count": len(comparisons),
                    "mean_jaccard_overlap": round(avg_overlap, 4),
                    "recent": comparisons[-10:],
                },
                "last_evaluation": (
                    self._last_evaluation.to_dict()
                    if self._last_evaluation is not None
                    else None
                ),
            }

    def export_state(self) -> dict[str, Any]:
        """Export full router state as a JSON-serializable dictionary."""
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "policy": self.policy.to_dict(),
                "guardrails": self.guardrails.to_dict(),
                "rollback_reason": self._rollback_reason,
                "champion_metrics": self._champion_metrics.to_dict(),
                "canary_metrics": self._canary_metrics.to_dict(),
                "shadow_comparisons": list(self._shadow_comparisons),
            }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore router state from an exported dictionary."""
        with self._lock:
            if "policy" in state:
                self.policy = CanaryPolicy.from_dict(state["policy"])
            if "guardrails" in state:
                self.guardrails = ABGuardrailConfig.from_dict(state["guardrails"])
            self._rollback_reason = state.get("rollback_reason")
            if "champion_metrics" in state:
                self._champion_metrics = ArmMetrics.from_dict(state["champion_metrics"])
            if "canary_metrics" in state:
                self._canary_metrics = ArmMetrics.from_dict(state["canary_metrics"])
            self._shadow_comparisons = deque(
                state.get("shadow_comparisons", []),
                maxlen=200,
            )

    def save_state(self, path: str | Path) -> Path:
        """Persist router state snapshot to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> bool:
        """Load router state snapshot from a JSON file if it exists."""
        target = Path(path)
        if not target.exists():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.import_state(payload)
        return True


_global_canary_router = ShadowCanaryRouter()


def get_global_canary_router() -> ShadowCanaryRouter:
    """Return the process-wide ShadowCanaryRouter singleton."""
    return _global_canary_router


def set_global_canary_router(router: ShadowCanaryRouter) -> None:
    """Replace the process-wide ShadowCanaryRouter singleton."""
    global _global_canary_router
    _global_canary_router = router
