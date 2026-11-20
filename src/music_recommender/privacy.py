"""Federated privacy-preserving personalization, DP budgets, and consent ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ConsentScope(StrEnum):
    """Supported user privacy consent scopes."""

    PERSONALIZATION = "personalization"
    FEDERATED_UPDATES = "federated_updates"
    ANALYTICS = "analytics"


@dataclass(slots=True)
class ConsentRecord:
    """Auditable consent ledger entry for a user."""

    user_id: str
    granted_scopes: set[ConsentScope] = field(default_factory=set)
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id must be non-empty")

    def has_scope(self, scope: ConsentScope | str) -> bool:
        """Return True if the requested consent scope is granted."""
        resolved = (
            scope
            if isinstance(scope, ConsentScope)
            else ConsentScope(str(scope).lower())
        )
        return resolved in self.granted_scopes

    def to_dict(self) -> dict[str, Any]:
        """Serialize consent record to a JSON-compatible dictionary."""
        return {
            "user_id": self.user_id,
            "granted_scopes": sorted(s.value for s in self.granted_scopes),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConsentRecord:
        """Deserialize a ConsentRecord from a dictionary."""
        return cls(
            user_id=str(payload["user_id"]),
            granted_scopes={ConsentScope(s) for s in payload.get("granted_scopes", [])},
            updated_at=str(payload.get("updated_at", datetime.now(UTC).isoformat())),
        )


@dataclass(frozen=True, slots=True)
class PrivacyBudgetConfig:
    """Differential privacy noise and per-user budget parameters."""

    max_epsilon: float = 10.0
    default_delta: float = 1e-5
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1

    def __post_init__(self) -> None:
        if self.max_epsilon <= 0.0:
            raise ValueError("max_epsilon must be > 0.0")
        if not 0.0 <= self.default_delta < 1.0:
            raise ValueError("default_delta must be in [0.0, 1.0)")
        if self.clip_norm <= 0.0:
            raise ValueError("clip_norm must be > 0.0")
        if self.noise_multiplier < 0.0:
            raise ValueError("noise_multiplier must be >= 0.0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize DP budget config to a JSON-compatible dictionary."""
        return {
            "max_epsilon": self.max_epsilon,
            "default_delta": self.default_delta,
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PrivacyBudgetConfig:
        """Deserialize a PrivacyBudgetConfig from a dictionary."""
        return cls(
            max_epsilon=float(payload.get("max_epsilon", 10.0)),
            default_delta=float(payload.get("default_delta", 1e-5)),
            clip_norm=float(payload.get("clip_norm", 1.0)),
            noise_multiplier=float(payload.get("noise_multiplier", 1.1)),
        )


@dataclass(slots=True)
class PrivacyBudgetState:
    """Per-user differential privacy expenditure ledger."""

    user_id: str
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    queries_count: int = 0
    max_epsilon: float = 10.0

    @property
    def remaining_epsilon(self) -> float:
        """Return remaining epsilon budget for this user."""
        return max(0.0, self.max_epsilon - self.epsilon_spent)

    @property
    def exhausted(self) -> bool:
        """Return True if the user's epsilon budget is exhausted."""
        return self.epsilon_spent >= self.max_epsilon

    def to_dict(self) -> dict[str, Any]:
        """Serialize privacy budget state to a JSON-compatible dictionary."""
        return {
            "user_id": self.user_id,
            "epsilon_spent": round(self.epsilon_spent, 6),
            "delta_spent": round(self.delta_spent, 8),
            "remaining_epsilon": round(self.remaining_epsilon, 6),
            "max_epsilon": self.max_epsilon,
            "queries_count": self.queries_count,
            "exhausted": self.exhausted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PrivacyBudgetState:
        """Deserialize a PrivacyBudgetState from a dictionary."""
        return cls(
            user_id=str(payload["user_id"]),
            epsilon_spent=float(payload.get("epsilon_spent", 0.0)),
            delta_spent=float(payload.get("delta_spent", 0.0)),
            queries_count=int(payload.get("queries_count", 0)),
            max_epsilon=float(payload.get("max_epsilon", 10.0)),
        )
