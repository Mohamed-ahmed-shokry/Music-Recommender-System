"""Federated privacy-preserving personalization, DP budgets, and consent ledger."""

from __future__ import annotations

import json
import math
import random
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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


class FederatedPrivacyManager:
    """Thread-safe DP budget accountant, consent ledger, and federated aggregator."""

    def __init__(
        self,
        *,
        config: PrivacyBudgetConfig | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.config = config or PrivacyBudgetConfig()
        self._consents: dict[str, ConsentRecord] = {}
        self._budgets: dict[str, PrivacyBudgetState] = {}
        self._federated_rounds_total: int = 0
        self._total_epsilon_spent: float = 0.0

    def set_consent(
        self,
        user_id: str,
        scopes: Sequence[ConsentScope | str],
    ) -> ConsentRecord:
        """Grant the specified consent scopes for a user."""
        clean_id = user_id.strip()
        if not clean_id:
            raise ValueError("user_id must be non-empty")
        resolved = {
            s if isinstance(s, ConsentScope) else ConsentScope(str(s).lower())
            for s in scopes
        }
        record = ConsentRecord(user_id=clean_id, granted_scopes=resolved)
        with self._lock:
            self._consents[clean_id] = record
        return record

    def revoke_consent(self, user_id: str) -> ConsentRecord:
        """Revoke all consent scopes for a user."""
        return self.set_consent(user_id, [])

    def get_consent(self, user_id: str) -> ConsentRecord:
        """Return stored ConsentRecord or an empty record for a user."""
        clean_id = user_id.strip()
        if not clean_id:
            raise ValueError("user_id must be non-empty")
        with self._lock:
            return self._consents.get(clean_id, ConsentRecord(user_id=clean_id))

    def has_consent(self, user_id: str, scope: ConsentScope | str) -> bool:
        """Check whether a user has granted a specific consent scope."""
        return self.get_consent(user_id).has_scope(scope)

    def get_budget(self, user_id: str) -> PrivacyBudgetState:
        """Return current DP budget state for a user."""
        clean_id = user_id.strip()
        if not clean_id:
            raise ValueError("user_id must be non-empty")
        with self._lock:
            existing = self._budgets.get(clean_id)
            if existing is not None:
                return existing
            return PrivacyBudgetState(
                user_id=clean_id,
                max_epsilon=self.config.max_epsilon,
            )

    def can_spend(self, user_id: str, epsilon: float) -> bool:
        """Return True if the user has sufficient remaining epsilon budget."""
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.0")
        with self._lock:
            state = self.get_budget(user_id)
            return state.epsilon_spent + epsilon <= state.max_epsilon + 1e-12

    def consume_privacy_budget(
        self,
        user_id: str,
        *,
        epsilon: float,
        delta: float | None = None,
    ) -> PrivacyBudgetState:
        """Deduct epsilon/delta from a user's privacy budget or raise ValueError."""
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.0")
        clean_id = user_id.strip()
        if not clean_id:
            raise ValueError("user_id must be non-empty")
        delta_inc = self.config.default_delta if delta is None else float(delta)
        if delta_inc < 0.0:
            raise ValueError("delta must be >= 0.0")

        with self._lock:
            state = self._budgets.get(clean_id)
            if state is None:
                state = PrivacyBudgetState(
                    user_id=clean_id,
                    max_epsilon=self.config.max_epsilon,
                )
                self._budgets[clean_id] = state
            if state.epsilon_spent + epsilon > state.max_epsilon + 1e-12:
                raise ValueError(
                    f"Privacy budget exhausted for user '{clean_id}' "
                    f"(remaining={state.remaining_epsilon:.4f}, "
                    f"requested={epsilon:.4f})"
                )
            state.epsilon_spent += float(epsilon)
            state.delta_spent += delta_inc
            state.queries_count += 1
            self._total_epsilon_spent += float(epsilon)
            return state

    def clip_and_add_dp_noise(
        self,
        vector: Sequence[float],
        *,
        epsilon: float = 1.0,
        seed: int | None = None,
    ) -> list[float]:
        """L2-clip vector to clip_norm and add Gaussian DP noise."""
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.0")
        vals = [float(x) for x in vector]
        if not vals:
            return []
        norm = math.sqrt(sum(v * v for v in vals))
        scale = min(1.0, self.config.clip_norm / norm) if norm > 0.0 else 1.0
        clipped = [v * scale for v in vals]

        sigma = (
            self.config.clip_norm
            * self.config.noise_multiplier
            / max(0.01, float(epsilon))
        )
        rng = random.Random(seed) if seed is not None else random.Random()
        return [round(c + rng.gauss(0.0, sigma), 6) for c in clipped]

    def aggregate_federated_updates(
        self,
        updates: Sequence[tuple[str, Sequence[float]]],
        *,
        epsilon_per_user: float = 0.5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Aggregate consented, budget-valid user updates with DP noise."""
        if epsilon_per_user <= 0.0:
            raise ValueError("epsilon_per_user must be > 0.0")

        accepted_users: list[str] = []
        rejected_users: dict[str, str] = {}
        privatized_vectors: list[list[float]] = []

        with self._lock:
            expected_dim: int | None = None
            for idx, (user_id, vec) in enumerate(updates):
                clean_id = str(user_id).strip()
                if not clean_id:
                    continue
                if not self.has_consent(clean_id, ConsentScope.FEDERATED_UPDATES):
                    rejected_users[clean_id] = "missing_federated_updates_consent"
                    continue
                if not self.can_spend(clean_id, epsilon_per_user):
                    rejected_users[clean_id] = "privacy_budget_exhausted"
                    continue
                if expected_dim is None:
                    expected_dim = len(vec)
                elif len(vec) != expected_dim:
                    rejected_users[clean_id] = "dimension_mismatch"
                    continue

                self.consume_privacy_budget(clean_id, epsilon=epsilon_per_user)
                user_seed = (seed + idx) if seed is not None else None
                privatized = self.clip_and_add_dp_noise(
                    vec, epsilon=epsilon_per_user, seed=user_seed
                )
                privatized_vectors.append(privatized)
                accepted_users.append(clean_id)

            if privatized_vectors:
                dim = len(privatized_vectors[0])
                count = float(len(privatized_vectors))
                aggregated = [
                    round(sum(v[d] for v in privatized_vectors) / count, 6)
                    for d in range(dim)
                ]
            else:
                aggregated = []

            self._federated_rounds_total += 1
            return {
                "round_id": self._federated_rounds_total,
                "accepted_count": len(accepted_users),
                "rejected_count": len(rejected_users),
                "accepted_users": accepted_users,
                "rejected_users": rejected_users,
                "epsilon_per_user": epsilon_per_user,
                "aggregated_update": aggregated,
            }

    def get_status(self) -> dict[str, Any]:
        """Return consent ledger counts, DP config, and total epsilon spent."""
        with self._lock:
            exhausted_count = sum(1 for b in self._budgets.values() if b.exhausted)
            return {
                "config": self.config.to_dict(),
                "consent_records_count": len(self._consents),
                "tracked_budgets_count": len(self._budgets),
                "exhausted_budgets_count": exhausted_count,
                "total_epsilon_spent": round(self._total_epsilon_spent, 6),
                "federated_rounds_total": self._federated_rounds_total,
            }

    def export_state(self) -> dict[str, Any]:
        """Export full privacy ledger and budgets to a JSON-serializable dict."""
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "config": self.config.to_dict(),
                "federated_rounds_total": self._federated_rounds_total,
                "total_epsilon_spent": self._total_epsilon_spent,
                "consents": {k: v.to_dict() for k, v in sorted(self._consents.items())},
                "budgets": {k: v.to_dict() for k, v in sorted(self._budgets.items())},
            }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore privacy ledger and budgets from an exported dictionary."""
        with self._lock:
            if "config" in state:
                self.config = PrivacyBudgetConfig.from_dict(state["config"])
            self._federated_rounds_total = int(state.get("federated_rounds_total", 0))
            self._total_epsilon_spent = float(state.get("total_epsilon_spent", 0.0))
            self._consents = {
                str(k): ConsentRecord.from_dict(v)
                for k, v in state.get("consents", {}).items()
            }
            self._budgets = {
                str(k): PrivacyBudgetState.from_dict(v)
                for k, v in state.get("budgets", {}).items()
            }

    def save_state(self, path: str | Path) -> Path:
        """Persist privacy manager state snapshot to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> bool:
        """Load privacy manager state snapshot from a JSON file if it exists."""
        target = Path(path)
        if not target.exists():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.import_state(payload)
        return True


_global_privacy_manager = FederatedPrivacyManager()


def get_global_privacy_manager() -> FederatedPrivacyManager:
    """Return the process-wide FederatedPrivacyManager singleton."""
    return _global_privacy_manager


def set_global_privacy_manager(manager: FederatedPrivacyManager) -> None:
    """Replace the process-wide FederatedPrivacyManager singleton."""
    global _global_privacy_manager
    _global_privacy_manager = manager
