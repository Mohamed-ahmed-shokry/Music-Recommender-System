"""Feature store drift monitoring, embedding staleness, and retraining triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class DriftSeverity(StrEnum):
    """Severity level for feature drift and embedding staleness evaluations."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class FeatureBaseline:
    """Reference distribution summary and histogram bins for a numeric feature."""

    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    histogram_probs: tuple[float, ...]
    bin_edges: tuple[float, ...]
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize baseline statistics to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min_val": round(self.min_val, 6),
            "max_val": round(self.max_val, 6),
            "histogram_probs": [round(p, 6) for p in self.histogram_probs],
            "bin_edges": [round(e, 6) for e in self.bin_edges],
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureBaseline:
        """Deserialize a FeatureBaseline from a dictionary."""
        return cls(
            name=str(payload["name"]),
            mean=float(payload["mean"]),
            std=float(payload["std"]),
            min_val=float(payload["min_val"]),
            max_val=float(payload["max_val"]),
            histogram_probs=tuple(float(p) for p in payload["histogram_probs"]),
            bin_edges=tuple(float(e) for e in payload["bin_edges"]),
            sample_count=int(payload["sample_count"]),
        )


@dataclass(frozen=True, slots=True)
class DriftMonitorConfig:
    """Thresholds for PSI drift, z-shift, embedding staleness, and cooldown."""

    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25
    z_shift_threshold: float = 2.0
    embedding_max_age_hours: float = 168.0
    embedding_min_cosine_retention: float = 0.85
    retrain_cooldown_seconds: float = 3600.0
    min_samples: int = 20

    def __post_init__(self) -> None:
        if self.psi_warning_threshold < 0.0:
            raise ValueError("psi_warning_threshold must be >= 0.0")
        if self.psi_critical_threshold < self.psi_warning_threshold:
            raise ValueError("psi_critical_threshold must be >= psi_warning_threshold")
        if self.z_shift_threshold <= 0.0:
            raise ValueError("z_shift_threshold must be > 0.0")
        if self.embedding_max_age_hours <= 0.0:
            raise ValueError("embedding_max_age_hours must be > 0.0")
        if not 0.0 <= self.embedding_min_cosine_retention <= 1.0:
            raise ValueError(
                "embedding_min_cosine_retention must be between 0.0 and 1.0"
            )
        if self.retrain_cooldown_seconds < 0.0:
            raise ValueError("retrain_cooldown_seconds must be >= 0.0")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Serialize drift monitor config to a JSON-compatible dictionary."""
        return {
            "psi_warning_threshold": self.psi_warning_threshold,
            "psi_critical_threshold": self.psi_critical_threshold,
            "z_shift_threshold": self.z_shift_threshold,
            "embedding_max_age_hours": self.embedding_max_age_hours,
            "embedding_min_cosine_retention": self.embedding_min_cosine_retention,
            "retrain_cooldown_seconds": self.retrain_cooldown_seconds,
            "min_samples": self.min_samples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DriftMonitorConfig:
        """Deserialize a DriftMonitorConfig from a dictionary."""
        return cls(
            psi_warning_threshold=float(payload.get("psi_warning_threshold", 0.1)),
            psi_critical_threshold=float(payload.get("psi_critical_threshold", 0.25)),
            z_shift_threshold=float(payload.get("z_shift_threshold", 2.0)),
            embedding_max_age_hours=float(
                payload.get("embedding_max_age_hours", 168.0)
            ),
            embedding_min_cosine_retention=float(
                payload.get("embedding_min_cosine_retention", 0.85)
            ),
            retrain_cooldown_seconds=float(
                payload.get("retrain_cooldown_seconds", 3600.0)
            ),
            min_samples=int(payload.get("min_samples", 20)),
        )


@dataclass(slots=True)
class FeatureDriftReport:
    """Drift evaluation result for a single monitored feature."""

    feature_name: str
    psi: float
    mean_shift_z: float
    current_mean: float
    baseline_mean: float
    severity: DriftSeverity
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize feature drift report to a JSON-compatible dictionary."""
        return {
            "feature_name": self.feature_name,
            "psi": round(self.psi, 6),
            "mean_shift_z": round(self.mean_shift_z, 4),
            "current_mean": round(self.current_mean, 6),
            "baseline_mean": round(self.baseline_mean, 6),
            "severity": self.severity.value,
            "sample_count": self.sample_count,
        }


@dataclass(slots=True)
class EmbeddingStalenessReport:
    """Freshness and semantic retention evaluation for model embeddings."""

    age_hours: float
    max_age_hours: float
    cosine_retention: float
    min_cosine_retention: float
    stale: bool
    severity: DriftSeverity
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize embedding staleness report to a JSON-compatible dictionary."""
        return {
            "age_hours": round(self.age_hours, 3),
            "max_age_hours": self.max_age_hours,
            "cosine_retention": round(self.cosine_retention, 4),
            "min_cosine_retention": self.min_cosine_retention,
            "stale": self.stale,
            "severity": self.severity.value,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class RetrainTriggerDecision:
    """Automated retraining recommendation produced from drift & staleness signals."""

    should_retrain: bool
    severity: DriftSeverity
    reasons: list[str]
    drifted_features: list[str]
    embedding_stale: bool
    cooldown_active: bool
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize retrain trigger decision to a JSON-compatible dictionary."""
        return {
            "should_retrain": self.should_retrain,
            "severity": self.severity.value,
            "reasons": list(self.reasons),
            "drifted_features": list(self.drifted_features),
            "embedding_stale": self.embedding_stale,
            "cooldown_active": self.cooldown_active,
            "evaluated_at": self.evaluated_at,
        }
