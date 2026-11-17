"""Feature store drift monitoring, embedding staleness, and retraining triggers."""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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


def _build_histogram(
    values: Sequence[float],
    bin_edges: Sequence[float] | None = None,
    bins: int = 10,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute smoothed bin probabilities and bin edges for PSI."""
    if not values:
        raise ValueError("values must be non-empty")
    edges: tuple[float, ...]
    if bin_edges is None:
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            edges = (lo - 0.5, hi + 0.5)
        else:
            step = (hi - lo) / max(1, bins)
            edges = tuple(lo + i * step for i in range(bins + 1))
    else:
        edges = tuple(float(e) for e in bin_edges)

    num_bins = max(1, len(edges) - 1)
    counts = [0] * num_bins
    for val in values:
        placed = False
        for idx in range(num_bins):
            left = edges[idx]
            right = edges[idx + 1]
            if idx == 0 and val < left:
                counts[0] += 1
                placed = True
                break
            if idx == num_bins - 1 and val >= left:
                counts[num_bins - 1] += 1
                placed = True
                break
            if left <= val < right:
                counts[idx] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    total = float(len(values))
    eps = 1e-6
    probs = tuple(max(eps, c / total) for c in counts)
    norm = sum(probs)
    return tuple(p / norm for p in probs), edges


class FeatureDriftMonitor:
    """Thread-safe feature distribution drift, embedding staleness, and
    automated retraining trigger monitor.
    """

    def __init__(
        self,
        *,
        config: DriftMonitorConfig | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.config = config or DriftMonitorConfig()
        self._time_fn: Callable[[], float] = time_fn or time.time

        self._baselines: dict[str, FeatureBaseline] = {}
        self._observations: dict[str, deque[float]] = {}
        self._embedding_updated_at: float | None = None
        self._embedding_cosine_retention: float = 1.0
        self._last_retrain_at: float | None = None
        self._retrain_trigger_count: int = 0
        self._last_evaluation: dict[str, Any] | None = None

    def _now(self, now: float | None = None) -> float:
        return self._time_fn() if now is None else now

    def register_baseline(
        self,
        name: str,
        values: Sequence[float],
        *,
        bins: int = 10,
    ) -> FeatureBaseline:
        """Compute and store a reference baseline distribution for a feature."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("feature name must be non-empty")
        nums = [float(v) for v in values]
        if not nums:
            raise ValueError("baseline values must be non-empty")

        probs, edges = _build_histogram(nums, bins=bins)
        mean_val = statistics.fmean(nums)
        std_val = statistics.pstdev(nums) if len(nums) > 1 else 0.0
        baseline = FeatureBaseline(
            name=clean_name,
            mean=mean_val,
            std=std_val,
            min_val=min(nums),
            max_val=max(nums),
            histogram_probs=probs,
            bin_edges=edges,
            sample_count=len(nums),
        )
        with self._lock:
            self._baselines[clean_name] = baseline
            self._observations.setdefault(clean_name, deque(maxlen=2000))
        return baseline

    def record_observations(self, name: str, values: Sequence[float]) -> int:
        """Append recent observed values for a monitored feature."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("feature name must be non-empty")
        nums = [float(v) for v in values]
        with self._lock:
            buf = self._observations.setdefault(clean_name, deque(maxlen=2000))
            buf.extend(nums)
            return len(buf)

    def record_embedding_snapshot(
        self,
        *,
        updated_at: float | None = None,
        cosine_retention: float = 1.0,
    ) -> None:
        """Record embedding freshness timestamp and cosine similarity retention."""
        if not 0.0 <= cosine_retention <= 1.0:
            raise ValueError("cosine_retention must be between 0.0 and 1.0")
        with self._lock:
            self._embedding_updated_at = self._now(updated_at)
            self._embedding_cosine_retention = float(cosine_retention)

    @staticmethod
    def compute_psi(
        baseline: FeatureBaseline,
        current_values: Sequence[float],
    ) -> float:
        """Compute Population Stability Index between baseline and current values."""
        if not current_values:
            return 0.0
        cur_probs, _ = _build_histogram(current_values, bin_edges=baseline.bin_edges)
        eps = 1e-6
        psi = 0.0
        for base_p, cur_p in zip(baseline.histogram_probs, cur_probs, strict=True):
            bp = max(eps, base_p)
            cp = max(eps, cur_p)
            psi += (cp - bp) * math.log(cp / bp)
        return max(0.0, psi)

    def evaluate_embedding_staleness(
        self,
        *,
        now: float | None = None,
    ) -> EmbeddingStalenessReport:
        """Evaluate embedding age and cosine retention against thresholds."""
        with self._lock:
            current = self._now(now)
            if self._embedding_updated_at is None:
                age_hours = 0.0
            else:
                age_hours = max(0.0, (current - self._embedding_updated_at) / 3600.0)
            retention = self._embedding_cosine_retention
            reasons: list[str] = []
            if age_hours > self.config.embedding_max_age_hours:
                reasons.append(
                    f"Embedding age {age_hours:.1f}h exceeds max "
                    f"{self.config.embedding_max_age_hours:.1f}h"
                )
            if retention < self.config.embedding_min_cosine_retention:
                reasons.append(
                    f"Cosine retention {retention:.3f} below min "
                    f"{self.config.embedding_min_cosine_retention:.3f}"
                )
            stale = len(reasons) > 0
            severity = DriftSeverity.CRITICAL if stale else DriftSeverity.NONE
            return EmbeddingStalenessReport(
                age_hours=age_hours,
                max_age_hours=self.config.embedding_max_age_hours,
                cosine_retention=retention,
                min_cosine_retention=self.config.embedding_min_cosine_retention,
                stale=stale,
                severity=severity,
                reasons=reasons,
            )

    def evaluate_drift(
        self,
        *,
        now: float | None = None,
        record_trigger: bool = True,
    ) -> dict[str, Any]:
        """Evaluate registered features, embedding staleness, and retrain trigger."""
        with self._lock:
            current = self._now(now)
            feature_reports: list[FeatureDriftReport] = []
            drifted_features: list[str] = []
            reasons: list[str] = []
            max_severity = DriftSeverity.NONE

            for name, baseline in sorted(self._baselines.items()):
                obs = list(self._observations.get(name, ()))
                if len(obs) < self.config.min_samples:
                    cur_mean = statistics.fmean(obs) if obs else baseline.mean
                    feature_reports.append(
                        FeatureDriftReport(
                            feature_name=name,
                            psi=0.0,
                            mean_shift_z=0.0,
                            current_mean=cur_mean,
                            baseline_mean=baseline.mean,
                            severity=DriftSeverity.NONE,
                            sample_count=len(obs),
                        )
                    )
                    continue

                psi = self.compute_psi(baseline, obs)
                cur_mean = statistics.fmean(obs)
                denom = baseline.std if baseline.std > 1e-9 else 1.0
                z_shift = abs(cur_mean - baseline.mean) / denom

                if (
                    psi >= self.config.psi_critical_threshold
                    or z_shift >= self.config.z_shift_threshold * 1.5
                ):
                    sev = DriftSeverity.CRITICAL
                    drifted_features.append(name)
                    reasons.append(
                        f"Feature '{name}' critical drift "
                        f"(PSI={psi:.3f}, z={z_shift:.2f})"
                    )
                    max_severity = DriftSeverity.CRITICAL
                elif (
                    psi >= self.config.psi_warning_threshold
                    or z_shift >= self.config.z_shift_threshold
                ):
                    sev = DriftSeverity.WARNING
                    drifted_features.append(name)
                    reasons.append(
                        f"Feature '{name}' warning drift "
                        f"(PSI={psi:.3f}, z={z_shift:.2f})"
                    )
                    if max_severity == DriftSeverity.NONE:
                        max_severity = DriftSeverity.WARNING
                else:
                    sev = DriftSeverity.NONE

                feature_reports.append(
                    FeatureDriftReport(
                        feature_name=name,
                        psi=psi,
                        mean_shift_z=z_shift,
                        current_mean=cur_mean,
                        baseline_mean=baseline.mean,
                        severity=sev,
                        sample_count=len(obs),
                    )
                )

            emb_report = self.evaluate_embedding_staleness(now=current)
            if emb_report.stale:
                reasons.extend(emb_report.reasons)
                max_severity = DriftSeverity.CRITICAL

            cooldown_active = (
                self._last_retrain_at is not None
                and (current - self._last_retrain_at)
                < self.config.retrain_cooldown_seconds
            )
            should_retrain = (
                max_severity == DriftSeverity.CRITICAL or emb_report.stale
            ) and not cooldown_active
            if should_retrain and record_trigger:
                self._retrain_trigger_count += 1

            decision = RetrainTriggerDecision(
                should_retrain=should_retrain,
                severity=max_severity,
                reasons=reasons,
                drifted_features=drifted_features,
                embedding_stale=emb_report.stale,
                cooldown_active=cooldown_active,
            )
            result = {
                "features": [r.to_dict() for r in feature_reports],
                "embedding": emb_report.to_dict(),
                "decision": decision.to_dict(),
                "max_psi": round(max((r.psi for r in feature_reports), default=0.0), 6),
                "retrain_trigger_count": self._retrain_trigger_count,
            }
            self._last_evaluation = result
            return result

    def acknowledge_retrain(
        self,
        *,
        now: float | None = None,
        clear_observations: bool = True,
    ) -> dict[str, Any]:
        """Record retrain completion, start cooldown, and refresh embedding age."""
        with self._lock:
            current = self._now(now)
            self._last_retrain_at = current
            self._embedding_updated_at = current
            self._embedding_cosine_retention = 1.0
            if clear_observations:
                for buf in self._observations.values():
                    buf.clear()
            return {
                "acknowledged": True,
                "last_retrain_at": current,
                "retrain_trigger_count": self._retrain_trigger_count,
            }

    def get_status(self, *, now: float | None = None) -> dict[str, Any]:
        """Return drift monitor config, baselines, embedding freshness, and eval."""
        with self._lock:
            current = self._now(now)
            emb_report = self.evaluate_embedding_staleness(now=current)
            return {
                "config": self.config.to_dict(),
                "baseline_count": len(self._baselines),
                "baselines": {
                    k: v.to_dict() for k, v in sorted(self._baselines.items())
                },
                "observation_counts": {
                    k: len(v) for k, v in sorted(self._observations.items())
                },
                "embedding": emb_report.to_dict(),
                "last_retrain_at": self._last_retrain_at,
                "retrain_trigger_count": self._retrain_trigger_count,
                "last_evaluation": self._last_evaluation,
            }

    def export_state(self) -> dict[str, Any]:
        """Export full drift monitor state to a JSON-serializable dictionary."""
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "config": self.config.to_dict(),
                "baselines": {k: v.to_dict() for k, v in self._baselines.items()},
                "observations": {k: list(v) for k, v in self._observations.items()},
                "embedding_updated_at": self._embedding_updated_at,
                "embedding_cosine_retention": self._embedding_cosine_retention,
                "last_retrain_at": self._last_retrain_at,
                "retrain_trigger_count": self._retrain_trigger_count,
            }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore drift monitor state from an exported dictionary."""
        with self._lock:
            if "config" in state:
                self.config = DriftMonitorConfig.from_dict(state["config"])
            self._baselines = {
                str(k): FeatureBaseline.from_dict(v)
                for k, v in state.get("baselines", {}).items()
            }
            self._observations = {
                str(k): deque((float(x) for x in v), maxlen=2000)
                for k, v in state.get("observations", {}).items()
            }
            self._embedding_updated_at = state.get("embedding_updated_at")
            self._embedding_cosine_retention = float(
                state.get("embedding_cosine_retention", 1.0)
            )
            self._last_retrain_at = state.get("last_retrain_at")
            self._retrain_trigger_count = int(state.get("retrain_trigger_count", 0))

    def save_state(self, path: str | Path) -> Path:
        """Persist drift monitor state snapshot to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> bool:
        """Load drift monitor state snapshot from a JSON file if it exists."""
        target = Path(path)
        if not target.exists():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.import_state(payload)
        return True


_global_drift_monitor = FeatureDriftMonitor()


def get_global_drift_monitor() -> FeatureDriftMonitor:
    """Return the process-wide FeatureDriftMonitor singleton."""
    return _global_drift_monitor


def set_global_drift_monitor(monitor: FeatureDriftMonitor) -> None:
    """Replace the process-wide FeatureDriftMonitor singleton."""
    global _global_drift_monitor
    _global_drift_monitor = monitor
