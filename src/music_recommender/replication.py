"""Multi-region model replication, edge artifact caching, and geo-failover routing."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class RegionHealth(StrEnum):
    """Operational health classification for a serving region."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Immutable model artifact manifest for multi-region replication."""

    model_version: str
    checksum_sha256: str
    size_bytes: int
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")
        if not self.checksum_sha256.strip():
            raise ValueError("checksum_sha256 must be non-empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")

    @classmethod
    def from_bytes(
        cls,
        model_version: str,
        payload: bytes,
    ) -> ArtifactManifest:
        """Build an ArtifactManifest by computing SHA-256 over raw payload bytes."""
        digest = hashlib.sha256(payload).hexdigest()
        return cls(
            model_version=model_version,
            checksum_sha256=digest,
            size_bytes=len(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a JSON-compatible dictionary."""
        return {
            "model_version": self.model_version,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactManifest:
        """Deserialize an ArtifactManifest from a dictionary."""
        return cls(
            model_version=str(payload["model_version"]),
            checksum_sha256=str(payload["checksum_sha256"]),
            size_bytes=int(payload["size_bytes"]),
            created_at=str(payload.get("created_at", datetime.now(UTC).isoformat())),
        )


@dataclass(slots=True)
class RegionEndpoint:
    """State of a regional serving endpoint and its edge artifact cache."""

    region_id: str
    endpoint_url: str
    priority: int = 1
    weight: float = 1.0
    health: RegionHealth = RegionHealth.HEALTHY
    latency_ms: float = 20.0
    error_rate: float = 0.0
    cached_version: str | None = None
    cached_checksum: str | None = None
    last_synced_at: float | None = None

    def is_synced(self, manifest: ArtifactManifest | None) -> bool:
        """Return True if this region's edge cache matches the active manifest."""
        if manifest is None:
            return True
        return (
            self.cached_version == manifest.model_version
            and self.cached_checksum == manifest.checksum_sha256
        )

    def health_score(self, manifest: ArtifactManifest | None = None) -> float:
        """Compute composite routing score (higher is better)."""
        if self.health == RegionHealth.UNHEALTHY:
            return -1.0
        base = 100.0 if self.health == RegionHealth.HEALTHY else 40.0
        sync_bonus = 20.0 if self.is_synced(manifest) else 0.0
        latency_penalty = min(30.0, self.latency_ms / 20.0)
        error_penalty = min(40.0, self.error_rate * 100.0)
        priority_bonus = max(0.0, 10.0 - float(self.priority))
        return round(
            max(
                0.0,
                base + sync_bonus + priority_bonus - latency_penalty - error_penalty,
            ),
            3,
        )

    def to_dict(self, manifest: ArtifactManifest | None = None) -> dict[str, Any]:
        """Serialize regional endpoint state to a JSON-compatible dictionary."""
        return {
            "region_id": self.region_id,
            "endpoint_url": self.endpoint_url,
            "priority": self.priority,
            "weight": self.weight,
            "health": self.health.value,
            "latency_ms": round(self.latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "cached_version": self.cached_version,
            "cached_checksum": self.cached_checksum,
            "last_synced_at": self.last_synced_at,
            "synced": self.is_synced(manifest),
            "health_score": self.health_score(manifest),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RegionEndpoint:
        """Deserialize a RegionEndpoint from a dictionary."""
        return cls(
            region_id=str(payload["region_id"]),
            endpoint_url=str(payload["endpoint_url"]),
            priority=int(payload.get("priority", 1)),
            weight=float(payload.get("weight", 1.0)),
            health=RegionHealth(payload.get("health", RegionHealth.HEALTHY.value)),
            latency_ms=float(payload.get("latency_ms", 20.0)),
            error_rate=float(payload.get("error_rate", 0.0)),
            cached_version=payload.get("cached_version"),
            cached_checksum=payload.get("cached_checksum"),
            last_synced_at=payload.get("last_synced_at"),
        )


@dataclass(slots=True)
class GeoRoutingDecision:
    """Outcome of geo-failover routing across regional endpoints."""

    target_region: str
    endpoint_url: str
    failover_used: bool
    reason: str
    cache_hit: bool
    health: RegionHealth
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize geo routing decision to a JSON-compatible dictionary."""
        return {
            "target_region": self.target_region,
            "endpoint_url": self.endpoint_url,
            "failover_used": self.failover_used,
            "reason": self.reason,
            "cache_hit": self.cache_hit,
            "health": self.health.value,
            "decided_at": self.decided_at,
        }


class MultiRegionReplicator:
    """Thread-safe multi-region model manifest replicator and geo-failover router."""

    def __init__(
        self,
        *,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._time_fn: Callable[[], float] = time_fn or time.time
        self._manifest: ArtifactManifest | None = None
        self._regions: dict[str, RegionEndpoint] = {}
        self._failover_events_total: int = 0

    def _now(self, now: float | None = None) -> float:
        return self._time_fn() if now is None else now

    def publish_manifest(
        self,
        model_version: str,
        checksum_sha256: str | None = None,
        size_bytes: int = 0,
    ) -> ArtifactManifest:
        """Publish a new active model artifact manifest for regional replication."""
        digest = (
            checksum_sha256.strip()
            if checksum_sha256
            else hashlib.sha256(model_version.encode("utf-8")).hexdigest()
        )
        manifest = ArtifactManifest(
            model_version=model_version.strip(),
            checksum_sha256=digest,
            size_bytes=size_bytes,
        )
        with self._lock:
            self._manifest = manifest
        return manifest

    def register_region(
        self,
        region_id: str,
        endpoint_url: str,
        *,
        priority: int = 1,
        weight: float = 1.0,
    ) -> RegionEndpoint:
        """Register or update a regional serving endpoint."""
        clean_id = region_id.strip()
        clean_url = endpoint_url.strip()
        if not clean_id:
            raise ValueError("region_id must be non-empty")
        if not clean_url.startswith(("http://", "https://")):
            raise ValueError("endpoint_url must start with http:// or https://")
        with self._lock:
            existing = self._regions.get(clean_id)
            if existing is not None:
                existing.endpoint_url = clean_url
                existing.priority = priority
                existing.weight = weight
                return existing
            endpoint = RegionEndpoint(
                region_id=clean_id,
                endpoint_url=clean_url,
                priority=priority,
                weight=weight,
            )
            self._regions[clean_id] = endpoint
            return endpoint

    def sync_region(
        self,
        region_id: str,
        *,
        now: float | None = None,
    ) -> RegionEndpoint:
        """Synchronize the active manifest into a region's edge cache."""
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                raise KeyError(f"Unknown region '{region_id}'")
            if self._manifest is None:
                raise ValueError("No active ArtifactManifest published")
            region.cached_version = self._manifest.model_version
            region.cached_checksum = self._manifest.checksum_sha256
            region.last_synced_at = self._now(now)
            return region

    def sync_all_regions(self, *, now: float | None = None) -> dict[str, Any]:
        """Synchronize the active manifest across all non-unhealthy regions."""
        with self._lock:
            if self._manifest is None:
                raise ValueError("No active ArtifactManifest published")
            current = self._now(now)
            synced: list[str] = []
            skipped: list[str] = []
            for region_id, region in sorted(self._regions.items()):
                if region.health == RegionHealth.UNHEALTHY:
                    skipped.append(region_id)
                    continue
                region.cached_version = self._manifest.model_version
                region.cached_checksum = self._manifest.checksum_sha256
                region.last_synced_at = current
                synced.append(region_id)
            return {
                "model_version": self._manifest.model_version,
                "synced_regions": synced,
                "skipped_regions": skipped,
            }

    def record_region_health(
        self,
        region_id: str,
        *,
        latency_ms: float,
        error_rate: float,
    ) -> RegionEndpoint:
        """Update region telemetry and derive RegionHealth status."""
        if latency_ms < 0.0:
            raise ValueError("latency_ms must be >= 0.0")
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0.0 and 1.0")
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                raise KeyError(f"Unknown region '{region_id}'")
            region.latency_ms = float(latency_ms)
            region.error_rate = float(error_rate)
            if error_rate >= 0.25 or latency_ms >= 1000.0:
                region.health = RegionHealth.UNHEALTHY
            elif error_rate >= 0.05 or latency_ms >= 300.0:
                region.health = RegionHealth.DEGRADED
            else:
                region.health = RegionHealth.HEALTHY
            return region

    def route_request(
        self,
        preferred_region: str | None = None,
    ) -> GeoRoutingDecision:
        """Select optimal serving region with automatic geo-failover."""
        with self._lock:
            if not self._regions:
                raise ValueError("No regions registered for geo-routing")

            if preferred_region is not None:
                pref = self._regions.get(preferred_region)
                if (
                    pref is not None
                    and pref.health == RegionHealth.HEALTHY
                    and pref.is_synced(self._manifest)
                ):
                    return GeoRoutingDecision(
                        target_region=pref.region_id,
                        endpoint_url=pref.endpoint_url,
                        failover_used=False,
                        reason="preferred_region_healthy",
                        cache_hit=pref.is_synced(self._manifest),
                        health=pref.health,
                    )

            candidates = [
                r for r in self._regions.values() if r.health != RegionHealth.UNHEALTHY
            ]
            if not candidates:
                # Last-resort fallback to lowest-error region if all are unhealthy
                fallback = min(
                    self._regions.values(),
                    key=lambda r: (r.error_rate, r.latency_ms, r.region_id),
                )
                self._failover_events_total += 1
                return GeoRoutingDecision(
                    target_region=fallback.region_id,
                    endpoint_url=fallback.endpoint_url,
                    failover_used=True,
                    reason="all_regions_unhealthy_fallback",
                    cache_hit=fallback.is_synced(self._manifest),
                    health=fallback.health,
                )

            best = max(
                candidates,
                key=lambda r: (
                    r.health_score(self._manifest),
                    -r.latency_ms,
                    r.region_id,
                ),
            )
            failover_used = (
                preferred_region is not None and best.region_id != preferred_region
            )
            if failover_used:
                self._failover_events_total += 1
                reason = f"failover_from_{preferred_region}"
            else:
                reason = "highest_scoring_region"

            return GeoRoutingDecision(
                target_region=best.region_id,
                endpoint_url=best.endpoint_url,
                failover_used=failover_used,
                reason=reason,
                cache_hit=best.is_synced(self._manifest),
                health=best.health,
            )

    def get_status(self) -> dict[str, Any]:
        """Return active manifest, regional edge cache status, and failover count."""
        with self._lock:
            regions = [
                r.to_dict(self._manifest) for _, r in sorted(self._regions.items())
            ]
            healthy_count = sum(
                1 for r in self._regions.values() if r.health == RegionHealth.HEALTHY
            )
            synced_count = sum(
                1 for r in self._regions.values() if r.is_synced(self._manifest)
            )
            return {
                "active_manifest": (
                    self._manifest.to_dict() if self._manifest is not None else None
                ),
                "regions_total": len(regions),
                "regions_healthy": healthy_count,
                "regions_synced": synced_count,
                "failover_events_total": self._failover_events_total,
                "regions": regions,
            }

    def export_state(self) -> dict[str, Any]:
        """Export full replicator state to a JSON-serializable dictionary."""
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "manifest": (
                    self._manifest.to_dict() if self._manifest is not None else None
                ),
                "failover_events_total": self._failover_events_total,
                "regions": {
                    k: v.to_dict(self._manifest)
                    for k, v in sorted(self._regions.items())
                },
            }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore replicator state from an exported dictionary."""
        with self._lock:
            manifest_data = state.get("manifest")
            self._manifest = (
                ArtifactManifest.from_dict(manifest_data)
                if manifest_data is not None
                else None
            )
            self._failover_events_total = int(state.get("failover_events_total", 0))
            self._regions = {
                str(k): RegionEndpoint.from_dict(v)
                for k, v in state.get("regions", {}).items()
            }

    def save_state(self, path: str | Path) -> Path:
        """Persist replicator state snapshot to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> bool:
        """Load replicator state snapshot from a JSON file if it exists."""
        target = Path(path)
        if not target.exists():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.import_state(payload)
        return True


_global_replicator = MultiRegionReplicator()


def get_global_replicator() -> MultiRegionReplicator:
    """Return the process-wide MultiRegionReplicator singleton."""
    return _global_replicator


def set_global_replicator(replicator: MultiRegionReplicator) -> None:
    """Replace the process-wide MultiRegionReplicator singleton."""
    global _global_replicator
    _global_replicator = replicator
