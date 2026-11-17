"""Multi-region model replication, edge artifact caching, and geo-failover routing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
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
