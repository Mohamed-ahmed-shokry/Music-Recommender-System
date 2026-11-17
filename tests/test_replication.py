"""Tests for multi-region replication, edge caching, and geo-failover routing."""

from __future__ import annotations

import pytest

from music_recommender.replication import (
    ArtifactManifest,
    GeoRoutingDecision,
    RegionEndpoint,
    RegionHealth,
)


def test_manifest_and_region_endpoint_models() -> None:
    """ArtifactManifest and RegionEndpoint should compute checksums, sync, and score."""
    manifest = ArtifactManifest.from_bytes("v1.0.0", b"als-artifact-bytes")
    assert manifest.size_bytes == len(b"als-artifact-bytes")
    assert len(manifest.checksum_sha256) == 64
    assert ArtifactManifest.from_dict(manifest.to_dict()) == manifest

    with pytest.raises(ValueError, match="model_version"):
        ArtifactManifest(model_version="", checksum_sha256="abc", size_bytes=1)

    region = RegionEndpoint(
        region_id="us-east-1",
        endpoint_url="https://us-east.rec.example.com",
        priority=1,
        latency_ms=15.0,
        error_rate=0.0,
    )
    assert region.is_synced(manifest) is False
    unsynced_score = region.health_score(manifest)

    region.cached_version = manifest.model_version
    region.cached_checksum = manifest.checksum_sha256
    assert region.is_synced(manifest) is True
    assert region.health_score(manifest) > unsynced_score

    restored = RegionEndpoint.from_dict(region.to_dict(manifest))
    assert restored.region_id == "us-east-1"
    assert restored.cached_checksum == manifest.checksum_sha256

    decision = GeoRoutingDecision(
        target_region="us-east-1",
        endpoint_url="https://us-east.rec.example.com",
        failover_used=False,
        reason="preferred_region_healthy",
        cache_hit=True,
        health=RegionHealth.HEALTHY,
    )
    assert decision.to_dict()["health"] == "healthy"
    assert decision.to_dict()["cache_hit"] is True
