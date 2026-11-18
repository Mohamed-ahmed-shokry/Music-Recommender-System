"""Tests for multi-region replication, edge caching, and geo-failover routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.replication import (
    ArtifactManifest,
    GeoRoutingDecision,
    MultiRegionReplicator,
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


def test_multi_region_sync_and_health_classification() -> None:
    """Replicator should sync healthy/degraded regions and skip unhealthy ones."""
    now = [5000.0]
    rep = MultiRegionReplicator(time_fn=lambda: now[0])
    rep.publish_manifest("v2.0.0", size_bytes=2048)

    rep.register_region("us-east-1", "https://us-east.example.com", priority=1)
    rep.register_region("eu-west-1", "https://eu-west.example.com", priority=2)
    rep.register_region("ap-south-1", "https://ap-south.example.com", priority=3)

    rep.record_region_health("us-east-1", latency_ms=25.0, error_rate=0.0)
    rep.record_region_health("eu-west-1", latency_ms=350.0, error_rate=0.02)
    rep.record_region_health("ap-south-1", latency_ms=1200.0, error_rate=0.3)

    sync_res = rep.sync_all_regions()
    assert sync_res["model_version"] == "v2.0.0"
    assert set(sync_res["synced_regions"]) == {"eu-west-1", "us-east-1"}
    assert sync_res["skipped_regions"] == ["ap-south-1"]

    status = rep.get_status()
    assert status["regions_total"] == 3
    assert status["regions_healthy"] == 1
    assert status["regions_synced"] == 2


def test_geo_failover_routing() -> None:
    """Geo-router should use preferred healthy region and failover when unhealthy."""
    rep = MultiRegionReplicator()
    rep.publish_manifest("v2.1.0")
    rep.register_region("us-east-1", "https://us-east.example.com", priority=1)
    rep.register_region("eu-west-1", "https://eu-west.example.com", priority=2)
    rep.sync_all_regions()

    # Healthy preferred region -> direct hit
    d1 = rep.route_request(preferred_region="us-east-1")
    assert d1.target_region == "us-east-1"
    assert d1.failover_used is False
    assert d1.cache_hit is True

    # Preferred region becomes unhealthy -> failover to eu-west-1
    rep.record_region_health("us-east-1", latency_ms=1500.0, error_rate=0.5)
    d2 = rep.route_request(preferred_region="us-east-1")
    assert d2.target_region == "eu-west-1"
    assert d2.failover_used is True
    assert d2.reason == "failover_from_us-east-1"
    assert rep.get_status()["failover_events_total"] == 1


def test_replication_state_persistence(tmp_path: Path) -> None:
    """MultiRegionReplicator should save and load state snapshots cleanly."""
    rep_a = MultiRegionReplicator()
    rep_a.publish_manifest("v3.0.0", size_bytes=1024)
    rep_a.register_region("us-west-2", "https://us-west.example.com")
    rep_a.sync_region("us-west-2")

    snapshot = tmp_path / "replication_state.json"
    rep_a.save_state(snapshot)

    rep_b = MultiRegionReplicator()
    assert rep_b.load_state(snapshot) is True
    status = rep_b.get_status()
    assert status["active_manifest"]["model_version"] == "v3.0.0"
    assert status["regions_synced"] == 1
