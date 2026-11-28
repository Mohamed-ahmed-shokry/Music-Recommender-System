"""Tests for cross-platform cold-start metadata transfer and artist onboarding."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from music_recommender.cold_start_transfer import (
    AnchorArtist,
    ColdStartTransferEngine,
    ExternalArtistProfile,
    OnboardedArtistRecord,
    OnboardingStatus,
    TransferNeighbor,
    get_global_transfer_engine,
    set_global_transfer_engine,
)


def test_external_artist_profile_validation_and_roundtrip() -> None:
    profile = ExternalArtistProfile(
        artist_id="ext_001",
        artist_name="Nova Synth",
        source_platform="spotify",
        genres=("synthwave", "electronic"),
        mood_tags=("energetic", "nostalgic"),
        country="SE",
        era="2020s",
        external_popularity=0.72,
    )
    restored = ExternalArtistProfile.from_dict(profile.to_dict())
    assert restored == profile

    with pytest.raises(ValueError, match="artist_id"):
        ExternalArtistProfile(artist_id=" ", artist_name="X")
    with pytest.raises(ValueError, match="artist_name"):
        ExternalArtistProfile(artist_id="ext_x", artist_name=" ")
    with pytest.raises(ValueError, match="source_platform"):
        ExternalArtistProfile(artist_id="ext_x", artist_name="X", source_platform=" ")
    with pytest.raises(ValueError, match="external_popularity"):
        ExternalArtistProfile(
            artist_id="ext_x", artist_name="X", external_popularity=1.5
        )


def test_anchor_and_onboarded_record_roundtrip() -> None:
    anchor = AnchorArtist(
        artist_id="artist_1",
        artist_name="Anchor A",
        embedding=(0.6, 0.8),
        genres=("electronic",),
        mood_tags=("energetic",),
        country="SE",
        era="2020s",
    )
    assert AnchorArtist.from_dict(anchor.to_dict()) == anchor

    with pytest.raises(ValueError, match="embedding"):
        AnchorArtist(artist_id="a", artist_name="A", embedding=())

    neighbor = TransferNeighbor(
        anchor_artist_id="artist_1",
        anchor_artist_name="Anchor A",
        similarity=0.85,
        shared_tags=("genre:electronic", "mood:energetic"),
    )
    record = OnboardedArtistRecord(
        profile=ExternalArtistProfile(
            artist_id="ext_001",
            artist_name="Nova Synth",
            genres=("electronic",),
        ),
        status=OnboardingStatus.ACTIVE,
        projected_embedding=(0.6, 0.8),
        confidence_score=0.85,
        nearest_anchors=(neighbor,),
    )
    restored = OnboardedArtistRecord.from_dict(record.to_dict())
    assert restored.status == OnboardingStatus.ACTIVE
    assert restored.nearest_anchors[0].anchor_artist_id == "artist_1"
    assert restored.to_dict()["embedding_dim"] == 2


def test_transfer_engine_projection_and_onboarding() -> None:
    engine = ColdStartTransferEngine(default_embedding_dim=4)
    engine.register_anchor(
        AnchorArtist(
            artist_id="a_pop",
            artist_name="Pop Star",
            embedding=(1.0, 0.0, 0.0, 0.0),
            genres=("pop", "dance"),
            mood_tags=("bright", "energetic"),
            country="US",
            era="2020s",
        )
    )
    engine.register_anchor(
        AnchorArtist(
            artist_id="a_rock",
            artist_name="Rock Band",
            embedding=(0.0, 1.0, 0.0, 0.0),
            genres=("rock", "indie"),
            mood_tags=("moody",),
            country="UK",
            era="2010s",
        )
    )

    profile = ExternalArtistProfile(
        artist_id="ext_pop_1",
        artist_name="New Pop Duo",
        source_platform="spotify",
        genres=("pop", "dance"),
        mood_tags=("bright",),
        country="US",
        era="2020s",
        external_popularity=0.8,
    )
    record = engine.onboard_artist(profile)
    assert record.status == OnboardingStatus.ACTIVE
    assert record.nearest_anchors[0].anchor_artist_id == "a_pop"
    assert record.confidence_score > 0.5
    norm = math.sqrt(sum(v * v for v in record.projected_embedding))
    assert norm == pytest.approx(1.0, rel=1e-4)
    assert engine.get_onboarded("ext_pop_1") == record

    # Novel unmatched genre falls back to deterministic hash vector & PENDING
    novel = ExternalArtistProfile(
        artist_id="ext_novel",
        artist_name="Microtonal Choir",
        genres=("microtonal",),
    )
    novel_rec = engine.onboard_artist(novel, min_confidence=0.25)
    assert novel_rec.status == OnboardingStatus.PENDING
    assert novel_rec.nearest_anchors == ()
    assert len(novel_rec.projected_embedding) == 4


def test_transfer_engine_artifact_sync_and_persistence(tmp_path: Path) -> None:
    artifact = SimpleNamespace(
        model=SimpleNamespace(item_factors=[[0.8, 0.6], [0.0, 1.0]]),
        mappings={
            "artist_id_to_index": {"art_1": 0, "art_2": 1},
            "artist_id_to_name": {"art_1": "Synth Alpha", "art_2": "Jazz Beta"},
        },
        content_artifacts=SimpleNamespace(
            metadata=pd.DataFrame(
                {
                    "artist_id": ["art_1", "art_2"],
                    "artist_name": ["Synth Alpha", "Jazz Beta"],
                    "genres": ["synthwave;electronic", "jazz;ambient"],
                    "mood_tags": ["energetic", "calm"],
                    "country": ["SE", "JP"],
                    "era": ["2020s", "2010s"],
                }
            )
        ),
    )

    engine = ColdStartTransferEngine()
    synced = engine.sync_from_artifact(artifact)
    assert synced == 2
    assert engine.anchor_count == 2

    rec = engine.onboard_artist(
        ExternalArtistProfile(
            artist_id="ext_jazz",
            artist_name="Tokyo Quartet",
            source_platform="bandcamp",
            genres=("jazz",),
            mood_tags=("calm",),
            country="JP",
        )
    )
    assert rec.nearest_anchors[0].anchor_artist_id == "art_2"

    state_file = tmp_path / "transfer_state.json"
    engine.save_state(state_file)

    restored = ColdStartTransferEngine()
    restored.load_state(state_file)
    assert restored.anchor_count == 2
    assert restored.get_onboarded("ext_jazz") is not None
    status = restored.get_status()
    assert status["onboarded_count"] == 1
    assert status["active_onboarded_count"] == 1


def test_global_transfer_engine_accessor() -> None:
    original = get_global_transfer_engine()
    custom = ColdStartTransferEngine()
    try:
        set_global_transfer_engine(custom)
        assert get_global_transfer_engine() is custom
    finally:
        set_global_transfer_engine(original)
