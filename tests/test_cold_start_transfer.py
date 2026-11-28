"""Tests for cross-platform cold-start metadata transfer and artist onboarding."""

from __future__ import annotations

import pytest

from music_recommender.cold_start_transfer import (
    AnchorArtist,
    ExternalArtistProfile,
    OnboardedArtistRecord,
    OnboardingStatus,
    TransferNeighbor,
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
