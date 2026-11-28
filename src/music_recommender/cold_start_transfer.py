"""Cross-platform cold-start metadata transfer learning & semantic artist onboarding."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class OnboardingStatus(StrEnum):
    """Lifecycle status of an onboarded cold-start artist."""

    PENDING = "pending"
    EMBEDDED = "embedded"
    ACTIVE = "active"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExternalArtistProfile:
    """Incoming cross-platform artist metadata profile for cold-start onboarding."""

    artist_id: str
    artist_name: str
    source_platform: str = "spotify"
    genres: tuple[str, ...] = ()
    mood_tags: tuple[str, ...] = ()
    country: str = ""
    era: str = ""
    external_popularity: float = 0.0

    def __post_init__(self) -> None:
        if not self.artist_id.strip():
            raise ValueError("artist_id must not be empty")
        if not self.artist_name.strip():
            raise ValueError("artist_name must not be empty")
        if not self.source_platform.strip():
            raise ValueError("source_platform must not be empty")
        if not (0.0 <= self.external_popularity <= 1.0):
            raise ValueError("external_popularity must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artist_id": self.artist_id,
            "artist_name": self.artist_name,
            "source_platform": self.source_platform,
            "genres": list(self.genres),
            "mood_tags": list(self.mood_tags),
            "country": self.country,
            "era": self.era,
            "external_popularity": round(self.external_popularity, 4),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExternalArtistProfile:
        return cls(
            artist_id=str(payload["artist_id"]),
            artist_name=str(payload["artist_name"]),
            source_platform=str(payload.get("source_platform", "spotify")),
            genres=tuple(str(g) for g in payload.get("genres", ())),
            mood_tags=tuple(str(m) for m in payload.get("mood_tags", ())),
            country=str(payload.get("country", "")),
            era=str(payload.get("era", "")),
            external_popularity=float(payload.get("external_popularity", 0.0)),
        )


@dataclass(frozen=True)
class AnchorArtist:
    """Catalog anchor artist with collaborative embedding and metadata tags."""

    artist_id: str
    artist_name: str
    embedding: tuple[float, ...]
    genres: tuple[str, ...] = ()
    mood_tags: tuple[str, ...] = ()
    country: str = ""
    era: str = ""

    def __post_init__(self) -> None:
        if not self.artist_id.strip():
            raise ValueError("artist_id must not be empty")
        if not self.embedding:
            raise ValueError("embedding must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artist_id": self.artist_id,
            "artist_name": self.artist_name,
            "embedding": list(self.embedding),
            "genres": list(self.genres),
            "mood_tags": list(self.mood_tags),
            "country": self.country,
            "era": self.era,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnchorArtist:
        return cls(
            artist_id=str(payload["artist_id"]),
            artist_name=str(payload["artist_name"]),
            embedding=tuple(float(v) for v in payload["embedding"]),
            genres=tuple(str(g) for g in payload.get("genres", ())),
            mood_tags=tuple(str(m) for m in payload.get("mood_tags", ())),
            country=str(payload.get("country", "")),
            era=str(payload.get("era", "")),
        )


@dataclass(frozen=True)
class TransferNeighbor:
    """Matched catalog anchor artist contributing to embedding transfer."""

    anchor_artist_id: str
    anchor_artist_name: str
    similarity: float
    shared_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_artist_id": self.anchor_artist_id,
            "anchor_artist_name": self.anchor_artist_name,
            "similarity": round(self.similarity, 6),
            "shared_tags": list(self.shared_tags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransferNeighbor:
        return cls(
            anchor_artist_id=str(payload["anchor_artist_id"]),
            anchor_artist_name=str(payload["anchor_artist_name"]),
            similarity=float(payload["similarity"]),
            shared_tags=tuple(str(t) for t in payload.get("shared_tags", ())),
        )


@dataclass(frozen=True)
class OnboardedArtistRecord:
    """Result of projecting and onboarding a zero-interaction artist."""

    profile: ExternalArtistProfile
    status: OnboardingStatus
    projected_embedding: tuple[float, ...]
    confidence_score: float
    nearest_anchors: tuple[TransferNeighbor, ...] = ()
    onboarded_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "status": self.status.value,
            "embedding_dim": len(self.projected_embedding),
            "projected_embedding": [round(v, 6) for v in self.projected_embedding],
            "confidence_score": round(self.confidence_score, 6),
            "nearest_anchors": [n.to_dict() for n in self.nearest_anchors],
            "onboarded_at": self.onboarded_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnboardedArtistRecord:
        return cls(
            profile=ExternalArtistProfile.from_dict(payload["profile"]),
            status=OnboardingStatus(str(payload["status"])),
            projected_embedding=tuple(
                float(v) for v in payload.get("projected_embedding", ())
            ),
            confidence_score=float(payload["confidence_score"]),
            nearest_anchors=tuple(
                TransferNeighbor.from_dict(n)
                for n in payload.get("nearest_anchors", ())
            ),
            onboarded_at=str(payload.get("onboarded_at", _utc_now_iso())),
        )
