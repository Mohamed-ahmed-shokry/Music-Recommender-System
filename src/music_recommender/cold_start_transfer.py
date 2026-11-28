"""Cross-platform cold-start metadata transfer learning & semantic artist onboarding."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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


def _normalize_vector(values: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple(v / norm for v in values)


def _hashed_fallback_embedding(artist_id: str, dim: int) -> tuple[float, ...]:
    raw: list[float] = []
    for idx in range(dim):
        digest = hashlib.sha256(f"{artist_id}:{idx}".encode()).digest()
        val = (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
        raw.append(val)
    return _normalize_vector(raw)


def _metadata_weighted_similarity(
    profile: ExternalArtistProfile,
    anchor: AnchorArtist,
) -> tuple[float, tuple[str, ...]]:
    """Compute weighted semantic overlap between an external profile and an anchor."""
    p_genres = {g.strip().lower() for g in profile.genres if g.strip()}
    a_genres = {g.strip().lower() for g in anchor.genres if g.strip()}
    p_moods = {m.strip().lower() for m in profile.mood_tags if m.strip()}
    a_moods = {m.strip().lower() for m in anchor.mood_tags if m.strip()}

    shared: list[str] = []
    weighted_num = 0.0
    weighted_den = 0.0

    if p_genres or a_genres:
        g_inter = sorted(p_genres & a_genres)
        g_union = p_genres | a_genres
        weighted_num += 2.0 * (len(g_inter) / len(g_union))
        weighted_den += 2.0
        shared.extend(f"genre:{g}" for g in g_inter)

    if p_moods or a_moods:
        m_inter = sorted(p_moods & a_moods)
        m_union = p_moods | a_moods
        weighted_num += 1.5 * (len(m_inter) / len(m_union))
        weighted_den += 1.5
        shared.extend(f"mood:{m}" for m in m_inter)

    p_country = profile.country.strip().lower()
    a_country = anchor.country.strip().lower()
    if p_country and a_country:
        weighted_den += 0.5
        if p_country == a_country:
            weighted_num += 0.5
            shared.append(f"country:{p_country}")

    p_era = profile.era.strip().lower()
    a_era = anchor.era.strip().lower()
    if p_era and a_era:
        weighted_den += 0.5
        if p_era == a_era:
            weighted_num += 0.5
            shared.append(f"era:{p_era}")

    if weighted_den <= 0.0:
        return 0.0, ()
    return weighted_num / weighted_den, tuple(shared)


def _split_semicolon_tags(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple, set)):
        return tuple(str(v).strip() for v in raw if str(v).strip())
    return tuple(part.strip() for part in str(raw).split(";") if part.strip())


class ColdStartTransferEngine:
    """Thread-safe metadata transfer learning & semantic artist onboarding engine."""

    def __init__(self, default_embedding_dim: int = 8) -> None:
        if default_embedding_dim <= 0:
            raise ValueError("default_embedding_dim must be > 0")
        self._lock = threading.RLock()
        self._default_dim = int(default_embedding_dim)
        self._anchors: dict[str, AnchorArtist] = {}
        self._onboarded: dict[str, OnboardedArtistRecord] = {}
        self._onboarding_total: int = 0

    @property
    def anchor_count(self) -> int:
        with self._lock:
            return len(self._anchors)

    def _embedding_dim(self) -> int:
        for anchor in self._anchors.values():
            return len(anchor.embedding)
        return self._default_dim

    def register_anchor(self, anchor: AnchorArtist) -> AnchorArtist:
        with self._lock:
            self._anchors[anchor.artist_id] = anchor
            return anchor

    def list_anchors(self) -> list[AnchorArtist]:
        with self._lock:
            return list(self._anchors.values())

    def sync_from_artifact(self, artifact: Any) -> int:
        """Populate catalog anchors from a RecommenderArtifact's item factors."""
        model = getattr(artifact, "model", None)
        item_factors = getattr(model, "item_factors", None)
        mappings = getattr(artifact, "mappings", {}) or {}
        id_to_idx = mappings.get("artist_id_to_index", {}) or {}
        id_to_name = mappings.get("artist_id_to_name", {}) or {}
        content = getattr(artifact, "content_artifacts", None)
        metadata_df = getattr(content, "metadata", None)

        meta_lookup: dict[str, dict[str, Any]] = {}
        if metadata_df is not None and hasattr(metadata_df, "to_dict"):
            for row in metadata_df.to_dict("records"):
                aid = str(row.get("artist_id", ""))
                if aid:
                    meta_lookup[aid] = row

        synced = 0
        with self._lock:
            for artist_id, idx in id_to_idx.items():
                aid = str(artist_id)
                if item_factors is not None and 0 <= int(idx) < len(item_factors):
                    vec = _normalize_vector([float(x) for x in item_factors[int(idx)]])
                else:
                    vec = _hashed_fallback_embedding(aid, self._default_dim)
                meta = meta_lookup.get(aid, {})
                anchor = AnchorArtist(
                    artist_id=aid,
                    artist_name=str(id_to_name.get(aid, meta.get("artist_name", aid))),
                    embedding=vec,
                    genres=_split_semicolon_tags(meta.get("genres")),
                    mood_tags=_split_semicolon_tags(meta.get("mood_tags")),
                    country=str(meta.get("country", "") or ""),
                    era=str(meta.get("era", "") or ""),
                )
                self._anchors[aid] = anchor
                synced += 1
        return synced

    def project_embedding(
        self,
        profile: ExternalArtistProfile,
        *,
        top_k_anchors: int = 5,
    ) -> tuple[tuple[float, ...], float, tuple[TransferNeighbor, ...]]:
        """Project external metadata into the collaborative embedding space."""
        with self._lock:
            dim = self._embedding_dim()
            scored: list[tuple[float, str, AnchorArtist, tuple[str, ...]]] = []
            for anchor in self._anchors.values():
                sim, shared = _metadata_weighted_similarity(profile, anchor)
                if sim > 0.0:
                    scored.append((sim, anchor.artist_id, anchor, shared))

            scored.sort(key=lambda item: (-item[0], item[1]))
            top = scored[: max(1, top_k_anchors)]

            if not top:
                fallback = _hashed_fallback_embedding(profile.artist_id, dim)
                return fallback, 0.1, ()

            accum = [0.0] * dim
            weight_sum = 0.0
            neighbors: list[TransferNeighbor] = []
            for sim, _aid, anchor, shared in top:
                weight_sum += sim
                for i in range(min(dim, len(anchor.embedding))):
                    accum[i] += sim * anchor.embedding[i]
                neighbors.append(
                    TransferNeighbor(
                        anchor_artist_id=anchor.artist_id,
                        anchor_artist_name=anchor.artist_name,
                        similarity=sim,
                        shared_tags=shared,
                    )
                )

            projected = _normalize_vector([v / max(weight_sum, 1e-12) for v in accum])
            mean_sim = weight_sum / len(top)
            # Boost confidence slightly when external popularity signal is strong
            confidence = min(1.0, mean_sim * 0.9 + profile.external_popularity * 0.1)
            return projected, confidence, tuple(neighbors)

    def onboard_artist(
        self,
        profile: ExternalArtistProfile,
        *,
        top_k_anchors: int = 5,
        min_confidence: float = 0.25,
    ) -> OnboardedArtistRecord:
        """Project and register a zero-interaction external artist."""
        projected, confidence, neighbors = self.project_embedding(
            profile, top_k_anchors=top_k_anchors
        )
        status = (
            OnboardingStatus.ACTIVE
            if confidence >= min_confidence
            else OnboardingStatus.PENDING
        )
        record = OnboardedArtistRecord(
            profile=profile,
            status=status,
            projected_embedding=projected,
            confidence_score=confidence,
            nearest_anchors=neighbors,
        )
        with self._lock:
            self._onboarded[profile.artist_id] = record
            self._onboarding_total += 1
        return record

    def get_onboarded(self, artist_id: str) -> OnboardedArtistRecord | None:
        with self._lock:
            return self._onboarded.get(artist_id)

    def list_onboarded(self) -> list[OnboardedArtistRecord]:
        with self._lock:
            return list(self._onboarded.values())

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._onboarded.values())
            active_count = sum(
                1 for r in records if r.status == OnboardingStatus.ACTIVE
            )
            mean_conf = (
                sum(r.confidence_score for r in records) / len(records)
                if records
                else 0.0
            )
            return {
                "anchors_registered": len(self._anchors),
                "embedding_dim": self._embedding_dim(),
                "onboarded_count": len(records),
                "active_onboarded_count": active_count,
                "onboarding_total": self._onboarding_total,
                "mean_confidence": round(mean_conf, 6),
                "recent_artist_ids": [r.profile.artist_id for r in records[-10:]],
            }

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "default_dim": self._default_dim,
                "anchors": [a.to_dict() for a in self._anchors.values()],
                "onboarded": [r.to_dict() for r in self._onboarded.values()],
                "onboarding_total": self._onboarding_total,
            }

    def import_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._default_dim = int(state.get("default_dim", self._default_dim))
            if "anchors" in state:
                self._anchors = {
                    a.artist_id: a
                    for a in (AnchorArtist.from_dict(item) for item in state["anchors"])
                }
            if "onboarded" in state:
                self._onboarded = {
                    r.profile.artist_id: r
                    for r in (
                        OnboardedArtistRecord.from_dict(item)
                        for item in state["onboarded"]
                    )
                }
            self._onboarding_total = int(
                state.get("onboarding_total", len(self._onboarded))
            )

    def save_state(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> None:
        source = Path(path)
        self.import_state(json.loads(source.read_text(encoding="utf-8")))


_GLOBAL_TRANSFER_ENGINE: ColdStartTransferEngine | None = None
_GLOBAL_LOCK = threading.Lock()


def get_global_transfer_engine() -> ColdStartTransferEngine:
    global _GLOBAL_TRANSFER_ENGINE
    with _GLOBAL_LOCK:
        if _GLOBAL_TRANSFER_ENGINE is None:
            _GLOBAL_TRANSFER_ENGINE = ColdStartTransferEngine()
        return _GLOBAL_TRANSFER_ENGINE


def set_global_transfer_engine(engine: ColdStartTransferEngine | None) -> None:
    global _GLOBAL_TRANSFER_ENGINE
    with _GLOBAL_LOCK:
        _GLOBAL_TRANSFER_ENGINE = engine
