"""Interactive explainability graph, counterfactual recourse, and user preferences."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ExplanationNode:
    """Node in a user-to-factor-to-artist recommendation explanation graph."""

    node_id: str
    kind: Literal["user", "artist", "genre", "mood", "factor"]
    label: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph node to a JSON-compatible dictionary."""
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "weight": round(self.weight, 4),
        }


@dataclass(frozen=True, slots=True)
class ExplanationEdge:
    """Directed weighted edge in a recommendation explanation graph."""

    source: str
    target: str
    relation: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph edge to a JSON-compatible dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 4),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualSuggestion:
    """Actionable counterfactual recourse to change an item's ranking."""

    action: Literal[
        "boost_genre",
        "boost_mood",
        "unmute_genre",
        "unblock_artist",
        "adjust_content_weight",
    ]
    target: str
    score_delta: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize counterfactual suggestion to a JSON-compatible dictionary."""
        return {
            "action": self.action,
            "target": self.target,
            "score_delta": round(self.score_delta, 4),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class UserPreferenceProfile:
    """User-controlled preference overrides for recommendation re-ranking."""

    user_id: str
    boosted_genres: dict[str, float] = field(default_factory=dict)
    muted_genres: set[str] = field(default_factory=set)
    boosted_moods: dict[str, float] = field(default_factory=dict)
    blocked_artists: set[str] = field(default_factory=set)
    content_weight_override: float | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if (
            self.content_weight_override is not None
            and not 0.0 <= self.content_weight_override <= 1.0
        ):
            raise ValueError("content_weight_override must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize preference profile to a JSON-compatible dictionary."""
        return {
            "user_id": self.user_id,
            "boosted_genres": {
                k: round(v, 4) for k, v in sorted(self.boosted_genres.items())
            },
            "muted_genres": sorted(self.muted_genres),
            "boosted_moods": {
                k: round(v, 4) for k, v in sorted(self.boosted_moods.items())
            },
            "blocked_artists": sorted(self.blocked_artists),
            "content_weight_override": self.content_weight_override,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UserPreferenceProfile:
        """Deserialize a UserPreferenceProfile from a dictionary."""
        return cls(
            user_id=str(payload["user_id"]),
            boosted_genres={
                str(k): float(v) for k, v in payload.get("boosted_genres", {}).items()
            },
            muted_genres={str(g) for g in payload.get("muted_genres", [])},
            boosted_moods={
                str(k): float(v) for k, v in payload.get("boosted_moods", {}).items()
            },
            blocked_artists={str(a) for a in payload.get("blocked_artists", [])},
            content_weight_override=(
                float(payload["content_weight_override"])
                if payload.get("content_weight_override") is not None
                else None
            ),
            updated_at=str(payload.get("updated_at", datetime.now(UTC).isoformat())),
        )
