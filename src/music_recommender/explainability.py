"""Interactive explainability graph, counterfactual recourse, and user preferences."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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


def _split_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [t.strip().lower() for t in raw.split(";") if t.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(t).strip().lower() for t in raw if str(t).strip()]
    return []


class ExplainabilityEngine:
    """Thread-safe explanation graph, counterfactual recourse, and user preferences."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._profiles: dict[str, UserPreferenceProfile] = {}
        self._graphs_generated_total: int = 0
        self._counterfactuals_generated_total: int = 0

    def get_user_preferences(self, user_id: str) -> UserPreferenceProfile:
        """Return stored UserPreferenceProfile or a default empty profile."""
        clean_id = user_id.strip()
        if not clean_id:
            raise ValueError("user_id must be non-empty")
        with self._lock:
            existing = self._profiles.get(clean_id)
            if existing is not None:
                return existing
            return UserPreferenceProfile(user_id=clean_id)

    def set_user_preferences(
        self,
        user_id: str,
        *,
        boosted_genres: dict[str, float] | None = None,
        muted_genres: Sequence[str] | None = None,
        boosted_moods: dict[str, float] | None = None,
        blocked_artists: Sequence[str] | None = None,
        content_weight_override: float | None = None,
    ) -> UserPreferenceProfile:
        """Create or update user preference overrides."""
        clean_id = user_id.strip()
        if not clean_id:
            raise ValueError("user_id must be non-empty")
        with self._lock:
            current = self._profiles.get(
                clean_id, UserPreferenceProfile(user_id=clean_id)
            )
            new_profile = UserPreferenceProfile(
                user_id=clean_id,
                boosted_genres=(
                    {
                        k.strip().lower(): float(v)
                        for k, v in boosted_genres.items()
                        if k.strip()
                    }
                    if boosted_genres is not None
                    else dict(current.boosted_genres)
                ),
                muted_genres=(
                    {g.strip().lower() for g in muted_genres if g.strip()}
                    if muted_genres is not None
                    else set(current.muted_genres)
                ),
                boosted_moods=(
                    {
                        k.strip().lower(): float(v)
                        for k, v in boosted_moods.items()
                        if k.strip()
                    }
                    if boosted_moods is not None
                    else dict(current.boosted_moods)
                ),
                blocked_artists=(
                    {a.strip() for a in blocked_artists if a.strip()}
                    if blocked_artists is not None
                    else set(current.blocked_artists)
                ),
                content_weight_override=(
                    content_weight_override
                    if content_weight_override is not None
                    else current.content_weight_override
                ),
            )
            self._profiles[clean_id] = new_profile
            return new_profile

    def clear_user_preferences(self, user_id: str) -> bool:
        """Remove stored preference overrides for a user."""
        with self._lock:
            return self._profiles.pop(user_id.strip(), None) is not None

    def apply_preferences(
        self,
        user_id: str,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter blocked artists and re-rank candidates using user preferences."""
        with self._lock:
            profile = self.get_user_preferences(user_id)

        reranked: list[dict[str, Any]] = []
        for item in candidates:
            artist_id = str(item.get("artist_id") or "")
            if artist_id in profile.blocked_artists:
                continue

            base_score = float(item.get("score", 0.0))
            genres = _split_tags(item.get("genres"))
            moods = _split_tags(item.get("mood_tags"))

            multiplier = 1.0
            if any(g in profile.muted_genres for g in genres):
                multiplier *= 0.25

            genre_boost = sum(profile.boosted_genres.get(g, 0.0) for g in genres)
            mood_boost = sum(profile.boosted_moods.get(m, 0.0) for m in moods)
            adjusted = round((base_score * multiplier) + genre_boost + mood_boost, 6)

            enriched = dict(item)
            enriched["base_score"] = round(base_score, 6)
            enriched["score"] = adjusted
            enriched["preference_adjusted"] = (
                multiplier != 1.0 or genre_boost != 0.0 or mood_boost != 0.0
            )
            reranked.append(enriched)

        reranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return reranked

    def build_explanation_graph(
        self,
        user_id: str,
        recommendations: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Construct a 3-layer user -> factor/genre/mood -> artist graph."""
        clean_id = user_id.strip()
        if not clean_id:
            raise ValueError("user_id must be non-empty")

        with self._lock:
            profile = self.get_user_preferences(clean_id)
            self._graphs_generated_total += 1

        nodes: dict[str, ExplanationNode] = {}
        edges: list[ExplanationEdge] = []

        user_node_id = f"user:{clean_id}"
        nodes[user_node_id] = ExplanationNode(
            node_id=user_node_id,
            kind="user",
            label=clean_id,
            weight=1.0,
        )

        for rec in recommendations:
            artist_id = str(rec.get("artist_id") or "unknown")
            artist_name = str(rec.get("artist_name") or artist_id)
            score = float(rec.get("score", 0.0))
            artist_node_id = f"artist:{artist_id}"
            nodes[artist_node_id] = ExplanationNode(
                node_id=artist_node_id,
                kind="artist",
                label=artist_name,
                weight=score,
            )

            genres = _split_tags(rec.get("genres"))
            moods = _split_tags(rec.get("mood_tags"))

            for genre in genres[:3]:
                g_node_id = f"genre:{genre}"
                boost = profile.boosted_genres.get(genre, 0.0)
                if g_node_id not in nodes:
                    nodes[g_node_id] = ExplanationNode(
                        node_id=g_node_id,
                        kind="genre",
                        label=genre,
                        weight=1.0 + boost,
                    )
                    edges.append(
                        ExplanationEdge(
                            source=user_node_id,
                            target=g_node_id,
                            relation="boosts_genre" if boost > 0 else "matches_genre",
                            weight=max(0.1, 1.0 + boost),
                        )
                    )
                edges.append(
                    ExplanationEdge(
                        source=g_node_id,
                        target=artist_node_id,
                        relation="has_genre",
                        weight=max(0.1, score),
                    )
                )

            for mood in moods[:2]:
                m_node_id = f"mood:{mood}"
                m_boost = profile.boosted_moods.get(mood, 0.0)
                if m_node_id not in nodes:
                    nodes[m_node_id] = ExplanationNode(
                        node_id=m_node_id,
                        kind="mood",
                        label=mood,
                        weight=1.0 + m_boost,
                    )
                    edges.append(
                        ExplanationEdge(
                            source=user_node_id,
                            target=m_node_id,
                            relation="boosts_mood" if m_boost > 0 else "matches_mood",
                            weight=max(0.1, 1.0 + m_boost),
                        )
                    )
                edges.append(
                    ExplanationEdge(
                        source=m_node_id,
                        target=artist_node_id,
                        relation="has_mood",
                        weight=max(0.1, score),
                    )
                )

            if not genres and not moods:
                factor_id = "factor:collaborative_als"
                if factor_id not in nodes:
                    nodes[factor_id] = ExplanationNode(
                        node_id=factor_id,
                        kind="factor",
                        label="Collaborative ALS",
                        weight=1.0,
                    )
                    edges.append(
                        ExplanationEdge(
                            source=user_node_id,
                            target=factor_id,
                            relation="latent_affinity",
                            weight=1.0,
                        )
                    )
                edges.append(
                    ExplanationEdge(
                        source=factor_id,
                        target=artist_node_id,
                        relation="scores",
                        weight=max(0.1, score),
                    )
                )

        return {
            "user_id": clean_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": [n.to_dict() for n in nodes.values()],
            "edges": [e.to_dict() for e in edges],
        }

    def generate_counterfactuals(
        self,
        user_id: str,
        target_artist: dict[str, Any],
        *,
        top_cutoff_score: float = 1.0,
    ) -> list[CounterfactualSuggestion]:
        """Generate actionable counterfactual recourse to promote target_artist."""
        with self._lock:
            profile = self.get_user_preferences(user_id)
            self._counterfactuals_generated_total += 1

        artist_id = str(target_artist.get("artist_id") or "")
        current_score = float(target_artist.get("score", 0.0))
        gap = max(0.05, round(top_cutoff_score - current_score + 0.05, 4))
        suggestions: list[CounterfactualSuggestion] = []

        if artist_id in profile.blocked_artists:
            suggestions.append(
                CounterfactualSuggestion(
                    action="unblock_artist",
                    target=artist_id,
                    score_delta=gap,
                    rationale=f"Unblock artist '{artist_id}' so it is eligible.",
                )
            )

        genres = _split_tags(target_artist.get("genres"))
        muted_overlap = [g for g in genres if g in profile.muted_genres]
        for g in muted_overlap:
            suggestions.append(
                CounterfactualSuggestion(
                    action="unmute_genre",
                    target=g,
                    score_delta=gap,
                    rationale=f"Unmute genre '{g}' to remove 0.25x suppression.",
                )
            )

        if genres:
            suggestions.append(
                CounterfactualSuggestion(
                    action="boost_genre",
                    target=genres[0],
                    score_delta=gap,
                    rationale=(
                        f"Boost genre '{genres[0]}' by +{gap:.2f} to surpass "
                        f"cutoff {top_cutoff_score:.2f}."
                    ),
                )
            )

        moods = _split_tags(target_artist.get("mood_tags"))
        if moods:
            suggestions.append(
                CounterfactualSuggestion(
                    action="boost_mood",
                    target=moods[0],
                    score_delta=gap,
                    rationale=f"Boost mood '{moods[0]}' by +{gap:.2f}.",
                )
            )

        if not suggestions:
            suggestions.append(
                CounterfactualSuggestion(
                    action="adjust_content_weight",
                    target="content_weight",
                    score_delta=gap,
                    rationale="Increase content_weight to surface metadata matches.",
                )
            )

        return suggestions

    def get_status(self) -> dict[str, Any]:
        """Return explainability engine telemetry and active profile summary."""
        with self._lock:
            return {
                "preference_profiles_count": len(self._profiles),
                "graphs_generated_total": self._graphs_generated_total,
                "counterfactuals_generated_total": (
                    self._counterfactuals_generated_total
                ),
                "users": sorted(self._profiles.keys()),
            }

    def export_state(self) -> dict[str, Any]:
        """Export preference profiles and counters to a JSON-serializable dict."""
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "graphs_generated_total": self._graphs_generated_total,
                "counterfactuals_generated_total": (
                    self._counterfactuals_generated_total
                ),
                "profiles": {k: v.to_dict() for k, v in sorted(self._profiles.items())},
            }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore preference profiles and counters from an exported dict."""
        with self._lock:
            self._graphs_generated_total = int(state.get("graphs_generated_total", 0))
            self._counterfactuals_generated_total = int(
                state.get("counterfactuals_generated_total", 0)
            )
            self._profiles = {
                str(k): UserPreferenceProfile.from_dict(v)
                for k, v in state.get("profiles", {}).items()
            }

    def save_state(self, path: str | Path) -> Path:
        """Persist explainability state snapshot to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> bool:
        """Load explainability state snapshot from a JSON file if it exists."""
        target = Path(path)
        if not target.exists():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.import_state(payload)
        return True


_global_explainability_engine = ExplainabilityEngine()


def get_global_explainability_engine() -> ExplainabilityEngine:
    """Return the process-wide ExplainabilityEngine singleton."""
    return _global_explainability_engine


def set_global_explainability_engine(engine: ExplainabilityEngine) -> None:
    """Replace the process-wide ExplainabilityEngine singleton."""
    global _global_explainability_engine
    _global_explainability_engine = engine
