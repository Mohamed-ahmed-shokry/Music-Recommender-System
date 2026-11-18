"""Tests for explainability graph, counterfactual recourse, and preferences."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_recommender.explainability import (
    CounterfactualSuggestion,
    ExplainabilityEngine,
    ExplanationEdge,
    ExplanationNode,
    UserPreferenceProfile,
)


def test_explainability_models_and_preference_roundtrip() -> None:
    """Graph primitives, counterfactuals, and UserPreferenceProfile should serialize."""
    node = ExplanationNode(node_id="genre:pop", kind="genre", label="pop", weight=0.8)
    edge = ExplanationEdge(
        source="user:u1", target="genre:pop", relation="prefers", weight=0.8
    )
    cf = CounterfactualSuggestion(
        action="boost_genre",
        target="jazz",
        score_delta=0.35,
        rationale="Boost 'jazz' to surpass cutoff.",
    )
    assert node.to_dict()["kind"] == "genre"
    assert edge.to_dict()["relation"] == "prefers"
    assert cf.to_dict()["action"] == "boost_genre"

    profile = UserPreferenceProfile(
        user_id="u1",
        boosted_genres={"pop": 0.4},
        muted_genres={"metal"},
        boosted_moods={"bright": 0.25},
        blocked_artists={"a_99"},
        content_weight_override=0.6,
    )
    restored = UserPreferenceProfile.from_dict(profile.to_dict())
    assert restored.user_id == "u1"
    assert restored.boosted_genres == {"pop": 0.4}
    assert restored.muted_genres == {"metal"}
    assert restored.blocked_artists == {"a_99"}
    assert restored.content_weight_override == 0.6

    with pytest.raises(ValueError, match="user_id"):
        UserPreferenceProfile(user_id="  ")
    with pytest.raises(ValueError, match="content_weight_override"):
        UserPreferenceProfile(user_id="u1", content_weight_override=1.5)


def test_preference_filtering_and_reranking() -> None:
    """apply_preferences should block artists, suppress muted genres, and boost."""
    engine = ExplainabilityEngine()
    engine.set_user_preferences(
        "u1",
        boosted_genres={"jazz": 0.6},
        muted_genres=["rock"],
        blocked_artists=["a_blocked"],
    )

    candidates = [
        {
            "artist_id": "a_blocked",
            "artist_name": "Blocked",
            "score": 1.0,
            "genres": "pop",
        },
        {"artist_id": "a_rock", "artist_name": "Rock", "score": 0.9, "genres": "rock"},
        {"artist_id": "a_jazz", "artist_name": "Jazz", "score": 0.5, "genres": "jazz"},
    ]
    reranked = engine.apply_preferences("u1", candidates)
    ids = [r["artist_id"] for r in reranked]
    assert ids == ["a_jazz", "a_rock"]
    assert reranked[0]["score"] == pytest.approx(1.1)
    assert reranked[1]["score"] == pytest.approx(0.225)


def test_build_explanation_graph_and_counterfactuals() -> None:
    """Engine should build 3-layer explanation graphs and counterfactual recourse."""
    engine = ExplainabilityEngine()
    engine.set_user_preferences(
        "u2",
        boosted_genres={"indie": 0.3},
        muted_genres=["electronic"],
        blocked_artists=["a_hidden"],
    )

    recs = [
        {
            "artist_id": "a1",
            "artist_name": "Indie Band",
            "score": 0.95,
            "genres": "indie;pop",
            "mood_tags": "bright",
        },
        {
            "artist_id": "a2",
            "artist_name": "ALS Only",
            "score": 0.7,
        },
    ]
    graph = engine.build_explanation_graph("u2", recs)
    assert graph["user_id"] == "u2"
    assert graph["node_count"] >= 5
    assert graph["edge_count"] >= 4
    assert any(n["kind"] == "genre" and n["label"] == "indie" for n in graph["nodes"])

    cfs = engine.generate_counterfactuals(
        "u2",
        {
            "artist_id": "a_hidden",
            "score": 0.4,
            "genres": "electronic",
            "mood_tags": "dark",
        },
        top_cutoff_score=1.0,
    )
    actions = {c.action for c in cfs}
    assert "unblock_artist" in actions
    assert "unmute_genre" in actions
    assert "boost_genre" in actions
    assert "boost_mood" in actions


def test_explainability_state_persistence(tmp_path: Path) -> None:
    """ExplainabilityEngine should export, save, load, and restore state."""
    engine_a = ExplainabilityEngine()
    engine_a.set_user_preferences("u3", boosted_genres={"pop": 0.5})
    engine_a.build_explanation_graph("u3", [{"artist_id": "a1", "score": 0.8}])

    snapshot = tmp_path / "explainability_state.json"
    engine_a.save_state(snapshot)

    engine_b = ExplainabilityEngine()
    assert engine_b.load_state(snapshot) is True
    assert engine_b.get_user_preferences("u3").boosted_genres == {"pop": 0.5}
    assert engine_b.get_status()["graphs_generated_total"] == 1
