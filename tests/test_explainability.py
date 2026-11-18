"""Tests for explainability graph, counterfactual recourse, and preferences."""

from __future__ import annotations

import pytest

from music_recommender.explainability import (
    CounterfactualSuggestion,
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
