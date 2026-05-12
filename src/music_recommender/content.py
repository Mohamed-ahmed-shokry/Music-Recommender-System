"""Content-based artist representation and recommendation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from music_recommender.preprocessing import Mappings
from music_recommender.ranking import validate_ranking_parameters

MetadataLookup = dict[str, dict[str, Any]]


@dataclass
class ContentArtifacts:
    """Content representation needed for metadata-aware recommendations."""

    metadata: pd.DataFrame
    vectorizer: TfidfVectorizer
    content_matrix: csr_matrix
    feature_names: list[str]
    artist_id_to_content_index: dict[str, int]
    content_index_to_artist_id: dict[int, str]
    metadata_lookup: MetadataLookup


def build_content_artifacts(
    metadata_df: pd.DataFrame,
    artist_ids: list[str],
) -> ContentArtifacts:
    """Vectorize artist metadata in the same order as model artist IDs."""
    metadata_by_artist = metadata_df.set_index("artist_id", drop=False)
    ordered_metadata = metadata_by_artist.loc[artist_ids].reset_index(drop=True)
    content_text = ordered_metadata.apply(_metadata_row_to_text, axis=1).tolist()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w_]+\b", lowercase=False)
    content_matrix = vectorizer.fit_transform(content_text).tocsr()

    artist_id_to_content_index = {
        artist_id: index for index, artist_id in enumerate(artist_ids)
    }
    content_index_to_artist_id = {
        index: artist_id for artist_id, index in artist_id_to_content_index.items()
    }

    return ContentArtifacts(
        metadata=ordered_metadata,
        vectorizer=vectorizer,
        content_matrix=content_matrix,
        feature_names=vectorizer.get_feature_names_out().tolist(),
        artist_id_to_content_index=artist_id_to_content_index,
        content_index_to_artist_id=content_index_to_artist_id,
        metadata_lookup=_build_metadata_lookup(ordered_metadata),
    )


def content_similarity_scores(
    artist_id: str,
    content_artifacts: ContentArtifacts,
) -> np.ndarray:
    """Return content similarity scores between one artist and all artists."""
    if artist_id not in content_artifacts.artist_id_to_content_index:
        raise ValueError(f"Unknown artist_id: {artist_id}")

    artist_index = content_artifacts.artist_id_to_content_index[artist_id]
    return cosine_similarity(
        content_artifacts.content_matrix[artist_index],
        content_artifacts.content_matrix,
    ).ravel()


def content_similar_artists(
    artist_id: str,
    content_artifacts: ContentArtifacts,
    artist_stats: dict[str, dict[str, Any]] | None,
    top_k: int,
    explain: bool = False,
) -> list[dict[str, Any]]:
    """Return artists with similar metadata."""
    validate_ranking_parameters(top_k)
    scores = content_similarity_scores(artist_id, content_artifacts)
    ranked_indices = np.argsort(scores)[::-1]

    recommendations: list[dict[str, Any]] = []
    for content_index in ranked_indices:
        candidate_artist_id = content_artifacts.content_index_to_artist_id[
            int(content_index)
        ]
        if candidate_artist_id == artist_id:
            continue
        recommendations.append(
            build_content_recommendation(
                artist_id=candidate_artist_id,
                score=float(scores[content_index]),
                content_artifacts=content_artifacts,
                artist_stats=artist_stats,
                reference_artist_ids=[artist_id],
                explain=explain,
            )
        )
        if len(recommendations) == top_k:
            break

    return recommendations


def user_content_scores(
    user_id: str,
    user_item_matrix: csr_matrix,
    mappings: Mappings,
    content_artifacts: ContentArtifacts,
) -> tuple[np.ndarray, list[str]]:
    """Build a content profile from a user's listened artists and score artists."""
    user_id_to_index = mappings["user_id_to_index"]
    index_to_artist_id = mappings["index_to_artist_id"]
    if user_id not in user_id_to_index:
        raise ValueError(f"Unknown user_id: {user_id}")

    user_index = user_id_to_index[user_id]
    listened_indices = user_item_matrix[user_index].indices
    listened_artist_ids = [index_to_artist_id[int(index)] for index in listened_indices]
    content_indices = [
        content_artifacts.artist_id_to_content_index[artist_id]
        for artist_id in listened_artist_ids
        if artist_id in content_artifacts.artist_id_to_content_index
    ]
    if not content_indices:
        return np.zeros(content_artifacts.content_matrix.shape[0]), listened_artist_ids

    weights = user_item_matrix[user_index].data.astype(float)
    if len(weights) != len(content_indices):
        weights = np.ones(len(content_indices))
    profile = _weighted_profile(
        content_artifacts.content_matrix[content_indices], weights
    )
    scores = cosine_similarity(profile, content_artifacts.content_matrix).ravel()
    return scores, listened_artist_ids


def profile_content_scores(
    content_artifacts: ContentArtifacts,
    artist_ids: list[str] | None = None,
    genres: list[str] | None = None,
    mood_tags: list[str] | None = None,
) -> tuple[np.ndarray, list[str], set[str]]:
    """Build a content profile from onboarding preferences and score artists."""
    artist_ids = artist_ids or []
    genres = genres or []
    mood_tags = mood_tags or []
    if not artist_ids and not genres and not mood_tags:
        raise ValueError("Provide at least one artist ID, genre, or mood tag.")

    unknown_artist_ids = [
        artist_id
        for artist_id in artist_ids
        if artist_id not in content_artifacts.artist_id_to_content_index
    ]
    if unknown_artist_ids:
        raise ValueError(f"Unknown artist IDs: {unknown_artist_ids}")

    content_rows = []
    if artist_ids:
        content_rows.append(
            content_artifacts.content_matrix[
                [
                    content_artifacts.artist_id_to_content_index[artist_id]
                    for artist_id in artist_ids
                ]
            ]
        )

    vector_tokens, preference_values = _preference_tokens(genres, mood_tags)
    if vector_tokens:
        content_rows.append(
            content_artifacts.vectorizer.transform([" ".join(sorted(vector_tokens))])
        )

    profile = _average_profile(content_rows)
    scores = cosine_similarity(profile, content_artifacts.content_matrix).ravel()
    return scores, artist_ids, preference_values


def recommend_from_scores(
    scores: np.ndarray,
    content_artifacts: ContentArtifacts,
    artist_stats: dict[str, dict[str, Any]] | None,
    top_k: int,
    exclude_artist_ids: set[str] | None = None,
    reference_artist_ids: list[str] | None = None,
    preference_tokens: set[str] | None = None,
    explain: bool = False,
) -> list[dict[str, Any]]:
    """Convert content scores into recommendation dictionaries."""
    validate_ranking_parameters(top_k)
    excluded = exclude_artist_ids or set()
    ranked_indices = np.argsort(scores)[::-1]

    recommendations: list[dict[str, Any]] = []
    for content_index in ranked_indices:
        artist_id = content_artifacts.content_index_to_artist_id[int(content_index)]
        if artist_id in excluded:
            continue
        recommendations.append(
            build_content_recommendation(
                artist_id=artist_id,
                score=float(scores[content_index]),
                content_artifacts=content_artifacts,
                artist_stats=artist_stats,
                reference_artist_ids=reference_artist_ids,
                preference_tokens=preference_tokens,
                explain=explain,
            )
        )
        if len(recommendations) == top_k:
            break

    return recommendations


def hybrid_scores(
    collaborative_scores: np.ndarray,
    content_scores: np.ndarray,
    content_weight: float,
) -> np.ndarray:
    """Blend collaborative and content scores after min-max normalization."""
    validate_content_weight(content_weight)
    collaborative = _min_max_normalize(collaborative_scores)
    content = _min_max_normalize(content_scores)
    return ((1 - content_weight) * collaborative) + (content_weight * content)


def validate_content_weight(content_weight: float) -> None:
    """Validate hybrid content weight."""
    if not 0 <= content_weight <= 1:
        raise ValueError("content_weight must be between 0 and 1.")


def build_content_recommendation(
    artist_id: str,
    score: float,
    content_artifacts: ContentArtifacts,
    artist_stats: dict[str, dict[str, Any]] | None,
    reference_artist_ids: list[str] | None = None,
    preference_tokens: set[str] | None = None,
    explain: bool = False,
) -> dict[str, Any]:
    """Build a recommendation payload with optional metadata explanations."""
    metadata = content_artifacts.metadata_lookup[artist_id]
    recommendation: dict[str, Any] = {
        "artist_id": artist_id,
        "artist_name": metadata["artist_name"],
        "score": score,
        "matched_metadata": {
            "genres": metadata["genres"],
            "mood_tags": metadata["mood_tags"],
        },
    }
    if artist_stats and artist_id in artist_stats:
        recommendation["popularity_rank"] = int(
            artist_stats[artist_id]["popularity_rank"]
        )
    if explain:
        recommendation["reasons"] = build_reasons(
            artist_id=artist_id,
            content_artifacts=content_artifacts,
            reference_artist_ids=reference_artist_ids,
            preference_tokens=preference_tokens,
        )
    return recommendation


def build_reasons(
    artist_id: str,
    content_artifacts: ContentArtifacts,
    reference_artist_ids: list[str] | None = None,
    preference_tokens: set[str] | None = None,
) -> list[str]:
    """Create simple human-readable explanation snippets."""
    metadata = content_artifacts.metadata_lookup[artist_id]
    artist_tokens = set(metadata["token_values"])
    reasons: list[str] = []

    if preference_tokens:
        matches = sorted(artist_tokens & preference_tokens)
        if matches:
            reasons.append(
                f"Matches your selected preferences: {', '.join(matches[:4])}"
            )

    if reference_artist_ids:
        for reference_artist_id in reference_artist_ids[:3]:
            if reference_artist_id not in content_artifacts.metadata_lookup:
                continue
            reference = content_artifacts.metadata_lookup[reference_artist_id]
            overlap = sorted(artist_tokens & set(reference["token_values"]))
            if overlap:
                reasons.append(
                    f"Shares {', '.join(overlap[:3])} with {reference['artist_name']}"
                )

    if not reasons:
        genres = ", ".join(metadata["genres"][:2])
        if genres:
            reasons.append(f"Recommended from content profile: {genres}")

    return reasons


def metadata_tokens_for_artist(
    artist_id: str,
    content_artifacts: ContentArtifacts,
) -> set[str]:
    """Return searchable metadata token values for one artist."""
    return set(content_artifacts.metadata_lookup[artist_id]["token_values"])


def _metadata_row_to_text(row: pd.Series) -> str:
    tokens = []
    tokens.extend(_prefixed_tokens("genre", row["genres"]))
    tokens.extend(_prefixed_tokens("mood", row["mood_tags"]))
    tokens.extend(_prefixed_tokens("country", row["country"]))
    tokens.extend(_prefixed_tokens("era", row["era"]))
    return " ".join(tokens)


def _build_metadata_lookup(metadata_df: pd.DataFrame) -> MetadataLookup:
    lookup: MetadataLookup = {}
    for row in metadata_df.itertuples(index=False):
        genres = _split_values(row.genres)
        mood_tags = _split_values(row.mood_tags)
        country = _split_values(row.country)
        era = _split_values(row.era)
        lookup[str(row.artist_id)] = {
            "artist_id": str(row.artist_id),
            "artist_name": str(row.artist_name),
            "genres": genres,
            "mood_tags": mood_tags,
            "country": country,
            "era": era,
            "token_values": set(genres + mood_tags + country + era),
        }
    return lookup


def _split_values(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(";") if item.strip()]


def _prefixed_tokens(prefix: str, value: str) -> list[str]:
    return [f"{prefix}_{_sanitize_token(item)}" for item in _split_values(value)]


def _sanitize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_").replace("&", "and")


def _preference_tokens(
    genres: list[str],
    mood_tags: list[str],
) -> tuple[set[str], set[str]]:
    vector_tokens = {
        f"genre_{_sanitize_token(value)}" for value in genres if value.strip()
    }
    vector_tokens.update(
        f"mood_{_sanitize_token(value)}" for value in mood_tags if value.strip()
    )
    explanation_values = {
        _sanitize_token(value).replace("_", " ")
        for value in genres + mood_tags
        if value.strip()
    }
    return vector_tokens, explanation_values


def _weighted_profile(matrix: csr_matrix, weights: np.ndarray) -> csr_matrix:
    weights = (
        weights / weights.sum()
        if weights.sum()
        else np.ones_like(weights) / len(weights)
    )
    return csr_matrix(weights @ matrix)


def _average_profile(rows: list[csr_matrix]) -> csr_matrix:
    if len(rows) == 1:
        matrix = rows[0]
    else:
        matrix = vstack(rows)
    return csr_matrix(matrix.mean(axis=0))


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max == score_min:
        return np.zeros_like(scores, dtype=float)
    return (scores - score_min) / (score_max - score_min)
