"""Interactive Streamlit dashboard for exploring recommendations."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from music_recommender.config import ARTIFACT_BUNDLE_PATH
from music_recommender.service import RecommenderService

DASHBOARD_ARTIFACT_ENV_VAR = "MUSIC_RECOMMENDER_ARTIFACT_PATH"


def resolve_dashboard_artifact_path() -> Path:
    """Return the artifact path configured for the dashboard."""
    configured_path = os.getenv(DASHBOARD_ARTIFACT_ENV_VAR)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return ARTIFACT_BUNDLE_PATH


@st.cache_resource(show_spinner=False)
def load_dashboard_service(artifact_path: str) -> RecommenderService:
    """Load and cache the serving artifact across dashboard reruns."""
    return RecommenderService.from_artifacts(artifact_path)


def split_metadata_terms(values: Iterable[Any]) -> list[str]:
    """Normalize semicolon-delimited metadata values into sorted choices."""
    terms: set[str] = set()
    for value in values:
        if isinstance(value, str):
            terms.update(term.strip() for term in value.split(";") if term.strip())
        elif isinstance(value, (list, tuple, set)):
            terms.update(str(term).strip() for term in value if str(term).strip())
    return sorted(terms, key=str.casefold)


def recommendation_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a service response into a dashboard-friendly table."""
    recommendations = payload.get("recommendations")
    if recommendations is None:
        recommendations = payload.get("similar_artists", [])

    rows = []
    for rank, recommendation in enumerate(recommendations, start=1):
        reasons = recommendation.get("reasons") or []
        rows.append(
            {
                "Rank": rank,
                "Artist": recommendation.get("artist_name", "Unknown artist"),
                "Artist ID": recommendation.get("artist_id", ""),
                "Score": round(float(recommendation.get("score", 0.0)), 4),
                "Popularity rank": recommendation.get("popularity_rank"),
                "Why": " · ".join(str(reason) for reason in reasons),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Rank",
            "Artist",
            "Artist ID",
            "Score",
            "Popularity rank",
            "Why",
        ],
    )


def catalog_frame(
    service: RecommenderService,
    query: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Build a dashboard table from the shared catalog service."""
    payload = service.browse_artists(query=query, limit=limit)
    catalog = pd.DataFrame(payload["artists"])
    for column in ("genres", "mood_tags"):
        if column in catalog:
            catalog[column] = catalog[column].apply(
                lambda values: "; ".join(str(value) for value in values)
            )
    catalog.attrs.update(
        total=payload["total"],
        has_more=payload["has_more"],
    )
    return catalog


def _artist_choices(service: RecommenderService) -> dict[str, str]:
    names = service.artifact.mappings["artist_id_to_name"]
    return {
        f"{artist_name} · {artist_id}": str(artist_id)
        for artist_id, artist_name in sorted(
            names.items(),
            key=lambda item: (str(item[1]).casefold(), str(item[0])),
        )
    }


def _render_results(payload: dict[str, Any]) -> None:
    frame = recommendation_frame(payload)
    strategy = str(payload.get("strategy", "recommendation")).replace("_", " ").title()

    st.subheader("Your results")
    st.caption(f"Strategy: {strategy}")
    if message := payload.get("message"):
        st.info(str(message))

    if frame.empty:
        st.warning("No recommendations matched the selected controls.")
        return

    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Score": st.column_config.NumberColumn(format="%.4f"),
            "Why": st.column_config.TextColumn(width="large"),
        },
    )


def _run_recommendation(action: Callable[[], dict[str, Any]]) -> None:
    try:
        with st.spinner("Building your recommendations..."):
            payload = action()
    except ValueError as error:
        st.error(str(error))
        return
    _render_results(payload)


def _render_personalized_tab(
    service: RecommenderService,
    user_ids: list[str],
    max_top_k: int,
) -> None:
    st.write("Blend collaborative listening history with artist metadata.")
    with st.form("personalized_recommendations"):
        user_id = st.selectbox("Listener", user_ids)
        top_k = st.slider("Number of recommendations", 1, max_top_k, min(10, max_top_k))
        content_weight = st.slider("Content weight", 0.0, 1.0, 0.25, 0.05)
        diversity = st.slider("Diversity", 0.0, 1.0, 0.0, 0.05)
        popularity_penalty = st.slider(
            "Popularity penalty",
            0.0,
            1.0,
            0.0,
            0.05,
        )
        include_listened = st.checkbox("Include previously listened artists")
        explain = st.checkbox("Show recommendation reasons", value=True)
        submitted = st.form_submit_button(
            "Recommend for this listener",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        _run_recommendation(
            lambda: service.recommend_user(
                user_id=user_id,
                top_k=top_k,
                include_listened=include_listened,
                diversity=diversity,
                popularity_penalty=popularity_penalty,
                content_weight=content_weight,
                explain=explain,
            )
        )


def _render_profile_tab(
    service: RecommenderService,
    artist_choices: dict[str, str],
    genres: list[str],
    moods: list[str],
    max_top_k: int,
) -> None:
    st.write("Create a cold-start profile without an existing listener account.")
    with st.form("profile_recommendations"):
        selected_artists = st.multiselect(
            "Favorite artists",
            list(artist_choices),
        )
        selected_genres = st.multiselect("Favorite genres", genres)
        selected_moods = st.multiselect("Mood and style tags", moods)
        top_k = st.slider(
            "Number of recommendations",
            1,
            max_top_k,
            min(10, max_top_k),
            key="profile_top_k",
        )
        explain = st.checkbox(
            "Show recommendation reasons",
            value=True,
            key="profile_explain",
        )
        submitted = st.form_submit_button(
            "Build my taste profile",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        artist_ids = [artist_choices[label] for label in selected_artists]
        _run_recommendation(
            lambda: service.recommend_profile(
                artist_ids=artist_ids,
                genres=selected_genres,
                mood_tags=selected_moods,
                top_k=top_k,
                explain=explain,
            )
        )


def _render_session_tab(
    service: RecommenderService,
    user_ids: list[str],
    artist_choices: dict[str, str],
    genres: list[str],
    moods: list[str],
    max_top_k: int,
) -> None:
    st.write("Mix long-term taste with what fits the current listening session.")
    with st.form("session_recommendations"):
        selected_user = st.selectbox(
            "Listener profile",
            ["New listener", *user_ids],
        )
        selected_artists = st.multiselect(
            "Seed artists",
            list(artist_choices),
            key="session_artists",
        )
        selected_genres = st.multiselect(
            "Session genres",
            genres,
            key="session_genres",
        )
        selected_moods = st.multiselect(
            "Session moods",
            moods,
            key="session_moods",
        )
        excluded_artists = st.multiselect(
            "Exclude artists",
            list(artist_choices),
        )
        top_k = st.slider(
            "Number of recommendations",
            1,
            max_top_k,
            min(10, max_top_k),
            key="session_top_k",
        )
        content_weight = st.slider(
            "Short-term content weight",
            0.0,
            1.0,
            0.35,
            0.05,
        )
        diversity = st.slider(
            "Diversity",
            0.0,
            1.0,
            0.0,
            0.05,
            key="session_diversity",
        )
        popularity_penalty = st.slider(
            "Popularity penalty",
            0.0,
            1.0,
            0.0,
            0.05,
            key="session_popularity_penalty",
        )
        explain = st.checkbox(
            "Show recommendation reasons",
            value=True,
            key="session_explain",
        )
        submitted = st.form_submit_button(
            "Build a session mix",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        artist_ids = [artist_choices[label] for label in selected_artists]
        exclude_artist_ids = [artist_choices[label] for label in excluded_artists]
        user_id = None if selected_user == "New listener" else selected_user
        _run_recommendation(
            lambda: service.recommend_session(
                artist_ids=artist_ids,
                genres=selected_genres,
                mood_tags=selected_moods,
                user_id=user_id,
                top_k=top_k,
                exclude_artist_ids=exclude_artist_ids,
                diversity=diversity,
                popularity_penalty=popularity_penalty,
                content_weight=content_weight,
                explain=explain,
            )
        )


def _render_similarity_tab(
    service: RecommenderService,
    artist_choices: dict[str, str],
    max_top_k: int,
) -> None:
    st.write("Explore the catalog through collaborative and metadata similarity.")
    method_labels = {
        "Hybrid": "hybrid",
        "Collaborative (ALS)": "als",
        "Metadata": "content",
    }
    with st.form("similar_artists"):
        selected_artist = st.selectbox("Starting artist", list(artist_choices))
        method_label = st.selectbox("Similarity method", list(method_labels))
        top_k = st.slider(
            "Number of similar artists",
            1,
            max_top_k,
            min(10, max_top_k),
            key="similar_top_k",
        )
        content_weight = st.slider(
            "Content weight",
            0.0,
            1.0,
            0.25,
            0.05,
            disabled=method_label != "Hybrid",
            key="similar_content_weight",
        )
        explain = st.checkbox(
            "Show similarity reasons",
            value=True,
            key="similar_explain",
        )
        submitted = st.form_submit_button(
            "Find similar artists",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        _run_recommendation(
            lambda: service.similar_artists(
                artist_id=artist_choices[selected_artist],
                top_k=top_k,
                method=method_labels[method_label],  # type: ignore[arg-type]
                content_weight=content_weight,
                explain=explain,
            )
        )


def _render_catalog_tab(service: RecommenderService) -> None:
    st.write("Inspect artist metadata and the popularity signals used by the model.")
    search = st.text_input(
        "Search artists, genres, countries, or moods",
        placeholder="Try electronic, Canada, or atmospheric",
    )
    catalog = catalog_frame(service, query=search or None)
    total = int(catalog.attrs["total"])
    if catalog.attrs["has_more"]:
        st.caption(f"Showing the first {len(catalog)} of {total} matching artists.")
    else:
        st.caption(f"Showing {total} matching artist{'s' if total != 1 else ''}.")

    st.dataframe(
        catalog,
        hide_index=True,
        width="stretch",
        column_config={
            "total_plays": st.column_config.NumberColumn(format="%d"),
            "listener_count": st.column_config.NumberColumn(format="%d"),
            "popularity_rank": st.column_config.NumberColumn(format="%d"),
        },
    )


def render_dashboard(service: RecommenderService) -> None:
    """Render the dashboard using an already loaded recommender service."""
    health = service.health()
    metadata = service.artifact.content_artifacts.metadata
    artist_choices = _artist_choices(service)
    user_ids = sorted(
        (str(user_id) for user_id in service.artifact.mappings["user_id_to_index"]),
        key=str.casefold,
    )
    genres = split_metadata_terms(metadata["genres"])
    moods = split_metadata_terms(metadata["mood_tags"])
    max_top_k = max(1, min(25, len(artist_choices)))

    st.title("Music Recommender Studio")
    st.caption(
        "Explore personalized, cold-start, session-aware, and artist-similarity "
        "recommendations from one trained hybrid model."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Listeners", health["num_users"])
    metric_columns[1].metric("Artists", health["num_artists"])
    metric_columns[2].metric("Interactions", health["num_interactions"])
    metric_columns[3].metric("Content features", health["content_features"])

    with st.sidebar:
        st.header("Model status")
        st.success("Artifact loaded")
        st.write(f"Version `{health['artifact_version']}`")
        st.write(f"Training device `{service.artifact.metadata['training_device']}`")
        st.caption("Controls are applied locally against the cached artifact.")

    tabs = st.tabs(
        [
            "For You",
            "Taste Profile",
            "Session Mix",
            "Similar Artists",
            "Catalog",
        ]
    )
    with tabs[0]:
        _render_personalized_tab(service, user_ids, max_top_k)
    with tabs[1]:
        _render_profile_tab(
            service,
            artist_choices,
            genres,
            moods,
            max_top_k,
        )
    with tabs[2]:
        _render_session_tab(
            service,
            user_ids,
            artist_choices,
            genres,
            moods,
            max_top_k,
        )
    with tabs[3]:
        _render_similarity_tab(service, artist_choices, max_top_k)
    with tabs[4]:
        _render_catalog_tab(service)


def main() -> None:
    """Load the artifact and launch the Streamlit dashboard."""
    st.set_page_config(
        page_title="Music Recommender Studio",
        page_icon="🎧",
        layout="wide",
    )
    artifact_path = resolve_dashboard_artifact_path()
    try:
        with st.spinner("Loading the recommendation model..."):
            service = load_dashboard_service(str(artifact_path))
    except (FileNotFoundError, ValueError) as error:
        st.title("Music Recommender Studio")
        st.error(str(error))
        st.info("Train the model before launching the dashboard.")
        st.code("uv run python -m music_recommender.cli train --no-use-gpu")
        st.stop()

    render_dashboard(service)


if __name__ == "__main__":
    main()
