"""Interactive Streamlit dashboard for exploring recommendations."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from music_recommender.config import (
    ARTIFACT_BUNDLE_PATH,
    COLD_START_POLICY_PATH,
)
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
        recommendations = payload.get("similar_artists")
    if recommendations is None:
        recommendations = payload.get("similar_tracks", [])

    rows = []
    for rank, recommendation in enumerate(recommendations, start=1):
        reasons = recommendation.get("reasons") or []
        if recommendation.get("track_name"):
            name = str(recommendation["track_name"])
            if recommendation.get("artist_name"):
                name = f"{name} by {recommendation['artist_name']}"
            rec_id = str(recommendation.get("track_id", ""))
        else:
            name = str(recommendation.get("artist_name", "Unknown artist"))
            rec_id = str(recommendation.get("artist_id", ""))
        components = recommendation.get("score_components")
        scores = ""
        if isinstance(components, dict):
            scores = "content {content:.3f} · collab {collab:.3f}".format(
                content=float(components.get("content_score", 0.0)),
                collab=float(components.get("collaborative_score", 0.0)),
            )
        rows.append(
            {
                "Rank": rank,
                "Artist": name,
                "Artist ID": rec_id,
                "Score": round(float(recommendation.get("score", 0.0)), 4),
                "Score components": scores,
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
            "Score components",
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


def track_catalog_frame(
    service: RecommenderService,
    query: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Build a dashboard table from the shared track catalog service."""
    payload = service.browse_tracks(query=query, limit=limit)
    catalog = pd.DataFrame(payload["tracks"])
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
    ltr_available = service.artifact.ltr_model is not None
    with st.form("personalized_recommendations"):
        user_id = st.selectbox("Listener", user_ids)
        top_k = st.slider("Number of recommendations", 1, max_top_k, min(10, max_top_k))
        content_weight = st.slider("Content weight", 0.0, 1.0, 0.25, 0.05)
        diversity = st.slider("Diversity", 0.0, 1.0, 0.0, 0.05)
        novelty_weight = st.slider(
            "Novelty weight",
            0.0,
            1.0,
            0.0,
            0.05,
            help="Balance discovery of novel / less popular artists.",
        )
        popularity_penalty = st.slider(
            "Popularity penalty",
            0.0,
            1.0,
            0.0,
            0.05,
        )
        include_listened = st.checkbox("Include previously listened artists")
        explain = st.checkbox("Show recommendation reasons", value=True)
        use_ltr = st.checkbox(
            "Use Learning-to-Rank re-ranking",
            value=False,
            disabled=not ltr_available,
            help=(
                "Re-rank ALS candidates with the bundled LTR model "
                "(requires artifact trained with --learn-to-rank)."
            ),
        )
        submitted = st.form_submit_button(
            "Recommend for this listener",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if use_ltr and ltr_available:
            _run_recommendation(
                lambda: service.recommend_user_ltr(
                    user_id=user_id,
                    top_k=top_k,
                    include_listened=include_listened,
                    diversity=diversity,
                    popularity_penalty=popularity_penalty,
                    novelty_weight=novelty_weight,
                )
            )
        else:
            _run_recommendation(
                lambda: service.recommend_user(
                    user_id=user_id,
                    top_k=top_k,
                    include_listened=include_listened,
                    diversity=diversity,
                    popularity_penalty=popularity_penalty,
                    novelty_weight=novelty_weight,
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


def _track_choices(service: RecommenderService) -> dict[str, str]:
    """Return track select options, with a fallback when data is unavailable."""
    try:
        resources = service._track_resources()
    except (AttributeError, ValueError):
        return {"Track 1 · track_1": "track_1", "Track 2 · track_2": "track_2"}
    choices = {}
    for track_id in resources.track_ids:
        meta = resources.track_lookup.get(track_id, {})
        label = f"{meta.get('track_name', track_id)} · {track_id}"
        choices[label] = track_id
    return dict(sorted(choices.items(), key=lambda item: str(item[0]).casefold()))


def _render_tracks_tab(
    service: RecommenderService,
    user_ids: list[str],
    max_top_k: int,
) -> None:
    st.write("Recommend tracks with audio-feature similarity.")
    track_choices = _track_choices(service)
    search = st.text_input(
        "Search tracks",
        placeholder="Try a track, artist, or album name",
        key="tracks_search",
    )
    if search and search.strip():
        query = search.strip().casefold()
        track_choices = {
            label: track_id
            for label, track_id in track_choices.items()
            if query in label.casefold()
        }
    st.caption(f"Showing {len(track_choices)} matching track(s).")
    with st.form("track_recommendations"):
        user_id = st.selectbox("Listener", user_ids, key="tracks_listener")
        top_k = st.slider(
            "Number of recommendations",
            1,
            max_top_k,
            min(10, max_top_k),
            key="tracks_top_k",
        )
        include_listened = st.checkbox(
            "Include previously listened tracks",
            key="tracks_include_listened",
        )
        popularity_penalty = st.slider(
            "Popularity penalty",
            0.0,
            1.0,
            0.0,
            0.05,
            key="tracks_popularity_penalty",
            help="Reduce scores for globally popular tracks.",
        )
        diversity = st.slider(
            "Diversity",
            0.0,
            1.0,
            0.0,
            0.05,
            key="tracks_diversity",
            help="Diversify recommendations by audio features.",
        )
        novelty_weight = st.slider(
            "Novelty weight",
            0.0,
            1.0,
            0.0,
            0.05,
            key="tracks_novelty_weight",
            help="Balance discovery of novel / less popular tracks.",
        )
        explain = st.checkbox(
            "Show recommendation reasons",
            value=True,
            key="tracks_explain",
        )
        method = st.selectbox(
            "Method",
            ("similarity", "hybrid"),
            key="tracks_method",
            help="Hybrid blends artist taste with audio-feature similarity.",
        )
        content_weight = st.slider(
            "Artist taste vs audio features",
            0.0,
            1.0,
            0.25,
            0.05,
            key="tracks_content_weight",
            help="1.0 favors audio features, 0.0 favors artist taste.",
        )
        track_ltr_available = (
            getattr(service.artifact, "track_ltr_model", None) is not None
        )
        use_ltr = st.checkbox(
            "Use Learning-to-Rank re-ranking",
            value=False,
            disabled=not track_ltr_available,
            key="tracks_use_ltr",
            help=(
                "Re-rank candidates with the bundled track LTR model "
                "(requires artifact trained with track LTR)."
            ),
        )
        submitted = st.form_submit_button(
            "Recommend tracks",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        _run_recommendation(
            lambda: service.recommend_tracks(
                user_id=user_id,
                top_k=top_k,
                include_listened=include_listened,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                novelty_weight=novelty_weight,
                explain=explain,
                method=method,
                content_weight=content_weight,
                ltr=use_ltr,
            )
        )

    if not track_choices:
        st.warning("No tracks match the current search.")
    else:
        with st.form("similar_tracks"):
            selected_track = st.selectbox("Starting track", list(track_choices))
            similar_top_k = st.slider(
                "Number of similar tracks",
                1,
                max_top_k,
                min(10, max_top_k),
                key="similar_tracks_top_k",
            )
            similar_submitted = st.form_submit_button(
                "Find similar tracks",
                type="primary",
                use_container_width=True,
            )

        if similar_submitted:
            _run_recommendation(
                lambda: service.similar_tracks(
                    track_id=track_choices[selected_track],
                    top_k=similar_top_k,
                )
            )

    st.write("Browse the track catalog.")
    catalog = track_catalog_frame(service, query=search or None)
    total = int(catalog.attrs["total"])
    if catalog.attrs["has_more"]:
        st.caption(f"Showing the first {len(catalog)} of {total} matching tracks.")
    else:
        st.caption(f"Showing {total} matching track{'s' if total != 1 else ''}.")

    st.dataframe(
        catalog,
        hide_index=True,
        width="stretch",
        column_config={
            "total_plays": st.column_config.NumberColumn(format="%d"),
            "listener_count": st.column_config.NumberColumn(format="%d"),
            "popularity_rank": st.column_config.NumberColumn(format="%d"),
            "popularity": st.column_config.NumberColumn(format="%d"),
        },
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


def _ablation_ranking_rows(summary: dict[str, Any]) -> pd.DataFrame:
    """Shape the persisted ablation summary into a comparison table."""
    ranking_data = summary["ranking"]
    knobs_data = summary["knobs"]
    rows = []
    for item in ranking_data:
        knob = item["knob"]
        knob_info = knobs_data.get(knob, {})
        rows.append(
            {
                "Knob": knob,
                "Mean Impact": round(item["mean_impact"], 4),
                "Std Impact": round(knob_info.get("std_impact", 0.0), 4),
                "Runs": knob_info.get("count", 0),
            }
        )
    return pd.DataFrame(rows)


def _render_ablation_summary_body(summary: dict[str, Any]) -> None:
    """Render the loaded ablation summary as a comparison table."""
    st.subheader("Knob Importance by Mean Total Impact")
    ranking_data = summary["ranking"]

    if ranking_data:
        df = _ablation_ranking_rows(summary)
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={
                "Mean Impact": st.column_config.NumberColumn(format="%.4f"),
                "Std Impact": st.column_config.NumberColumn(format="%.4f"),
                "Runs": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            f"Aggregated from {summary['reports_loaded']} ablation report(s). "
            "A small standard deviation relative to the mean indicates a stable knob."
        )
    else:
        st.info("No knob importance data in the summary.")


def _render_ablation_summary_tab(service: RecommenderService) -> None:
    st.write("Aggregated knob-importance summary across ablation reports.")
    st.caption(
        "Run `music_recommender.cli ablation-summary` to generate the summary from "
        "persisted ablation reports."
    )
    try:
        from music_recommender.config import REPORTS_DIR
        from music_recommender.evaluate import load_ablation_summary_report

        summary = load_ablation_summary_report(REPORTS_DIR / "ablation_summary.json")
    except (FileNotFoundError, ValueError) as error:
        st.warning(f"No ablation summary available: {error}")
        st.info("Run `uv run music-recommender ablation-summary` to generate one.")
        return

    _render_ablation_summary_body(summary)


def _bandit_arm_rows(status: dict[str, Any]) -> pd.DataFrame:
    """Shape the bandit lifecycle summary into a per-arm comparison table."""
    rows = []
    for arm, stats in status["state"]["arms"].items():
        rows.append(
            {
                "Arm": arm,
                "Selections": stats["selections"],
                "Cum. Reward": round(stats["total_reward"], 4),
                "Mean Reward": round(stats["mean_reward"], 4),
            }
        )
    return pd.DataFrame(rows)


def _bandit_context_features(
    service: RecommenderService, status: dict[str, Any]
) -> tuple[str, ...]:
    """Return the active context feature set, preferring state-recorded names."""
    if status["available"] and status["state"]:
        recorded = status["state"].get("context_features")
        if recorded:
            return tuple(recorded)
    return service.context_features


def _render_bandit_tab(service: RecommenderService) -> None:
    st.write(
        "Cold-start bandit lifecycle: persisted state, active policy, and served "
        "feedback journal."
    )
    st.caption(
        "Served feedback is recorded to the journal by `recommend-user "
        "--record-feedback` and folded into the state by the idempotent sweep "
        "below (safe to rerun)."
    )
    try:
        status = service.bandit_status()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Bandit lifecycle unavailable: {error}")
        return

    if hasattr(service, "streaming_health"):
        try:
            health_status = service.streaming_health()
        except Exception:
            health_status = None
        if health_status:
            h_stat = str(health_status.get("status", "unknown")).upper()
            if h_stat == "HEALTHY":
                st.success(f"Streaming & Maintenance Health: **{h_stat}**")
            elif h_stat == "DEGRADED":
                st.warning(f"Streaming & Maintenance Health: **{h_stat}**")
            elif h_stat == "UNHEALTHY":
                st.error(f"Streaming & Maintenance Health: **{h_stat}**")
            for warning_msg in health_status.get("warnings", []):
                st.warning(f"⚠️ {warning_msg}")

    active_features = _bandit_context_features(service, status)
    st.caption(
        f"Context features ({len(active_features)}): "
        + ", ".join(f"`{feature}`" for feature in active_features)
        + "."
    )
    threshold = status.get("auto_sweep_threshold")
    if threshold:
        st.caption(
            f"Online auto-sweep threshold: **{threshold}** cold-start impressions."
        )

    if not status["available"]:
        st.info(
            "No bandit state yet. Run `simulate-bandit --write-state` or "
            "`bandit-update`."
        )
    else:
        state = status["state"]
        cols = st.columns(4)
        cols[0].metric("Arms", len(state["arms"]))
        cols[1].metric("Context dim", state["context_dim"])
        eff_alpha = state.get("effective_alpha", state.get("alpha", 1.0))
        cols[2].metric("Alpha", f"{eff_alpha:.2f}")
        cols[3].metric("Total selections", state["total_selections"])
        st.dataframe(
            _bandit_arm_rows(status),
            hide_index=True,
            width="stretch",
            column_config={
                "Selections": st.column_config.NumberColumn(format="%d"),
                "Cum. Reward": st.column_config.NumberColumn(format="%.4f"),
                "Mean Reward": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            f"Policy model: **{state.get('policy_type', 'linucb').upper()}** | "
            f"Effective alpha: **{eff_alpha:.3f}** | "
            f"State generated at {state['generated_at']} "
            f"(alpha_decay={state.get('alpha_decay', 0.0)})."
        )

    if status["policy"]:
        st.write(
            "Active policy: "
            + ", ".join(
                f"**{arm}** {weight:.3f}" for arm, weight in status["policy"].items()
            )
        )
    else:
        st.write("Active policy: none (cold start serves popular items).")

    fold = status["last_fold"]
    if fold:
        st.write(
            f"Journal: **{status['journal']['length']}** record(s), "
            f"**{status['journal']['pending']}** pending (last fold {fold['at']} "
            f"at offset {fold['offset']})."
        )
    else:
        st.write(
            f"Journal: **{status['journal']['length']}** record(s), "
            f"**{status['journal']['pending']}** pending (no fold yet)."
        )

    fold_col1, fold_col2 = st.columns([2, 2])
    with fold_col1:
        gamma_input = st.slider(
            "Recency discount factor (gamma)",
            min_value=0.1,
            max_value=1.0,
            value=float(status.get("gamma", 1.0)),
            step=0.05,
            key="bandit_gamma_slider",
            help="Weight given to prior statistics during fold (gamma * A + x x^T).",
        )
    with fold_col2:
        st.write("")
        st.write("")
        fold_clicked = st.button(
            "Fold pending feedback", type="primary", key="fold_bandit_feedback_btn"
        )

    if fold_clicked:
        try:
            updated = service.sweep_bandit_feedback(gamma=gamma_input)
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Fold failed: {error}")
        else:
            st.success(
                f"Folded pending feedback with gamma={gamma_input:.2f}. "
                f"Journal now {updated['journal']['pending']} pending."
            )

    queue_metrics = status.get("streaming_queue")
    if queue_metrics:
        st.markdown("##### Streaming Feedback Ingestion Queue")
        q_cols = st.columns(4)
        q_cols[0].metric(
            "Queue Depth",
            f"{queue_metrics['queue_depth']} / {queue_metrics['max_queue_size']}",
        )
        q_cols[1].metric(
            "Utilization",
            f"{float(queue_metrics['utilization_pct']):.1f}%",
        )
        q_cols[2].metric(
            "Backpressure",
            str(queue_metrics.get("backpressure", "drop_oldest")),
        )
        flushed_batches = queue_metrics.get("total_flushed_batches", 0)
        flushed_recs = queue_metrics.get("flushed_count", 0)
        q_cols[3].metric(
            "Flushed Batches",
            f"{flushed_batches} ({flushed_recs} recs)",
        )
        if (
            queue_metrics.get("dropped_count", 0) > 0
            or queue_metrics.get("flush_errors", 0) > 0
        ):
            st.caption(
                f"Dropped records: **{queue_metrics.get('dropped_count', 0)}** | "
                f"Flush errors: **{queue_metrics.get('flush_errors', 0)}**"
            )
        flush_q_btn = st.button(
            "Flush Feedback Queue", key="btn_flush_feedback_queue"
        )
        if flush_q_btn:
            if hasattr(service, "flush_feedback"):
                try:
                    service.flush_feedback()
                    st.success("Streaming feedback queue flushed successfully.")
                except Exception as error:
                    st.error(f"Queue flush failed: {error}")
            else:
                st.info("Service does not support flush_feedback.")

    worker_metrics = status.get("maintenance_worker")
    if worker_metrics:
        st.markdown("##### Asynchronous Maintenance Daemon")
        w_cols = st.columns(4)
        daemon_status = "Running" if worker_metrics.get("is_running") else "Stopped"
        w_cols[0].metric("Daemon Status", daemon_status)
        w_cols[1].metric("Sweep Interval", f"{worker_metrics['interval_seconds']}s")
        w_cols[2].metric("Cycles Executed", worker_metrics["cycles_count"])
        folded_recs = worker_metrics.get("records_folded_count", 0)
        w_cols[3].metric(
            "Sweeps / Folded",
            f"{worker_metrics['sweeps_count']} ({folded_recs} recs)",
        )
        last_sw = worker_metrics.get("last_sweep_duration_seconds")
        avg_sw = worker_metrics.get("avg_sweep_duration_seconds")
        min_sw = worker_metrics.get("min_sweep_duration_seconds")
        max_sw = worker_metrics.get("max_sweep_duration_seconds")
        if last_sw is not None or avg_sw is not None:
            w_time_cols = st.columns(4)
            w_time_cols[0].metric(
                "Last Sweep",
                f"{last_sw:.4f}s" if last_sw is not None else "-",
            )
            w_time_cols[1].metric(
                "Avg Sweep",
                f"{avg_sw:.4f}s" if avg_sw is not None else "-",
            )
            w_time_cols[2].metric(
                "Min Sweep",
                f"{min_sw:.4f}s" if min_sw is not None else "-",
            )
            w_time_cols[3].metric(
                "Max Sweep",
                f"{max_sw:.4f}s" if max_sw is not None else "-",
            )
        trigger_sweep_btn = st.button(
            "Trigger Maintenance Sweep", key="btn_trigger_maintenance_sweep"
        )
        if trigger_sweep_btn:
            if hasattr(service, "trigger_maintenance_sweep"):
                try:
                    sw_res = service.trigger_maintenance_sweep()
                    recs_folded = sw_res.get("records_folded", 0)
                    st.success(
                        f"Maintenance sweep triggered: {recs_folded} record(s) folded."
                    )
                except Exception as error:
                    st.error(f"Maintenance sweep failed: {error}")
            else:
                st.info("Service does not support trigger_maintenance_sweep.")

    st.divider()
    st.subheader("Bandit Snapshots & Drift Tracking")
    st.caption(
        "Persist timestamped state snapshots to monitor policy drift and track "
        "parameter evolution over time."
    )

    snap_col1, snap_col2 = st.columns([3, 1])
    with snap_col1:
        snapshot_label = st.text_input(
            "Snapshot label (optional)",
            placeholder="e.g. baseline, pre-campaign, audit",
            key="bandit_snapshot_label",
        )
    with snap_col2:
        st.write("")
        st.write("")
        create_snap_clicked = st.button("Create Snapshot", key="create_bandit_snapshot")

    if create_snap_clicked:
        try:
            created_path = service.create_bandit_snapshot(
                label=snapshot_label.strip() if snapshot_label else None
            )
            st.success(f"Created snapshot `{created_path.name}`.")
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Snapshot creation failed: {error}")

    champion_snap = status.get("champion_snapshot")
    if champion_snap:
        champ_label = champion_snap.get("label") or "none"
        champ_sels = champion_snap.get("total_selections", 0)
        st.info(
            f"🏆 Active Champion Snapshot: `{champion_snap['filename']}` "
            f"(label: `{champ_label}`, selections: {champ_sels})"
        )
    else:
        st.caption("Active Champion Snapshot: None tagged")

    try:
        snapshots = service.list_bandit_snapshots()
    except Exception as error:
        st.error(f"Failed to list snapshots: {error}")
        snapshots = []

    if snapshots:
        st.caption(f"Persisted snapshots ({len(snapshots)}):")
        snap_df = pd.DataFrame(
            [
                {
                    "Filename": s["filename"],
                    "Champion": (
                        "🏆 Champion"
                        if (
                            champion_snap is not None
                            and s["filename"] == champion_snap.get("filename")
                        )
                        or bool(s.get("is_champion", False))
                        else "-"
                    ),
                    "Timestamp": s.get("timestamp", "-"),
                    "Label": s.get("label") or "-",
                    "Size (bytes)": s.get("size_bytes", 0),
                }
                for s in snapshots
            ]
        )
        st.dataframe(snap_df, hide_index=True, width="stretch")

        snapshot_options = [str(s["filename"]) for s in snapshots]
        st.markdown("##### Champion Snapshot Management & State Rollback")
        tag_col, rollback_col = st.columns(2)
        with tag_col:
            snap_to_tag = st.selectbox(
                "Select snapshot to tag as champion",
                options=snapshot_options,
                index=0,
                key="bandit_snapshot_to_tag",
            )
            if st.button("Tag as Champion", key="btn_tag_champion_snapshot"):
                if hasattr(service, "tag_champion_snapshot"):
                    try:
                        service.tag_champion_snapshot(snap_to_tag)
                        st.success(
                            f"Snapshot `{snap_to_tag}` tagged as champion."
                        )
                    except Exception as error:
                        st.error(f"Champion tagging failed: {error}")
                else:
                    st.info("Service does not support tag_champion_snapshot.")
        with rollback_col:
            snap_to_rollback = st.selectbox(
                "Select snapshot to restore active state",
                options=snapshot_options,
                index=0,
                key="bandit_snapshot_to_rollback",
            )
            if st.button("Roll Back State", key="btn_rollback_bandit_state"):
                if hasattr(service, "rollback_bandit_state"):
                    try:
                        service.rollback_bandit_state(
                            snapshot_path=snap_to_rollback
                        )
                        st.success(
                            f"Rolled back active state to `{snap_to_rollback}`."
                        )
                    except Exception as error:
                        st.error(f"State rollback failed: {error}")
                else:
                    st.info("Service does not support rollback_bandit_state.")

        st.markdown("#### Policy Drift Analysis")
        st.caption(
            "Compare current bandit parameters against a reference snapshot to "
            "inspect ridge coefficient drift (L2 norm, cosine similarity) and arm "
            "dominance."
        )
        snapshot_options = [str(s["filename"]) for s in snapshots]
        selected_ref = st.selectbox(
            "Reference snapshot",
            options=snapshot_options,
            index=0,
            key="bandit_drift_reference",
        )
        if selected_ref:
            try:
                drift = service.compute_bandit_drift(reference_path=selected_ref)
            except (FileNotFoundError, ValueError) as error:
                st.error(f"Drift computation failed: {error}")
            else:
                drift_cols = st.columns(4)
                dominant_b = str(drift["dominant_arm_b"] or "none")
                dominant_a = str(drift["dominant_arm_a"] or "none")
                drift_cols[0].metric("Dominant Arm (Current)", dominant_b)
                drift_cols[1].metric("Dominant Arm (Ref)", dominant_a)
                drift_cols[2].metric(
                    "Dominant Arm Changed",
                    "Yes" if drift["dominant_arm_changed"] else "No",
                )
                drift_cols[3].metric(
                    "Max L2 Drift",
                    f"{float(str(drift['max_l2_drift'])):.4f}",
                )

                arm_drift_rows = []
                drift_arms = drift.get("arms", {})
                if isinstance(drift_arms, dict):
                    for arm, arm_data in drift_arms.items():
                        cos_sim = arm_data.get("theta_cosine_similarity")
                        arm_drift_rows.append(
                            {
                                "Arm": arm,
                                "L2 Drift": round(float(arm_data["theta_l2_drift"]), 4),
                                "Cosine Sim": (
                                    round(float(cos_sim), 4)
                                    if cos_sim is not None
                                    else "-"
                                ),
                                "Selections Δ": arm_data["selections_growth"],
                                "Reward Δ": round(
                                    float(arm_data["total_reward_diff"]), 4
                                ),
                                "Mean Reward Δ": round(
                                    float(arm_data["mean_reward_diff"]), 4
                                ),
                            }
                        )
                if arm_drift_rows:
                    st.dataframe(
                        pd.DataFrame(arm_drift_rows),
                        hide_index=True,
                        width="stretch",
                    )

                guardrails = status.get("drift_guardrails", {})
                thresholds = guardrails.get("thresholds", {})
                st.markdown("##### Automated Drift Guardrails")
                g_enabled = (
                    "Enabled" if guardrails.get("enabled") else "Disabled"
                )
                g_rollback = (
                    "Enabled" if guardrails.get("auto_rollback") else "Disabled"
                )
                st.caption(
                    f"Guardrails: **{g_enabled}** | "
                    f"Auto-rollback: **{g_rollback}** | "
                    f"Max L2: `{thresholds.get('max_l2_drift', 0.5)}` | "
                    f"Min Cosine: `{thresholds.get('min_cosine_similarity', 0.7)}`"
                )
                if st.button("Verify Drift Safety", key="btn_verify_drift_safety"):
                    if hasattr(service, "evaluate_bandit_drift_safety"):
                        try:
                            safety_res = service.evaluate_bandit_drift_safety(
                                reference_path=selected_ref
                            )
                            ref_name = safety_res.get(
                                "reference_snapshot", selected_ref
                            )
                            if safety_res.get("is_safe"):
                                st.success(
                                    "Drift Safety Check: **PASSED** against "
                                    f"`{ref_name}`."
                                )
                            else:
                                st.error(
                                    "Drift Safety Check: **BREACH DETECTED** against "
                                    f"`{ref_name}`."
                                )
                            for violation in safety_res.get("violations", []):
                                st.warning(f"Violation: {violation}")
                        except Exception as error:
                            st.error(f"Drift safety evaluation failed: {error}")
                    else:
                        st.info(
                            "Service does not support evaluate_bandit_drift_safety."
                        )
    else:
        st.info(
            "No snapshots found in `reports/bandit_snapshots/`. "
            "Create a snapshot above to begin tracking policy drift."
        )

    st.divider()
    st.subheader("Cold-Start Policy Derivation")
    st.caption(
        "Derive serving softmax arm weights from the active bandit state with "
        "optional temperature annealing."
    )
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        temp_input = st.slider(
            "Temperature (tau_0)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.05,
            key="bandit_derive_temperature",
            help="Higher temperature yields more uniform exploration.",
        )
    with p_col2:
        decay_input = st.slider(
            "Temperature decay rate",
            min_value=0.0,
            max_value=0.1,
            value=0.0,
            step=0.005,
            key="bandit_derive_temp_decay",
            help="Annealing schedule decay lambda: tau(t) = max(min_t, tau_0/(1+l*t)).",
        )
    with p_col3:
        min_temp_input = st.slider(
            "Min temperature floor",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            key="bandit_derive_min_temp",
            help="Minimum temperature boundary under annealing schedule.",
        )
    persist_check = st.checkbox(
        "Persist derived weights to active serving policy location",
        value=True,
        key="bandit_derive_persist_check",
    )
    if st.button("Derive Policy", key="btn_bandit_derive_policy"):
        try:
            derived_weights = service.derive_cold_start_policy_from_state(
                temperature=temp_input,
                temperature_decay=decay_input,
                min_temperature=min_temp_input,
                persist_path=COLD_START_POLICY_PATH if persist_check else None,
                update_active_policy=persist_check,
            )
            st.success("Derived cold-start serving weights successfully.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Arm": arm, "Weight": weight}
                        for arm, weight in sorted(
                            derived_weights.items(), key=lambda x: -x[1]
                        )
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Derivation failed: {error}")

    st.divider()
    st.subheader("Offline Evaluation & Policy Benchmarking")

    with st.expander("Offline Policy Evaluation (OPE) on Logged Feedback"):
        st.caption(
            "Evaluate candidate policies counterfactually using Inverse Propensity "
            "Scoring (IPS), SnIPS, Direct Method (DM), and Doubly Robust (DR)."
        )
        ope_col1, ope_col2 = st.columns(2)
        with ope_col1:
            candidate_policy = st.selectbox(
                "Candidate Target Policy",
                options=[
                    "linucb",
                    "thompson_sampling",
                    "epsilon_greedy",
                    "active_state",
                    "popular",
                ],
                index=0,
                key="bandit_ope_candidate_policy",
            )
        with ope_col2:
            propensity_clip = st.slider(
                "Min Propensity Clip Threshold",
                min_value=0.01,
                max_value=0.20,
                value=0.01,
                step=0.01,
                key="bandit_ope_propensity_clip",
            )
        if st.button("Run Off-Policy Evaluation", key="btn_run_bandit_ope"):
            try:
                target_arg: Any = (
                    candidate_policy if candidate_policy != "active_state" else None
                )
                ope_res = service.evaluate_bandit_off_policy(
                    target_policy=target_arg,
                    min_propensity=propensity_clip,
                )
                ope_summary = ope_res["summary"]
                ope_metrics = ope_res["metrics"]
                m_cols = st.columns(4)
                m_cols[0].metric(
                    "Records Evaluated", ope_summary["records_evaluated"]
                )
                m_cols[1].metric("Match Rate", f"{ope_summary['match_rate']:.1%}")
                m_cols[2].metric(
                    "Effective Sample Size",
                    f"{ope_summary['effective_sample_size']:.1f}",
                )
                m_cols[3].metric(
                    "Logged Mean Reward",
                    f"{ope_summary['logging_mean_reward']:.4f}",
                )

                est_rows = [
                    {
                        "Estimator": "Inverse Propensity (IPS)",
                        "Estimated Value": ope_metrics["ips"]["value"],
                        "Std Error": ope_metrics["ips"].get("standard_error", "-"),
                    },
                    {
                        "Estimator": "Self-Normalized (SnIPS)",
                        "Estimated Value": ope_metrics["snips"]["value"],
                        "Std Error": "-",
                    },
                    {
                        "Estimator": "Direct Method (DM)",
                        "Estimated Value": ope_metrics["direct_method"]["value"],
                        "Std Error": ope_metrics["direct_method"].get(
                            "standard_error", "-"
                        ),
                    },
                    {
                        "Estimator": "Doubly Robust (DR)",
                        "Estimated Value": ope_metrics["doubly_robust"]["value"],
                        "Std Error": ope_metrics["doubly_robust"].get(
                            "standard_error", "-"
                        ),
                    },
                ]
                st.dataframe(pd.DataFrame(est_rows), hide_index=True, width="stretch")
            except (FileNotFoundError, ValueError) as error:
                st.error(f"Off-policy evaluation failed: {error}")

    with st.expander("Multi-Policy Comparative Simulation Benchmark"):
        st.caption(
            "Benchmark candidate policies side-by-side across identical holdout splits."
        )
        bench_col1, bench_col2 = st.columns(2)
        with bench_col1:
            bench_rounds = st.number_input(
                "Simulation Rounds",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="bandit_bench_rounds",
            )
        with bench_col2:
            bench_policies = st.multiselect(
                "Benchmark Policies",
                options=["linucb", "thompson_sampling", "epsilon_greedy"],
                default=["linucb", "thompson_sampling", "epsilon_greedy"],
                key="bandit_bench_policies",
            )
        if st.button("Run Policy Benchmark", key="btn_run_bandit_bench"):
            if not bench_policies:
                st.warning("Select at least one policy to benchmark.")
            else:
                try:
                    bench_res = service.compare_bandit_policies(
                        policies=bench_policies,
                        rounds=int(bench_rounds),
                    )
                    champ = bench_res["summary"]["champion"]
                    st.success(f"Benchmark completed! Champion: **{champ}**")
                    board_items = bench_res["summary"]["leaderboard"]
                    leaderboard_rows = []
                    for rank, b_item in enumerate(board_items, start=1):
                        p_name = b_item["name"]
                        p_stats = bench_res["policies"][p_name]
                        leaderboard_rows.append(
                            {
                                "Rank": rank,
                                "Policy": p_name,
                                "Cum. Reward": round(p_stats["cumulative_reward"], 4),
                                "Mean Reward": round(p_stats["mean_reward"], 4),
                                "Regret": round(p_stats["regret"], 4),
                                "Win Rate": f"{b_item['win_rate']:.1%}",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(leaderboard_rows),
                        hide_index=True,
                        width="stretch",
                    )
                except (FileNotFoundError, ValueError) as error:
                    st.error(f"Policy benchmark failed: {error}")


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
            "Tracks",
            "Catalog",
            "Ablation Summary",
            "Cold-Start Bandit",
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
        _render_tracks_tab(service, user_ids, max_top_k)
    with tabs[5]:
        _render_catalog_tab(service)
    with tabs[6]:
        _render_ablation_summary_tab(service)
    with tabs[7]:
        _render_bandit_tab(service)


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
    main()  # pragma: no cover - dashboard entry point invoked by Streamlit
