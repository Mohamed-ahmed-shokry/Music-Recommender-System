"""Command line interface for the music recommender."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import typer

from music_recommender import __version__
from music_recommender.bandit import (
    DEFAULT_COLD_START_ARMS,
    BaseContextualBandit,
    DriftSafetyThresholds,
    EpsilonGreedyContextualBandit,
    LinUCBContextualBandit,
    ThompsonSamplingContextualBandit,
    append_bandit_feedback,
    compare_bandit_simulation_policies,
    compute_bandit_drift,
    compute_off_policy_evaluation,
    derive_cold_start_policy,
    evaluate_drift_safety,
    feedback_from_report,
    fold_bandit_state,
    get_champion_snapshot,
    list_bandit_snapshots,
    load_bandit_feedback,
    load_bandit_report,
    load_bandit_snapshot,
    load_bandit_state,
    load_cold_start_policy,
    pending_feedback_count,
    prune_bandit_snapshots,
    resolve_context_features,
    restore_bandit_snapshot,
    rollback_bandit_state,
    save_bandit_context_features,
    save_bandit_snapshot,
    simulate_cold_start_exploration,
    snapshot_bandit_state,
    summarize_bandit_lifecycle,
    sweep_bandit_journal,
    tag_champion_snapshot,
    validate_state_context_features,
    write_bandit_comparison_report,
    write_bandit_report,
    write_bandit_state,
    write_cold_start_policy,
    write_ope_report,
)
from music_recommender.config import (
    ARTIFACT_BUNDLE_PATH,
    BANDIT_CONTEXT_FEATURES_PATH,
    BANDIT_FEEDBACK_PATH,
    BANDIT_SNAPSHOTS_DIR,
    BANDIT_STATE_PATH,
    COLD_START_POLICY_PATH,
    CONTEXT_FEATURES_ENV_VAR,
    DATA_DIR,
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
    DEFAULT_BANDIT_DRIFT_MAX_L2,
    DEFAULT_BANDIT_DRIFT_MAX_REWARD_DROP,
    DEFAULT_BANDIT_DRIFT_MIN_COSINE,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_MIN_ARTIST_INTERACTIONS,
    DEFAULT_MIN_USER_INTERACTIONS,
    DEFAULT_TOP_K,
    DEFAULT_USE_GPU,
    MAPPINGS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    RAW_METADATA_PATH,
    RAW_TRACK_DATA_PATH,
    RAW_TRACK_METADATA_PATH,
    REPORTS_DIR,
)
from music_recommender.data import load_and_validate_interactions
from music_recommender.evaluate import (
    ablation_importances,
    aggregate_ablation_reports,
    build_ablation_settings,
    compare_parameter_settings,
    evaluate_pareto_frontier,
    evaluate_repeated_holdout,
    ranking_params_for_training,
    select_winning_strategies,
    strategy_leaderboard,
    write_ablation_report,
    write_ablation_summary_report,
    write_pareto_report,
)
from music_recommender.logging_setup import configure_logging
from music_recommender.metadata import load_and_validate_artist_metadata
from music_recommender.model import train_and_save_model
from music_recommender.preprocessing import prepare_training_data
from music_recommender.recommend import format_recommendations
from music_recommender.service import RecommenderService
from music_recommender.surfaces import (
    compare_surface_metrics,
    write_surface_comparison_report,
)
from music_recommender.track_evaluate import (
    ablate_track_parameter_settings,
    compare_track_parameter_settings,
    evaluate_track_holdout,
    write_track_report,
)
from music_recommender.tracking import (
    DEFAULT_EVALUATION_EXPERIMENT,
    DEFAULT_TRAINING_EXPERIMENT,
    ExperimentTrackingError,
    tracking_run,
)
from music_recommender.tracks import (
    artist_affinity_to_track_scores,
    artist_taste_scores_for_user,
    build_track_content_matrix,
    build_track_stats,
    get_similar_tracks,
    load_and_validate_track_interactions,
    load_and_validate_track_metadata,
    recommend_tracks_for_user,
    train_track_artist_taste,
)
from music_recommender.tracks import (
    popular_tracks as rank_tracks_by_popularity,
)

# Optional Spotify imports
try:
    from music_recommender.spotify import (
        SpotifyConfig,
        build_track_metadata_frame,
        create_spotify_client,
        fetch_artist,
        fetch_artist_top_tracks,
        fetch_artists,
        fetch_audio_features,
        get_artist_related_artists,
        search_artists,
        search_tracks,
    )

    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False

logger = logging.getLogger(__name__)

app = typer.Typer(help="Train and use an ALS music artist recommender.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed package version and exit.",
    ),
) -> None:
    """Train, evaluate, and serve hybrid artist recommendations."""
    configure_logging()


def _format_artifact_age(created_at: str) -> str:
    created = datetime.fromisoformat(created_at)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age = datetime.now(UTC) - created
    total_seconds = int(age.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m"
    return f"{total_seconds // 3600}h"


def _parse_csv_option(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command()
def prepare_data(
    min_user_interactions: int = DEFAULT_MIN_USER_INTERACTIONS,
    min_artist_interactions: int = DEFAULT_MIN_ARTIST_INTERACTIONS,
) -> None:
    """Validate sample data, build the interaction matrix, and save mappings."""
    df, user_item_matrix, mappings = prepare_training_data(
        raw_data_path=RAW_DATA_PATH,
        mappings_path=MAPPINGS_PATH,
        min_user_interactions=min_user_interactions,
        min_artist_interactions=min_artist_interactions,
    )
    typer.echo("Data prepared successfully.")
    typer.echo(f"Users: {len(mappings['user_id_to_index'])}")
    typer.echo(f"Artists: {len(mappings['artist_id_to_index'])}")
    typer.echo(f"Interactions: {len(df)}")
    typer.echo(f"Matrix shape: {user_item_matrix.shape}")


@app.command()
def prepare_metadata(
    metadata_path: Path = RAW_METADATA_PATH,
    data_path: Path = RAW_DATA_PATH,
) -> None:
    """Validate artist metadata and sample interaction coverage."""
    try:
        interactions_df = load_and_validate_interactions(data_path)
        metadata_df = load_and_validate_artist_metadata(metadata_path, interactions_df)
    except ValueError as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Metadata validated successfully.")
    typer.echo(f"Metadata rows: {len(metadata_df)}")
    typer.echo(f"Interaction artists covered: {interactions_df['artist_id'].nunique()}")
    typer.echo(f"Metadata path: {metadata_path}")


@app.command()
def train(
    data_path: Path = RAW_DATA_PATH,
    metadata_path: Path = RAW_METADATA_PATH,
    track_data_path: Path = RAW_TRACK_DATA_PATH,
    track_metadata_path: Path = RAW_TRACK_METADATA_PATH,
    factors: int = DEFAULT_ALS_FACTORS,
    regularization: float = DEFAULT_ALS_REGULARIZATION,
    iterations: int = DEFAULT_ALS_ITERATIONS,
    alpha: float = DEFAULT_ALS_ALPHA,
    use_gpu: bool = DEFAULT_USE_GPU,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    popularity_penalty: float = 0.0,
    diversity: float = 0.0,
    include_listened: bool = typer.Option(
        False,
        "--include-listened/--no-include-listened",
        help="Include previously listened artists by default.",
    ),
    track: bool = typer.Option(
        False,
        "--track/--no-track",
        help="Log this training run to MLflow.",
    ),
    tracking_uri: str | None = typer.Option(
        None,
        help="Remote MLflow server URI; defaults to MLFLOW_TRACKING_URI.",
    ),
    experiment_name: str = typer.Option(
        DEFAULT_TRAINING_EXPERIMENT,
        help="MLflow experiment name.",
    ),
    run_name: str | None = typer.Option(None, help="Optional MLflow run name."),
    log_artifact: bool = typer.Option(
        True,
        "--log-artifact/--no-log-artifact",
        help="Upload the serving artifact when tracking is enabled.",
    ),
) -> None:
    """Train and save the ALS model."""
    try:
        with tracking_run(
            enabled=track,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            tags={"workflow": "training", "model_type": "implicit_als"},
        ) as tracked_run:
            tracked_run.log_params(
                {
                    "data_path": str(data_path),
                    "metadata_path": str(metadata_path),
                    "track_data_path": str(track_data_path),
                    "track_metadata_path": str(track_metadata_path),
                    "factors": factors,
                    "regularization": regularization,
                    "iterations": iterations,
                    "alpha": alpha,
                    "use_gpu": use_gpu,
                    "content_weight": content_weight,
                    "popularity_penalty": popularity_penalty,
                    "diversity": diversity,
                    "include_listened": include_listened,
                }
            )
            model, user_item_matrix, mappings = train_and_save_model(
                raw_data_path=data_path,
                metadata_path=metadata_path,
                track_data_path=track_data_path,
                track_metadata_path=track_metadata_path,
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                alpha=alpha,
                use_gpu=use_gpu,
                content_weight=content_weight,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                include_listened=include_listened,
            )
            matrix_size = user_item_matrix.shape[0] * user_item_matrix.shape[1]
            tracked_run.log_metrics(
                {
                    "num_users": len(mappings["user_id_to_index"]),
                    "num_artists": len(mappings["artist_id_to_index"]),
                    "num_interactions": user_item_matrix.nnz,
                    "matrix_density": (
                        user_item_matrix.nnz / matrix_size if matrix_size else 0.0
                    ),
                }
            )
            tracked_run.set_tags(
                {
                    "training_device": getattr(
                        model,
                        "training_device",
                        "unknown",
                    ),
                    "gpu_fallback": bool(getattr(model, "gpu_fallback_reason", None)),
                }
            )
            if track and log_artifact:
                tracked_run.log_artifact(
                    ARTIFACT_BUNDLE_PATH,
                    artifact_path="serving",
                )
    except (ExperimentTrackingError, FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    except RuntimeError as error:
        typer.secho(
            f"Error: training failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo("Model trained successfully.")
    typer.echo(f"Training device: {getattr(model, 'training_device', 'unknown')}")
    fallback_reason = getattr(model, "gpu_fallback_reason", None)
    if fallback_reason:
        typer.echo(f"GPU fallback reason: {fallback_reason}")
    typer.echo(f"Saved model to: {MODEL_PATH}")
    typer.echo(f"Saved mappings to: {MAPPINGS_PATH}")
    typer.echo(f"Saved artifact bundle to: {ARTIFACT_BUNDLE_PATH}")
    typer.echo(f"Training matrix shape: {user_item_matrix.shape}")
    typer.echo(f"Users: {len(mappings['user_id_to_index'])}")
    typer.echo(f"Artists: {len(mappings['artist_id_to_index'])}")
    typer.echo(f"Default content weight: {content_weight}")
    typer.echo(
        f"Default ranking settings: penalty={popularity_penalty}, "
        f"diversity={diversity}, include_listened={include_listened}"
    )
    if tracked_run.enabled:
        typer.echo(f"MLflow run ID: {tracked_run.run_id}")
        typer.echo(f"MLflow tracking URI: {tracked_run.tracking_uri}")


@app.command()
def artifact_info() -> None:
    """Print details about the saved recommender artifact."""
    try:
        service = RecommenderService.from_artifacts()
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    metadata = service.metadata()
    artifact_metadata = metadata["metadata"]
    training_config = metadata["training_config"]
    hybrid_config = metadata["hybrid_config"]
    content_metadata = metadata["content"]
    typer.echo(f"Artifact version: {metadata['version']}")
    typer.echo(f"Created at: {artifact_metadata['created_at']}")
    typer.echo(f"Artifact age: {_format_artifact_age(artifact_metadata['created_at'])}")
    typer.echo(f"Users: {artifact_metadata['num_users']}")
    typer.echo(f"Artists: {artifact_metadata['num_artists']}")
    typer.echo(f"Interactions: {artifact_metadata['num_interactions']}")
    typer.echo(f"Training device: {artifact_metadata['training_device']}")
    if artifact_metadata.get("gpu_fallback_reason"):
        typer.echo(f"GPU fallback reason: {artifact_metadata['gpu_fallback_reason']}")
    typer.echo(f"Factors: {training_config['factors']}")
    typer.echo(f"Regularization: {training_config['regularization']}")
    typer.echo(f"Iterations: {training_config['iterations']}")
    typer.echo(f"Alpha: {training_config['alpha']}")
    typer.echo(f"Default content weight: {hybrid_config['default_content_weight']}")
    ranking_config = metadata.get("ranking_config", {})
    typer.echo(
        "Default ranking settings: "
        f"penalty={ranking_config.get('popularity_penalty', 0.0)}, "
        f"diversity={ranking_config.get('diversity', 0.0)}, "
        f"include_listened={ranking_config.get('include_listened', False)}"
    )
    typer.echo(f"Content features: {content_metadata['num_features']}")
    typer.echo(f"Dataset hash: {artifact_metadata['dataset']['sha256']}")
    typer.echo(
        f"Metadata dataset hash: {artifact_metadata['metadata_dataset']['sha256']}"
    )


@app.command()
def recommend_user(
    user_id: str = typer.Option(..., help="Original user ID, for example user_1."),
    top_k: int = DEFAULT_TOP_K,
    include_listened: bool = typer.Option(
        False,
        "--include-listened/--exclude-listened",
        help="Include or exclude artists the user already listened to.",
    ),
    popularity_penalty: float = 0.0,
    diversity: float = 0.0,
    novelty_weight: float = typer.Option(
        0.0,
        "--novelty-weight",
        min=0.0,
        max=1.0,
        help="Weight for novelty/unpopular discovery in multi-objective ranking.",
    ),
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    explain: bool = False,
    cold_start_policy_path: str | None = typer.Option(
        None,
        "--cold-start-policy-path",
        help="Path to a cold-start bandit policy JSON for unknown-user serving.",
    ),
    record_feedback: bool = typer.Option(
        False,
        "--record-feedback",
        help=(
            "Record served bandit feedback (context, arm, reward) for unknown "
            "users to the feedback journal."
        ),
    ),
    feedback_journal_path: str = typer.Option(
        str(BANDIT_FEEDBACK_PATH),
        "--feedback-path",
        help="Journal path for recorded serve feedback.",
    ),
    context: str | None = typer.Option(
        None,
        "--context",
        help=(
            "Comma-separated cold-start context features observed for this "
            "request (log_plays, log_unique_artists, mean_popularity_rank); "
            "recorded with --record-feedback instead of the neutral context."
        ),
    ),
    ltr: bool = typer.Option(
        False,
        "--ltr/--no-ltr",
        help="Re-rank the ALS candidates with the bundled learning-to-rank model.",
    ),
) -> None:
    """Recommend artists for a user."""
    try:
        if cold_start_policy_path is not None:
            service = RecommenderService.from_artifacts(
                cold_start_policy_path=cold_start_policy_path
            )
        else:
            service = RecommenderService.from_artifacts()
        if ltr:
            if record_feedback:
                raise ValueError(
                    "--record-feedback only applies to the cold-start bandit "
                    "serving branch (use without --ltr)."
                )
            if context is not None:
                raise ValueError(
                    "--context only applies to the cold-start bandit "
                    "serving branch (use without --ltr)."
                )
            response = service.recommend_user_ltr(
                user_id=user_id,
                top_k=top_k,
                include_listened=include_listened,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                novelty_weight=novelty_weight,
            )
        else:
            parsed_context: list[float] | None = None
            if context is not None:
                if not record_feedback:
                    raise ValueError(
                        "--context requires --record-feedback so the served "
                        "observation is actually journaled."
                    )
                parsed_context = [
                    float(value.strip())
                    for value in context.split(",")
                    if value.strip()
                ]
            response = service.recommend_user(
                user_id=user_id,
                top_k=top_k,
                include_listened=include_listened,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                novelty_weight=novelty_weight,
                content_weight=content_weight,
                explain=explain,
                record_feedback=record_feedback,
                feedback_journal_path=feedback_journal_path,
                context=parsed_context,
            )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Recommendations for {user_id}:")
    typer.echo(f"Strategy: {response['strategy']}")
    if response.get("message"):
        typer.echo(response["message"])
    typer.echo(format_recommendations(response["recommendations"]))
    if response.get("feedback"):
        feedback = response["feedback"]
        arms = ", ".join(feedback["arms"]) if feedback.get("arms") else feedback["arm"]
        typer.secho(
            f"Served feedback recorded to {feedback['journal_path']} "
            f"(arm='{feedback['arm']}', reward={feedback['reward']}, "
            f"credited arms: {arms})",
            fg=typer.colors.GREEN,
        )


@app.command()
def recommend_profile(
    artist_ids: str = typer.Option(
        "",
        help="Comma-separated favorite artist IDs, for example artist_1,artist_6.",
    ),
    genres: str = typer.Option(
        "",
        help="Comma-separated preferred genres, for example pop,electronic.",
    ),
    mood_tags: str = typer.Option(
        "",
        help="Comma-separated mood tags, for example bright,dancefloor.",
    ),
    top_k: int = DEFAULT_TOP_K,
    explain: bool = False,
) -> None:
    """Recommend artists from onboarding preferences."""
    try:
        service = RecommenderService.from_artifacts()
        response = service.recommend_profile(
            artist_ids=_parse_csv_option(artist_ids),
            genres=_parse_csv_option(genres),
            mood_tags=_parse_csv_option(mood_tags),
            top_k=top_k,
            explain=explain,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Profile recommendations:")
    typer.echo(f"Strategy: {response['strategy']}")
    typer.echo(format_recommendations(response["recommendations"]))


@app.command()
def recommend_session(
    user_id: str | None = typer.Option(
        None,
        help="Optional known user ID to blend long-term taste into the session.",
    ),
    artist_ids: str = typer.Option(
        "",
        help="Comma-separated seed artist IDs, for example artist_1,artist_6.",
    ),
    genres: str = typer.Option(
        "",
        help="Comma-separated session genres, for example pop,electronic.",
    ),
    mood_tags: str = typer.Option(
        "",
        help="Comma-separated session moods, for example bright,dancefloor.",
    ),
    exclude_artist_ids: str = typer.Option(
        "",
        help="Comma-separated artist IDs to exclude from the session.",
    ),
    top_k: int = DEFAULT_TOP_K,
    include_listened: bool = typer.Option(
        False,
        "--include-listened/--exclude-listened",
        help="Include or exclude artists the user already listened to.",
    ),
    popularity_penalty: float = 0.0,
    diversity: float = 0.0,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    explain: bool = False,
) -> None:
    """Recommend artists for a short-term listening session."""
    try:
        service = RecommenderService.from_artifacts()
        response = service.recommend_session(
            artist_ids=_parse_csv_option(artist_ids),
            genres=_parse_csv_option(genres),
            mood_tags=_parse_csv_option(mood_tags),
            user_id=user_id,
            top_k=top_k,
            exclude_artist_ids=_parse_csv_option(exclude_artist_ids),
            include_listened=include_listened,
            popularity_penalty=popularity_penalty,
            diversity=diversity,
            content_weight=content_weight,
            explain=explain,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Session recommendations:")
    typer.echo(f"Strategy: {response['strategy']}")
    if response.get("message"):
        typer.echo(response["message"])
    typer.echo(format_recommendations(response["recommendations"]))


@app.command()
def popular_artists(top_k: int = DEFAULT_TOP_K) -> None:
    """Show globally popular artists from the training data."""
    try:
        service = RecommenderService.from_artifacts()
        response = service.popular_artists(top_k=top_k)
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Popular artists:")
    typer.echo(f"Strategy: {response['strategy']}")
    typer.echo(format_recommendations(response["recommendations"]))


@app.command()
def similar_artists(
    artist_id: str = typer.Option(
        ..., help="Original artist ID, for example artist_2."
    ),
    top_k: int = DEFAULT_TOP_K,
    method: Literal["als", "content", "hybrid"] = typer.Option(
        "als", help="Similarity method: als, content, hybrid."
    ),
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    explain: bool = False,
) -> None:
    """Find artists similar to a selected artist."""
    try:
        service = RecommenderService.from_artifacts()
        response = service.similar_artists(
            artist_id=artist_id,
            top_k=top_k,
            method=method,
            content_weight=content_weight,
            explain=explain,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Artists similar to {artist_id}:")
    typer.echo(f"Strategy: {response['strategy']}")
    typer.echo(format_recommendations(response["similar_artists"]))


@app.command()
def content_similar_artists(
    artist_id: str = typer.Option(
        ..., help="Original artist ID, for example artist_2."
    ),
    top_k: int = DEFAULT_TOP_K,
    explain: bool = False,
) -> None:
    """Find artists similar to a selected artist using metadata only."""
    try:
        service = RecommenderService.from_artifacts()
        response = service.content_similar_artists(
            artist_id=artist_id,
            top_k=top_k,
            explain=explain,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Content-similar artists for {artist_id}:")
    typer.echo(f"Strategy: {response['strategy']}")
    typer.echo(format_recommendations(response["similar_artists"]))


@app.command()
def evaluate(
    top_k: int = DEFAULT_TOP_K,
    folds: int = 1,
    compare_baseline: bool = False,
    compare_all: bool = False,
    use_gpu: bool = DEFAULT_USE_GPU,
    track: bool = typer.Option(
        False,
        "--track/--no-track",
        help="Log this evaluation run to MLflow.",
    ),
    tracking_uri: str | None = typer.Option(
        None,
        help="Remote MLflow server URI; defaults to MLFLOW_TRACKING_URI.",
    ),
    experiment_name: str = typer.Option(
        DEFAULT_EVALUATION_EXPERIMENT,
        help="MLflow experiment name.",
    ),
    run_name: str | None = typer.Option(None, help="Optional MLflow run name."),
    compare_settings: str | None = typer.Option(
        None,
        "--compare-settings",
        help=(
            "A/B test ALS reranking settings as 'label:key=value,...;label2:...'. "
            "Same holdout for every label. Example: "
            "'control:;diversity:popularity_penalty=0.2,diversity=0.5'."
        ),
    ),
    promote_winner: bool = typer.Option(
        False,
        "--promote-winner/--no-promote-winner",
        help=(
            "After an A/B comparison, retrain the model with the winning "
            "setting's ranking parameters and save the new artifact."
        ),
    ),
    min_quality_threshold: str | None = typer.Option(
        None,
        "--min-quality-threshold",
        help=(
            "Minimum quality thresholds for auto-promotion as 'metric=value,...'. "
            "Example: 'ndcg_at_k=0.3,precision_at_k=0.15'. "
            "Only promotes winner if all thresholds are met."
        ),
    ),
    fail_on_quality_gate: bool = typer.Option(
        False,
        "--fail-on-quality-gate/--no-fail-on-quality-gate",
        help=(
            "Exit with a non-zero status when the winning setting does not "
            "meet --min-quality-threshold, for CI quality gates."
        ),
    ),
    learn_to_rank: bool = typer.Option(
        False,
        "--learn-to-rank/--no-learn-to-rank",
        help=(
            "Fit a lightweight ranking model on the training fold and re-rank "
            "the ALS candidates, reporting the additional 'ltr' arm."
        ),
    ),
    ablations: str | None = typer.Option(
        None,
        "--ablations",
        help=(
            "Ablate each active knob of the champion ranking config "
            "'key=value,...' (e.g. 'popularity_penalty=0.2,diversity=0.5') and "
            "report per-metric impact plus knob importance."
        ),
    ),
    pareto_frontier: bool = typer.Option(
        False,
        "--pareto-frontier/--no-pareto-frontier",
        help="Sweep multi-objective trade-offs and compute the Pareto frontier.",
    ),
    report_dir: str = typer.Option(
        str(REPORTS_DIR),
        "--report-dir",
        help="Directory for the persistent ablation-importance report.",
    ),
) -> None:
    """Evaluate recommendations with ranking metrics."""
    if promote_winner and compare_settings is None:
        raise typer.BadParameter("--promote-winner requires --compare-settings.")
    if min_quality_threshold is not None and not promote_winner:
        raise typer.BadParameter("--min-quality-threshold requires --promote-winner.")
    if fail_on_quality_gate and min_quality_threshold is None:
        raise typer.BadParameter(
            "--fail-on-quality-gate requires --min-quality-threshold."
        )
    if compare_settings is not None and (compare_baseline or compare_all):
        raise typer.BadParameter(
            "--compare-settings cannot be combined with"
            " --compare-baseline or --compare-all."
        )
    if ablations is not None and (
        compare_settings is not None
        or compare_baseline
        or compare_all
        or promote_winner
        or learn_to_rank
        or pareto_frontier
    ):
        raise typer.BadParameter(
            "--ablations cannot be combined with --compare-settings,"
            " --compare-baseline, --compare-all, --promote-winner,"
            " --learn-to-rank, or --pareto-frontier."
        )
    if pareto_frontier and (
        compare_settings is not None
        or compare_baseline
        or compare_all
        or promote_winner
    ):
        raise typer.BadParameter(
            "--pareto-frontier cannot be combined with --compare-settings,"
            " --compare-baseline, --compare-all, or --promote-winner."
        )
    try:
        with tracking_run(
            enabled=track,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            tags={"workflow": "evaluation", "model_type": "implicit_als"},
        ) as tracked_run:
            tracked_run.log_params(
                {
                    "top_k": top_k,
                    "folds": folds,
                    "compare_baseline": compare_baseline,
                    "compare_all": compare_all,
                    "compare_settings": compare_settings,
                    "use_gpu": use_gpu,
                    "learn_to_rank": learn_to_rank,
                    "ablations": str(ablations),
                    "pareto_frontier": pareto_frontier,
                    "data_path": str(RAW_DATA_PATH),
                    "metadata_path": str(RAW_METADATA_PATH) if compare_all else None,
                }
            )
            df = load_and_validate_interactions(RAW_DATA_PATH)
            metadata_df = (
                load_and_validate_artist_metadata(RAW_METADATA_PATH, df)
                if compare_all
                else None
            )
            if pareto_frontier:
                pareto_report = evaluate_pareto_frontier(
                    df,
                    top_k=top_k,
                    folds=folds,
                    use_gpu=use_gpu,
                )
                tracked_run.log_dict(pareto_report, "evaluation/pareto_frontier.json")
                tracked_run.set_tags({"analysis": "pareto_frontier"})
                pareto_report_path = write_pareto_report(
                    pareto_report,
                    Path(report_dir),
                )
            elif ablations is not None:
                champion = _parse_parameter_value_dict(ablations)
                ablation_settings = build_ablation_settings(champion)
                arm_metrics = compare_parameter_settings(
                    df,
                    top_k=top_k,
                    parameter_sets=ablation_settings,
                    folds=folds,
                    use_gpu=use_gpu,
                )
                tracked_run.log_dict(arm_metrics, "evaluation/metrics.json")
                tracked_run.set_tags({"strategies": ",".join(arm_metrics.keys())})
                ablation_report_path = write_ablation_report(
                    arm_metrics,
                    Path(report_dir),
                )
            elif compare_settings is not None:
                parameter_sets = _parse_parameter_settings(compare_settings)
                comparison_metrics = compare_parameter_settings(
                    df,
                    top_k=top_k,
                    parameter_sets=parameter_sets,
                    folds=folds,
                    use_gpu=use_gpu,
                )
                tracked_run.log_dict(comparison_metrics, "evaluation/metrics.json")
                tracked_run.set_tags(
                    {"strategies": ",".join(comparison_metrics.keys())}
                )
                metrics = cast(
                    dict[str, float] | dict[str, dict[str, float]],
                    comparison_metrics,
                )
            else:
                metrics = evaluate_repeated_holdout(
                    df,
                    top_k=top_k,
                    folds=folds,
                    compare_baseline=compare_baseline,
                    compare_all=compare_all,
                    metadata_df=metadata_df,
                    use_gpu=use_gpu,
                    learn_to_rank=learn_to_rank,
                )
                tracked_run.log_metrics(metrics)
                tracked_run.log_dict(metrics, "evaluation/metrics.json")
                strategy_list = []
                if compare_all:
                    strategy_list = ["als", "popularity", "content", "hybrid"]
                elif compare_baseline:
                    strategy_list = ["als", "popularity"]
                else:
                    strategy_list = ["als"]
                if learn_to_rank:
                    strategy_list.append("ltr")
                tracked_run.set_tags({"strategies": ",".join(strategy_list)})
    except (ExperimentTrackingError, FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    if pareto_frontier:
        typer.echo(
            f"Multi-Objective Pareto Frontier Evaluation ({folds} fold(s), "
            f"top_k={top_k}):"
        )
        typer.echo(
            f"{'Label':<20} {'Weights (Rel/Div/Nov)':<24} "
            f"{'NDCG@K':<10} {'Diversity':<10} {'Novelty':<10} {'Pareto?'}"
        )
        typer.echo("-" * 84)
        for config in pareto_report["configurations"]:
            w = config["weights"]
            weights_str = (
                f"{w['relevance']:.2f} / {w['diversity']:.2f} / {w['novelty']:.2f}"
            )
            is_opt = "*" if config.get("is_pareto_optimal") else ""
            m = config["metrics"]
            typer.echo(
                f"{config['label']:<20} {weights_str:<24} "
                f"{m.get('ndcg_at_k', 0.0):<10.4f} "
                f"{m.get('intra_list_diversity', 0.0):<10.4f} "
                f"{m.get('novelty_at_k', 0.0):<10.4f} {is_opt}"
            )
        typer.echo("-" * 84)
        typer.echo(
            "* Marked configurations belong to the non-dominated Pareto frontier."
        )
        best = pareto_report.get("best_balanced_configuration")
        if best:
            bw = best["weights"]
            typer.echo(
                f"Best balanced configuration: {best['label']} "
                f"(relevance={bw['relevance']:.2f}, diversity={bw['diversity']:.2f}, "
                f"novelty={bw['novelty']:.2f})"
            )
        typer.echo(f"Pareto report written to: {pareto_report_path}")
    elif ablations is not None:
        _print_ablation_report(arm_metrics, top_k, folds)
        typer.echo(f"Ablation report written to: {ablation_report_path}")
    elif compare_settings is not None:
        typer.echo(f"Evaluation over {folds} fold(s):")
        for label, label_metrics in comparison_metrics.items():
            _print_metric_row(label, label_metrics, top_k)
        winners = select_winning_strategies(comparison_metrics)
        typer.echo("Winners by metric:")
        for metric, label in winners.items():
            typer.echo(f"  {metric}: {label}")
        best_label, wins = strategy_leaderboard(comparison_metrics)[0]
        typer.echo(f"Overall: {best_label} won {wins} of {len(winners)} metrics.")
        if promote_winner:
            if min_quality_threshold and not _check_quality_threshold(
                comparison_metrics[best_label], min_quality_threshold
            ):
                if fail_on_quality_gate:
                    typer.secho(
                        "Error: quality gate failed: winning setting does not "
                        "meet minimum thresholds.",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(code=1)
                typer.secho(
                    "Quality gate failed: winning setting does not meet "
                    "minimum thresholds. Promotion skipped.",
                    fg=typer.colors.YELLOW,
                )
            else:
                _promote_ranking_settings(best_label, parameter_sets)
    elif compare_all:
        comparison_metrics = cast(dict[str, dict[str, float]], metrics)
        typer.echo(f"Evaluation over {folds} fold(s):")
        _print_metric_row("ALS", comparison_metrics["als"], top_k)
        _print_metric_row("Popularity", comparison_metrics["popularity"], top_k)
        _print_metric_row("Content", comparison_metrics["content"], top_k)
        _print_metric_row("Hybrid", comparison_metrics["hybrid"], top_k)
        if learn_to_rank:
            _print_metric_row("LTR", comparison_metrics["ltr"], top_k)
    elif compare_baseline:
        comparison_metrics = cast(dict[str, dict[str, float]], metrics)
        typer.echo(f"Evaluation over {folds} fold(s):")
        _print_metric_row("ALS", comparison_metrics["als"], top_k)
        _print_metric_row("Popularity", comparison_metrics["popularity"], top_k)
        if learn_to_rank:
            _print_metric_row("LTR", comparison_metrics["ltr"], top_k)
    elif learn_to_rank:
        comparison_metrics = cast(dict[str, dict[str, float]], metrics)
        typer.echo(f"Evaluation over {folds} fold(s):")
        _print_metric_row("ALS", comparison_metrics["als"], top_k)
        _print_metric_row("LTR", comparison_metrics["ltr"], top_k)
    else:
        _print_metric_row("ALS", cast(dict[str, float], metrics), top_k)

    if tracked_run.enabled:
        typer.echo(f"MLflow run ID: {tracked_run.run_id}")
        typer.echo(f"MLflow tracking URI: {tracked_run.tracking_uri}")


def _print_metric_row(name: str, metrics: dict[str, float], top_k: int) -> None:
    typer.echo(f"{name}:")
    typer.echo(f"  Precision@{top_k}: {metrics['precision_at_k']:.4f}")
    typer.echo(f"  Recall@{top_k}: {metrics['recall_at_k']:.4f}")
    typer.echo(f"  MAP@{top_k}: {metrics['map_at_k']:.4f}")
    typer.echo(f"  NDCG@{top_k}: {metrics['ndcg_at_k']:.4f}")
    typer.echo(f"  Catalog coverage: {metrics['catalog_coverage']:.4f}")
    typer.echo(f"  Average popularity: {metrics['average_popularity']:.4f}")
    typer.echo(f"  Novelty@{top_k}: {metrics['novelty_at_k']:.4f}")
    typer.echo(f"  Unexpectedness@{top_k}: {metrics['unexpectedness_at_k']:.4f}")
    typer.echo(f"  Serendipity@{top_k}: {metrics['serendipity_at_k']:.4f}")
    typer.echo(f"  Explanation coverage: {metrics['explanation_coverage']:.4f}")
    typer.echo(f"  Intra-list diversity: {metrics['intra_list_diversity']:.4f}")


def _print_ablation_report(
    arm_metrics: dict[str, dict[str, float]],
    top_k: int,
    folds: int,
) -> None:
    """Print the champion-first arm rows and the knob importance ranking."""
    typer.echo(f"Ablation over {folds} fold(s):")
    champion = arm_metrics["champion"]
    _print_metric_row("Champion", champion, top_k)
    for label, label_metrics in arm_metrics.items():
        if label == "champion":
            continue
        _print_metric_row(label, label_metrics, top_k)
    _, ranking = ablation_importances(arm_metrics)
    typer.echo("Knob importance (absolute per-metric impact vs champion):")
    for knob, impact in ranking:
        typer.echo(f"  {knob}: {impact:.4f}")


def _promote_ranking_settings(
    label: str,
    parameter_sets: dict[str, dict[str, float | int | bool | str]],
) -> None:
    """Retrain the model with the winning setting's ranking parameters."""
    typer.echo("Promoting the winning setting into the serving bundle...")
    params = ranking_params_for_training(parameter_sets[label])
    try:
        train_and_save_model(
            popularity_penalty=float(params.get("popularity_penalty", 0.0)),
            diversity=float(params.get("diversity", 0.0)),
            include_listened=bool(params.get("include_listened", False)),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        typer.secho(
            f"Error: promotion failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error
    typer.echo(f"Promoted '{label}' ranking settings into the artifact.")


def _parse_parameter_settings(
    text: str,
) -> dict[str, dict[str, float | int | bool | str]]:
    """Parse 'label:key=value,key2=value2;label2:...' into labeled parameter sets."""
    parameter_sets: dict[str, dict[str, float | int | bool | str]] = {}
    for item in text.split(";"):
        label, separator, config = item.partition(":")
        label = label.strip()
        if not label or not separator:
            raise ValueError(
                f"Invalid parameter setting '{item}'. Expected 'label:key=value,...'."
            )
        if label in parameter_sets:
            raise ValueError(f"Duplicate parameter setting label: {label}.")
        kwargs: dict[str, float | int | bool | str] = {}
        if config.strip():
            for pair in config.split(","):
                key, equals, raw_value = pair.partition("=")
                key = key.strip()
                raw_value = raw_value.strip()
                if not key or not equals:
                    raise ValueError(f"Invalid pair '{pair}'. Expected 'key=value'.")
                kwargs[key] = _parse_parameter_value(raw_value)
        parameter_sets[label] = kwargs
    return parameter_sets


def _parse_parameter_value(raw: str) -> float | int | bool | str:
    """Parse a setting value as bool, then int, then float, else keep a string."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_parameter_value_dict(text: str) -> dict[str, float | int | bool | str]:
    """Parse 'key=value,key2=value2' into a setting dictionary."""
    settings: dict[str, float | int | bool | str] = {}
    if not text.strip():
        raise ValueError("--ablations requires a non-empty 'key=value,...' config.")
    for pair in text.split(","):
        key, equals, raw_value = pair.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not equals:
            raise ValueError(f"Invalid pair '{pair}'. Expected 'key=value'.")
        settings[key] = _parse_parameter_value(raw_value)
    return settings


def _check_quality_threshold(metrics: dict[str, float], threshold_str: str) -> bool:
    """Check if all metrics meet their minimum thresholds.

    Args:
        metrics: Dictionary of metric names to values.
        threshold_str: Comma-separated 'metric=value' pairs, e.g.
            'ndcg_at_k=0.3,precision_at_k=0.15'.

    Returns:
        True if all thresholds are met, False otherwise.
    """
    for pair in threshold_str.split(","):
        key, equals, raw_value = pair.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not equals:
            raise ValueError(f"Invalid threshold '{pair}'. Expected 'metric=value'.")
        try:
            threshold = float(raw_value)
        except ValueError as err:
            raise ValueError(
                f"Invalid threshold value '{raw_value}' for metric '{key}'. "
                "Must be a number."
            ) from err
        if key not in metrics:
            raise ValueError(f"Unknown metric '{key}' in threshold specification.")
        if metrics[key] < threshold:
            return False
    return True


@app.command()
def demo(use_gpu: bool = DEFAULT_USE_GPU) -> None:
    """Train when needed and show example recommendations."""
    try:
        if not ARTIFACT_BUNDLE_PATH.exists():
            typer.echo("No saved model found. Training on the sample dataset first.")
            train_and_save_model(use_gpu=use_gpu)

        service = RecommenderService.from_artifacts()
        response = service.recommend_user(user_id="user_1", top_k=5, explain=True)
        similar_response = service.content_similar_artists(
            artist_id="artist_2",
            top_k=5,
            explain=True,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    except RuntimeError as error:
        typer.secho(
            f"Error: demo failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo("Recommendations for user_1:")
    typer.echo(f"Strategy: {response['strategy']}")
    typer.echo(format_recommendations(response["recommendations"]))
    typer.echo("")
    typer.echo("Content-similar artists for artist_2:")
    typer.echo(format_recommendations(similar_response["similar_artists"]))


@app.command()
def ablation_summary(
    report_dir: str = typer.Option(
        str(REPORTS_DIR),
        "--report-dir",
        help="Directory of persisted ablation reports to aggregate.",
    ),
    summary_path: str = typer.Option(
        str(REPORTS_DIR / "ablation_summary.json"),
        "--summary-path",
        help="Where to write the aggregated summary JSON report.",
    ),
) -> None:
    """Aggregate ablation-importance reports across runs or datasets."""
    try:
        summary = aggregate_ablation_reports(Path(report_dir))
        written = write_ablation_summary_report(summary, Path(summary_path))
    except (FileNotFoundError, ValueError, OSError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(
        f"Aggregated {summary['reports_loaded']} ablation report(s) from {report_dir}:"
    )
    typer.echo("Knob importance by mean total impact:")
    for item in summary["ranking"]:
        knob = summary["knobs"][item["knob"]]
        typer.echo(
            f"  {item['knob']}: mean={item['mean_impact']:.4f} "
            f"std={knob['std_impact']:.4f} runs={knob['count']}"
        )
    typer.echo(f"Aggregated summary written to: {written}")


def _get_spotify_client() -> tuple[object, SpotifyConfig]:
    """Get Spotify client and config, handling missing credentials gracefully."""
    if not SPOTIFY_AVAILABLE:
        typer.secho(
            "Error: Spotify integration not installed. "
            "Install with: uv sync --extra spotify",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        config = SpotifyConfig.from_env()
    except ValueError as error:
        typer.secho(
            f"Error: {error}. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error
    return create_spotify_client(config), config


@app.command()
def spotify_artist(
    artist_id: str = typer.Argument(..., help="Spotify artist ID."),
) -> None:
    """Fetch and display artist information from Spotify."""
    try:
        client, _ = _get_spotify_client()
        artist = fetch_artist(client, artist_id)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Artist: {artist.name}")
    typer.echo(f"Spotify ID: {artist.id}")
    typer.echo(f"Genres: {', '.join(artist.genres) if artist.genres else 'N/A'}")
    typer.echo(f"Popularity: {artist.popularity}")
    typer.echo(f"Followers: {artist.followers:,}")
    if artist.external_urls:
        typer.echo(f"Spotify URL: {artist.external_urls.get('spotify', 'N/A')}")


@app.command()
def spotify_artists(
    artist_ids: str = typer.Option(
        ..., "--ids", help="Comma-separated Spotify artist IDs."
    ),
) -> None:
    """Fetch and display multiple artists from Spotify."""
    ids = [aid.strip() for aid in artist_ids.split(",") if aid.strip()]
    if not ids:
        typer.secho("Error: No artist IDs provided.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        client, _ = _get_spotify_client()
        artists = fetch_artists(client, ids)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    for artist in artists:
        typer.echo(f"  {artist.name} ({artist.id}) - Popularity: {artist.popularity}")


@app.command()
def spotify_artist_top_tracks(
    artist_id: str = typer.Argument(..., help="Spotify artist ID."),
    country: str = typer.Option("US", help="Country code for top tracks."),
) -> None:
    """Fetch and display an artist's top tracks from Spotify."""
    try:
        client, _ = _get_spotify_client()
        tracks = fetch_artist_top_tracks(client, artist_id, country=country)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Top tracks for artist {artist_id} (country: {country}):")
    for i, track in enumerate(tracks, 1):
        artists_str = ", ".join(track.artist_names)
        typer.echo(
            f"  {i}. {track.name} by {artists_str} (popularity: {track.popularity})"
        )


@app.command()
def spotify_related_artists(
    artist_id: str = typer.Argument(..., help="Spotify artist ID."),
) -> None:
    """Fetch and display related artists for a given artist."""
    try:
        client, _ = _get_spotify_client()
        artists = get_artist_related_artists(client, artist_id)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Related artists for {artist_id}:")
    for i, artist in enumerate(artists, 1):
        typer.echo(
            f"  {i}. {artist.name} ({artist.id}) - Popularity: {artist.popularity}"
        )


@app.command()
def spotify_search_artists(
    query: str = typer.Argument(..., help="Search query for artist name."),
    limit: int = typer.Option(20, help="Maximum number of results."),
) -> None:
    """Search for artists by name on Spotify."""
    try:
        client, _ = _get_spotify_client()
        artists = search_artists(client, query, limit=limit)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Search results for '{query}':")
    for i, artist in enumerate(artists, 1):
        typer.echo(
            f"  {i}. {artist.name} ({artist.id}) - Popularity: {artist.popularity}"
        )


@app.command()
def spotify_search_tracks(
    query: str = typer.Argument(..., help="Search query for track name."),
    limit: int = typer.Option(20, help="Maximum number of results."),
) -> None:
    """Search for tracks by name on Spotify."""
    try:
        client, _ = _get_spotify_client()
        tracks = search_tracks(client, query, limit=limit)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Search results for '{query}':")
    for i, track in enumerate(tracks, 1):
        artists_str = ", ".join(track.artist_names)
        typer.echo(
            f"  {i}. {track.name} by {artists_str} (popularity: {track.popularity})"
        )


@app.command()
def spotify_audio_features(
    track_ids: str = typer.Option(
        ..., "--ids", help="Comma-separated Spotify track IDs."
    ),
) -> None:
    """Fetch and display audio features for tracks."""
    ids = [tid.strip() for tid in track_ids.split(",") if tid.strip()]
    if not ids:
        typer.secho("Error: No track IDs provided.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        client, _ = _get_spotify_client()
        features = fetch_audio_features(client, ids)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    for i, feature in enumerate(features):
        if feature:
            typer.echo(
                f"  {ids[i]}: danceability={feature.danceability:.3f}, "
                f"energy={feature.energy:.3f}, valence={feature.valence:.3f}, "
                f"tempo={feature.tempo:.1f}, key={feature.key}, mode={feature.mode}"
            )
        else:
            typer.echo(f"  {ids[i]}: No audio features available")


@app.command()
def spotify_import_catalog(
    artist_ids: str = typer.Option(
        ..., "--artist-ids", help="Comma-separated Spotify artist IDs."
    ),
    country: str = typer.Option("US", help="Country code for top tracks."),
    output: str = typer.Option(
        str(DATA_DIR / "raw" / "spotify_track_metadata.csv"),
        "--output",
        help="Where to write the track metadata CSV.",
    ),
) -> None:
    """Import a track metadata catalog from Spotify top tracks.

    Fetches each artist's top tracks plus audio features and writes a CSV
    matching the track metadata contract, ready for track recommendations.
    """
    ids = [aid.strip() for aid in artist_ids.split(",") if aid.strip()]
    if not ids:
        typer.secho("Error: No artist IDs provided.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        client, _ = _get_spotify_client()
        all_tracks = []
        for artist_id in ids:
            all_tracks.extend(
                fetch_artist_top_tracks(client, artist_id, country=country)
            )
        seen: set[str] = set()
        unique_tracks = []
        for track in all_tracks:
            if track.id not in seen:
                seen.add(track.id)
                unique_tracks.append(track)
        features = fetch_audio_features(client, [track.id for track in unique_tracks])
        frame = build_track_metadata_frame(unique_tracks, features)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    except Exception as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    skipped = len(unique_tracks) - len(frame)
    typer.echo(f"Imported {len(frame)} tracks from {len(ids)} artist(s).")
    if skipped:
        typer.echo(f"Skipped {skipped} track(s) without audio features.")
    typer.echo(f"Wrote track metadata to: {output_path}")


@app.command()
def prepare_track_data(
    min_user_interactions: int = DEFAULT_MIN_USER_INTERACTIONS,
    min_track_interactions: int = DEFAULT_MIN_ARTIST_INTERACTIONS,
) -> None:
    """Validate track sample data, build interaction matrix, and save mappings."""
    df = load_and_validate_track_interactions(RAW_TRACK_DATA_PATH)
    metadata_df = load_and_validate_track_metadata(RAW_TRACK_METADATA_PATH, df)

    typer.echo("Track data prepared successfully.")
    typer.echo(f"Users: {df['user_id'].nunique()}")
    typer.echo(f"Tracks: {df['track_id'].nunique()}")
    typer.echo(f"Artists: {df['artist_id'].nunique()}")
    typer.echo(f"Interactions: {len(df)}")
    typer.echo(f"Track metadata rows: {len(metadata_df)}")


@app.command()
def popular_tracks(top_k: int = DEFAULT_TOP_K) -> None:
    """Show globally popular tracks from the track data."""
    try:
        df = load_and_validate_track_interactions(RAW_TRACK_DATA_PATH)
        track_stats = build_track_stats(df)
        recommendations = rank_tracks_by_popularity(track_stats, top_k=top_k)

        metadata_df = load_and_validate_track_metadata(RAW_TRACK_METADATA_PATH, df)
        lookup = {
            row.track_id: row.track_name for row in metadata_df.itertuples(index=False)
        }
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Popular tracks:")
    if not recommendations:
        typer.echo("  No tracks available.")
    else:
        for i, rec in enumerate(recommendations, 1):
            track_name = lookup.get(rec["track_id"], rec["track_id"])
            typer.echo(f"  {i}. {track_name} (plays: {rec['score']:.0f})")


@app.command()
def track_recommendations(
    user_id: str = typer.Option(..., help="User ID for recommendations."),
    top_k: int = DEFAULT_TOP_K,
    include_listened: bool = typer.Option(
        False,
        "--include-listened/--exclude-listened",
        help="Include or exclude listened tracks in the recommendations.",
    ),
    popularity_penalty: float = typer.Option(
        0.0,
        "--popularity-penalty",
        min=0.0,
        max=1.0,
        help="Penalize globally popular tracks (0.0 to 1.0).",
    ),
    diversity: float = typer.Option(
        0.0,
        "--diversity",
        min=0.0,
        max=1.0,
        help="Diversify recommendations by audio features (0.0 to 1.0).",
    ),
    novelty_weight: float = typer.Option(
        0.0,
        "--novelty-weight",
        min=0.0,
        max=1.0,
        help="Weight for novelty in multi-objective ranking (0.0 to 1.0).",
    ),
    explain: bool = typer.Option(
        False,
        "--explain/--no-explain",
        help="Show why each track is recommended.",
    ),
    method: str = typer.Option(
        "similarity",
        "--method",
        help="Recommendation method: similarity or hybrid.",
    ),
    content_weight: float = typer.Option(
        DEFAULT_CONTENT_WEIGHT,
        "--content-weight",
        min=0.0,
        max=1.0,
        help="Balance hybrid tracks between artist taste and audio features.",
    ),
    ltr: bool = typer.Option(
        False,
        "--ltr/--no-ltr",
        help="Re-rank track candidates with the learning-to-rank model.",
    ),
) -> None:
    """Recommend tracks for a user using track similarity (or hybrid)."""
    if method not in ("similarity", "hybrid"):
        raise typer.BadParameter("method must be one of: similarity, hybrid.")
    try:
        import numpy as np

        df = load_and_validate_track_interactions(RAW_TRACK_DATA_PATH)
        metadata_df = load_and_validate_track_metadata(RAW_TRACK_METADATA_PATH, df)

        # Build user-track matrix
        user_track_matrix = df.pivot_table(
            index="user_id",
            columns="track_id",
            values="play_count",
            fill_value=0,
        )

        # Build track similarity matrix from audio features
        feature_df, feature_names = build_track_content_matrix(metadata_df)
        track_ids = feature_df.index.tolist()
        track_id_to_index = {tid: i for i, tid in enumerate(track_ids)}

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        track_similarity_matrix = cosine_similarity(feature_df.values)

        track_stats = build_track_stats(df)
        track_name_lookup = dict(
            zip(
                metadata_df["track_id"].astype(str),
                metadata_df["track_name"].astype(str),
                strict=True,
            )
        )
        track_artist_name_lookup = dict(
            zip(
                metadata_df["track_id"].astype(str),
                metadata_df["artist_name"].astype(str),
                strict=True,
            )
        )

        artist_taste_per_track = None
        if method == "hybrid":
            from music_recommender.artifacts import load_artifact

            artifact = load_artifact()
            if (
                artifact.taste_model is not None
                and artifact.taste_user_id_to_index is not None
                and artifact.taste_artist_id_to_index is not None
            ):
                taste_model = artifact.taste_model
                user_id_to_index = artifact.taste_user_id_to_index
                artist_id_to_index = artifact.taste_artist_id_to_index
                logger.info("taste_model_source=artifact")
            else:
                taste_model, user_id_to_index, artist_id_to_index = (
                    train_track_artist_taste(df)
                )
                logger.info("taste_model_source=trained")
            artist_scores = artist_taste_scores_for_user(
                taste_model, user_id_to_index, artist_id_to_index, user_id
            )
            if artist_scores is None:
                raise ValueError(
                    f"Unknown user_id for hybrid track recommendations: {user_id}"
                )
            track_artists = dict(
                zip(
                    metadata_df["track_id"].astype(str),
                    metadata_df["artist_id"].astype(str),
                    strict=True,
                )
            )
            artist_taste_per_track = artist_affinity_to_track_scores(
                artist_scores=artist_scores,
                artist_id_to_index=artist_id_to_index,
                track_id_to_index=track_id_to_index,
                track_artists=track_artists,
            )

        # Get recommendations
        recommendations = recommend_tracks_for_user(
            user_id=user_id,
            user_track_matrix=user_track_matrix,
            track_similarity_matrix=track_similarity_matrix,
            track_id_to_index=track_id_to_index,
            top_k=top_k,
            include_listened=include_listened,
            track_stats=track_stats,
            feature_matrix=np.asarray(feature_df.values, dtype=float),
            popularity_penalty=popularity_penalty,
            diversity=diversity,
            novelty_weight=novelty_weight,
            explain=explain,
            track_name_lookup=track_name_lookup,
            track_artist_lookup=track_artist_name_lookup,
            artist_taste_per_track=artist_taste_per_track,
            content_weight=content_weight,
        )

        if ltr:
            from music_recommender.artifacts import load_artifact
            from music_recommender.ltr import (
                rank_tracks_with_ltr,
                train_track_ltr_ranker,
            )
            from music_recommender.tracks import build_track_serving_resources

            track_resources = build_track_serving_resources(df, metadata_df)
            ranker = None
            try:
                artifact = load_artifact()
                ranker = getattr(artifact, "track_ltr_model", None)
            except Exception:
                pass
            if ranker is None:
                ranker = train_track_ltr_ranker(
                    train_df=df,
                    resources=track_resources,
                )
            recommendations = rank_tracks_with_ltr(
                ranker,
                recommendations=recommendations,
                user_id=user_id,
                resources=track_resources,
                top_k=top_k,
            )

    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    ltr_suffix = ", LTR" if ltr else ""
    typer.echo(f"Track recommendations for {user_id} (method: {method}{ltr_suffix}):")
    if not recommendations:
        typer.echo("  No recommendations available.")
    else:
        for i, rec in enumerate(recommendations, 1):
            track_id = rec["track_id"]
            track_name = metadata_df.loc[
                metadata_df["track_id"] == track_id, "track_name"
            ].values[0]
            artist_name = metadata_df.loc[
                metadata_df["track_id"] == track_id, "artist_name"
            ].values[0]
            typer.echo(
                f"  {i}. {track_name} by {artist_name} (score: {rec['score']:.4f})"
            )
            for reason in rec.get("reasons") or []:
                typer.echo(f"     - {reason}")


@app.command()
def similar_tracks(
    track_id: str = typer.Option(..., help="Track ID to find similar tracks for."),
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Find tracks similar to a given track using audio features."""
    try:
        df = load_and_validate_track_interactions(RAW_TRACK_DATA_PATH)
        metadata_df = load_and_validate_track_metadata(RAW_TRACK_METADATA_PATH, df)

        feature_df, feature_names = build_track_content_matrix(metadata_df)
        track_ids = feature_df.index.tolist()
        track_id_to_index = {tid: i for i, tid in enumerate(track_ids)}

        from sklearn.metrics.pairwise import cosine_similarity

        track_similarity_matrix = cosine_similarity(feature_df.values)

        similar = get_similar_tracks(
            track_id=track_id,
            track_similarity_matrix=track_similarity_matrix,
            track_id_to_index=track_id_to_index,
            top_k=top_k,
        )

    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Tracks similar to {track_id}:")
    if not similar:
        typer.echo("  No similar tracks found.")
    else:
        for i, rec in enumerate(similar, 1):
            track_id = rec["track_id"]
            track_name = metadata_df.loc[
                metadata_df["track_id"] == track_id, "track_name"
            ].values[0]
            artist_name = metadata_df.loc[
                metadata_df["track_id"] == track_id, "artist_name"
            ].values[0]
            typer.echo(
                f"  {i}. {track_name} by {artist_name} (score: {rec['score']:.4f})"
            )


@app.command()
def evaluate_tracks(
    top_k: int = DEFAULT_TOP_K,
    folds: int = 1,
    include_listened: bool = typer.Option(
        False,
        "--include-listened/--exclude-listened",
        help="Include or exclude listened tracks when evaluating.",
    ),
    compare_baseline: bool = typer.Option(
        False,
        "--compare-baseline/--no-compare-baseline",
        help="Compare track similarity against a global-popularity baseline.",
    ),
    compare_all: bool = typer.Option(
        False,
        "--compare-all/--no-compare-all",
        help="Compare similarity, popularity, and hybrid strategies in a single pass.",
    ),
    compare_settings: str | None = typer.Option(
        None,
        "--compare-settings",
        help=(
            "A/B test track reranking settings as 'label:key=value,...;label2:...'. "
            "Same holdout for every label. Example: "
            "'control:;penalty:popularity_penalty=0.2,diversity=0.5'."
        ),
    ),
    ablations: str | None = typer.Option(
        None,
        "--ablations",
        help=(
            "Ablate each active knob of the champion ranking config "
            "'key=value,...' (e.g. 'popularity_penalty=0.2,diversity=0.5') and "
            "report per-metric impact plus knob importance."
        ),
    ),
    popularity_penalty: float = typer.Option(
        0.0,
        "--popularity-penalty",
        min=0.0,
        max=1.0,
        help="Penalize globally popular tracks during evaluation (0.0 to 1.0).",
    ),
    diversity: float = typer.Option(
        0.0,
        "--diversity",
        min=0.0,
        max=1.0,
        help="Diversify recommendations by audio features (0.0 to 1.0).",
    ),
    method: str = typer.Option(
        "similarity",
        "--method",
        help="Evaluation method: similarity or hybrid.",
    ),
    content_weight: float = typer.Option(
        DEFAULT_CONTENT_WEIGHT,
        "--content-weight",
        min=0.0,
        max=1.0,
        help="Balance hybrid tracks between artist taste and audio features.",
    ),
    learn_to_rank: bool = typer.Option(
        False,
        "--learn-to-rank/--no-learn-to-rank",
        help=(
            "Fit a lightweight ranking model on the training fold and re-rank "
            "the track candidates, reporting the additional 'ltr' arm."
        ),
    ),
    report_name: str | None = typer.Option(
        None,
        "--report-name",
        help=(
            "Name for the evaluation JSON report (stored under the reports directory)."
        ),
    ),
    report_dir: str | None = typer.Option(
        None,
        "--report-dir",
        help="Directory for the persistent evaluation or ablation report.",
    ),
) -> None:
    """Evaluate track similarity with repeated per-user holdout splits."""
    if compare_settings is not None and (compare_baseline or compare_all):
        raise typer.BadParameter(
            "--compare-settings cannot be combined with"
            " --compare-baseline or --compare-all."
        )
    if ablations is not None and (
        compare_settings is not None or compare_baseline or compare_all or learn_to_rank
    ):
        raise typer.BadParameter(
            "--ablations cannot be combined with --compare-settings,"
            " --compare-baseline, --compare-all, or --learn-to-rank."
        )
    resolved_report_dir = Path(report_dir) if report_dir is not None else REPORTS_DIR
    try:
        df = load_and_validate_track_interactions(RAW_TRACK_DATA_PATH)
        metadata_df = load_and_validate_track_metadata(RAW_TRACK_METADATA_PATH, df)
        compare_metrics: dict[str, dict[str, float]] | None = None
        single_metrics: dict[str, float] | dict[str, dict[str, float]] | None = None
        ablation_arm_metrics: dict[str, dict[str, float]] | None = None
        ablation_report_path: Path | None = None
        if ablations is not None:
            champion = _parse_parameter_value_dict(ablations)
            ablation_arm_metrics, _, _ = ablate_track_parameter_settings(
                df,
                metadata_df,
                champion=champion,
                top_k=top_k,
                folds=folds,
                method=method,
                content_weight=content_weight,
            )
            ablation_report_path = write_ablation_report(
                ablation_arm_metrics,
                resolved_report_dir,
                report_name=report_name or "track_ablation_importance",
            )
        elif compare_settings is not None:
            parameter_sets = _parse_parameter_settings(compare_settings)
            compare_metrics = compare_track_parameter_settings(
                df,
                metadata_df,
                top_k=top_k,
                parameter_sets=parameter_sets,
                folds=folds,
            )
        else:
            single_metrics = evaluate_track_holdout(
                df,
                metadata_df,
                top_k=top_k,
                folds=folds,
                include_listened=include_listened,
                compare_baseline=compare_baseline,
                compare_all=compare_all,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                method=method,
                content_weight=content_weight,
                learn_to_rank=learn_to_rank,
            )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    if ablations is not None:
        assert ablation_arm_metrics is not None
        typer.echo(f"Ablation over {folds} fold(s):")
        champion_metrics = ablation_arm_metrics["champion"]
        _print_track_metric_row("Champion", champion_metrics, top_k)
        for label, label_metrics in ablation_arm_metrics.items():
            if label == "champion":
                continue
            _print_track_metric_row(label, label_metrics, top_k)
        _, ranking = ablation_importances(ablation_arm_metrics)
        typer.echo("Knob importance (absolute per-metric impact vs champion):")
        for knob, impact in ranking:
            typer.echo(f"  {knob}: {impact:.4f}")
        typer.echo(f"Ablation report written to: {ablation_report_path}")
    elif compare_metrics is not None:
        typer.echo(f"Track evaluation over {folds} fold(s):")
        for label, label_metrics in compare_metrics.items():
            _print_track_metric_row(label, label_metrics, top_k)
        winners = select_winning_strategies(compare_metrics)
        typer.echo("Winners by metric:")
        for metric, label in winners.items():
            typer.echo(f"  {metric}: {label}")
        best_label, wins = strategy_leaderboard(compare_metrics)[0]
        typer.echo(f"Overall: {best_label} won {wins} of {len(winners)} metrics.")
    elif compare_all:
        typer.echo(f"Track evaluation over {folds} fold(s):")
        arm_metrics = cast(dict[str, dict[str, float]], single_metrics)
        for arm in ("similarity", "popularity", "hybrid"):
            _print_track_metric_row(arm.title(), arm_metrics[arm], top_k)
        if learn_to_rank:
            _print_track_metric_row("LTR", arm_metrics["ltr"], top_k)
    elif compare_baseline:
        typer.echo(f"Track evaluation over {folds} fold(s):")
        arm_metrics = cast(dict[str, dict[str, float]], single_metrics)
        for arm in ("similarity", "popularity"):
            _print_track_metric_row(arm.title(), arm_metrics[arm], top_k)
        if learn_to_rank:
            _print_track_metric_row("LTR", arm_metrics["ltr"], top_k)
    elif learn_to_rank:
        typer.echo(f"Track evaluation over {folds} fold(s):")
        arm_metrics = cast(dict[str, dict[str, float]], single_metrics)
        _print_track_metric_row("Similarity", arm_metrics["similarity"], top_k)
        _print_track_metric_row("LTR", arm_metrics["ltr"], top_k)
    else:
        typer.echo(f"Track evaluation over {folds} fold(s):")
        _print_track_metric_row(
            "Similarity", cast(dict[str, float], single_metrics), top_k, header=False
        )
    if report_name and ablations is None:
        report_data = cast(
            dict[str, float] | dict[str, dict[str, float]],
            compare_metrics if compare_metrics is not None else single_metrics,
        )
        written = write_track_report(
            report_data,
            resolved_report_dir,
            top_k=top_k,
            folds=folds,
            report_name=report_name,
        )
        typer.echo(f"Wrote track evaluation report to: {written}")


def _print_track_metric_row(
    name: str, metrics: dict[str, float], top_k: int, header: bool = True
) -> None:
    """Print one labeled row of track evaluation metrics."""
    if header:
        typer.echo(f"{name}:")
    typer.echo(f"  Precision@{top_k}: {metrics['precision_at_k']:.4f}")
    typer.echo(f"  Recall@{top_k}: {metrics['recall_at_k']:.4f}")
    typer.echo(f"  MAP@{top_k}: {metrics['map_at_k']:.4f}")
    typer.echo(f"  NDCG@{top_k}: {metrics['ndcg_at_k']:.4f}")
    typer.echo(f"  Catalog coverage: {metrics['catalog_coverage']:.4f}")
    typer.echo(f"  Average popularity: {metrics['average_popularity']:.4f}")
    typer.echo(f"  Novelty@{top_k}: {metrics['novelty_at_k']:.4f}")
    typer.echo(f"  Serendipity@{top_k}: {metrics['serendipity_at_k']:.4f}")
    typer.echo(f"  Unexpectedness@{top_k}: {metrics['unexpectedness_at_k']:.4f}")
    typer.echo(f"  Intra-list diversity: {metrics['intra_list_diversity']:.4f}")
    typer.echo(f"  Explanation coverage: {metrics['explanation_coverage']:.4f}")


@app.command()
def evaluate_surfaces(
    top_k: int = DEFAULT_TOP_K,
    folds: int = 1,
    compare_all: bool = False,
    report_name: str | None = typer.Option(
        None,
        "--report-name",
        help="Name for the surface comparison JSON report.",
    ),
    report_dir: str | None = typer.Option(
        None,
        "--report-dir",
        help="Directory for the persistent surface comparison report.",
    ),
) -> None:
    """Evaluate and compare artist and track recommendation surfaces side by side."""
    resolved_report_dir = Path(report_dir) if report_dir is not None else REPORTS_DIR
    try:
        df = load_and_validate_interactions(RAW_DATA_PATH)
        metadata_df = (
            load_and_validate_artist_metadata(RAW_METADATA_PATH, df)
            if compare_all
            else None
        )
        track_df = load_and_validate_track_interactions(RAW_TRACK_DATA_PATH)
        track_metadata_df = load_and_validate_track_metadata(
            RAW_TRACK_METADATA_PATH, track_df
        )

        artist_metrics = evaluate_repeated_holdout(
            df=df,
            top_k=top_k,
            folds=folds,
            compare_baseline=False,
            compare_all=compare_all,
            metadata_df=metadata_df,
            use_gpu=DEFAULT_USE_GPU,
        )
        track_metrics = evaluate_track_holdout(
            df=track_df,
            metadata_df=track_metadata_df,
            top_k=top_k,
            folds=folds,
            compare_baseline=False,
            compare_all=compare_all,
        )
        comparison = compare_surface_metrics(
            artist_metrics=cast(dict[str, Any], artist_metrics),
            track_metrics=cast(dict[str, Any], track_metrics),
            top_k=top_k,
            folds=folds,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Cross-surface evaluation over {folds} fold(s) (top_k={top_k}):")
    typer.echo(f"{'Metric':<25} {'Artist (ALS)':>14} {'Track (Sim)':>14} {'Delta':>10}")
    typer.echo("-" * 65)
    for metric, stats in comparison["comparison"].items():
        typer.echo(
            f"{metric:<25} {stats['artist']:>14.4f} "
            f"{stats['track']:>14.4f} {stats['delta']:>+10.4f}"
        )
    if compare_all:
        typer.echo("\nDetailed surface arm breakdowns:")
        typer.echo("Artist surface arms:")
        for arm, arm_metrics in comparison["artist"].items():
            _print_metric_row(arm.upper(), arm_metrics, top_k)
        typer.echo("Track surface arms:")
        for arm, arm_metrics in comparison["track"].items():
            _print_track_metric_row(arm.title(), arm_metrics, top_k)

    written = write_surface_comparison_report(
        comparison,
        resolved_report_dir,
        report_name=report_name,
    )
    typer.echo(f"Surface comparison report written to: {written}")


@app.command()
def simulate_bandit(
    top_k: int = DEFAULT_TOP_K,
    rounds: int = 50,
    seed: int = 42,
    arms: str = ",".join(DEFAULT_COLD_START_ARMS),
    holdout_ratio: float = 0.25,
    alpha: float = 1.0,
    policy_type: str = typer.Option(
        "linucb",
        "--policy-type",
        help="Bandit policy algorithm: 'linucb' or 'thompson_sampling'.",
    ),
    alpha_decay: float = typer.Option(
        0.0,
        "--alpha-decay",
        help="Exploration decay rate lambda >= 0 for alpha(t) = alpha/(1 + lambda*t).",
    ),
    gamma: float = typer.Option(
        1.0,
        "--gamma",
        help="Recency discount factor gamma in range (0, 1].",
    ),
    context_features: str | None = typer.Option(
        None,
        "--context-features",
        help=(
            "Comma-separated bandit context feature names, e.g. "
            "'log_plays,mean_popularity_rank'. Overrides the "
            "MUSIC_RECOMMENDER_CONTEXT_FEATURES env var and any persisted "
            "bandit_context_features.json config."
        ),
    ),
    compare_policies: bool = typer.Option(
        False,
        "--compare-policies",
        help="Run comparative multi-policy benchmarking on identical holdouts.",
    ),
    policies: str = typer.Option(
        "linucb,thompson_sampling,epsilon_greedy",
        "--policies",
        help=(
            "Comma-separated policy types to benchmark when "
            "--compare-policies is set."
        ),
    ),
    report_name: str | None = typer.Option(
        None,
        "--report-name",
        help="Name for the bandit simulation JSON report.",
    ),
    report_dir: str | None = typer.Option(
        None,
        "--report-dir",
        help="Directory for the persistent bandit simulation report.",
    ),
    from_state: str | None = typer.Option(
        None,
        "--from-state",
        help="Path to a persisted bandit state to resume learning from (prior).",
    ),
    write_state: str | None = typer.Option(
        None,
        "--write-state",
        help="Path to persist the trained bandit state after the simulation.",
    ),
) -> None:
    """Simulate a contextual cold-start bandit against the current fallback."""
    resolved_report_dir = Path(report_dir) if report_dir is not None else REPORTS_DIR
    try:
        df = load_and_validate_interactions(RAW_DATA_PATH)
        selected_arms = tuple(arm.strip() for arm in arms.split(",") if arm.strip())
        features_arg = (
            tuple(
                feature.strip()
                for feature in context_features.split(",")
                if feature.strip()
            )
            if context_features is not None
            else None
        )
        resolved_features = resolve_context_features(
            features=features_arg,
            env=os.getenv(CONTEXT_FEATURES_ENV_VAR),
            config_path=BANDIT_CONTEXT_FEATURES_PATH,
        )

        if compare_policies:
            policy_names = tuple(
                p.strip() for p in policies.split(",") if p.strip()
            )
            policy_specs = [
                {
                    "name": p,
                    "policy_type": p,
                    "alpha": alpha if p != "epsilon_greedy" else min(alpha, 0.5),
                    "alpha_decay": alpha_decay,
                    "gamma": gamma,
                }
                for p in policy_names
            ]
            comparison_report = compare_bandit_simulation_policies(
                df,
                policies=policy_specs,
                top_k=top_k,
                rounds=rounds,
                seed=seed,
                arms=selected_arms,
                holdout_ratio=holdout_ratio,
                context_features=resolved_features,
            )
            typer.echo(
                f"Multi-Policy Contextual Bandit Benchmark "
                f"(rounds={rounds}, top_k={top_k}):"
            )
            typer.echo(f"Context features: {', '.join(resolved_features)}")
            typer.echo(
                f"{'Rank':<6} {'Policy':<20} {'Cum. Reward':>12} "
                f"{'Mean Reward':>12} {'Regret':>10} {'Win Rate':>10}"
            )
            typer.echo("-" * 74)
            leaderboard = comparison_report["summary"]["leaderboard"]
            for rank, item in enumerate(leaderboard, start=1):
                p_name = item["name"]
                p_stats = comparison_report["policies"][p_name]
                typer.echo(
                    f"{rank:<6} {p_name:<20} "
                    f"{p_stats['cumulative_reward']:>12.4f} "
                    f"{p_stats['mean_reward']:>12.4f} "
                    f"{p_stats['regret']:>10.4f} "
                    f"{item['win_rate']:>9.1%}"
                )
            typer.echo("-" * 74)
            champ_name = comparison_report["summary"]["champion"]
            champ_rew = comparison_report["policies"][champ_name]["cumulative_reward"]
            typer.echo(
                f"Best performing policy: {champ_name} "
                f"(cumulative_reward={champ_rew:.4f})"
            )
            written = write_bandit_comparison_report(
                comparison_report,
                resolved_report_dir,
                report_name=report_name,
            )
            typer.echo(f"Bandit comparison report written to: {written}")
            return

        initial_state = (
            load_bandit_state(from_state) if from_state is not None else None
        )
        report = simulate_cold_start_exploration(
            df,
            top_k=top_k,
            rounds=rounds,
            seed=seed,
            arms=selected_arms,
            holdout_ratio=holdout_ratio,
            alpha=alpha,
            alpha_decay=alpha_decay,
            gamma=gamma,
            policy_type=policy_type,
            context_features=resolved_features,
            initial_state=initial_state,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    if from_state is not None:
        prior = report["config"]["prior"]
        typer.echo(
            f"Resumed from a prior state ({prior['selections']} selections, "
            f"total reward {prior['total_reward']:.4f})."
        )
    typer.echo(
        f"Cold-start exploration bandit simulation "
        f"(policy={policy_type}, top_k={top_k}):"
    )
    typer.echo(f"Context features: {', '.join(resolved_features)}")
    typer.echo(f"{'Arm':<12} {'Selected':>9} {'Cum. Reward':>12} {'Mean Reward':>12}")
    typer.echo("-" * 48)
    for arm, stats in report["arms"].items():
        typer.echo(
            f"{arm:<12} {stats['selections']:>9} {stats['total_reward']:>12.4f} "
            f"{stats['mean_reward']:>12.4f}"
        )
    summary = report["summary"]
    typer.echo("-" * 48)
    typer.echo(f"Rounds completed: {summary['rounds_completed']}")
    typer.echo(f"Cumulative reward: {summary['cumulative_reward']:.4f}")
    typer.echo(f"Best in hindsight: {summary['best_in_hindsight']:.4f}")
    typer.echo(f"Regret: {summary['regret']:.4f}")
    typer.echo(
        f"Always-popular control mean reward: "
        f"{summary['always_popular']['mean_reward']:.4f}"
    )

    written = write_bandit_report(
        report,
        resolved_report_dir,
        report_name=report_name,
    )
    typer.echo(f"Bandit simulation report written to: {written}")

    if write_state is not None:
        target_cls: Any
        if policy_type == "thompson_sampling":
            target_cls = ThompsonSamplingContextualBandit
        elif policy_type == "epsilon_greedy":
            target_cls = EpsilonGreedyContextualBandit
        else:
            target_cls = LinUCBContextualBandit

        base_state = (
            initial_state
            if initial_state is not None
            else snapshot_bandit_state(
                target_cls(
                    selected_arms,
                    len(report["config"]["context_features"]),
                    alpha=report["config"]["alpha"],
                    alpha_decay=alpha_decay,
                    context_features=report["config"]["context_features"],
                )
            )
        )
        trained = fold_bandit_state(
            base_state, feedback_from_report(report), gamma=gamma
        )
        state_path = write_bandit_state(
            trained,
            Path(write_state).parent,
            state_name=Path(write_state).stem,
        )
        typer.echo(f"Bandit state written to: {state_path}")


@app.command()
def bandit_policy(
    report_path: str = typer.Option(
        REPORTS_DIR / "bandit_simulation.json",
        "--report-path",
        help="Path to a bandit simulation JSON report to derive weights from.",
    ),
    from_state: str | None = typer.Option(
        None,
        "--from-state",
        help="Path to a bandit state JSON snapshot to derive weights directly from.",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        min=0.01,
        help="Softmax temperature scaling arm weights from mean rewards.",
    ),
    temperature_decay: float = typer.Option(
        0.0,
        "--temperature-decay",
        help="Temperature annealing decay rate lambda >= 0.",
    ),
    min_temperature: float = typer.Option(
        0.05,
        "--min-temperature",
        min=0.001,
        help="Minimum temperature floor under annealing schedule.",
    ),
    policy_name: str | None = typer.Option(
        None,
        "--policy-name",
        help="Name for the cold-start policy JSON file.",
    ),
    policy_dir: str | None = typer.Option(
        None,
        "--policy-dir",
        help="Directory for the persisted cold-start policy.",
    ),
) -> None:
    """Derive cold-start arm weights from a bandit report or state."""
    resolved_policy_dir = Path(policy_dir) if policy_dir is not None else REPORTS_DIR
    try:
        if from_state is not None:
            source_data = load_bandit_state(from_state)
            source_desc = f"state '{from_state}'"
        else:
            source_data = load_bandit_report(report_path)
            source_desc = f"report '{report_path}'"

        policy = derive_cold_start_policy(
            source_data,
            temperature=temperature,
            temperature_decay=temperature_decay,
            min_temperature=min_temperature,
        )
        written = write_cold_start_policy(
            policy,
            resolved_policy_dir,
            policy_name=policy_name,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Learned cold-start policy arm weights:")
    typer.echo(f"Source: {source_desc}")
    typer.echo(f"{'Arm':<12} {'Weight':>12}")
    typer.echo("-" * 26)
    for arm, weight in sorted(policy.items(), key=lambda item: (-item[1], item[0])):
        typer.echo(f"{arm:<12} {weight:>12.6f}")
    typer.echo(f"Cold-start policy written to: {written}")
    if Path(written).resolve() == COLD_START_POLICY_PATH.resolve():
        typer.echo(
            "Policy saved to the default serving location; "
            "recommend-user will now use it for unknown users."
        )


@app.command()
def bandit_eval_offline(
    feedback_path: str = typer.Option(
        BANDIT_FEEDBACK_PATH,
        "--feedback-path",
        help="Path to JSONL feedback journal for off-policy evaluation.",
    ),
    policy_type: str = typer.Option(
        "linucb",
        "--policy-type",
        help=(
            "Candidate policy algorithm: 'linucb', 'thompson_sampling', "
            "or 'epsilon_greedy'."
        ),
    ),
    alpha: float = typer.Option(
        1.0,
        "--alpha",
        help="Candidate policy exploration parameter alpha (or epsilon).",
    ),
    alpha_decay: float = typer.Option(
        0.0,
        "--alpha-decay",
        help="Exploration decay rate lambda for candidate policy.",
    ),
    from_state: str | None = typer.Option(
        None,
        "--from-state",
        help="Path to a candidate bandit state JSON snapshot to evaluate.",
    ),
    policy_path: str | None = typer.Option(
        None,
        "--policy-path",
        help="Path to a cold-start static policy weights JSON to evaluate.",
    ),
    min_propensity: float = typer.Option(
        0.01,
        "--min-propensity",
        help="Minimum propensity clipping threshold.",
    ),
    ridge_lambda: float = typer.Option(
        1.0,
        "--ridge-lambda",
        help="L2 regularization for direct method reward regression model.",
    ),
    write_report: bool = typer.Option(
        False,
        "--write-report",
        help="Persist evaluation results to a JSON report.",
    ),
    report_name: str | None = typer.Option(
        None,
        "--report-name",
        help="Name for the OPE JSON report.",
    ),
    report_dir: str | None = typer.Option(
        None,
        "--report-dir",
        help="Directory for the persistent OPE report.",
    ),
) -> None:
    """Evaluate candidate bandit policies offline using OPE (IPS/SnIPS/DM/DR)."""
    journal_path = Path(feedback_path)
    try:
        records = load_bandit_feedback(journal_path)
        if not records:
            raise ValueError(f"Feedback journal '{journal_path}' contains no records.")

        target: BaseContextualBandit | dict[str, float] | str
        if from_state is not None:
            target = BaseContextualBandit.from_state(load_bandit_state(from_state))
        elif policy_path is not None:
            target = load_cold_start_policy(policy_path)
        else:
            ctx_array = np.asarray(records[0].get("context"), dtype=float)
            dim = int(ctx_array.size)
            if dim == 0:
                raise ValueError(
                    "Logged feedback records contain empty context vectors."
                )
            behavior_props = records[0].get("behavior_propensities")
            arm_names = (
                tuple(behavior_props.keys())
                if isinstance(behavior_props, dict) and behavior_props
                else DEFAULT_COLD_START_ARMS
            )
            target_cls: Any
            if policy_type == "thompson_sampling":
                target_cls = ThompsonSamplingContextualBandit
            elif policy_type == "epsilon_greedy":
                target_cls = EpsilonGreedyContextualBandit
            else:
                target_cls = LinUCBContextualBandit
            target = target_cls(arm_names, dim, alpha=alpha, alpha_decay=alpha_decay)

        results = compute_off_policy_evaluation(
            records,
            target,
            min_propensity=min_propensity,
            ridge_lambda=ridge_lambda,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    summary = results["summary"]
    metrics = results["metrics"]
    typer.echo(f"Off-Policy Evaluation (Target Policy: {results['target_policy']}):")
    typer.echo(f"Logged records evaluated: {summary['records_evaluated']}")
    typer.echo(f"Match rate: {summary['match_rate']:.1%}")
    typer.echo(f"Effective Sample Size (ESS): {summary['effective_sample_size']:.2f}")
    typer.echo(f"Mean logged reward: {summary['logging_mean_reward']:.4f}")
    typer.echo("-" * 52)
    typer.echo(f"{'Estimator':<24} {'Value':>12} {'Std Error':>12}")
    typer.echo("-" * 52)
    estimators_table = [
        (
            "Inverse Propensity (IPS)",
            metrics["ips"].get("value"),
            metrics["ips"].get("standard_error"),
        ),
        (
            "Self-Normalized (SnIPS)",
            metrics["snips"].get("value"),
            None,
        ),
        (
            "Direct Method (DM)",
            metrics["direct_method"].get("value"),
            metrics["direct_method"].get("standard_error"),
        ),
        (
            "Doubly Robust (DR)",
            metrics["doubly_robust"].get("value"),
            metrics["doubly_robust"].get("standard_error"),
        ),
    ]
    for est, v, se in estimators_table:
        v_str = f"{v:.4f}" if v is not None else "N/A"
        se_str = f"{se:.4f}" if se is not None else "N/A"
        typer.echo(f"{est:<24} {v_str:>12} {se_str:>12}")
    typer.echo("-" * 52)

    if write_report:
        resolved_report_dir = (
            Path(report_dir) if report_dir is not None else REPORTS_DIR
        )
        written = write_ope_report(
            results,
            resolved_report_dir,
            report_name=report_name,
        )
        typer.echo(f"OPE report written to: {written}")


@app.command()
def bandit_update(
    state_path: str = typer.Option(
        BANDIT_STATE_PATH,
        "--state-path",
        help="Path to the persisted bandit state to fold feedback into.",
    ),
    report_path: str | None = typer.Option(
        None,
        "--report-path",
        help="Bandit report whose per-round observations become offline feedback.",
    ),
    journal_path: str | None = typer.Option(
        BANDIT_FEEDBACK_PATH,
        "--journal-path",
        help="Feedback journal to fold (skipped when the journal is absent).",
    ),
    output_state: str | None = typer.Option(
        None,
        "--output-state",
        help="Path to write the updated state (defaults to --state-path).",
    ),
    gamma: float = typer.Option(
        1.0,
        "--gamma",
        help="Recency discount factor gamma in range (0, 1].",
    ),
    context_features: str | None = typer.Option(
        None,
        "--context-features",
        help=(
            "Comma-separated feature names expected by the state. Verified "
            "against the state's recorded feature set when present, so folding "
            "with the wrong context dimension fails fast."
        ),
    ),
) -> None:
    """Fold observed feedback into a bandit state as its new prior.

    Offline observations come from --report-path; served observations are
    folded from the feedback journal through an idempotent sweep, so reruns
    (or a scheduled maintenance loop) never double-count already-folded
    records.
    """
    try:
        if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"gamma must be in the range (0, 1], got {gamma}.")
        state = load_bandit_state(state_path)
        report = load_bandit_report(report_path) if report_path else None

        if context_features is not None:
            requested_features = tuple(
                feature.strip()
                for feature in context_features.split(",")
                if feature.strip()
            )
            validate_state_context_features(state, requested_features)

        journal = (
            Path(journal_path) if journal_path is not None else BANDIT_FEEDBACK_PATH
        )
        if report is None and not journal.exists():
            raise ValueError(
                "No feedback to fold: provide --report-path or a journal file."
            )

        updated = state
        if report is not None:
            updated = fold_bandit_state(
                updated, feedback_from_report(report), gamma=gamma
            )

        if journal.exists():
            updated, sweep = sweep_bandit_journal(
                updated, load_bandit_feedback(journal), gamma=gamma
            )
            typer.echo(
                f"Folded {sweep['folded_count']} pending journal record(s) "
                f"(gamma={gamma}); "
                f"{sweep['journal_length'] - sweep['offset']} remain pending."
            )

        target = Path(output_state) if output_state is not None else Path(state_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(updated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Bandit state after folding feedback:")
    typer.echo(f"{'Arm':<12} {'Selected':>9} {'Cum. Reward':>12} {'Mean Reward':>12}")
    typer.echo("-" * 48)
    for arm, stats in updated["arms"].items():
        selections = stats["selections"]
        total_reward = stats["rewards"]
        mean_reward = total_reward / selections if selections else 0.0
        typer.echo(
            f"{arm:<12} {selections:>9} {total_reward:>12.4f} {mean_reward:>12.4f}"
        )
    typer.echo(f"Updated bandit state written to: {target}")


@app.command()
def bandit_status(
    state_path: str = typer.Option(
        str(BANDIT_STATE_PATH),
        "--state-path",
        help="Path to the persisted bandit state to inspect.",
    ),
    journal_path: str = typer.Option(
        str(BANDIT_FEEDBACK_PATH),
        "--journal-path",
        help="Feedback journal to account for in the summary.",
    ),
    policy_path: str = typer.Option(
        str(COLD_START_POLICY_PATH),
        "--policy-path",
        help="Cold-start policy file to report as active.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON output for monitoring pipelines.",
    ),
) -> None:
    """Report the cold-start bandit lifecycle (state, policy, journal, folds)."""
    state_path_obj, journal_path_obj = Path(state_path), Path(journal_path)
    try:
        state = load_bandit_state(state_path) if state_path_obj.exists() else None
        journal = (
            load_bandit_feedback(journal_path) if journal_path_obj.exists() else []
        )
        policy = (
            load_cold_start_policy(policy_path) if Path(policy_path).exists() else None
        )
        summary = summarize_bandit_lifecycle(
            state=state, policy=policy, feedback=journal
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.echo("Cold-start bandit lifecycle:")
    if summary["available"]:
        typer.echo(f"State: available ({state_path_obj})")
        st_info = summary["state"]
        pol_type = st_info.get("policy_type", "linucb")
        alpha_val = st_info.get("alpha", 1.0)
        decay_val = st_info.get("alpha_decay", 0.0)
        eff_alpha = st_info.get("effective_alpha", alpha_val)
        typer.echo(
            f"Policy model: {pol_type} (alpha={alpha_val}, "
            f"alpha_decay={decay_val}, effective_alpha={eff_alpha})"
        )
    else:
        typer.echo("State: none (run simulate-bandit --write-state or bandit-update)")
    recorded_features = None
    if summary["available"] and summary["state"]:
        recorded_features = summary["state"].get("context_features")
    active_features = (
        tuple(recorded_features)
        if recorded_features
        else resolve_context_features(
            env=os.getenv(CONTEXT_FEATURES_ENV_VAR),
            config_path=BANDIT_CONTEXT_FEATURES_PATH,
        )
    )
    typer.echo(
        f"Context features ({len(active_features)}): {', '.join(active_features)}"
    )
    typer.echo(f"{'Arm':<12} {'Selected':>9} {'Cum. Reward':>12} {'Mean Reward':>12}")
    typer.echo("-" * 48)
    arms = summary["state"]["arms"] if summary["available"] else {}
    for arm, stats in arms.items():
        typer.echo(
            f"{arm:<12} {stats['selections']:>9} {stats['total_reward']:>12.4f} "
            f"{stats['mean_reward']:>12.4f}"
        )
    if summary["policy"]:
        typer.echo(
            "Policy: "
            + " | ".join(
                f"{arm} {weight:.4f}" for arm, weight in summary["policy"].items()
            )
        )
    else:
        typer.echo("Policy: none (cold start serves popular items)")
    last_fold = summary["last_fold"]
    if last_fold:
        typer.echo(
            f"Journal: {summary['journal']['length']} record(s), "
            f"{summary['journal']['pending']} pending (last fold {last_fold['at']})"
        )
    else:
        typer.echo(
            f"Journal: {summary['journal']['length']} record(s), "
            f"{summary['journal']['pending']} pending (no fold yet)"
        )


@app.command()
def bandit_observability(
    state_path: str = typer.Option(
        str(BANDIT_STATE_PATH),
        "--state-path",
        help="Path to the persisted bandit state to inspect.",
    ),
    journal_path: str = typer.Option(
        str(BANDIT_FEEDBACK_PATH),
        "--journal-path",
        help="Path to the feedback journal.",
    ),
    snapshot_dir: str = typer.Option(
        str(BANDIT_SNAPSHOTS_DIR),
        "--snapshot-dir",
        help="Directory where bandit snapshots are stored.",
    ),
    policy_path: str = typer.Option(
        str(COLD_START_POLICY_PATH),
        "--policy-path",
        help="Cold-start policy file to evaluate.",
    ),
    include_ope: bool = typer.Option(
        False,
        "--include-ope",
        help="Calculate and include off-policy evaluation on the feedback journal.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON output for monitoring systems.",
    ),
) -> None:
    """Report observability telemetry across streaming, maintenance, and guardrails."""
    st_path_obj = Path(state_path)
    j_path_obj = Path(journal_path)
    sn_dir_obj = Path(snapshot_dir)
    pol_path_obj = Path(policy_path)

    state = load_bandit_state(st_path_obj) if st_path_obj.exists() else None
    journal = load_bandit_feedback(j_path_obj) if j_path_obj.exists() else []
    snapshots = list_bandit_snapshots(sn_dir_obj) if sn_dir_obj.exists() else []
    champ_path = get_champion_snapshot(sn_dir_obj) if sn_dir_obj.exists() else None
    champ_name = champ_path.name if champ_path is not None else None

    warnings: list[str] = []
    if not state:
        warnings.append("No active bandit state file found.")
    pending = pending_feedback_count(state, journal) if state else len(journal)
    if pending >= 50:
        warnings.append(f"{pending} pending feedback records waiting to be folded.")

    health_status = "healthy" if not warnings else "degraded"

    ope_data: dict[str, Any] | None = None
    if include_ope:
        if not journal:
            ope_data = {"available": False, "reason": "No feedback records found."}
        else:
            pol = (
                load_cold_start_policy(pol_path_obj)
                if pol_path_obj.exists()
                else "popular"
            )
            try:
                ope_data = compute_off_policy_evaluation(journal, pol)
            except Exception as ope_err:
                ope_data = {"available": False, "error": str(ope_err)}

    st_cfg = state.get("config", {}) if state else {}
    pol_model = str(st_cfg.get("policy_type", "linucb")) if state else None
    alpha_val = float(st_cfg.get("alpha", 1.0)) if state else None
    ctx_feat: list[str] = (
        list(state.get("context_features") or []) if state else []
    )
    total_sel = (
        sum(
            int(arm.get("selections", 0))
            for arm in state.get("arms", {}).values()
            if isinstance(arm, dict)
        )
        if state
        else 0
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "health": {
            "status": health_status,
            "warnings": warnings,
        },
        "state": {
            "available": state is not None,
            "path": str(st_path_obj),
            "policy_type": pol_model,
            "alpha": alpha_val,
            "context_features": ctx_feat if state else None,
            "total_selections": total_sel,
        },
        "journal": {
            "path": str(j_path_obj),
            "total_records": len(journal),
            "pending_records": pending,
        },
        "snapshots": {
            "directory": str(sn_dir_obj),
            "total_count": len(snapshots),
            "champion": champ_name,
            "latest": snapshots[0]["filename"] if snapshots else None,
        },
        "off_policy_evaluation": ope_data,
    }

    if as_json:
        typer.echo(json.dumps(report, indent=2))
        return

    typer.echo("Bandit Observability Diagnostics:")
    typer.echo("=================================")
    if state:
        typer.echo(f"State: available ({st_path_obj})")
        typer.echo(f"  Policy model: {pol_model} (alpha={alpha_val})")
        typer.echo(f"  Context features ({len(ctx_feat)}): {', '.join(ctx_feat)}")
        typer.echo(f"  Total selections: {total_sel}")
    else:
        typer.echo("State: unavailable (run simulate-bandit or bandit-update)")

    typer.echo(f"Journal: {j_path_obj}")
    typer.echo(f"  Total records: {len(journal)}, Pending records: {pending}")

    typer.echo(f"Snapshots: {sn_dir_obj} ({len(snapshots)} total)")
    typer.echo(f"  Champion: {champ_name or 'none'}")
    if snapshots:
        typer.echo(f"  Latest: {snapshots[0]['filename']}")

    typer.echo(f"Health: {health_status} ({len(warnings)} warning(s))")
    for w in warnings:
        typer.echo(f"  - Warning: {w}")

    if include_ope and ope_data:
        typer.echo("Off-Policy Evaluation (OPE):")
        if ope_data.get("available") is False:
            err_msg = ope_data.get("reason") or ope_data.get("error")
            typer.echo(f"  OPE unavailable: {err_msg}")
        else:
            summ = ope_data.get("summary", {})
            mets = ope_data.get("metrics", {})
            typer.echo(f"  Target policy: {ope_data.get('target_policy')}")
            typer.echo(f"  Records evaluated: {summ.get('records_evaluated')}")
            typer.echo(f"  Effective sample size: {summ.get('effective_sample_size')}")
            for m_key, m_val in mets.items():
                if isinstance(m_val, dict):
                    v = m_val.get("value")
                    ci = m_val.get("ci_95")
                    if ci and v is not None:
                        typer.echo(
                            f"    {m_key.upper():<14}: {v:.4f} "
                            f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])"
                        )
                    elif v is not None:
                        typer.echo(f"    {m_key.upper():<14}: {v:.4f}")



@app.command()
def bandit_context(
    features: str | None = typer.Option(
        None,
        "--set",
        help="Comma-separated feature names to persist as the active set.",
    ),
    config_path: str = typer.Option(
        BANDIT_CONTEXT_FEATURES_PATH,
        "--config-path",
        help="Path to the bandit context feature config to inspect or update.",
    ),
) -> None:
    """Show or update the active bandit context feature set.

    Without --set, prints the effective feature set by precedence (env var,
    then the persisted config, then the project default) together with its
    source. With --set, validates and persists the feature names as the
    project's active config.
    """
    config = Path(config_path)
    try:
        if features is not None:
            names = tuple(part.strip() for part in features.split(",") if part.strip())
            save_bandit_context_features(names, config)
            typer.echo(
                f"Bandit context features saved to: {config} ({', '.join(names)})."
            )
            return
        resolved = resolve_context_features(
            env=os.getenv(CONTEXT_FEATURES_ENV_VAR),
            config_path=config,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    if os.getenv(CONTEXT_FEATURES_ENV_VAR):
        source = "environment variable MUSIC_RECOMMENDER_CONTEXT_FEATURES"
    elif config.exists():
        source = f"persisted config ({config})"
    else:
        source = "project default"
    typer.echo(f"Bandit context features ({len(resolved)}): {', '.join(resolved)}")
    typer.echo(f"Source: {source}.")


@app.command()
def record_bandit_feedback(
    context: str = typer.Option(
        ...,
        "--context",
        help="Comma-separated context feature values served for the request.",
    ),
    arm: str = typer.Option(
        ...,
        "--arm",
        help="Arm the service chose for this request.",
    ),
    reward: float = typer.Option(
        ...,
        "--reward",
        help="Engagement reward observed for the served recommendation.",
    ),
    user_id: str | None = typer.Option(
        None,
        "--user-id",
        help="Optional identifier of the served user.",
    ),
    journal_path: str = typer.Option(
        BANDIT_FEEDBACK_PATH,
        "--journal-path",
        help="Feedback journal to append the observation to.",
    ),
) -> None:
    """Record a served-request bandit observation into the feedback journal."""
    try:
        parsed = [float(value.strip()) for value in context.split(",") if value.strip()]
        record = {"context": parsed, "arm": arm, "reward": reward}
        if user_id:
            record["user_id"] = user_id
        journal = append_bandit_feedback(record, journal_path)
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(
        f"Recorded reward {reward:.4f} for arm '{arm}' into feedback journal: {journal}"
    )


@app.command()
def bandit_snapshot(
    create: bool = typer.Option(
        False,
        "--create",
        help="Create a new timestamped snapshot of the bandit state.",
    ),
    label: str | None = typer.Option(
        None,
        "--label",
        help="Optional label to attach to the created snapshot.",
    ),
    list_snapshots: bool = typer.Option(
        False,
        "--list",
        help="List existing bandit state snapshots.",
    ),
    restore: str | None = typer.Option(
        None,
        "--restore",
        help="Path to snapshot file to restore into the active bandit state.",
    ),
    prune: int | None = typer.Option(
        None,
        "--prune",
        help="Prune older snapshots keeping the specified number of newest snapshots.",
    ),
    tag_champion: str | None = typer.Option(
        None,
        "--tag-champion",
        help="Path to snapshot file to pin as verified champion.",
    ),
    state_path: str = typer.Option(
        str(BANDIT_STATE_PATH),
        "--state-path",
        help="Path to the active bandit state file.",
    ),
    snapshot_dir: str = typer.Option(
        str(BANDIT_SNAPSHOTS_DIR),
        "--snapshot-dir",
        help="Directory where snapshots are saved and listed.",
    ),
) -> None:
    """Manage cold-start bandit state snapshots (create, list, restore, prune)."""
    sn_dir = Path(snapshot_dir)
    st_path = Path(state_path)

    try:
        if create:
            state = load_bandit_state(st_path)
            created = save_bandit_snapshot(state, snapshot_dir=sn_dir, label=label)
            typer.echo(f"Bandit state snapshot saved to: {created}")
            return

        if tag_champion is not None:
            tagged = tag_champion_snapshot(Path(tag_champion), snapshot_dir=sn_dir)
            typer.echo(f"Tagged snapshot as verified champion: {tagged}")
            return

        if restore is not None:
            restored = restore_bandit_snapshot(Path(restore), target_state_path=st_path)
            typer.echo(f"Restored snapshot '{restore}' into active state: {restored}")
            return

        if prune is not None:
            deleted = prune_bandit_snapshots(sn_dir, max_keep=prune)
            typer.echo(
                f"Pruned {len(deleted)} snapshot(s), keeping {prune} newest in: "
                f"{sn_dir}"
            )
            return

        # default or explicit --list
        snapshots = list_bandit_snapshots(sn_dir)
        if not snapshots:
            typer.echo(f"No bandit state snapshots found in: {sn_dir}")
            return

        typer.echo(f"Bandit state snapshots ({len(snapshots)}) in {sn_dir}:")
        typer.echo(
            f"{'Filename':<36} {'Created At':<22} {'Label':<15} "
            f"{'Champion':<10} {'Selections':>10}"
        )
        typer.echo("-" * 98)
        for s in snapshots:
            fn = s["filename"]
            ca = s["created_at"] or "-"
            lbl = s["label"] or "-"
            champ = "yes" if s.get("is_champion") else "no"
            sel = s["total_selections"]
            typer.echo(f"{fn:<36} {ca:<22} {lbl:<15} {champ:<10} {sel:>10}")
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


@app.command()
def bandit_rollback(
    snapshot_path: str | None = typer.Option(
        None,
        "--snapshot-path",
        help="Explicit snapshot file to roll back to (defaults to champion snapshot).",
    ),
    state_path: str = typer.Option(
        str(BANDIT_STATE_PATH),
        "--state-path",
        help="Path to the active bandit state file to overwrite.",
    ),
    snapshot_dir: str = typer.Option(
        str(BANDIT_SNAPSHOTS_DIR),
        "--snapshot-dir",
        help="Directory to search for champion snapshot.",
    ),
    update_policy: bool = typer.Option(
        True,
        "--update-policy",
        help="Whether to derive and overwrite the active cold-start policy file.",
    ),
    policy_path: str = typer.Option(
        str(COLD_START_POLICY_PATH),
        "--policy-path",
        help="Path to write updated policy JSON if --update-policy is set.",
    ),
) -> None:
    """Roll back the active bandit state to a champion or specified snapshot."""
    st_path = Path(state_path)
    sn_dir = Path(snapshot_dir)
    pol_path = Path(policy_path)

    try:
        res = rollback_bandit_state(
            target_state_path=st_path,
            snapshot_path=Path(snapshot_path) if snapshot_path is not None else None,
            snapshot_dir=sn_dir,
        )
        is_champ_str = "yes" if res["is_champion"] else "no"
        lbl_str = res["label"] or "none"
        typer.echo(
            f"Rolled back bandit state to: {res['restored_from']} "
            f"(champion={is_champ_str}, label={lbl_str})"
        )
        if update_policy:
            restored_state = load_bandit_state(st_path)
            new_policy = derive_cold_start_policy(restored_state)
            pol_path.parent.mkdir(parents=True, exist_ok=True)
            pol_path.write_text(
                json.dumps(new_policy, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            typer.echo(f"Refreshed active cold-start policy at: {pol_path}")
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


@app.command()
def bandit_drift(
    reference: str | None = typer.Option(
        None,
        "--reference",
        help="Reference snapshot or prior state path (defaults to champion/newest).",
    ),
    state_path: str = typer.Option(
        str(BANDIT_STATE_PATH),
        "--state-path",
        help="Path to the current bandit state to evaluate.",
    ),
    snapshot_dir: str = typer.Option(
        str(BANDIT_SNAPSHOTS_DIR),
        "--snapshot-dir",
        help="Directory to look up snapshots when --reference is omitted.",
    ),
    check_safety: bool = typer.Option(
        False,
        "--check-safety",
        help="Evaluate drift against safety guardrails and exit 1 on violation.",
    ),
    max_l2: float = typer.Option(
        DEFAULT_BANDIT_DRIFT_MAX_L2,
        "--max-l2",
        help="Maximum allowable L2 parameter drift threshold.",
    ),
    min_cosine: float = typer.Option(
        DEFAULT_BANDIT_DRIFT_MIN_COSINE,
        "--min-cosine",
        help="Minimum allowable cosine similarity threshold.",
    ),
    max_reward_drop: float = typer.Option(
        DEFAULT_BANDIT_DRIFT_MAX_REWARD_DROP,
        "--max-reward-drop",
        help="Maximum allowable mean reward drop threshold.",
    ),
) -> None:
    """Compute and display parameter and metric drift between bandit states."""
    st_path = Path(state_path)
    sn_dir = Path(snapshot_dir)
    try:
        current_state = load_bandit_state(st_path)
        if reference is not None:
            ref_path = Path(reference)
            if not ref_path.is_file():
                candidate = sn_dir / ref_path.name
                if candidate.is_file():
                    ref_path = candidate
                else:
                    raise FileNotFoundError(
                        f"Reference snapshot not found: {reference}"
                    )
        else:
            champ = get_champion_snapshot(sn_dir)
            if champ is not None and champ.is_file():
                ref_path = champ
            else:
                snapshots = list_bandit_snapshots(sn_dir)
                if snapshots:
                    ref_path = Path(snapshots[0]["path"])
                else:
                    raise FileNotFoundError(
                        "No reference state provided and no snapshots found in "
                        f"'{sn_dir}' to compare against."
                    )
        ref_state = load_bandit_snapshot(ref_path)
        drift = compute_bandit_drift(ref_state, current_state)
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    summary = drift["summary"]
    typer.echo(f"Bandit state drift (reference: {ref_path.name}):")
    typer.echo(
        f"{'Arm':<12} {'L2 Drift':>10} {'Cosine Sim':>11} {'Delta Sel':>10} "
        f"{'Delta Rew':>11} {'Delta Mean':>11}"
    )
    typer.echo("-" * 70)
    for arm, d in drift["arms"].items():
        typer.echo(
            f"{arm:<12} {d['l2_drift']:>10.4f} {d['cosine_similarity']:>11.4f} "
            f"{d['delta_selections']:>10} {d['delta_rewards']:>11.4f} "
            f"{d['delta_mean_reward']:>11.4f}"
        )
    typer.echo("-" * 70)
    typer.echo(
        f"Summary: max L2 drift={summary['max_l2_drift']:.4f}, "
        f"mean L2 drift={summary['mean_l2_drift']:.4f}, "
        f"dominant arm={summary['dominant_arm_a']} -> {summary['dominant_arm_b']} "
        f"(changed={'yes' if summary['dominant_arm_changed'] else 'no'}), "
        f"drift detected={'yes' if summary['has_drift'] else 'no'}"
    )

    if check_safety:
        thresholds = DriftSafetyThresholds(
            max_l2_drift=max_l2,
            min_cosine_similarity=min_cosine,
            max_reward_drop=max_reward_drop,
        )
        safety = evaluate_drift_safety(
            target=current_state,
            baseline=ref_state,
            thresholds=thresholds,
            reference_snapshot=ref_path.name,
        )
        typer.echo("\nDrift Safety Evaluation:")
        if safety.is_safe:
            typer.secho(
                "PASSED: Drift is within safe operating thresholds.",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"FAILED: Drift safety guardrails violated "
                f"({len(safety.violations)} violation(s)):",
                fg=typer.colors.RED,
                err=True,
            )
            for v in safety.violations:
                typer.secho(f"  - {v}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)


@app.command()
def bandit_sweep(
    state_path: str = typer.Option(
        str(BANDIT_STATE_PATH),
        "--state-path",
        help="Path to the active bandit state file.",
    ),
    journal_path: str = typer.Option(
        str(BANDIT_FEEDBACK_PATH),
        "--journal-path",
        help="Path to the feedback journal.",
    ),
    threshold: int = typer.Option(
        1,
        "--threshold",
        help="Minimum pending feedback records required to execute a fold.",
    ),
    interval: float = typer.Option(
        5.0,
        "--interval",
        help="Poll interval in seconds when running in --loop mode.",
    ),
    loop: bool = typer.Option(
        False,
        "--loop",
        help="Run continuously as a daemon, sweeping pending feedback periodically.",
    ),
    gamma: float = typer.Option(
        1.0,
        "--gamma",
        help="Recency discount factor gamma in range (0, 1].",
    ),
    snapshot_on_sweep: bool = typer.Option(
        False,
        "--snapshot-on-sweep",
        help="Automatically create a state snapshot when new records are folded.",
    ),
    snapshot_dir: str = typer.Option(
        str(BANDIT_SNAPSHOTS_DIR),
        "--snapshot-dir",
        help="Directory where snapshots are saved if --snapshot-on-sweep is active.",
    ),
    enable_guardrails: bool = typer.Option(
        False,
        "--enable-guardrails",
        help="Enable drift safety guardrails evaluation after folding.",
    ),
    auto_rollback: bool = typer.Option(
        False,
        "--auto-rollback",
        help="Automatically revert state if drift safety guardrails are violated.",
    ),
    max_l2: float = typer.Option(
        DEFAULT_BANDIT_DRIFT_MAX_L2,
        "--max-l2",
        help="Maximum allowable L2 drift threshold for guardrails.",
    ),
    min_cosine: float = typer.Option(
        DEFAULT_BANDIT_DRIFT_MIN_COSINE,
        "--min-cosine",
        help="Minimum allowable cosine similarity threshold for guardrails.",
    ),
    max_reward_drop: float = typer.Option(
        DEFAULT_BANDIT_DRIFT_MAX_REWARD_DROP,
        "--max-reward-drop",
        help="Maximum allowable mean reward drop threshold for guardrails.",
    ),
) -> None:
    """Automate online bandit journal sweeps (one-off or recurring loop)."""
    import time

    if threshold < 1:
        typer.secho(
            "Error: --threshold must be a positive integer.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if interval <= 0:
        typer.secho(
            "Error: --interval must be a positive number.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
        typer.secho(
            f"Error: gamma must be in the range (0, 1], got {gamma}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    st_p = Path(state_path)
    j_p = Path(journal_path)
    sn_dir = Path(snapshot_dir)

    def _execute_sweep_step() -> int:
        if not st_p.exists() or not j_p.exists():
            return 0
        state = load_bandit_state(st_p)
        feedback = load_bandit_feedback(j_p)
        pending = pending_feedback_count(state, feedback)
        if pending < threshold:
            return 0
        updated, sweep = sweep_bandit_journal(state, feedback, gamma=gamma)
        folded = int(sweep.get("folded_count", 0))

        if enable_guardrails and folded > 0:
            champ = get_champion_snapshot(sn_dir)
            ref_st = (
                load_bandit_snapshot(champ)
                if champ and champ.is_file()
                else state
            )
            ref_nm = champ.name if champ and champ.is_file() else "prior_state"
            drift_thresh = DriftSafetyThresholds(
                max_l2_drift=max_l2,
                min_cosine_similarity=min_cosine,
                max_reward_drop=max_reward_drop,
            )
            drift_res = evaluate_drift_safety(
                target=updated,
                baseline=ref_st,
                thresholds=drift_thresh,
                reference_snapshot=ref_nm,
            )
            if not drift_res.is_safe:
                typer.secho(
                    f"Warning: Drift safety guardrails violated "
                    f"({len(drift_res.violations)} violation(s)):",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
                for v in drift_res.violations:
                    typer.secho(f"  - {v}", fg=typer.colors.YELLOW, err=True)
                if auto_rollback:
                    typer.secho(
                        "Auto-rollback active: reverted state changes.",
                        fg=typer.colors.RED,
                    )
                    return 0

        st_p.write_text(
            json.dumps(updated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        typer.echo(
            f"Folded {folded} pending feedback record(s) "
            f"(gamma={gamma}) into: {st_p}"
        )
        if snapshot_on_sweep:
            snap = save_bandit_snapshot(
                updated, snapshot_dir=sn_dir, label="auto_sweep"
            )
            typer.echo(f"Created automatic snapshot: {snap}")
        return folded

    try:
        if not loop:
            if not st_p.exists():
                raise FileNotFoundError(f"Bandit state not found: {st_p}")
            if not j_p.exists():
                typer.echo(f"Feedback journal not found: {j_p}. No records to sweep.")
                return
            state = load_bandit_state(st_p)
            feedback = load_bandit_feedback(j_p)
            pending = pending_feedback_count(state, feedback)
            if pending < threshold:
                typer.echo(
                    f"Pending records ({pending}) below threshold ({threshold}). "
                    "No sweep executed."
                )
                return
            _execute_sweep_step()
            return

        typer.echo(
            f"Starting bandit sweep daemon (interval={interval}s, "
            f"threshold={threshold})... Press Ctrl+C to stop."
        )
        while True:
            try:
                _execute_sweep_step()
            except Exception as loop_err:
                typer.secho(
                    f"Warning during sweep iteration: {loop_err}",
                    fg=typer.colors.YELLOW,
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("\nBandit sweep daemon stopped.")
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()  # pragma: no cover - CLI entry point invoked by `python -m`
