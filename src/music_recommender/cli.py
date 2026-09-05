"""Command line interface for the music recommender."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import typer

from music_recommender import __version__
from music_recommender.config import (
    ARTIFACT_BUNDLE_PATH,
    DATA_DIR,
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
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
    evaluate_repeated_holdout,
    ranking_params_for_training,
    select_winning_strategies,
    strategy_leaderboard,
    write_ablation_report,
    write_ablation_summary_report,
)
from music_recommender.metadata import load_and_validate_artist_metadata
from music_recommender.model import train_and_save_model
from music_recommender.preprocessing import prepare_training_data
from music_recommender.recommend import format_recommendations
from music_recommender.service import RecommenderService
from music_recommender.tracking import (
    DEFAULT_EVALUATION_EXPERIMENT,
    DEFAULT_TRAINING_EXPERIMENT,
    ExperimentTrackingError,
    tracking_run,
)
from music_recommender.tracks import (
    build_track_content_matrix,
    get_similar_tracks,
    load_and_validate_track_interactions,
    load_and_validate_track_metadata,
    recommend_tracks_for_user,
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
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    explain: bool = False,
    ltr: bool = typer.Option(
        False,
        "--ltr/--no-ltr",
        help="Re-rank the ALS candidates with the bundled learning-to-rank model.",
    ),
) -> None:
    """Recommend artists for a user."""
    try:
        service = RecommenderService.from_artifacts()
        if ltr:
            response = service.recommend_user_ltr(
                user_id=user_id,
                top_k=top_k,
                include_listened=include_listened,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
            )
        else:
            response = service.recommend_user(
                user_id=user_id,
                top_k=top_k,
                include_listened=include_listened,
                popularity_penalty=popularity_penalty,
                diversity=diversity,
                content_weight=content_weight,
                explain=explain,
            )
    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Recommendations for {user_id}:")
    typer.echo(f"Strategy: {response['strategy']}")
    if response.get("message"):
        typer.echo(response["message"])
    typer.echo(format_recommendations(response["recommendations"]))


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
    ):
        raise typer.BadParameter(
            "--ablations cannot be combined with --compare-settings,"
            " --compare-baseline, --compare-all, --promote-winner, or"
            " --learn-to-rank."
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
            if ablations is not None:
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

    if ablations is not None:
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
    except (ValueError, Exception) as error:
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
    except (ValueError, Exception) as error:
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
    except (ValueError, Exception) as error:
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
    except (ValueError, Exception) as error:
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
    except (ValueError, Exception) as error:
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
    except (ValueError, Exception) as error:
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
    except (ValueError, Exception) as error:
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
    except (ValueError, Exception) as error:
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
def track_recommendations(
    user_id: str = typer.Option(..., help="User ID for recommendations."),
    top_k: int = DEFAULT_TOP_K,
    include_listened: bool = typer.Option(
        False, "--include-listened/--exclude-listened"
    ),
) -> None:
    """Recommend tracks for a user using track similarity."""
    try:
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

        # Get recommendations
        recommendations = recommend_tracks_for_user(
            user_id=user_id,
            user_track_matrix=user_track_matrix,
            track_similarity_matrix=track_similarity_matrix,
            track_id_to_index=track_id_to_index,
            top_k=top_k,
            include_listened=include_listened,
        )

    except (FileNotFoundError, ValueError) as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Track recommendations for {user_id}:")
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


if __name__ == "__main__":
    app()  # pragma: no cover - CLI entry point invoked by `python -m`
