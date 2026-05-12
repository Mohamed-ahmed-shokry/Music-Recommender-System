"""Command line interface for the music recommender."""

from datetime import UTC, datetime
from pathlib import Path

import typer

from music_recommender.config import (
    ARTIFACT_BUNDLE_PATH,
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
    DEFAULT_MIN_ARTIST_INTERACTIONS,
    DEFAULT_MIN_USER_INTERACTIONS,
    DEFAULT_TOP_K,
    DEFAULT_USE_GPU,
    MAPPINGS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)
from music_recommender.data import load_and_validate_interactions
from music_recommender.evaluate import evaluate_repeated_holdout
from music_recommender.model import train_and_save_model
from music_recommender.preprocessing import prepare_training_data
from music_recommender.recommend import format_recommendations
from music_recommender.service import RecommenderService

app = typer.Typer(help="Train and use an ALS music artist recommender.")


def _format_artifact_age(created_at: str) -> str:
    created = datetime.fromisoformat(created_at)
    age = datetime.now(UTC) - created
    total_seconds = int(age.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m"
    return f"{total_seconds // 3600}h"


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
def train(
    data_path: Path = RAW_DATA_PATH,
    factors: int = DEFAULT_ALS_FACTORS,
    regularization: float = DEFAULT_ALS_REGULARIZATION,
    iterations: int = DEFAULT_ALS_ITERATIONS,
    alpha: float = DEFAULT_ALS_ALPHA,
    use_gpu: bool = DEFAULT_USE_GPU,
) -> None:
    """Train and save the ALS model."""
    model, user_item_matrix, mappings = train_and_save_model(
        raw_data_path=data_path,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        use_gpu=use_gpu,
    )
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


@app.command()
def artifact_info() -> None:
    """Print details about the saved recommender artifact."""
    try:
        service = RecommenderService.from_artifacts()
    except FileNotFoundError as error:
        typer.echo(f"Error: {error}")
        raise typer.Exit(code=1) from error

    metadata = service.metadata()
    artifact_metadata = metadata["metadata"]
    training_config = metadata["training_config"]
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
    typer.echo(f"Dataset hash: {artifact_metadata['dataset']['sha256']}")


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
) -> None:
    """Recommend artists for a user."""
    try:
        service = RecommenderService.from_artifacts()
        response = service.recommend_user(
            user_id=user_id,
            top_k=top_k,
            include_listened=include_listened,
            popularity_penalty=popularity_penalty,
            diversity=diversity,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}")
        raise typer.Exit(code=1) from error

    typer.echo(f"Recommendations for {user_id}:")
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
        typer.echo(f"Error: {error}")
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
) -> None:
    """Find artists similar to a selected artist."""
    try:
        service = RecommenderService.from_artifacts()
        response = service.similar_artists(artist_id=artist_id, top_k=top_k)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}")
        raise typer.Exit(code=1) from error

    typer.echo(f"Artists similar to {artist_id}:")
    typer.echo(format_recommendations(response["similar_artists"]))


@app.command()
def evaluate(
    top_k: int = DEFAULT_TOP_K,
    folds: int = 1,
    compare_baseline: bool = False,
    use_gpu: bool = DEFAULT_USE_GPU,
) -> None:
    """Evaluate recommendations with ranking metrics."""
    try:
        df = load_and_validate_interactions(RAW_DATA_PATH)
        metrics = evaluate_repeated_holdout(
            df,
            top_k=top_k,
            folds=folds,
            compare_baseline=compare_baseline,
            use_gpu=use_gpu,
        )
    except ValueError as error:
        typer.echo(f"Error: {error}")
        raise typer.Exit(code=1) from error

    if compare_baseline:
        typer.echo(f"Evaluation over {folds} fold(s):")
        _print_metric_row("ALS", metrics["als"], top_k)
        _print_metric_row("Popularity", metrics["popularity"], top_k)
        return

    _print_metric_row("ALS", metrics, top_k)

def _print_metric_row(name: str, metrics: dict[str, float], top_k: int) -> None:
    typer.echo(f"{name}:")
    typer.echo(f"  Precision@{top_k}: {metrics['precision_at_k']:.4f}")
    typer.echo(f"  Recall@{top_k}: {metrics['recall_at_k']:.4f}")
    typer.echo(f"  MAP@{top_k}: {metrics['map_at_k']:.4f}")
    typer.echo(f"  NDCG@{top_k}: {metrics['ndcg_at_k']:.4f}")
    typer.echo(f"  Catalog coverage: {metrics['catalog_coverage']:.4f}")
    typer.echo(f"  Average popularity: {metrics['average_popularity']:.4f}")
    typer.echo(f"  Intra-list diversity: {metrics['intra_list_diversity']:.4f}")


@app.command()
def demo(use_gpu: bool = DEFAULT_USE_GPU) -> None:
    """Train when needed and show example recommendations."""
    if not ARTIFACT_BUNDLE_PATH.exists():
        typer.echo("No saved model found. Training on the sample dataset first.")
        train_and_save_model(use_gpu=use_gpu)

    service = RecommenderService.from_artifacts()
    typer.echo("Recommendations for user_1:")
    response = service.recommend_user(user_id="user_1", top_k=5)
    typer.echo(f"Strategy: {response['strategy']}")
    typer.echo(format_recommendations(response["recommendations"]))
    typer.echo("")
    typer.echo("Artists similar to artist_2:")
    similar_response = service.similar_artists(artist_id="artist_2", top_k=5)
    typer.echo(format_recommendations(similar_response["similar_artists"]))


if __name__ == "__main__":
    app()
