"""Command line interface for the music recommender."""

import typer

from music_recommender.config import (
    DEFAULT_ALS_ALPHA,
    DEFAULT_ALS_FACTORS,
    DEFAULT_ALS_ITERATIONS,
    DEFAULT_ALS_REGULARIZATION,
    DEFAULT_USE_GPU,
    DEFAULT_MIN_ARTIST_INTERACTIONS,
    DEFAULT_MIN_USER_INTERACTIONS,
    DEFAULT_TOP_K,
    MAPPINGS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)
from music_recommender.data import load_and_validate_interactions
from music_recommender.evaluate import evaluate_model
from music_recommender.model import train_and_save_model
from music_recommender.preprocessing import prepare_training_data
from music_recommender.recommend import (
    format_recommendations,
    get_similar_artists,
    load_recommender_artifacts,
    recommend_artists_for_user,
)

app = typer.Typer(help="Train and use an ALS music artist recommender.")


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
    factors: int = DEFAULT_ALS_FACTORS,
    regularization: float = DEFAULT_ALS_REGULARIZATION,
    iterations: int = DEFAULT_ALS_ITERATIONS,
    alpha: float = DEFAULT_ALS_ALPHA,
    use_gpu: bool = DEFAULT_USE_GPU,
) -> None:
    """Train and save the ALS model."""
    _, user_item_matrix, mappings = train_and_save_model(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        use_gpu=use_gpu,
    )
    typer.echo("Model trained successfully.")
    typer.echo(f"Training device: {getattr(_, 'training_device', 'unknown')}")
    fallback_reason = getattr(_, "gpu_fallback_reason", None)
    if fallback_reason:
        typer.echo(f"GPU fallback reason: {fallback_reason}")
    typer.echo(f"Saved model to: {MODEL_PATH}")
    typer.echo(f"Saved mappings to: {MAPPINGS_PATH}")
    typer.echo(f"Training matrix shape: {user_item_matrix.shape}")
    typer.echo(f"Users: {len(mappings['user_id_to_index'])}")
    typer.echo(f"Artists: {len(mappings['artist_id_to_index'])}")


@app.command()
def recommend_user(
    user_id: str = typer.Option(..., help="Original user ID, for example user_1."),
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Recommend artists for a user."""
    try:
        model, user_item_matrix, mappings = load_recommender_artifacts()
        recommendations = recommend_artists_for_user(
            model=model,
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            mappings=mappings,
            top_k=top_k,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}")
        raise typer.Exit(code=1) from error

    typer.echo(f"Recommendations for {user_id}:")
    typer.echo(format_recommendations(recommendations))


@app.command()
def similar_artists(
    artist_id: str = typer.Option(..., help="Original artist ID, for example artist_2."),
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Find artists similar to a selected artist."""
    try:
        model, _, mappings = load_recommender_artifacts()
        recommendations = get_similar_artists(
            model=model,
            artist_id=artist_id,
            mappings=mappings,
            top_k=top_k,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}")
        raise typer.Exit(code=1) from error

    typer.echo(f"Artists similar to {artist_id}:")
    typer.echo(format_recommendations(recommendations))


@app.command()
def evaluate(top_k: int = DEFAULT_TOP_K, use_gpu: bool = DEFAULT_USE_GPU) -> None:
    """Evaluate recommendations with ranking metrics."""
    df = load_and_validate_interactions(RAW_DATA_PATH)
    metrics = evaluate_model(df, top_k=top_k, use_gpu=use_gpu)
    typer.echo(f"Precision@{top_k}: {metrics['precision_at_k']:.4f}")
    typer.echo(f"Recall@{top_k}: {metrics['recall_at_k']:.4f}")
    typer.echo(f"MAP@{top_k}: {metrics['map_at_k']:.4f}")
    typer.echo(f"NDCG@{top_k}: {metrics['ndcg_at_k']:.4f}")


@app.command()
def demo(use_gpu: bool = DEFAULT_USE_GPU) -> None:
    """Train when needed and show example recommendations."""
    if not MODEL_PATH.exists() or not MAPPINGS_PATH.exists():
        typer.echo("No saved model found. Training on the sample dataset first.")
        train_and_save_model(use_gpu=use_gpu)

    model, user_item_matrix, mappings = load_recommender_artifacts()
    typer.echo("Recommendations for user_1:")
    typer.echo(
        format_recommendations(
            recommend_artists_for_user(
                model=model,
                user_id="user_1",
                user_item_matrix=user_item_matrix,
                mappings=mappings,
                top_k=5,
            )
        )
    )
    typer.echo("")
    typer.echo("Artists similar to artist_2:")
    typer.echo(
        format_recommendations(
            get_similar_artists(
                model=model,
                artist_id="artist_2",
                mappings=mappings,
                top_k=5,
            )
        )
    )


if __name__ == "__main__":
    app()
