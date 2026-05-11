"""FastAPI app for serving music recommendations."""

from fastapi import FastAPI, HTTPException

from music_recommender.service import RecommenderService

app = FastAPI(title="Music Recommendation System API")
service: RecommenderService | None = None


@app.on_event("startup")
def load_service() -> None:
    """Load model artifacts once at API startup when available."""
    global service
    try:
        service = RecommenderService.from_artifacts()
    except FileNotFoundError:
        service = None


def get_service() -> RecommenderService:
    """Return the loaded service or a clear training error."""
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model first.",
        )
    return service


@app.get("/")
def root() -> dict[str, str]:
    """Return a simple API health message."""
    return {"message": "Music Recommendation System API"}


@app.get("/health")
def health() -> dict[str, object]:
    """Return service health and artifact availability."""
    return get_service().health()


@app.get("/metadata")
def metadata() -> dict[str, object]:
    """Return loaded artifact metadata."""
    return get_service().metadata()


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: str, top_k: int = 10) -> dict[str, object]:
    """Return artist recommendations for a user."""
    try:
        return get_service().recommend_user(user_id=user_id, top_k=top_k)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/similar-artists/{artist_id}")
def similar_artists(artist_id: str, top_k: int = 10) -> dict[str, object]:
    """Return artists similar to a selected artist."""
    try:
        return get_service().similar_artists(artist_id=artist_id, top_k=top_k)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
