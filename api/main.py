"""FastAPI app for serving music recommendations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from music_recommender.config import DEFAULT_CONTENT_WEIGHT
from music_recommender.service import RecommenderService

service: RecommenderService | None = None
service_load_error: str | None = None


class ProfileRecommendationRequest(BaseModel):
    """Onboarding preference payload for content-based recommendations."""

    artist_ids: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    mood_tags: list[str] = Field(default_factory=list)
    top_k: int = 10
    explain: bool = False


class SessionRecommendationRequest(BaseModel):
    """Short-term listening session payload for v4 recommendations."""

    artist_ids: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    mood_tags: list[str] = Field(default_factory=list)
    user_id: str | None = None
    top_k: int = 10
    exclude_artist_ids: list[str] = Field(default_factory=list)
    include_listened: bool = False
    diversity: float = 0.0
    popularity_penalty: float = 0.0
    content_weight: float = DEFAULT_CONTENT_WEIGHT
    explain: bool = False


def load_service() -> None:
    """Load model artifacts once at API startup when available."""
    global service, service_load_error
    try:
        service = RecommenderService.from_artifacts()
        service_load_error = None
    except (FileNotFoundError, ValueError) as error:
        service = None
        service_load_error = str(error)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize the recommender service once for the API process."""
    load_service()
    yield


app = FastAPI(title="Music Recommendation System API", lifespan=lifespan)


def get_service() -> RecommenderService:
    """Return the loaded service or a clear training error."""
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=service_load_error
            or "Model artifacts not found. Train the model first.",
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


@app.get("/popular-artists")
def popular_artists(top_k: int = 10) -> dict[str, object]:
    """Return globally popular artists."""
    try:
        return get_service().popular_artists(top_k=top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/recommend/user/{user_id}")
def recommend_user(
    user_id: str,
    top_k: int = 10,
    include_listened: bool = False,
    diversity: float = 0.0,
    popularity_penalty: float = 0.0,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    explain: bool = False,
) -> dict[str, object]:
    """Return artist recommendations for a user."""
    try:
        return get_service().recommend_user(
            user_id=user_id,
            top_k=top_k,
            include_listened=include_listened,
            diversity=diversity,
            popularity_penalty=popularity_penalty,
            content_weight=content_weight,
            explain=explain,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/recommend/profile")
def recommend_profile(request: ProfileRecommendationRequest) -> dict[str, object]:
    """Return recommendations from favorite artists and metadata preferences."""
    try:
        return get_service().recommend_profile(
            artist_ids=request.artist_ids,
            genres=request.genres,
            mood_tags=request.mood_tags,
            top_k=request.top_k,
            explain=request.explain,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/recommend/session")
def recommend_session(request: SessionRecommendationRequest) -> dict[str, object]:
    """Return session recommendations from short-term taste signals."""
    try:
        return get_service().recommend_session(
            artist_ids=request.artist_ids,
            genres=request.genres,
            mood_tags=request.mood_tags,
            user_id=request.user_id,
            top_k=request.top_k,
            exclude_artist_ids=request.exclude_artist_ids,
            include_listened=request.include_listened,
            diversity=request.diversity,
            popularity_penalty=request.popularity_penalty,
            content_weight=request.content_weight,
            explain=request.explain,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/similar-artists/{artist_id}")
def similar_artists(
    artist_id: str,
    top_k: int = 10,
    method: str = "als",
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    explain: bool = False,
) -> dict[str, object]:
    """Return artists similar to a selected artist."""
    try:
        return get_service().similar_artists(
            artist_id=artist_id,
            top_k=top_k,
            method=method,
            content_weight=content_weight,
            explain=explain,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/content-similar-artists/{artist_id}")
def content_similar_artists(
    artist_id: str,
    top_k: int = 10,
    explain: bool = False,
) -> dict[str, object]:
    """Return artists similar to a selected artist by metadata only."""
    try:
        return get_service().content_similar_artists(
            artist_id=artist_id,
            top_k=top_k,
            explain=explain,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
