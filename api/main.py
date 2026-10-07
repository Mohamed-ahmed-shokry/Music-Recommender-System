"""FastAPI app for serving music recommendations."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from api.middleware import RequestSafetyMiddleware
from music_recommender import __version__
from music_recommender.bandit import (
    DriftSafetyThresholds,
    derive_cold_start_policy,
    load_bandit_report,
)
from music_recommender.config import (
    BANDIT_SNAPSHOTS_DIR,
    COLD_START_POLICY_PATH,
    DEFAULT_CONTENT_WEIGHT,
    REPORTS_DIR,
)
from music_recommender.evaluate import load_ablation_summary_report
from music_recommender.logging_setup import configure_logging
from music_recommender.service import RecommenderService

configure_logging()

logger = logging.getLogger("music_recommender.api.main")

_CORS_ORIGINS_ENV = "CORS_ORIGINS"
_CORS_ORIGINS_RAW = os.getenv(_CORS_ORIGINS_ENV, "*")

service: RecommenderService | None = None
service_load_error: str | None = None

ABLATION_SUMMARY_PATH = REPORTS_DIR / "ablation_summary.json"

MAX_API_RESULTS = 100
MAX_REQUEST_VALUES = 100
MAX_REQUEST_BODY_BYTES = 64 * 1024
PositiveTopK = Annotated[int, Query(ge=1, le=MAX_API_RESULTS)]
UnitInterval = Annotated[float, Query(ge=0.0, le=1.0)]
SimilarityMethod = Literal["als", "content", "hybrid"]
OptionalCatalogText = Annotated[str | None, Query(max_length=100)]
CatalogOffset = Annotated[int, Query(ge=0)]
CatalogLimit = Annotated[int, Query(ge=1, le=MAX_API_RESULTS)]
RequestText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]


class ProfileRecommendationRequest(BaseModel):
    """Onboarding preference payload for content-based recommendations."""

    model_config = ConfigDict(strict=True, extra="forbid")

    artist_ids: list[RequestText] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_VALUES,
    )
    genres: list[RequestText] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_VALUES,
    )
    mood_tags: list[RequestText] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_VALUES,
    )
    top_k: int = Field(default=10, ge=1, le=MAX_API_RESULTS)
    explain: bool = False


class SessionRecommendationRequest(BaseModel):
    """Short-term listening session payload for v4 recommendations."""

    model_config = ConfigDict(strict=True, extra="forbid")

    artist_ids: list[RequestText] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_VALUES,
    )
    genres: list[RequestText] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_VALUES,
    )
    mood_tags: list[RequestText] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_VALUES,
    )
    user_id: RequestText | None = None
    top_k: int = Field(default=10, ge=1, le=MAX_API_RESULTS)
    exclude_artist_ids: list[RequestText] = Field(
        default_factory=list,
        max_length=MAX_REQUEST_VALUES,
    )
    include_listened: bool = False
    diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    popularity_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    content_weight: float = Field(
        default=DEFAULT_CONTENT_WEIGHT,
        ge=0.0,
        le=1.0,
    )
    explain: bool = False


def load_service() -> None:
    """Load model artifacts once at API startup when available."""
    global service, service_load_error
    try:
        service = RecommenderService.from_artifacts()
        service_load_error = None
        logger.info(
            "service_loaded users=%s",
            service.artifact.metadata.get("num_users"),
        )
    except (FileNotFoundError, ValueError) as error:
        service = None
        service_load_error = str(error)
        logger.warning("service_unavailable reason=%s", error)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize the recommender service once for the API process."""
    load_service()
    try:
        yield
    finally:
        if service is not None:
            try:
                service.close()
            except Exception as err:
                logger.warning(
                    "Error closing service during API lifespan shutdown: %s",
                    err,
                )


app = FastAPI(
    title="Music Recommendation System API",
    description=(
        "Hybrid ALS + content-based artist recommendations with explainable "
        "ranking, cold-start onboarding, and session-aware mixes."
    ),
    version=__version__,
    contact={
        "name": "Music Recommender System",
        "url": "https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)
_cors_origins = [
    origin.strip() for origin in _CORS_ORIGINS_RAW.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)
app.add_middleware(
    RequestSafetyMiddleware,
    max_body_bytes=MAX_REQUEST_BODY_BYTES,
)


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


@app.get("/health/streaming")
def health_streaming(response: Response) -> dict[str, object]:
    """Return streaming feedback ingestion and maintenance health."""
    health_data = get_service().streaming_health()
    if health_data.get("status") == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health_data


@app.get("/metadata")
def metadata() -> dict[str, object]:
    """Return loaded artifact metadata."""
    return get_service().metadata()


@app.get("/evaluation/ablation-summary")
def ablation_summary() -> dict[str, object]:
    """Return the persisted aggregated ablation-importance summary."""
    try:
        return load_ablation_summary_report(ABLATION_SUMMARY_PATH)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="No aggregated ablation summary found. Run "
            "`music_recommender.cli ablation-summary` first.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/popular-artists")
def popular_artists(top_k: PositiveTopK = 10) -> dict[str, object]:
    """Return globally popular artists."""
    try:
        return get_service().popular_artists(top_k=top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/catalog/artists")
def browse_artists(
    query: OptionalCatalogText = None,
    genre: OptionalCatalogText = None,
    mood_tag: OptionalCatalogText = None,
    country: OptionalCatalogText = None,
    era: OptionalCatalogText = None,
    offset: CatalogOffset = 0,
    limit: CatalogLimit = 25,
) -> dict[str, object]:
    """Search and page through artists available to the recommender."""
    try:
        return get_service().browse_artists(
            query=query,
            genre=genre,
            mood_tag=mood_tag,
            country=country,
            era=era,
            offset=offset,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/recommend/user/{user_id}")
def recommend_user(
    user_id: RequestText,
    top_k: PositiveTopK = 10,
    include_listened: bool = False,
    diversity: UnitInterval = 0.0,
    popularity_penalty: UnitInterval = 0.0,
    novelty_weight: UnitInterval = 0.0,
    content_weight: UnitInterval = DEFAULT_CONTENT_WEIGHT,
    explain: bool = False,
    record_feedback: bool = False,
    context: str | None = None,
) -> dict[str, object]:
    """Return artist recommendations for a user.

    When ``record_feedback`` is set and the user is served via the cold-start
    bandit branch, one feedback record per positive-weight policy arm is
    appended to the bandit feedback journal. ``context`` optionally carries the
    cold-start feature vector observed for this request, so the recorded
    feedback is contextual instead of the neutral vector.
    """
    try:
        parsed_context: list[float] | None = None
        if context is not None:
            parsed_context = [
                float(value.strip()) for value in context.split(",") if value.strip()
            ]
            if not parsed_context:
                raise ValueError("Serve context must be a non-empty vector.")
        return get_service().recommend_user(
            user_id=user_id,
            top_k=top_k,
            include_listened=include_listened,
            diversity=diversity,
            popularity_penalty=popularity_penalty,
            novelty_weight=novelty_weight,
            content_weight=content_weight,
            explain=explain,
            record_feedback=record_feedback,
            context=parsed_context,
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
            novelty_weight=request.novelty_weight,
            content_weight=request.content_weight,
            explain=request.explain,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/similar-artists/{artist_id}")
def similar_artists(
    artist_id: RequestText,
    top_k: PositiveTopK = 10,
    method: SimilarityMethod = "als",
    content_weight: UnitInterval = DEFAULT_CONTENT_WEIGHT,
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
    artist_id: RequestText,
    top_k: PositiveTopK = 10,
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


@app.get("/recommend/user/{user_id}/ltr")
def recommend_user_ltr(
    user_id: RequestText,
    top_k: PositiveTopK = 10,
    include_listened: bool = False,
    diversity: UnitInterval = 0.0,
    popularity_penalty: UnitInterval = 0.0,
    novelty_weight: UnitInterval = 0.0,
) -> dict[str, object]:
    """Return LTR re-ranked artist recommendations for a user."""
    try:
        return get_service().recommend_user_ltr(
            user_id=user_id,
            top_k=top_k,
            include_listened=include_listened,
            diversity=diversity,
            popularity_penalty=popularity_penalty,
            novelty_weight=novelty_weight,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/tracks/recommend/{user_id}")
def recommend_tracks(
    user_id: RequestText,
    top_k: PositiveTopK = 10,
    include_listened: bool = False,
    diversity: UnitInterval = 0.0,
    popularity_penalty: UnitInterval = 0.0,
    novelty_weight: UnitInterval = 0.0,
    explain: bool = False,
    method: Literal["similarity", "hybrid"] = "similarity",
    content_weight: UnitInterval | None = None,
    ltr: bool = False,
) -> dict[str, object]:
    """Return track recommendations for a user with audio-feature similarity.

    The ``hybrid`` method blends collaborative artist taste with the
    audio-feature content scores.
    """
    try:
        return get_service().recommend_tracks(
            user_id=user_id,
            top_k=top_k,
            include_listened=include_listened,
            diversity=diversity,
            popularity_penalty=popularity_penalty,
            novelty_weight=novelty_weight,
            explain=explain,
            method=method,
            content_weight=content_weight,
            ltr=ltr,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/tracks/recommend/{user_id}/ltr")
def recommend_tracks_ltr(
    user_id: RequestText,
    top_k: PositiveTopK = 10,
    include_listened: bool = False,
    diversity: UnitInterval = 0.0,
    popularity_penalty: UnitInterval = 0.0,
    novelty_weight: UnitInterval = 0.0,
    explain: bool = False,
    method: Literal["similarity", "hybrid"] = "similarity",
    content_weight: UnitInterval | None = None,
) -> dict[str, object]:
    """Return LTR re-ranked track recommendations for a user."""
    try:
        return get_service().recommend_tracks_ltr(
            user_id=user_id,
            top_k=top_k,
            include_listened=include_listened,
            diversity=diversity,
            popularity_penalty=popularity_penalty,
            novelty_weight=novelty_weight,
            explain=explain,
            method=method,
            content_weight=content_weight,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/tracks/similar/{track_id}")
def similar_tracks(
    track_id: RequestText,
    top_k: PositiveTopK = 10,
) -> dict[str, object]:
    """Return tracks similar to a selected track by audio features."""
    try:
        return get_service().similar_tracks(
            track_id=track_id,
            top_k=top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/tracks/popular")
def popular_tracks(top_k: PositiveTopK = 10) -> dict[str, object]:
    """Return globally popular tracks from the track data."""
    try:
        return get_service().popular_tracks(top_k=top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/tracks/catalog")
def browse_tracks(
    query: OptionalCatalogText = None,
    artist: OptionalCatalogText = None,
    offset: CatalogOffset = 0,
    limit: CatalogLimit = 25,
) -> dict[str, object]:
    """Search and page through tracks available to the recommender."""
    try:
        return get_service().browse_tracks(
            query=query,
            artist=artist,
            offset=offset,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/bandit/status")
def bandit_status() -> dict[str, object]:
    """Return the cold-start bandit lifecycle snapshot."""
    service = get_service()
    status = service.bandit_status()
    status["context_features"] = list(service.context_features)
    return status


class BanditUpdateRequest(BaseModel):
    """Optional validation body for folding bandit feedback."""

    model_config = ConfigDict(strict=True, extra="forbid")

    context_features: list[str] | None = Field(
        default=None,
        max_length=MAX_REQUEST_VALUES,
        description=(
            "Expected context feature names; verified against the state's "
            "recorded set before folding."
        ),
    )
    gamma: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Optional recency discount factor gamma in range (0, 1].",
    )


@app.post("/bandit/update")
def bandit_update(
    payload: BanditUpdateRequest | None = None,
) -> dict[str, object]:
    """Fold pending served feedback into the persisted bandit state."""
    try:
        return get_service().sweep_bandit_feedback(
            context_features=(
                payload.context_features if payload is not None else None
            ),
            gamma=(payload.gamma if payload is not None else None),
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Bandit state not found: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


class BanditSnapshotRequest(BaseModel):
    """Optional payload for creating a bandit state snapshot."""

    model_config = ConfigDict(strict=True, extra="forbid")

    label: str | None = Field(
        default=None,
        max_length=64,
        description="Optional human-readable label for the snapshot.",
    )


@app.get("/bandit/snapshots")
def list_bandit_snapshots() -> dict[str, object]:
    """List persisted bandit state snapshots sorted newest first."""
    snapshots = get_service().list_bandit_snapshots()
    return {
        "snapshots": snapshots,
        "count": len(snapshots),
    }


@app.post("/bandit/snapshots", status_code=201)
def create_bandit_snapshot(
    payload: BanditSnapshotRequest | None = None,
) -> dict[str, object]:
    """Create a new timestamped snapshot of the active bandit state."""
    try:
        path = get_service().create_bandit_snapshot(
            label=payload.label if payload is not None else None
        )
        return {
            "created": True,
            "filename": path.name,
            "path": str(path),
        }
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Bandit state not found: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/bandit/drift")
def get_bandit_drift(
    reference: str | None = Query(
        default=None,
        description=(
            "Optional snapshot path or filename to compare against; "
            "defaults to latest snapshot."
        ),
    ),
) -> dict[str, object]:
    """Compute drift between reference snapshot and current state."""
    try:
        service = get_service()
        ref_path = None
        if reference is not None:
            ref_candidate = Path(reference)
            if not ref_candidate.is_file():
                candidate = BANDIT_SNAPSHOTS_DIR / reference
                if candidate.is_file():
                    ref_candidate = candidate
            ref_path = ref_candidate
        return service.compute_bandit_drift(reference_path=ref_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"State or reference snapshot not found: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


class BanditChampionTagRequest(BaseModel):
    """Request payload for tagging a snapshot as champion."""

    model_config = ConfigDict(strict=True, extra="forbid")

    snapshot_path: str = Field(
        description="Snapshot filename or full path to pin as champion.",
    )


class BanditDriftSafetyRequest(BaseModel):
    """Request payload for evaluating bandit drift safety."""

    model_config = ConfigDict(strict=True, extra="forbid")

    reference_snapshot: str | None = Field(
        default=None,
        description="Optional path or filename of reference snapshot.",
    )
    max_l2_drift: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional maximum L2 drift threshold override.",
    )
    min_cosine_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity threshold override.",
    )
    max_reward_drop: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional maximum reward drop threshold override.",
    )
    allow_dominant_arm_change: bool = Field(
        default=True,
        description="Whether to permit changes in dominant arm without violation.",
    )


class BanditRollbackRequest(BaseModel):
    """Request payload for rolling back bandit state to champion or snapshot."""

    model_config = ConfigDict(strict=True, extra="forbid")

    snapshot_path: str | None = Field(
        default=None,
        description="Optional explicit snapshot path or filename to restore.",
    )
    update_active_policy: bool = Field(
        default=True,
        description="Whether to reload serving cold-start policy in memory.",
    )


@app.post("/bandit/snapshots/champion")
def bandit_tag_champion_snapshot(
    payload: BanditChampionTagRequest,
) -> dict[str, object]:
    """Tag a persisted bandit snapshot as the verified production champion."""
    try:
        srv = get_service()
        return srv.tag_champion_snapshot(snapshot_path=payload.snapshot_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/bandit/snapshots/champion")
def bandit_get_champion() -> dict[str, object]:
    """Get active champion snapshot metadata."""
    srv = get_service()
    champ = srv.get_champion_snapshot()
    if champ is None:
        raise HTTPException(status_code=404, detail="No champion snapshot found.")
    return champ


@app.post("/bandit/drift/safety")
def bandit_drift_safety(
    payload: BanditDriftSafetyRequest | None = None,
) -> dict[str, object]:
    """Evaluate drift safety between active state and reference snapshot."""
    try:
        srv = get_service()
        req = payload or BanditDriftSafetyRequest()
        thresholds: DriftSafetyThresholds | None = None
        if (
            req.max_l2_drift is not None
            or req.min_cosine_similarity is not None
            or req.max_reward_drop is not None
            or not req.allow_dominant_arm_change
        ):
            base_thresh = getattr(srv, "drift_thresholds", None)
            base_l2 = base_thresh.max_l2_drift if base_thresh else 0.5
            base_cos = base_thresh.min_cosine_similarity if base_thresh else 0.8
            base_drop = base_thresh.max_reward_drop if base_thresh else 0.2
            thresholds = DriftSafetyThresholds(
                max_l2_drift=(
                    req.max_l2_drift if req.max_l2_drift is not None else base_l2
                ),
                min_cosine_similarity=(
                    req.min_cosine_similarity
                    if req.min_cosine_similarity is not None
                    else base_cos
                ),
                max_reward_drop=(
                    req.max_reward_drop
                    if req.max_reward_drop is not None
                    else base_drop
                ),
                allow_dominant_arm_change=req.allow_dominant_arm_change,
            )
        return srv.evaluate_bandit_drift_safety(
            reference_path=req.reference_snapshot,
            thresholds=thresholds,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/bandit/rollback")
def bandit_rollback(
    payload: BanditRollbackRequest | None = None,
) -> dict[str, object]:
    """Roll back the active bandit state to champion or specified snapshot."""
    try:
        srv = get_service()
        req = payload or BanditRollbackRequest()
        return srv.rollback_bandit_state(
            snapshot_path=req.snapshot_path,
            update_active_policy=req.update_active_policy,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/bandit/observability")
def bandit_observability(
    include_ope: bool = Query(
        default=False,
        description="Whether to run and attach off-policy evaluation.",
    ),
) -> dict[str, object]:
    """Return consolidated bandit observability diagnostics and telemetry."""
    try:
        return get_service().bandit_observability(include_ope=include_ope)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Resource not found: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


class BanditOpeRequest(BaseModel):
    """Optional validation body for off-policy evaluation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    feedback_journal_path: str | None = Field(
        default=None,
        description="Optional path to logged feedback journal.",
    )
    target_policy: str | dict[str, float] | None = Field(
        default=None,
        description=(
            "Candidate policy algorithm ('linucb', 'thompson_sampling', "
            "'epsilon_greedy'), arm name, or static weights dict."
        ),
    )
    min_propensity: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Minimum propensity clipping threshold.",
    )
    ridge_lambda: float = Field(
        default=1.0,
        gt=0.0,
        description="Ridge regression L2 regularization for Direct Method.",
    )


class BanditCompareRequest(BaseModel):
    """Optional payload for multi-policy comparative benchmarking."""

    model_config = ConfigDict(strict=True, extra="forbid")

    policies: list[str | dict[str, object]] | None = Field(
        default=None,
        description="List of policy names or spec dicts to benchmark.",
    )
    top_k: int = Field(
        default=5,
        gt=0,
        le=MAX_API_RESULTS,
        description="Number of recommendations per cold-start user.",
    )
    rounds: int = Field(
        default=20,
        gt=0,
        le=500,
        description="Number of simulation rounds.",
    )
    seed: int = Field(
        default=42,
        description="RNG seed for simulation.",
    )
    holdout_ratio: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Fraction of users held out for evaluation.",
    )


class BanditDerivePolicyRequest(BaseModel):
    """Payload for deriving cold-start serving weights."""

    model_config = ConfigDict(strict=True, extra="forbid")

    from_state: str | None = Field(
        default=None,
        description="Optional path to a bandit state snapshot JSON.",
    )
    report_path: str | None = Field(
        default=None,
        description="Optional path to a bandit simulation report JSON.",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature for arm weights.",
    )
    temperature_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="Annealing decay rate lambda >= 0.",
    )
    min_temperature: float = Field(
        default=0.05,
        gt=0.0,
        description="Minimum temperature floor under annealing schedule.",
    )
    persist: bool = Field(
        default=False,
        description="Whether to persist derived policy to default serving location.",
    )


@app.post("/bandit/evaluate/off-policy")
def bandit_evaluate_off_policy(
    payload: BanditOpeRequest | None = None,
) -> dict[str, object]:
    """Evaluate candidate policy offline on logged feedback using OPE."""
    try:
        service = get_service()
        req = payload or BanditOpeRequest()
        return service.evaluate_bandit_off_policy(
            target_policy=req.target_policy,
            feedback_journal_path=req.feedback_journal_path,
            min_propensity=req.min_propensity,
            ridge_lambda=req.ridge_lambda,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Feedback journal or policy not found: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/bandit/evaluate/compare")
def bandit_evaluate_compare(
    payload: BanditCompareRequest | None = None,
) -> dict[str, object]:
    """Benchmark multiple contextual bandit policies on identical holdouts."""
    try:
        service = get_service()
        req = payload or BanditCompareRequest()
        return service.compare_bandit_policies(
            policies=req.policies,
            top_k=req.top_k,
            rounds=req.rounds,
            seed=req.seed,
            holdout_ratio=req.holdout_ratio,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Interactions data not found: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/bandit/policy/derive")
def bandit_policy_derive(
    payload: BanditDerivePolicyRequest | None = None,
) -> dict[str, object]:
    """Derive cold-start serving weights with optional temperature annealing."""
    try:
        service = get_service()
        req = payload or BanditDerivePolicyRequest()
        persist_path = COLD_START_POLICY_PATH if req.persist else None

        if req.report_path is not None:
            report_data = load_bandit_report(req.report_path)
            derived = derive_cold_start_policy(
                report_data,
                temperature=req.temperature,
                temperature_decay=req.temperature_decay,
                min_temperature=req.min_temperature,
            )
            if persist_path is not None:
                service.cold_start_policy = derived
            source = f"report:{req.report_path}"
        else:
            derived = service.derive_cold_start_policy_from_state(
                state_path=req.from_state,
                temperature=req.temperature,
                temperature_decay=req.temperature_decay,
                min_temperature=req.min_temperature,
                persist_path=persist_path,
                update_active_policy=req.persist,
            )
            source = (
                f"state:{req.from_state}"
                if req.from_state is not None
                else "active_state"
            )

        return {
            "source": source,
            "policy": derived,
            "temperature": req.temperature,
            "temperature_decay": req.temperature_decay,
            "min_temperature": req.min_temperature,
            "persisted": req.persist,
        }
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Source state or report not found: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
