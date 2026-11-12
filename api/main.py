"""FastAPI app for serving music recommendations."""

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from api.middleware import RequestSafetyMiddleware, TraceContextMiddleware
from music_recommender import __version__
from music_recommender.alerting import (
    AlertEvaluator,
    AlertHistoryBuffer,
    AlertIncidentTracker,
    AlertNotificationThrottler,
    AlertRule,
    AlertSilenceRule,
    AlertSilencingEngine,
    AlertSpilloverStorage,
    AlertTransition,
    DeadLetterAlertRecord,
    DLQRetryPolicy,
    IncidentRemediationRunner,
    MaintenanceWindow,
    MultiChannelDispatcher,
    OpsgenieGateway,
    PagerDutyGateway,
    ReceivedAlertEvent,
    RemediationActionType,
    RemediationRule,
    SilenceMatcher,
    parse_alert_payload,
)
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
from music_recommender.telemetry import (
    collect_service_metric_families,
    export_prometheus_metrics,
)

configure_logging()

logger = logging.getLogger("music_recommender.api.main")

_CORS_ORIGINS_ENV = "CORS_ORIGINS"
_CORS_ORIGINS_RAW = os.getenv(_CORS_ORIGINS_ENV, "*")

service: RecommenderService | None = None
service_load_error: str | None = None
_fallback_alert_evaluator = AlertEvaluator()
_fallback_storage: AlertSpilloverStorage | None = AlertSpilloverStorage(
    storage_dir=Path("data/alerts")
)
_fallback_alert_history = AlertHistoryBuffer(spillover_storage=_fallback_storage)
_fallback_alert_throttler = AlertNotificationThrottler()
_fallback_alert_multichannel_dispatcher = MultiChannelDispatcher(
    throttler=_fallback_alert_throttler
)
_fallback_remediation_runner = IncidentRemediationRunner()
_fallback_silencing_engine = AlertSilencingEngine()
_fallback_alert_multichannel_dispatcher.silencing_engine = _fallback_silencing_engine
_fallback_incident_tracker = AlertIncidentTracker()
_fallback_remediation_runner.register_rule(
    RemediationRule(
        rule_name="StreamingQueueHighUtilization",
        action_type=RemediationActionType.FLUSH_FEEDBACK_QUEUE,
        cooldown_seconds=60.0,
    )
)
_fallback_remediation_runner.register_rule(
    RemediationRule(
        rule_name="StreamingHealthDegraded",
        action_type=RemediationActionType.RESTART_MAINTENANCE_WORKER,
        cooldown_seconds=120.0,
    )
)
_fallback_remediation_runner.register_rule(
    RemediationRule(
        rule_name="DeadLetterSpilloverHigh",
        action_type=RemediationActionType.ROTATE_ALERT_STORAGE,
        cooldown_seconds=300.0,
    )
)
_fallback_remediation_runner.register_rule(
    RemediationRule(
        rule_name="AlertDeliveryFailureRateHigh",
        action_type=RemediationActionType.TRIP_CIRCUIT_BREAKER,
        cooldown_seconds=120.0,
        parameters={"channel": "webhook"},
    )
)
_fallback_remediation_runner.register_handler(
    RemediationActionType.FLUSH_FEEDBACK_QUEUE,
    lambda tr, params: (
        True,
        "Fallback flush feedback queue no-op",
        {"flushed": False},
    ),
)
_fallback_remediation_runner.register_handler(
    RemediationActionType.RESTART_MAINTENANCE_WORKER,
    lambda tr, params: (True, "Fallback restart worker no-op", {"restarted": False}),
)
_fallback_remediation_runner.register_handler(
    RemediationActionType.ROTATE_ALERT_STORAGE,
    lambda tr, params: (True, "Fallback rotate storage no-op", {"rotated": False}),
)
_fallback_remediation_runner.register_handler(
    RemediationActionType.TRIGGER_ONLINE_FOLD,
    lambda tr, params: (True, "Fallback online fold no-op", {"folded": False}),
)
_fallback_remediation_runner.register_handler(
    RemediationActionType.TRIP_CIRCUIT_BREAKER,
    lambda tr, params: (
        True,
        f"Fallback trip circuit breaker for {params.get('channel', 'webhook')}",
        {"channel": params.get("channel", "webhook")},
    ),
)


def _get_alert_storage() -> AlertSpilloverStorage | None:
    if service is not None and service.alert_spillover_storage is not None:
        return service.alert_spillover_storage
    return _fallback_storage


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
    expose_headers=[
        "X-Request-ID",
        "X-Process-Time",
        "traceparent",
        "tracestate",
        "X-Trace-ID",
        "X-Span-ID",
    ],
)
app.add_middleware(
    TraceContextMiddleware,
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


@app.get("/metrics")
def metrics() -> Response:
    """Return Prometheus text exposition format (0.0.4) metrics for scraping."""
    return PlainTextResponse(
        content=export_prometheus_metrics(service),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/metrics/influxdb")
def metrics_influxdb(
    measurement: str = "music_recommender",
) -> Response:
    """Return metrics in InfluxDB Line Protocol 2.0 format."""
    from music_recommender.telemetry import export_influxdb_metrics

    return PlainTextResponse(
        content=export_influxdb_metrics(
            service,
            default_measurement=measurement,
        ),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/metrics/push")
async def metrics_push(request: Request) -> dict[str, object]:
    """Push current service metrics to a remote telemetry collector."""
    from music_recommender.telemetry import (
        TelemetryPushAdapter,
        TelemetryPushConfig,
        collect_service_metric_families,
        get_global_request_tracker,
    )

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {exc}",
        ) from exc

    if not isinstance(body, dict) or "endpoint_url" not in body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must include 'endpoint_url'.",
        )

    try:
        cfg = TelemetryPushConfig(
            endpoint_url=str(body["endpoint_url"]),
            protocol=body.get("protocol", "influxdb"),
            headers=body.get("headers", {}),
            timeout_seconds=float(body.get("timeout_seconds", 10.0)),
            default_measurement=body.get("default_measurement", "music_recommender"),
            dry_run=bool(body.get("dry_run", False)),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid push configuration: {err}",
        ) from err

    families = collect_service_metric_families(service)
    families.extend(get_global_request_tracker().to_metric_families())

    adapter = TelemetryPushAdapter()
    res = adapter.push(families, cfg)
    return res.to_dict()


@app.get("/alerts")
def alerts(dispatch: bool = True) -> dict[str, object]:
    """Evaluate alert rules against live metrics and return their states.

    State changes are posted to the configured alert webhook unless
    ``dispatch=false``. When artifacts are not loaded the ``ServiceDown``
    alert is reported instead of an HTTP error.
    """
    if service is not None:
        return service.evaluate_alerts(dispatch=dispatch)
    transitions = _fallback_alert_evaluator.evaluate(
        collect_service_metric_families(None)
    )
    states = _fallback_alert_evaluator.states()
    return {
        "alerts": states,
        "firing_count": sum(1 for a in states if a["status"] == "firing"),
        "transitions": [t.to_dict() for t in transitions],
        "webhook_configured": False,
        "notifications_delivered": 0,
    }


@app.post("/alerts/webhook")
async def alerts_webhook(
    request: Request,
    channel: str | None = None,
) -> dict[str, object]:
    """Ingest alert notifications from Alertmanager or direct webhooks."""
    import json as _json

    raw_bytes = await request.body()
    raw_str = raw_bytes.decode("utf-8", errors="replace")

    try:
        body = _json.loads(raw_str)
    except Exception as exc:
        logger.warning("Failed to decode JSON payload in /alerts/webhook: %s", exc)
        dlq_rec = DeadLetterAlertRecord(
            reason="malformed_json",
            raw_payload=raw_str[:5000],
            channel=channel or "default",
            error_message=str(exc),
            headers=dict(request.headers),
            source=request.client.host if request.client else None,
        )
        storage = _get_alert_storage()
        if storage is not None:
            storage.record_dead_letter(dlq_rec)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {exc}",
        ) from exc

    if not isinstance(body, dict):
        dlq_rec = DeadLetterAlertRecord(
            reason="invalid_payload_type",
            raw_payload=raw_str[:5000],
            channel=channel or "default",
            error_message="Expected JSON object, got " + type(body).__name__,
            headers=dict(request.headers),
            source=request.client.host if request.client else None,
        )
        storage = _get_alert_storage()
        if storage is not None:
            storage.record_dead_letter(dlq_rec)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Alert payload must be a JSON object.",
        )

    events = parse_alert_payload(body, channel=channel)
    history = service.alert_history if service is not None else _fallback_alert_history
    history.record_many(events)

    for event in events:
        log_level = logging.ERROR if event.severity == "critical" else logging.WARNING
        logger.log(
            log_level,
            "Received alert [%s] %s (status=%s, channel=%s): %s",
            event.severity.upper(),
            event.alertname,
            event.status,
            event.channel or "default",
            event.summary,
        )

    return {
        "status": "received",
        "count": len(events),
        "channel": channel,
        "alerts": [e.alertname for e in events],
    }


@app.get("/alerts/history")
def alerts_history(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    channel: str | None = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    severity: str | None = None,
    include_spillover: bool = False,
) -> dict[str, object]:
    """Return recently received alert notifications matching criteria."""
    history = service.alert_history if service is not None else _fallback_alert_history
    if include_spillover and hasattr(history, "get_combined_events"):
        combined = history.get_combined_events(
            limit=limit,
            channel=channel,
            status=status_filter,
            severity=severity,
        )
        return {
            "total": len(combined),
            "events": combined,
            "spillover_included": True,
        }
    events = history.get_events(
        limit=limit,
        channel=channel,
        status=status_filter,
        severity=severity,
    )
    return {
        "total": len(events),
        "events": [e.to_dict() for e in events],
        "spillover_included": False,
    }


@app.get("/alerts/dlq")
def alerts_dlq(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> dict[str, object]:
    """Return captured dead-letter alert notifications from persistent storage."""
    storage = _get_alert_storage()
    records = storage.read_dead_letters(limit=limit) if storage is not None else []
    return {
        "total": len(records),
        "records": records,
    }


@app.delete("/alerts/dlq")
def purge_alerts_dlq() -> dict[str, object]:
    """Clear all records in the dead-letter queue."""
    storage = _get_alert_storage()
    purged_count = storage.purge_dead_letters() if storage is not None else 0
    return {
        "status": "purged",
        "purged_count": purged_count,
    }


@app.get("/alerts/storage/stats")
def alert_storage_stats() -> dict[str, object]:
    """Return storage metrics, file sizes, and rotation configurations."""
    storage = _get_alert_storage()
    if storage is None:
        return {
            "spillover_count": 0,
            "spillover_bytes": 0,
            "spillover_rotations": 0,
            "dlq_count": 0,
            "dlq_bytes": 0,
            "dlq_rotations": 0,
            "storage_dir": None,
        }
    return cast(dict[str, object], storage.get_stats())


@app.post("/alerts/rotate")
def alerts_rotate(
    max_backups: Annotated[int, Query(ge=1, le=20)] = 5,
    max_age_days: Annotated[float, Query(ge=0.1, le=365.0)] = 30.0,
) -> dict[str, object]:
    """Trigger file rotation and retention pruning for persistent alert storage."""
    storage = _get_alert_storage()
    if storage is None:
        return {
            "status": "noop",
            "spillover_rotated": False,
            "dlq_rotated": False,
            "pruned_files": 0,
        }
    rot = storage.rotate_storage(max_backups=max_backups)
    ret = storage.enforce_retention(max_age_days=max_age_days, max_backups=max_backups)
    return {
        "status": "rotated",
        "spillover_rotated": rot["spillover_rotated"],
        "dlq_rotated": rot["dlq_rotated"],
        "pruned_files": ret.get("pruned_files", 0),
    }


@app.post("/alerts/dlq/retry")
def alerts_dlq_retry(
    max_retries: Annotated[int, Query(ge=1, le=10)] = 3,
    channel: Annotated[str | None, Query(max_length=50)] = None,
    reason: Annotated[str | None, Query(max_length=50)] = None,
    dry_run: bool = False,
    target_url: Annotated[str | None, Query(max_length=255)] = None,
) -> dict[str, object]:
    """Retry dead-letter alert queue records with exponential backoff."""
    storage = _get_alert_storage()
    if storage is None:
        return {
            "total_scanned": 0,
            "retried": 0,
            "succeeded": 0,
            "failed": 0,
            "exhausted": 0,
            "dry_run": dry_run,
            "results": [],
        }
    policy = DLQRetryPolicy(max_retries=max_retries)
    report = storage.retry_dead_letters(
        policy=policy,
        target_url=target_url,
        channel=channel,
        reason=reason,
        dry_run=dry_run,
    )
    return cast(dict[str, object], report.to_dict())


@app.get("/alerts/archive")
def alerts_archive(
    format: Annotated[str, Query(pattern=r"^(tar\.gz|zip)$")] = "tar.gz",
) -> Response:
    """Create and stream a compressed disaster recovery archive of alert logs."""
    import tempfile

    storage = _get_alert_storage()
    if storage is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert storage is not configured",
        )
    temp_dir = Path(tempfile.gettempdir())
    ext = "zip" if format == "zip" else "tar.gz"
    archive_file = temp_dir / f"alert_archive_{int(time.time())}.{ext}"
    try:
        storage.create_archive_bundle(archive_file, format=format)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create archive bundle: {exc}",
        ) from exc

    media_type = "application/zip" if format == "zip" else "application/gzip"
    return FileResponse(
        archive_file,
        media_type=media_type,
        filename=archive_file.name,
    )


class GatewayTestRequest(BaseModel):
    """Payload to test alert dispatch to an external incident gateway."""

    model_config = ConfigDict(extra="forbid")

    gateway: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = (
        Field(..., description="Target gateway: pagerduty, opsgenie, or webhook")
    )
    routing_key: str | None = Field(
        default=None, description="Routing key for PagerDuty"
    )
    api_key: str | None = Field(default=None, description="API key for Opsgenie")
    endpoint_url: str | None = Field(default=None, description="Custom endpoint URL")
    alertname: str = Field(default="TestAlert", description="Alert rule name")
    severity: str = Field(
        default="warning", description="Severity: warning or critical"
    )
    summary: str = Field(default="Synthetic test alert", description="Alert summary")
    event: str = Field(default="firing", description="Event type: firing or resolved")


@app.get("/alerts/throttling/stats")
def alert_throttling_stats() -> dict[str, object]:
    """Return metrics and state of alert notification rate-limiting."""
    if service is not None:
        return cast(dict[str, object], service.get_alert_throttling_stats())
    return cast(dict[str, object], _fallback_alert_throttler.get_stats())


@app.post("/alerts/throttling/reset")
def alert_throttling_reset() -> dict[str, object]:
    """Reset the alert notification throttling counters and state."""
    if service is not None:
        service.reset_alert_throttling()
    else:
        _fallback_alert_throttler.reset()
    return {"status": "ok", "message": "Alert notification throttling state reset"}


@app.post("/alerts/gateway/test")
def alert_gateway_test(payload: GatewayTestRequest) -> dict[str, object]:
    """Test dispatching a synthetic alert notification to a gateway."""
    gateway_type = payload.gateway.lower()
    if gateway_type not in ("pagerduty", "opsgenie", "webhook"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported gateway type '{payload.gateway}'. "
                "Must be 'pagerduty', 'opsgenie', or 'webhook'."
            ),
        )

    rule = AlertRule(
        name=payload.alertname,
        metric="test_metric",
        comparator=">",
        threshold=1.0,
        severity=payload.severity,
        summary=payload.summary,
    )
    transition = AlertTransition(
        rule=rule,
        event=payload.event,
        value=1.5,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if gateway_type == "pagerduty":
        routing_key = payload.routing_key or os.getenv("PAGERDUTY_ROUTING_KEY", "")
        if not routing_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Routing key is required for PagerDuty gateway test.",
            )
        gw = PagerDutyGateway(
            routing_key=routing_key,
            endpoint_url=payload.endpoint_url
            or "https://events.pagerduty.com/v2/enqueue",
        )
        ok, code, msg = gw.send(transition)
        return {
            "gateway": "pagerduty",
            "success": ok,
            "status_code": code,
            "response": msg,
        }

    if gateway_type == "opsgenie":
        api_key = payload.api_key or os.getenv("OPSGENIE_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="API key is required for Opsgenie gateway test.",
            )
        gw_og = OpsgenieGateway(
            api_key=api_key,
            endpoint_url=payload.endpoint_url or "https://api.opsgenie.com/v2/alerts",
        )
        ok, code, msg = gw_og.send(transition)
        return {
            "gateway": "opsgenie",
            "success": ok,
            "status_code": code,
            "response": msg,
        }

    # webhook
    url = payload.endpoint_url or os.getenv("ALERT_WEBHOOK_URL", "")
    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="endpoint_url is required for webhook gateway test.",
        )
    from music_recommender.alerting import WebhookDispatcher

    try:
        dispatcher = WebhookDispatcher(url)
        delivered = dispatcher.dispatch([transition])
        return {
            "gateway": "webhook",
            "success": delivered > 0,
            "status_code": 200 if delivered > 0 else 500,
            "delivered": delivered,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Webhook delivery failed: {exc}",
        ) from exc


class SilenceMatcherModel(BaseModel):
    """Matcher criteria for alert silencing."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Target property: rule_name, alertname, severity, or label key",
    )
    value: str = Field(..., description="Value to match")
    is_regex: bool = Field(False, description="Match using regular expression")


class CreateSilenceRequest(BaseModel):
    """Request payload to create a new alert silence rule."""

    model_config = ConfigDict(extra="forbid")

    silence_id: str | None = Field(
        None, description="Optional custom silence identifier"
    )
    matchers: list[SilenceMatcherModel] = Field(
        default_factory=list, description="List of matchers"
    )
    duration_seconds: float | None = Field(
        None, description="Silence duration in seconds from now"
    )
    starts_at: float | None = Field(
        None, description="Silence start timestamp (epoch seconds)"
    )
    ends_at: float | None = Field(
        None, description="Silence end timestamp (epoch seconds)"
    )
    author: str = Field("operator", description="Author creating the silence")
    comment: str = Field("", description="Context or explanation for silence")


class CreateMaintenanceWindowRequest(BaseModel):
    """Request payload to schedule a maintenance window."""

    model_config = ConfigDict(extra="forbid")

    window_id: str | None = Field(None, description="Optional custom window identifier")
    name: str = Field(
        "scheduled_maintenance", description="Name of the maintenance window"
    )
    duration_seconds: float | None = Field(
        None, description="Window duration in seconds from now"
    )
    starts_at: float | None = Field(
        None, description="Window start timestamp (epoch seconds)"
    )
    ends_at: float | None = Field(
        None, description="Window end timestamp (epoch seconds)"
    )
    suppress_all: bool = Field(
        True, description="Suppress all alert rules during window"
    )
    affected_rules: list[str] = Field(
        default_factory=list, description="Specific rules to suppress"
    )
    author: str = Field("operator", description="Author scheduling the window")
    description: str = Field("", description="Details regarding maintenance tasks")


class RemediationTriggerRequest(BaseModel):
    """Payload to trigger an automated self-healing remediation action."""

    model_config = ConfigDict(extra="forbid")

    rule_name: str = Field(
        default="manual", description="Alert rule or trigger identifier"
    )
    action_type: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(..., description="Remediation action type")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )
    dry_run: bool = Field(
        default=False, description="Simulate action without side effects"
    )


@app.get("/alerts/remediation/rules")
def alerts_remediation_rules() -> dict[str, object]:
    """Return configured automated incident remediation rules."""
    rules = (
        service.get_remediation_rules()
        if service is not None
        else _fallback_remediation_runner.get_rules()
    )
    return {"rules": rules, "count": len(rules)}


@app.get("/alerts/remediation/history")
def alerts_remediation_history(
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> dict[str, object]:
    """Return execution history of self-healing remediation actions."""
    history = (
        service.get_remediation_history(limit=limit)
        if service is not None
        else _fallback_remediation_runner.get_history(limit=limit)
    )
    return {"history": history, "count": len(history)}


@app.post("/alerts/remediation/trigger")
def alerts_remediation_trigger(payload: RemediationTriggerRequest) -> dict[str, object]:
    """Manually trigger a self-healing remediation action on demand."""
    if service is not None:
        result = service.trigger_remediation_action(
            rule_name=payload.rule_name,
            action_type=payload.action_type,
            parameters=payload.parameters,
            dry_run=payload.dry_run,
        )
    else:
        res = _fallback_remediation_runner.execute_action(
            rule_name=payload.rule_name,
            action_type=payload.action_type,
            parameters=payload.parameters,
            dry_run=payload.dry_run,
        )
        result = res.to_dict()
    return {"status": "ok", "result": result}


@app.get("/alerts/circuit-breaker/stats")
def alert_circuit_breaker_stats() -> dict[str, object]:
    """Return health states and trip counters for outbound circuit breakers."""
    cbs = (
        service.get_circuit_breaker_stats()
        if service is not None
        else _fallback_alert_multichannel_dispatcher.get_stats().get(
            "circuit_breakers", {}
        )
    )
    return {"circuit_breakers": cbs}


@app.post("/alerts/circuit-breaker/reset")
def alert_circuit_breaker_reset() -> dict[str, object]:
    """Reset all outbound circuit breakers back to CLOSED."""
    if service is not None:
        service.reset_circuit_breakers()
    else:
        _fallback_alert_multichannel_dispatcher.reset()
    return {"status": "ok", "message": "Circuit breakers reset to CLOSED"}


@app.get("/alerts/silences")
def alerts_silences_list(
    active_only: bool = False,
) -> dict[str, object]:
    """Return active or all alert silence rules."""
    if service is not None:
        rules = service.get_silence_rules(active_only=active_only)
    else:
        raw_rules = _fallback_silencing_engine.list_silences(active_only=active_only)
        rules = [r.to_dict() for r in raw_rules]
    return {"silences": rules, "count": len(rules)}


@app.post("/alerts/silences")
def alerts_silences_create(payload: CreateSilenceRequest) -> dict[str, object]:
    """Create an alert silence rule."""
    now = time.time()
    starts_at = payload.starts_at if payload.starts_at is not None else now
    if payload.ends_at is not None:
        ends_at = payload.ends_at
    elif payload.duration_seconds is not None:
        ends_at = starts_at + payload.duration_seconds
    else:
        ends_at = starts_at + 3600.0

    matchers = [
        SilenceMatcher(name=m.name, value=m.value, is_regex=m.is_regex)
        for m in payload.matchers
    ]
    rule_kwargs: dict[str, Any] = {
        "matchers": matchers,
        "starts_at": starts_at,
        "ends_at": ends_at,
        "author": payload.author,
        "comment": payload.comment,
        "enabled": True,
    }
    if payload.silence_id:
        rule_kwargs["silence_id"] = payload.silence_id

    silence_rule = AlertSilenceRule(**rule_kwargs)
    if service is not None:
        created = service.add_silence_rule(silence_rule)
    else:
        created = _fallback_silencing_engine.add_silence(silence_rule)
    return {"status": "ok", "silence": created.to_dict()}


@app.delete("/alerts/silences/{silence_id}")
def alerts_silences_delete(silence_id: str) -> dict[str, object]:
    """Delete or cancel an alert silence rule."""
    if service is not None:
        removed = service.remove_silence_rule(silence_id)
    else:
        removed = _fallback_silencing_engine.remove_silence(silence_id)
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"Silence rule '{silence_id}' not found"
        )
    return {"status": "ok", "removed": True, "silence_id": silence_id}


@app.get("/alerts/maintenance")
def alerts_maintenance_list(
    active_only: bool = False,
) -> dict[str, object]:
    """Return active or all scheduled maintenance windows."""
    if service is not None:
        windows = service.get_maintenance_windows(active_only=active_only)
    else:
        raw_windows = _fallback_silencing_engine.list_maintenance_windows(
            active_only=active_only
        )
        windows = [w.to_dict() for w in raw_windows]
    return {"maintenance_windows": windows, "count": len(windows)}


@app.post("/alerts/maintenance")
def alerts_maintenance_create(
    payload: CreateMaintenanceWindowRequest,
) -> dict[str, object]:
    """Schedule a maintenance window for alert suppression."""
    now = time.time()
    starts_at = payload.starts_at if payload.starts_at is not None else now
    if payload.ends_at is not None:
        ends_at = payload.ends_at
    elif payload.duration_seconds is not None:
        ends_at = starts_at + payload.duration_seconds
    else:
        ends_at = starts_at + 7200.0

    mw_kwargs: dict[str, Any] = {
        "name": payload.name,
        "starts_at": starts_at,
        "ends_at": ends_at,
        "suppress_all": payload.suppress_all,
        "affected_rules": payload.affected_rules,
        "author": payload.author,
        "description": payload.description,
        "enabled": True,
    }
    if payload.window_id:
        mw_kwargs["window_id"] = payload.window_id

    window = MaintenanceWindow(**mw_kwargs)
    if service is not None:
        created = service.schedule_maintenance_window(window)
    else:
        created = _fallback_silencing_engine.add_maintenance_window(window)
    return {"status": "ok", "maintenance_window": created.to_dict()}


@app.delete("/alerts/maintenance/{window_id}")
def alerts_maintenance_delete(window_id: str) -> dict[str, object]:
    """Cancel a scheduled maintenance window."""
    if service is not None:
        removed = service.cancel_maintenance_window(window_id)
    else:
        removed = _fallback_silencing_engine.remove_maintenance_window(window_id)
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"Maintenance window '{window_id}' not found"
        )
    return {"status": "ok", "cancelled": True, "window_id": window_id}


@app.get("/alerts/analytics")
def alerts_analytics_summary(
    window_seconds: float = 86400.0,
    slo_target_pct: float = 99.9,
) -> dict[str, object]:
    """Return aggregated incident analytics (MTTD, MTTR, MTBF, and SLO compliance)."""
    if service is not None:
        summary = service.get_incident_analytics(
            window_seconds=window_seconds,
            slo_target_pct=slo_target_pct,
        )
    else:
        summary = _fallback_incident_tracker.get_analytics(
            window_seconds=window_seconds,
            slo_target_pct=slo_target_pct,
        )
    summary["window_seconds"] = window_seconds
    summary["slo_target_pct"] = slo_target_pct
    return summary


@app.get("/alerts/analytics/incidents")
def alerts_analytics_incidents(
    limit: int = 50,
    rule_name: str | None = None,
    active_only: bool = False,
) -> dict[str, object]:
    """Return historical or active incident lifecycle records."""
    if service is not None:
        incidents = service.get_incident_history(
            limit=limit,
            rule_name=rule_name,
            active_only=active_only,
        )
    else:
        if active_only:
            recs = _fallback_incident_tracker.get_active_incidents()
            if rule_name:
                recs = [r for r in recs if r.rule_name == rule_name]
            incidents = [r.to_dict() for r in recs[:limit]]
        else:
            recs = _fallback_incident_tracker.get_history(
                limit=limit, rule_name=rule_name
            )
            incidents = [r.to_dict() for r in recs]
    return {
        "incidents": incidents,
        "count": len(incidents),
        "active_only": active_only,
    }


@app.get("/alerts/analytics/slo")
def alerts_analytics_slo(
    window_seconds: float = 86400.0,
    slo_target_pct: float = 99.9,
) -> dict[str, object]:
    """Return Service Level Objective (SLO) compliance percentages per alert rule."""
    if service is not None:
        report = service.get_slo_compliance_report(
            window_seconds=window_seconds,
            slo_target_pct=slo_target_pct,
        )
    else:
        stats = _fallback_incident_tracker.get_all_rule_stats(
            window_seconds=window_seconds,
            slo_target_pct=slo_target_pct,
        )
        report = {
            "window_seconds": window_seconds,
            "slo_target_pct": slo_target_pct,
            "rules": {name: s.to_dict() for name, s in stats.items()},
        }
    return report


@app.post("/alerts/replay")
async def alerts_replay(
    request: Request,
    dry_run: bool = False,
) -> dict[str, object]:
    """Replay alert events into the active alert history buffer."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {exc}",
        ) from exc

    if not isinstance(body, dict) or "events" not in body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must be a JSON object containing an 'events' list.",
        )

    raw_events = body.get("events")
    if not isinstance(raw_events, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'events' must be a list of alert objects.",
        )

    parsed_events: list[ReceivedAlertEvent] = []
    for item in raw_events:
        if isinstance(item, dict):
            parsed_events.extend(parse_alert_payload(item))

    if not dry_run:
        history = (
            service.alert_history if service is not None else _fallback_alert_history
        )
        history.record_many(parsed_events)

    return {
        "status": "replayed" if not dry_run else "dry_run",
        "replayed_count": len(parsed_events),
        "dry_run": dry_run,
        "alerts": [e.alertname for e in parsed_events],
    }


@app.get("/alerts/correlation")
def alerts_correlation(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    window_seconds: Annotated[float, Query(ge=1.0, le=86400.0)] = 300.0,
    include_spillover: bool = False,
) -> dict[str, object]:
    """Perform incident root-cause correlation analysis over active alert history."""
    from music_recommender.alerting import AlertCorrelationGraph

    if service is not None:
        report = service.correlate_alert_history(
            limit=limit,
            window_seconds=window_seconds,
            include_spillover=include_spillover,
        )
        return cast(dict[str, object], report.to_dict())

    events = _fallback_alert_history.get_events(limit=limit)
    graph = AlertCorrelationGraph()
    fallback_report = graph.correlate(events, window_seconds=window_seconds)
    return cast(dict[str, object], fallback_report.to_dict())


@app.post("/alerts/correlate")
async def alerts_correlate(
    request: Request,
    window_seconds: float = 300.0,
) -> dict[str, object]:
    """Analyze an arbitrary collection of alert events and attribute root causes."""
    from music_recommender.alerting import AlertCorrelationGraph

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {exc}",
        ) from exc

    if not isinstance(body, dict) or "alerts" not in body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must be a JSON object containing an 'alerts' list.",
        )

    alerts_list = body.get("alerts")
    if not isinstance(alerts_list, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'alerts' must be a list of alert notifications.",
        )

    graph = AlertCorrelationGraph()
    report = graph.correlate(alerts_list, window_seconds=window_seconds)
    return report.to_dict()


class TraceCorrelationRequest(BaseModel):
    """Payload for correlating an alert or incident with candidate traces."""

    model_config = ConfigDict(extra="forbid")

    alert_name: str = Field(..., min_length=1, max_length=128)
    timestamp: float | None = Field(default=None, ge=0.0)
    window_seconds: float = Field(default=300.0, ge=1.0, le=86400.0)
    incident_id: str | None = Field(default=None, max_length=64)


@app.get("/traces")
def get_traces(
    trace_id: str | None = None,
    name: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Search buffered execution spans."""
    lim = max(1, min(limit, 1000))
    if service is not None:
        spans = service.search_spans(
            trace_id=trace_id, name=name, status=status, limit=lim
        )
    else:
        from music_recommender.tracing import get_global_tracer

        spans = get_global_tracer().get_spans(
            trace_id=trace_id, name=name, status=status, limit=lim
        )
    return {
        "count": len(spans),
        "spans": [s.to_dict() for s in spans],
    }


@app.get("/traces/export/otlp")
def export_traces_otlp(
    trace_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Export stored spans into canonical OpenTelemetry JSON format."""
    lim = max(1, min(limit, 1000))
    if service is not None:
        return service.export_otlp_traces(trace_id=trace_id, limit=lim)
    from music_recommender.tracing import get_global_tracer

    return get_global_tracer().to_otlp_export(trace_id=trace_id, limit=lim)


@app.get("/traces/{trace_id}")
def get_trace_by_id(trace_id: str) -> dict[str, Any]:
    """Retrieve all spans belonging to a trace, ordered chronologically."""
    tid = trace_id.strip().lower()
    if service is not None:
        spans = service.get_trace(tid)
    else:
        from music_recommender.tracing import get_global_tracer

        spans = get_global_tracer().get_trace(tid)

    if not spans:
        raise HTTPException(
            status_code=404,
            detail=f"Trace '{trace_id}' not found.",
        )
    return {
        "trace_id": tid,
        "span_count": len(spans),
        "spans": [s.to_dict() for s in spans],
    }


@app.post("/traces/correlate/alert")
def correlate_alert_traces(req: TraceCorrelationRequest) -> dict[str, Any]:
    """Correlate an alert or incident with candidate traces and error spans."""
    if service is not None:
        result = service.correlate_alert_traces(
            alert_name=req.alert_name,
            timestamp=req.timestamp,
            window_seconds=req.window_seconds,
            incident_id=req.incident_id,
        )
    else:
        from music_recommender.tracing import (
            AlertTraceCorrelator,
            get_global_tracer,
        )

        correlator = AlertTraceCorrelator(get_global_tracer())
        result = correlator.correlate(
            alert_name=req.alert_name,
            timestamp=req.timestamp,
            window_seconds=req.window_seconds,
            incident_id=req.incident_id,
        )
    return result.to_dict()


@app.get("/maintenance/status")
def maintenance_status() -> dict[str, object]:
    """Return the maintenance worker status and metrics if active."""
    svc = get_service()
    status = svc.maintenance_status()
    if status is None:
        return {"active": False, "message": "Maintenance worker not configured"}
    return {"active": True, **status}


class TriggerFoldRequest(BaseModel):
    """Request body for triggering a manual maintenance fold."""

    model_config = {"extra": "forbid"}


@app.post("/maintenance/trigger")
def trigger_fold(payload: TriggerFoldRequest | None = None) -> dict[str, object]:
    """Trigger an immediate bandit feedback fold (manual sweep)."""
    svc = get_service()
    try:
        result = svc.trigger_maintenance_sweep()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fold failed: {e}") from e
    return {"success": True, **result}


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
