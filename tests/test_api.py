import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

import api.main as api_main
from api.middleware import RequestSafetyMiddleware


class FakeService:
    def __init__(self) -> None:
        self.context_features: list[str] = [
            "log_plays",
            "log_unique_artists",
            "mean_popularity_rank",
        ]
        self.cold_start_policy: dict[str, float] = {"popular": 1.0}
        self.closed: bool = False

    def close(self, timeout: float | None = 5.0) -> None:
        self.closed = True

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "artifact_version": "4.0",
            "streaming_status": "healthy",
        }

    def streaming_health(self) -> dict[str, object]:
        return {
            "status": "healthy",
            "reasons": [],
            "queue": {
                "running": True,
                "buffered_events": 0,
                "max_queue_size": 1000,
                "utilization_pct": 0.0,
                "backpressure": False,
                "total_enqueued": 10,
                "total_flushed_batches": 2,
                "total_flushed_records": 10,
                "flush_errors": 0,
            },
            "maintenance": {
                "running": True,
                "total_sweeps": 1,
                "total_records_swept": 10,
                "failed_sweeps": 0,
                "last_sweep_at": "2026-10-06T00:00:00Z",
                "avg_sweep_duration_seconds": 0.02,
                "min_sweep_duration_seconds": 0.02,
                "max_sweep_duration_seconds": 0.02,
            },
            "drift": {
                "is_safe": True,
                "violations": [],
                "warnings": [],
            },
        }

    def bandit_observability(
        self, *, include_ope: bool = False
    ) -> dict[str, object]:
        obs: dict[str, object] = {
            "timestamp": "2026-10-06T00:00:00Z",
            "health": self.streaming_health(),
            "state": {
                "available": True,
                "policy_type": "linucb",
                "alpha": 0.5,
                "total_selections": 10,
            },
            "journal": {
                "path": "reports/bandit_feedback.json",
                "total_records": 10,
                "pending_records": 0,
            },
            "snapshots": {
                "total_count": 1,
                "champion": "bandit_state_champ.json",
                "latest": "bandit_state_champ.json",
            },
            "off_policy_evaluation": None,
        }
        if include_ope:
            obs["off_policy_evaluation"] = {
                "available": True,
                "target_policy": "popular",
                "summary": {"records_evaluated": 10, "effective_sample_size": 8.5},
                "metrics": {
                    "ips": {"value": 0.6, "ci_95": [0.5, 0.7]},
                },
            }
        return obs

    def metadata(self) -> dict[str, object]:
        return {"version": "4.0", "hybrid_config": {"default_content_weight": 0.25}}

    def bandit_status(self) -> dict[str, object]:
        return {
            "available": True,
            "auto_sweep_threshold": 50,
            "snapshots_count": 1,
            "state": {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "context_dim": 3,
                "alpha": 0.5,
                "total_selections": 2,
                "arms": {
                    "popular": {
                        "selections": 2,
                        "total_reward": 1.4,
                        "mean_reward": 0.7,
                    },
                },
            },
            "policy": {"popular": 1.0},
            "journal": {"length": 2, "pending": 0},
            "last_fold": {"offset": 2, "at": "2026-01-01T00:00:00+00:00"},
        }

    def list_bandit_snapshots(self) -> list[dict[str, object]]:
        return [
            {
                "filename": "bandit_state_20260927T000000Z.json",
                "path": "reports/bandit_snapshots/bandit_state_20260927T000000Z.json",
                "timestamp": "20260927T000000Z",
                "label": "baseline",
                "size_bytes": 1024,
            }
        ]

    def create_bandit_snapshot(self, label: str | None = None) -> Path:
        suffix = f"_{label}" if label else ""
        filename = f"bandit_state_20260927T000000Z{suffix}.json"
        return Path(f"reports/bandit_snapshots/{filename}")

    def compute_bandit_drift(
        self, reference_path: Path | str | None = None
    ) -> dict[str, object]:
        return {
            "current_state_path": "reports/bandit_state.json",
            "reference_state_path": str(
                reference_path
                or "reports/bandit_snapshots/bandit_state_20260927T000000Z.json"
            ),
            "has_drift": False,
            "max_l2_drift": 0.0,
            "mean_l2_drift": 0.0,
            "arms": {
                "popular": {
                    "theta_l2_drift": 0.0,
                    "theta_cosine_similarity": 1.0,
                    "selections_growth": 0,
                    "total_reward_diff": 0.0,
                    "mean_reward_diff": 0.0,
                }
            },
            "dominant_arm_a": "popular",
            "dominant_arm_b": "popular",
            "dominant_arm_changed": False,
        }

    def evaluate_bandit_drift_safety(
        self,
        reference_path: Path | str | None = None,
        state_path: Path | str | None = None,
        snapshot_dir: Path | str | None = None,
        thresholds: object = None,
    ) -> dict[str, object]:
        return {
            "is_safe": True,
            "violations": [],
            "warnings": [],
            "metrics": {
                "max_l2_drift": 0.05,
                "mean_l2_drift": 0.02,
                "min_cosine_similarity": 0.95,
                "max_reward_drop": 0.01,
                "dominant_arm_a": "popular",
                "dominant_arm_b": "popular",
                "dominant_arm_changed": False,
            },
            "reference_snapshot": str(reference_path or "champ.json"),
            "evaluated_at": "2026-10-06T00:00:00Z",
        }

    def rollback_bandit_state(
        self,
        snapshot_path: Path | str | None = None,
        target_state_path: Path | str | None = None,
        snapshot_dir: Path | str | None = None,
        *,
        update_active_policy: bool = True,
    ) -> dict[str, object]:
        self.cold_start_policy = {"popular": 1.0}
        return {
            "restored": True,
            "target_state_path": "reports/bandit_state.json",
            "restored_from": str(snapshot_path or "reports/champ.json"),
            "restored_at": "2026-10-06T00:00:00Z",
            "label": "champion",
            "is_champion": True,
            "policy_updated": update_active_policy,
            "policy": self.cold_start_policy,
        }

    def tag_champion_snapshot(
        self,
        snapshot_path: Path | str,
        snapshot_dir: Path | str | None = None,
    ) -> dict[str, object]:
        return {
            "tagged": True,
            "path": str(snapshot_path),
            "filename": Path(str(snapshot_path)).name,
            "is_champion": True,
            "label": "champion_v1",
            "total_selections": 10,
        }

    def get_champion_snapshot(
        self,
        snapshot_dir: Path | str | None = None,
    ) -> dict[str, object] | None:
        return {
            "path": "reports/bandit_snapshots/bandit_state_champ.json",
            "filename": "bandit_state_champ.json",
            "label": "champion_v1",
            "is_champion": True,
            "created_at": "2026-10-06T00:00:00Z",
            "total_selections": 10,
        }

    def sweep_bandit_feedback(
        self,
        *,
        context_features: list[str] | None = None,
        gamma: float | None = None,
    ) -> dict[str, object]:
        self.last_sweep_called = True
        self.last_gamma_passed = gamma
        if context_features is not None and context_features != self.context_features:
            raise ValueError(
                f"Requested context features {context_features} do not match "
                f"the state's recorded feature set {self.context_features}"
            )
        if gamma is not None and (gamma <= 0.0 or gamma > 1.0):
            raise ValueError(f"gamma must be in the range (0, 1], got {gamma}.")
        status = self.bandit_status()
        if gamma is not None:
            status["gamma"] = gamma
        return status

    def evaluate_bandit_off_policy(
        self,
        *,
        target_policy: object = None,
        feedback_journal_path: object = None,
        behavior_propensities: object = None,
        min_propensity: float = 0.01,
        ridge_lambda: float = 1.0,
    ) -> dict[str, object]:
        return {
            "target_policy": str(target_policy or "linucb_bandit"),
            "summary": {
                "records_evaluated": 10,
                "logging_mean_reward": 0.5,
                "match_rate": 0.8,
                "effective_sample_size": 8.5,
            },
            "metrics": {
                "ips": {"value": 0.6, "standard_error": 0.05},
                "snips": {"value": 0.58},
                "direct_method": {"value": 0.62, "standard_error": 0.04},
                "doubly_robust": {
                    "value": 0.61,
                    "standard_error": 0.04,
                    "lift_over_logging": 0.11,
                },
            },
            "arms": {"popular": {"target_selection_ratio": 0.5}},
        }

    def compare_bandit_policies(
        self,
        interactions_path: object = None,
        *,
        policies: object = None,
        top_k: int = 5,
        rounds: int = 20,
        seed: int = 42,
        holdout_seed: int = 42,
        holdout_ratio: float = 0.5,
        bootstrap_ratio: float = 0.5,
    ) -> dict[str, object]:
        return {
            "config": {"top_k": top_k, "rounds": rounds, "seed": seed},
            "policies": {
                "linucb": {
                    "cumulative_reward": 10.0,
                    "mean_reward": 0.5,
                    "regret": 2.0,
                },
                "thompson_sampling": {
                    "cumulative_reward": 11.0,
                    "mean_reward": 0.55,
                    "regret": 1.0,
                },
            },
            "summary": {
                "champion": "thompson_sampling",
                "leaderboard": [
                    {
                        "name": "thompson_sampling",
                        "mean_reward": 0.55,
                        "win_rate": 0.6,
                    },
                    {"name": "linucb", "mean_reward": 0.5, "win_rate": 0.4},
                ],
            },
        }

    def derive_cold_start_policy_from_state(
        self,
        *,
        state_path: object = None,
        temperature: float = 1.0,
        temperature_decay: float = 0.0,
        min_temperature: float = 0.01,
        persist_path: object = None,
        update_active_policy: bool = True,
    ) -> dict[str, float]:
        policy = {"popular": 0.6, "balanced": 0.25, "long_tail": 0.15}
        if update_active_policy:
            self.cold_start_policy = policy
        return policy

    def popular_artists(self, top_k: int) -> dict[str, object]:
        return {
            "strategy": "popular_baseline",
            "recommendations": [
                {"artist_id": "artist_1", "artist_name": "A", "score": 10.0}
            ][:top_k],
        }

    def popular_tracks(self, top_k: int) -> dict[str, object]:
        return {
            "strategy": "popular_baseline",
            "recommendations": [
                {
                    "track_id": "track_1",
                    "track_name": "Hit",
                    "artist_name": "Test Artist",
                    "score": 10.0,
                }
            ][:top_k],
        }

    def browse_artists(
        self,
        *,
        query: str | None,
        genre: str | None,
        mood_tag: str | None,
        country: str | None,
        era: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, object]:
        return {
            "total": 1,
            "offset": offset,
            "limit": limit,
            "has_more": False,
            "filters": {
                "query": query,
                "genre": genre,
                "mood_tag": mood_tag,
                "country": country,
                "era": era,
            },
            "artists": [
                {
                    "artist_id": "artist_1",
                    "artist_name": "A",
                    "genres": ["pop"],
                    "mood_tags": ["bright"],
                    "country": "Canada",
                    "era": "2020s",
                    "popularity_rank": 1,
                }
            ][:limit],
        }

    def recommend_user(
        self,
        user_id: str,
        top_k: int,
        include_listened: bool,
        diversity: float,
        popularity_penalty: float,
        content_weight: float,
        explain: bool,
        novelty_weight: float = 0.0,
        record_feedback: bool = False,
        context: list[float] | None = None,
    ) -> dict[str, object]:
        self.last_record_feedback = record_feedback
        self.last_context = context
        if record_feedback:
            return {
                "user_id": user_id,
                "strategy": "bandit_fallback",
                "feedback": {
                    "recorded": True,
                    "arm": "popular",
                    "reward": 1.0,
                    "context": (
                        list(context) if context is not None else [0.0, 0.0, 0.0]
                    ),
                },
                "recommendations": [
                    {
                        "artist_id": "artist_2",
                        "artist_name": "B",
                        "score": 0.9,
                    }
                ][:top_k],
            }
        return {
            "user_id": user_id,
            "strategy": "hybrid_personalized",
            "content_weight": content_weight,
            "recommendations": [
                {
                    "artist_id": "artist_2",
                    "artist_name": "B",
                    "score": 0.9,
                    "score_components": {"hybrid_score": 0.9},
                    "reasons": ["Matches your selected preferences: pop"]
                    if explain
                    else [],
                }
            ][:top_k],
            "include_listened": include_listened,
            "diversity": diversity,
            "popularity_penalty": popularity_penalty,
            "novelty_weight": novelty_weight,
        }

    def recommend_profile(
        self,
        artist_ids: list[str],
        genres: list[str],
        mood_tags: list[str],
        top_k: int,
        explain: bool,
    ) -> dict[str, object]:
        return {
            "strategy": "content_profile",
            "recommendations": [
                {
                    "artist_id": "artist_3",
                    "artist_name": "C",
                    "score": 0.8,
                    "reasons": [f"Seed artists: {', '.join(artist_ids)}"]
                    if explain
                    else [],
                    "genres": genres,
                    "mood_tags": mood_tags,
                }
            ][:top_k],
        }

    def recommend_user_ltr(
        self,
        user_id: str,
        top_k: int,
        include_listened: bool,
        diversity: float,
        popularity_penalty: float,
        novelty_weight: float = 0.0,
    ) -> dict[str, object]:
        return {
            "user_id": user_id,
            "strategy": "ltr_ranked",
            "recommendations": [
                {
                    "artist_id": "artist_4",
                    "artist_name": "D",
                    "score": 0.85,
                }
            ][:top_k],
            "include_listened": include_listened,
            "diversity": diversity,
            "popularity_penalty": popularity_penalty,
            "novelty_weight": novelty_weight,
        }

    def recommend_session(
        self,
        artist_ids: list[str],
        genres: list[str],
        mood_tags: list[str],
        user_id: str | None,
        top_k: int,
        exclude_artist_ids: list[str],
        include_listened: bool,
        diversity: float,
        popularity_penalty: float,
        content_weight: float,
        explain: bool,
        novelty_weight: float = 0.0,
    ) -> dict[str, object]:
        return {
            "user_id": user_id,
            "strategy": "session_hybrid" if user_id else "session_content",
            "content_weight": content_weight,
            "seed_artist_ids": artist_ids,
            "genres": genres,
            "mood_tags": mood_tags,
            "excluded_artist_ids": exclude_artist_ids,
            "include_listened": include_listened,
            "diversity": diversity,
            "popularity_penalty": popularity_penalty,
            "novelty_weight": novelty_weight,
            "recommendations": [
                {
                    "artist_id": "artist_6",
                    "artist_name": "F",
                    "score": 0.75,
                    "reasons": ["Shares pop"] if explain else [],
                }
            ][:top_k],
        }

    def similar_artists(
        self,
        artist_id: str,
        top_k: int,
        method: str,
        content_weight: float,
        explain: bool,
    ) -> dict[str, object]:
        return {
            "artist_id": artist_id,
            "strategy": f"{method}_similarity",
            "content_weight": content_weight,
            "similar_artists": [
                {
                    "artist_id": "artist_4",
                    "artist_name": "D",
                    "score": 0.7,
                    "reasons": ["Shares pop"] if explain else [],
                }
            ][:top_k],
        }

    def content_similar_artists(
        self,
        artist_id: str,
        top_k: int,
        explain: bool,
    ) -> dict[str, object]:
        return {
            "artist_id": artist_id,
            "strategy": "content_similarity",
            "similar_artists": [
                {
                    "artist_id": "artist_5",
                    "artist_name": "E",
                    "score": 0.6,
                    "reasons": ["Shares bright"] if explain else [],
                }
            ][:top_k],
        }

    def recommend_tracks(
        self,
        user_id: str,
        top_k: int,
        include_listened: bool,
        popularity_penalty: float = 0.0,
        diversity: float = 0.0,
        novelty_weight: float = 0.0,
        explain: bool = False,
        method: str = "similarity",
        content_weight: float | None = None,
        ltr: bool = False,
    ) -> dict[str, object]:
        if user_id == "ghost":
            raise ValueError(f"Unknown user_id: {user_id}")
        if ltr:
            strategy = "track_hybrid_ltr" if method == "hybrid" else "track_ltr"
        else:
            strategy = "track_hybrid" if method == "hybrid" else "track_similarity"
        return {
            "user_id": user_id,
            "strategy": strategy,
            "method": method,
            "recommendations": [
                {
                    "track_id": "track_1",
                    "track_name": "Hit",
                    "artist_name": "Test Artist",
                    "score": 0.95,
                    "reasons": ["Because you listened to Hit"] if explain else [],
                }
            ][:top_k],
            "include_listened": include_listened,
            "popularity_penalty": popularity_penalty,
            "diversity": diversity,
            "novelty_weight": novelty_weight,
        }

    def recommend_tracks_ltr(
        self,
        user_id: str,
        top_k: int,
        include_listened: bool,
        popularity_penalty: float = 0.0,
        diversity: float = 0.0,
        novelty_weight: float = 0.0,
        explain: bool = False,
        method: str = "similarity",
        content_weight: float | None = None,
    ) -> dict[str, object]:
        return self.recommend_tracks(
            user_id=user_id,
            top_k=top_k,
            include_listened=include_listened,
            popularity_penalty=popularity_penalty,
            diversity=diversity,
            novelty_weight=novelty_weight,
            explain=explain,
            method=method,
            content_weight=content_weight,
            ltr=True,
        )

    def similar_tracks(
        self,
        track_id: str,
        top_k: int,
    ) -> dict[str, object]:
        if track_id == "track_missing":
            raise ValueError(f"Unknown track_id: {track_id}")
        return {
            "track_id": track_id,
            "strategy": "track_similarity",
            "similar_tracks": [
                {
                    "track_id": "track_2",
                    "track_name": "Hit 2",
                    "artist_name": "Test Artist",
                    "score": 0.9,
                }
            ][:top_k],
        }

    def browse_tracks(
        self,
        *,
        query: str | None,
        artist: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, object]:
        if offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        if limit < 1:
            raise ValueError("limit must be a positive integer.")
        tracks = [
            {
                "track_id": "track_1",
                "track_name": "Hit",
                "artist_id": "artist_1",
                "artist_name": "Test Artist",
                "popularity_rank": 1,
            },
            {
                "track_id": "track_2",
                "track_name": "Hit 2",
                "artist_id": "artist_1",
                "artist_name": "Test Artist",
                "popularity_rank": 2,
            },
        ]
        if query:
            tracks = [
                track
                for track in tracks
                if query.casefold() in track["track_name"].casefold()
            ]
        if artist:
            tracks = [
                t
                for t in tracks
                if artist.casefold()
                in {t["artist_id"].casefold(), t["artist_name"].casefold()}
            ]
        page = tracks[offset : offset + limit]
        return {
            "total": len(tracks),
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < len(tracks),
            "tracks": page,
        }


def test_health_route_uses_loaded_service() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["artifact_version"] == "4.0"


def test_request_logging_emits_method_path_and_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with (
        TestClient(api_main.app) as client,
        caplog.at_level(logging.INFO, logger="music_recommender.api"),
    ):
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/health")

    assert response.status_code == 200
    request_logs = [
        record
        for record in caplog.records
        if "request method=GET path=/health" in record.getMessage()
    ]
    assert request_logs
    assert " status=200 " in request_logs[0].getMessage()


def test_openapi_document_exposes_project_metadata() -> None:
    with TestClient(api_main.app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert document["info"]["version"] == api_main.__version__
    assert document["info"]["license"]["name"] == "MIT"
    assert "Hybrid ALS" in document["info"]["description"]


def test_cors_exposes_request_context_headers() -> None:
    with TestClient(api_main.app) as client:
        response = client.get("/", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "X-Request-ID" in response.headers["access-control-expose-headers"]
    assert "X-Process-Time" in response.headers["access-control-expose-headers"]


def test_path_identifiers_reject_oversized_values() -> None:
    oversized_id = "a" * 101

    with TestClient(api_main.app) as client:
        user_response = client.get(f"/recommend/user/{oversized_id}")
        artist_response = client.get(f"/similar-artists/{oversized_id}")

    assert user_response.status_code == 422
    assert artist_response.status_code == 422


def test_responses_echo_valid_request_id_and_report_process_time() -> None:
    with TestClient(api_main.app) as client:
        response = client.get("/", headers={"X-Request-ID": "client-request_123"})

    assert response.headers["x-request-id"] == "client-request_123"
    assert float(response.headers["x-process-time"]) >= 0.0


def test_invalid_request_id_is_replaced() -> None:
    with TestClient(api_main.app) as client:
        response = client.get("/", headers={"X-Request-ID": "not valid"})

    request_id = response.headers["x-request-id"]
    assert request_id != "not valid"
    assert len(request_id) == 32


def test_declared_oversized_request_body_is_rejected() -> None:
    with TestClient(api_main.app) as client:
        response = client.post(
            "/recommend/profile",
            content=b"x" * (api_main.MAX_REQUEST_BODY_BYTES + 1),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        f"Request body exceeds the {api_main.MAX_REQUEST_BODY_BYTES}-byte limit."
    )
    assert response.headers["x-request-id"]
    assert float(response.headers["x-process-time"]) >= 0.0


def test_streamed_oversized_request_body_is_rejected() -> None:
    async def consume_body(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        del scope, send
        while True:
            message = await receive()
            if not message.get("more_body", False):
                return

    middleware = RequestSafetyMiddleware(consume_body, max_body_bytes=4)
    messages = iter(
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"45", "more_body": False},
        ]
    )
    sent_messages: list[Message] = []

    async def receive() -> Message:
        return next(messages)  # type: ignore[return-value]

    async def send(message: Message) -> None:
        sent_messages.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/recommend/profile",
        "raw_path": b"/recommend/profile",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "state": {},
    }

    asyncio.run(middleware(scope, receive, send))

    response_start = sent_messages[0]
    assert response_start["type"] == "http.response.start"
    assert response_start["status"] == 413


def http_scope_with_headers(headers: list[tuple[bytes, bytes]]) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/recommend/profile",
        "raw_path": b"/recommend/profile",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "state": {},
    }


def test_middleware_rejects_non_positive_body_limit() -> None:
    with pytest.raises(ValueError, match="max_body_bytes"):
        RequestSafetyMiddleware(lambda *_: None, max_body_bytes=0)


@pytest.mark.parametrize("content_length", [b"not-a-number", b"-5"])
def test_middleware_rejects_invalid_content_length(content_length: bytes) -> None:
    middleware = RequestSafetyMiddleware(
        lambda *_: None,
        max_body_bytes=100,
    )
    sent_messages: list[Message] = []

    async def send(message: Message) -> None:
        sent_messages.append(message)

    asyncio.run(
        middleware(
            http_scope_with_headers([(b"content-length", content_length)]),
            lambda: {},  # type: ignore[arg-type,return-value]
            send,
        )
    )

    assert sent_messages[0]["status"] == 400


def test_body_limit_after_response_start_does_not_reject() -> None:
    async def app_sends_head_then_reads(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        del scope
        await send({"type": "http.response.start", "status": 200, "headers": []})
        while True:
            message = await receive()
            if not message.get("more_body", False):
                return

    middleware = RequestSafetyMiddleware(app_sends_head_then_reads, max_body_bytes=4)
    messages = iter(
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"45", "more_body": False},
        ]
    )
    sent_messages: list[Message] = []

    async def receive() -> Message:
        return next(messages)  # type: ignore[return-value]

    async def send(message: Message) -> None:
        sent_messages.append(message)

    asyncio.run(
        middleware(
            http_scope_with_headers([]),
            receive,
            send,
        )
    )

    assert len(sent_messages) == 1
    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 200


def test_recommend_user_route_accepts_hybrid_params() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/recommend/user/user_1",
            params={"content_weight": 0.4, "explain": True, "top_k": 1},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "hybrid_personalized"
    assert body["content_weight"] == 0.4
    assert body["recommendations"][0]["reasons"]


def test_recommend_user_ltr_route_returns_ltr_ranked_hits() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/recommend/user/user_1/ltr",
            params={"top_k": 1, "diversity": 0.2, "popularity_penalty": 0.1},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "ltr_ranked"
    assert body["recommendations"][0]["artist_id"] == "artist_4"
    assert body["diversity"] == 0.2
    assert body["popularity_penalty"] == 0.1


def test_recommend_user_route_records_bandit_feedback_when_asked() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/recommend/user/new_user",
            params={"record_feedback": True, "top_k": 2},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "bandit_fallback"
    assert body["feedback"] == {
        "recorded": True,
        "arm": "popular",
        "reward": 1.0,
        "context": [0.0, 0.0, 0.0],
    }


def test_recommend_user_route_forwards_context_to_service() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/recommend/user/new_user",
            params={"record_feedback": True, "context": "0.1,0.2,0.3"},
        )

    assert response.status_code == 200
    assert api_main.service.last_context == [0.1, 0.2, 0.3]
    assert response.json()["feedback"]["context"] == [0.1, 0.2, 0.3]


def test_bandit_status_route_reports_lifecycle() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/bandit/status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["auto_sweep_threshold"] == 50
    assert body["snapshots_count"] == 1
    assert body["journal"] == {"length": 2, "pending": 0}
    assert body["state"]["arms"]["popular"]["selections"] == 2
    assert body["context_features"] == [
        "log_plays",
        "log_unique_artists",
        "mean_popularity_rank",
    ]


def test_bandit_update_route_folds_pending_feedback() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post("/bandit/update")

    assert response.status_code == 200
    assert api_main.service.last_sweep_called is True
    assert response.json()["available"] is True


def test_bandit_update_route_matching_context_features() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post(
            "/bandit/update",
            json={
                "context_features": [
                    "log_plays",
                    "log_unique_artists",
                    "mean_popularity_rank",
                ]
            },
        )

    assert response.status_code == 200
    assert api_main.service.last_sweep_called is True


def test_bandit_update_route_mismatched_context_features_is_unprocessable() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post(
            "/bandit/update",
            json={"context_features": ["log_plays", "mean_popularity_rank"]},
        )

    assert response.status_code == 422
    assert "do not match the state's recorded feature set" in response.json()["detail"]


def test_bandit_update_route_missing_state_is_not_found() -> None:
    class NoStateService(FakeService):
        def sweep_bandit_feedback(
            self,
            *,
            context_features: list[str] | None = None,
            gamma: float | None = None,
        ) -> dict[str, object]:
            raise FileNotFoundError("bandit_state.json")

    with TestClient(api_main.app) as client:
        api_main.service = NoStateService()
        api_main.service_load_error = None

        response = client.post("/bandit/update")

    assert response.status_code == 404
    assert "Bandit state not found" in response.json()["detail"]


def test_bandit_update_route_corrupt_state_is_unprocessable() -> None:
    class CorruptStateService(FakeService):
        def sweep_bandit_feedback(
            self,
            *,
            context_features: list[str] | None = None,
            gamma: float | None = None,
        ) -> dict[str, object]:
            raise ValueError("Failed to parse bandit state")

    with TestClient(api_main.app) as client:
        api_main.service = CorruptStateService()
        api_main.service_load_error = None

        response = client.post("/bandit/update")

    assert response.status_code == 422
    assert "Failed to parse bandit state" in response.json()["detail"]


def test_bandit_snapshots_list_route() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/bandit/snapshots")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert len(body["snapshots"]) == 1
    assert body["snapshots"][0]["label"] == "baseline"


def test_bandit_snapshots_create_route() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post("/bandit/snapshots", json={"label": "baseline"})

    assert response.status_code == 201
    body = response.json()
    assert body["created"] is True
    assert "baseline" in body["filename"]


def test_bandit_snapshots_create_route_without_body() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post("/bandit/snapshots")

    assert response.status_code == 201
    assert response.json()["created"] is True


def test_bandit_snapshots_create_route_missing_state_is_not_found() -> None:
    class NoStateSnapshotService(FakeService):
        def create_bandit_snapshot(self, label: str | None = None) -> Path:
            raise FileNotFoundError("reports/bandit_state.json not found")

    with TestClient(api_main.app) as client:
        api_main.service = NoStateSnapshotService()
        api_main.service_load_error = None

        response = client.post("/bandit/snapshots")

    assert response.status_code == 404
    assert "Bandit state not found" in response.json()["detail"]


def test_bandit_snapshots_create_route_corrupt_state_is_unprocessable() -> None:
    class CorruptSnapshotService(FakeService):
        def create_bandit_snapshot(self, label: str | None = None) -> Path:
            raise ValueError("Corrupt bandit state")

    with TestClient(api_main.app) as client:
        api_main.service = CorruptSnapshotService()
        api_main.service_load_error = None

        response = client.post("/bandit/snapshots")

    assert response.status_code == 422
    assert "Corrupt bandit state" in response.json()["detail"]


def test_bandit_drift_route_defaults_to_latest() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/bandit/drift")

    assert response.status_code == 200
    body = response.json()
    assert body["has_drift"] is False
    assert body["dominant_arm_a"] == "popular"
    assert "popular" in body["arms"]


def test_bandit_drift_route_with_reference() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/bandit/drift",
            params={"reference": "bandit_state_20260927T000000Z.json"},
        )

    assert response.status_code == 200
    assert response.json()["has_drift"] is False


def test_bandit_drift_route_missing_state_or_reference_is_not_found() -> None:
    class MissingReferenceService(FakeService):
        def compute_bandit_drift(
            self, reference_path: Path | str | None = None
        ) -> dict[str, object]:
            raise FileNotFoundError("Snapshot not found")

    with TestClient(api_main.app) as client:
        api_main.service = MissingReferenceService()
        api_main.service_load_error = None

        response = client.get("/bandit/drift?reference=nonexistent.json")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_bandit_drift_route_corrupt_reference_is_unprocessable() -> None:
    class CorruptReferenceService(FakeService):
        def compute_bandit_drift(
            self, reference_path: Path | str | None = None
        ) -> dict[str, object]:
            raise ValueError("Invalid snapshot format")

    with TestClient(api_main.app) as client:
        api_main.service = CorruptReferenceService()
        api_main.service_load_error = None

        response = client.get("/bandit/drift")

    assert response.status_code == 422
    assert "Invalid snapshot format" in response.json()["detail"]


def test_recommend_user_route_rejects_malformed_context() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/recommend/user/new_user",
            params={"record_feedback": True, "context": "0.1,abc"},
        )

    assert response.status_code == 422


def test_recommend_user_route_no_feedback_by_default() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/recommend/user/user_1")

    assert response.status_code == 200
    assert response.json()["strategy"] == "hybrid_personalized"


def test_recommend_user_ltr_route_turns_value_errors_into_422() -> None:
    class FailingLtrService(FakeService):
        def recommend_user_ltr(
            self,
            user_id: str,
            top_k: int,
            include_listened: bool,
            diversity: float,
            popularity_penalty: float,
            novelty_weight: float = 0.0,
        ) -> dict[str, object]:
            raise ValueError("The LTR model is not available for this user.")

    with TestClient(api_main.app) as client:
        api_main.service = FailingLtrService()
        api_main.service_load_error = None

        response = client.get("/recommend/user/user_1/ltr")

    assert response.status_code == 422
    assert "LTR model is not available" in response.json()["detail"]


def test_artist_catalog_route_accepts_search_filters_and_pagination() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/catalog/artists",
            params={
                "query": "week",
                "genre": "pop",
                "country": "Canada",
                "offset": 0,
                "limit": 10,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 1
    assert body["limit"] == 10
    assert body["filters"]["query"] == "week"
    assert body["filters"]["genre"] == "pop"
    assert body["artists"][0]["artist_id"] == "artist_1"


def test_recommend_profile_route_returns_content_profile() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post(
            "/recommend/profile",
            json={
                "artist_ids": ["artist_1"],
                "genres": ["pop"],
                "top_k": 1,
                "explain": True,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "content_profile"
    assert body["recommendations"][0]["reasons"]


def test_recommend_session_route_returns_session_recommendations() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post(
            "/recommend/session",
            json={
                "user_id": "user_1",
                "artist_ids": ["artist_1"],
                "genres": ["pop"],
                "exclude_artist_ids": ["artist_2"],
                "top_k": 1,
                "content_weight": 0.35,
                "explain": True,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "session_hybrid"
    assert body["content_weight"] == 0.35
    assert body["seed_artist_ids"] == ["artist_1"]
    assert body["excluded_artist_ids"] == ["artist_2"]
    assert body["recommendations"][0]["reasons"]


def test_recommendation_payload_text_is_trimmed() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post(
            "/recommend/session",
            json={
                "user_id": " user_1 ",
                "artist_ids": [" artist_1 "],
                "genres": [" pop "],
                "exclude_artist_ids": [" artist_2 "],
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["user_id"] == "user_1"
    assert body["seed_artist_ids"] == ["artist_1"]
    assert body["genres"] == ["pop"]
    assert body["excluded_artist_ids"] == ["artist_2"]


def test_content_similar_route_returns_content_similarity() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/content-similar-artists/artist_1",
            params={"top_k": 1, "explain": True},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "content_similarity"
    assert body["similar_artists"][0]["reasons"]


def test_missing_service_returns_training_error() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = None
        api_main.service_load_error = "Retrain the model."

        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Retrain the model."


def test_invalid_artifact_keeps_api_alive_but_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load(_: type[api_main.RecommenderService]) -> None:
        raise ValueError("Artifact structure is invalid. Retrain the model.")

    monkeypatch.setattr(
        api_main.RecommenderService,
        "from_artifacts",
        classmethod(fail_to_load),
    )

    with TestClient(api_main.app) as client:
        liveness_response = client.get("/")
        readiness_response = client.get("/health")

    assert liveness_response.status_code == 200
    assert readiness_response.status_code == 503
    assert "Artifact structure is invalid" in readiness_response.json()["detail"]


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        ("get", "/popular-artists?top_k=0", {}),
        ("get", "/popular-artists?top_k=101", {}),
        ("get", "/catalog/artists?offset=-1", {}),
        ("get", "/catalog/artists?limit=101", {}),
        ("get", "/tracks/catalog?offset=-1", {}),
        ("get", "/tracks/catalog?limit=101", {}),
        ("get", "/recommend/user/user_1?diversity=1.1", {}),
        ("get", "/tracks/recommend/user_1?diversity=1.1", {}),
        ("get", "/tracks/recommend/user_1?popularity_penalty=-0.1", {}),
        ("get", "/similar-artists/artist_1?method=unknown", {}),
        (
            "post",
            "/recommend/profile",
            {"json": {"artist_ids": ["artist_1"], "top_k": 0}},
        ),
        (
            "post",
            "/recommend/profile",
            {"json": {"artist_ids": ["artist_1"], "top_k": 101}},
        ),
        (
            "post",
            "/recommend/session",
            {"json": {"genres": ["pop"], "popularity_penalty": -0.1}},
        ),
    ],
)
def test_routes_reject_invalid_ranking_parameters(
    method: str,
    path: str,
    request_kwargs: dict[str, object],
) -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.request(method, path, **request_kwargs)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/recommend/profile", {"genres": [""]}),
        ("/recommend/profile", {"mood_tags": [" "]}),
        ("/recommend/profile", {"artist_ids": ["a" * 101]}),
        (
            "/recommend/profile",
            {"genres": ["pop"] * (api_main.MAX_REQUEST_VALUES + 1)},
        ),
        ("/recommend/session", {"user_id": " "}),
        (
            "/recommend/session",
            {"exclude_artist_ids": ["artist_1"] * (api_main.MAX_REQUEST_VALUES + 1)},
        ),
    ],
)
def test_recommendation_routes_reject_invalid_payload_text(
    path: str,
    payload: dict[str, object],
) -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post(path, json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/recommend/profile",
            {"artist_ids": ["artist_1"], "top_k": True},
        ),
        (
            "/recommend/profile",
            {"artist_ids": ["artist_1"], "explain": 1},
        ),
        (
            "/recommend/session",
            {"genres": ["pop"], "content_weight": True},
        ),
        (
            "/recommend/session",
            {"genres": ["pop"], "include_listened": "yes"},
        ),
        (
            "/recommend/session",
            {"genres": ["pop"], "content_weigth": 0.5},
        ),
    ],
)
def test_recommendation_routes_reject_coerced_or_unknown_fields(
    path: str,
    payload: dict[str, object],
) -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.post(path, json=payload)

    assert response.status_code == 422


class RaisingService:
    def close(self, timeout: float | None = 5.0) -> None:
        pass

    def __getattr__(self, _name: str) -> object:
        def raise_value_error(*_args: object, **_kwargs: object) -> None:
            raise ValueError("service operation failed")

        return raise_value_error


def test_metadata_route_returns_artifact_metadata() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/metadata")

    assert response.status_code == 200
    assert response.json()["version"] == "4.0"


def test_popular_artists_route_returns_recommendations() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/popular-artists", params={"top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "popular_baseline"
    assert body["recommendations"][0]["artist_name"] == "A"


def test_similar_artists_route_returns_similar_artists() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/similar-artists/artist_1",
            params={"method": "hybrid", "top_k": 1, "explain": True},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "hybrid_similarity"
    assert body["similar_artists"][0]["reasons"]


def test_similar_artists_route_returns_404_for_unknown_artist() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = RaisingService()
        api_main.service_load_error = None

        response = client.get("/similar-artists/missing_artist")

    assert response.status_code == 404
    assert response.json()["detail"] == "service operation failed"


def test_content_similar_route_returns_404_for_unknown_artist() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = RaisingService()
        api_main.service_load_error = None

        response = client.get("/content-similar-artists/missing_artist")

    assert response.status_code == 404
    assert response.json()["detail"] == "service operation failed"


def test_track_recommend_route_returns_track_recommendations() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/recommend/user_1", params={"top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "track_similarity"
    assert body["recommendations"][0]["track_id"] == "track_1"


def test_track_recommend_route_explains_reasons() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/tracks/recommend/user_1", params={"top_k": 1, "explain": True}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"][0]["reasons"] == ["Because you listened to Hit"]


def test_track_recommend_route_returns_422_for_unknown_user() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/recommend/ghost")

    assert response.status_code == 422
    assert "Unknown user_id" in response.json()["detail"]


def test_track_recommend_route_supports_hybrid_method() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/tracks/recommend/user_1",
            params={"top_k": 1, "method": "hybrid", "content_weight": 0.5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "hybrid"
    assert body["strategy"] == "track_hybrid"


def test_track_recommend_route_supports_ltr() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/tracks/recommend/user_1",
            params={"top_k": 1, "ltr": True},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "track_ltr"
    assert len(body["recommendations"]) == 1


def test_track_recommend_ltr_route() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/recommend/user_1/ltr", params={"top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "track_ltr"
    assert len(body["recommendations"]) == 1


def test_track_recommend_route_rejects_invalid_method() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/recommend/user_1", params={"method": "bogus"})

    assert response.status_code == 422
    assert (
        "Input should be 'similarity' or 'hybrid'"
        in response.json()["detail"][0]["msg"]
    )


def test_similar_tracks_route_returns_similar_tracks() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/similar/track_1", params={"top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "track_similarity"
    assert body["similar_tracks"][0]["track_id"] == "track_2"


def test_similar_tracks_route_returns_404_for_unknown_track() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/similar/track_missing")

    assert response.status_code == 404
    assert "Unknown track_id" in response.json()["detail"]


def test_track_catalog_route_accepts_search_filters_and_pagination() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get(
            "/tracks/catalog",
            params={"query": "hit 2", "artist": "Test Artist", "limit": 10},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["tracks"][0]["track_id"] == "track_2"
    assert body["has_more"] is False


def test_track_catalog_route_rejects_invalid_pagination() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/catalog", params={"offset": -1})

    assert response.status_code == 422


def test_popular_tracks_route_returns_hits() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/popular", params={"top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "popular_baseline"
    assert body["recommendations"][0]["track_id"] == "track_1"


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        ("get", "/popular-artists", {}),
        ("get", "/catalog/artists", {}),
        ("get", "/recommend/user/user_1", {}),
        ("get", "/tracks/recommend/user_1", {}),
        ("get", "/tracks/recommend/user_1/ltr", {}),
        ("get", "/tracks/catalog", {}),
        ("post", "/recommend/profile", {"json": {}}),
        ("post", "/recommend/session", {"json": {}}),
    ],
)
def test_service_value_errors_become_http_422(
    method: str,
    path: str,
    request_kwargs: dict[str, object],
) -> None:
    with TestClient(api_main.app) as client:
        api_main.service = RaisingService()
        api_main.service_load_error = None

        response = client.request(method, path, **request_kwargs)

    assert response.status_code == 422
    assert response.json()["detail"] == "service operation failed"


def test_ablation_summary_route_returns_persisted_summary(
    tmp_path, monkeypatch
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "reports_loaded": 2,
                "ranking": [{"knob": "diversity", "mean_impact": 0.2}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "ABLATION_SUMMARY_PATH", summary_path)

    with TestClient(api_main.app) as client:
        response = client.get("/evaluation/ablation-summary")

    assert response.status_code == 200
    assert response.json()["reports_loaded"] == 2
    assert response.json()["ranking"] == [{"knob": "diversity", "mean_impact": 0.2}]


def test_ablation_summary_route_returns_404_when_missing(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "absent.json"
    monkeypatch.setattr(api_main, "ABLATION_SUMMARY_PATH", missing)

    with TestClient(api_main.app) as client:
        response = client.get("/evaluation/ablation-summary")

    assert response.status_code == 404
    assert "No aggregated ablation summary found" in response.json()["detail"]


def test_ablation_summary_route_returns_422_on_invalid_summary(
    tmp_path, monkeypatch
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(api_main, "ABLATION_SUMMARY_PATH", summary_path)

    with TestClient(api_main.app) as client:
        response = client.get("/evaluation/ablation-summary")

    assert response.status_code == 422
    assert "Failed to parse ablation summary" in response.json()["detail"]


def test_recommend_user_novelty_weight() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.get("/recommend/user/user_1?novelty_weight=0.35")

    assert response.status_code == 200
    assert response.json()["novelty_weight"] == 0.35


def test_recommend_tracks_novelty_weight() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.get("/tracks/recommend/user_1?novelty_weight=0.45")

    assert response.status_code == 200
    assert response.json()["novelty_weight"] == 0.45


def test_bandit_update_route_supports_gamma() -> None:
    fake_service = FakeService()
    with TestClient(api_main.app) as client:
        api_main.service = fake_service
        api_main.service_load_error = None
        response = client.post("/bandit/update", json={"gamma": 0.85})

    assert response.status_code == 200
    assert fake_service.last_gamma_passed == 0.85
    assert response.json()["gamma"] == 0.85


def test_bandit_update_route_rejects_out_of_range_gamma() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.post("/bandit/update", json={"gamma": 1.5})

    assert response.status_code == 422


def test_bandit_evaluate_off_policy_route_success() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.post(
            "/bandit/evaluate/off-policy",
            json={
                "target_policy": "epsilon_greedy",
                "min_propensity": 0.05,
                "ridge_lambda": 0.5,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["target_policy"] == "epsilon_greedy"
    assert "metrics" in body
    assert body["summary"]["records_evaluated"] == 10


def test_bandit_evaluate_off_policy_route_not_found() -> None:
    class MissingJournalService(FakeService):
        def evaluate_bandit_off_policy(self, **kwargs: object) -> dict[str, object]:
            raise FileNotFoundError("Feedback journal not found: missing.jsonl")

    with TestClient(api_main.app) as client:
        api_main.service = MissingJournalService()
        api_main.service_load_error = None
        response = client.post("/bandit/evaluate/off-policy")
    assert response.status_code == 404
    assert "Feedback journal or policy not found" in response.json()["detail"]


def test_bandit_evaluate_off_policy_route_invalid_propensity() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.post(
            "/bandit/evaluate/off-policy",
            json={"min_propensity": -0.1},
        )
    assert response.status_code == 422


def test_bandit_evaluate_compare_route_success() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.post(
            "/bandit/evaluate/compare",
            json={
                "policies": ["linucb", "thompson_sampling"],
                "rounds": 10,
                "top_k": 3,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["config"]["rounds"] == 10
    assert body["summary"]["champion"] == "thompson_sampling"


def test_bandit_evaluate_compare_route_not_found() -> None:
    class MissingInteractionsService(FakeService):
        def compare_bandit_policies(self, **kwargs: object) -> dict[str, object]:
            raise FileNotFoundError("Interactions data not found: missing.csv")

    with TestClient(api_main.app) as client:
        api_main.service = MissingInteractionsService()
        api_main.service_load_error = None
        response = client.post("/bandit/evaluate/compare")
    assert response.status_code == 404
    assert "Interactions data not found" in response.json()["detail"]


def test_bandit_evaluate_compare_route_invalid_rounds() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.post(
            "/bandit/evaluate/compare",
            json={"rounds": 0},
        )
    assert response.status_code == 422


def test_bandit_policy_derive_route_from_state_success() -> None:
    fake_service = FakeService()
    with TestClient(api_main.app) as client:
        api_main.service = fake_service
        api_main.service_load_error = None
        response = client.post(
            "/bandit/policy/derive",
            json={
                "temperature": 1.5,
                "temperature_decay": 0.02,
                "min_temperature": 0.05,
                "persist": False,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["temperature"] == 1.5
    assert body["persisted"] is False
    assert "policy" in body
    assert body["policy"]["popular"] == 0.6


def test_bandit_policy_derive_route_with_persist() -> None:
    fake_service = FakeService()
    with TestClient(api_main.app) as client:
        api_main.service = fake_service
        api_main.service_load_error = None
        response = client.post(
            "/bandit/policy/derive",
            json={"persist": True},
        )
    assert response.status_code == 200
    assert response.json()["persisted"] is True
    assert fake_service.cold_start_policy["popular"] == 0.6


def test_bandit_policy_derive_route_not_found() -> None:
    class MissingStateService(FakeService):
        def derive_cold_start_policy_from_state(
            self, **kwargs: object
        ) -> dict[str, float]:
            raise FileNotFoundError("State file not found")

    with TestClient(api_main.app) as client:
        api_main.service = MissingStateService()
        api_main.service_load_error = None
        response = client.post(
            "/bandit/policy/derive",
            json={"from_state": "nonexistent_state.json"},
        )
    assert response.status_code == 404
    assert "Source state or report not found" in response.json()["detail"]


def test_bandit_policy_derive_route_invalid_temperature() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None
        response = client.post(
            "/bandit/policy/derive",
            json={"temperature": -1.0},
        )
    assert response.status_code == 422


def test_api_lifespan_closes_service(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_service = FakeService()

    def mock_load_service() -> None:
        api_main.service = fake_service
        api_main.service_load_error = None

    monkeypatch.setattr(api_main, "load_service", mock_load_service)
    with TestClient(api_main.app):
        assert fake_service.closed is False
    assert fake_service.closed is True


def test_bandit_snapshots_champion_tag_and_get() -> None:
    fake_service = FakeService()
    with TestClient(api_main.app) as client:
        api_main.service = fake_service
        api_main.service_load_error = None

        # Tag champion
        res_post = client.post(
            "/bandit/snapshots/champion",
            json={"snapshot_path": "reports/bandit_snapshots/snap_1.json"},
        )
        assert res_post.status_code == 200
        assert res_post.json()["tagged"] is True
        assert res_post.json()["is_champion"] is True

        # Get champion
        res_get = client.get("/bandit/snapshots/champion")
        assert res_get.status_code == 200
        assert res_get.json()["is_champion"] is True
        assert res_get.json()["filename"] == "bandit_state_champ.json"


def test_bandit_snapshots_champion_get_404_when_none() -> None:
    class NoChampionService(FakeService):
        def get_champion_snapshot(
            self, snapshot_dir: object = None
        ) -> dict[str, object] | None:
            return None

    with TestClient(api_main.app) as client:
        api_main.service = NoChampionService()
        api_main.service_load_error = None

        res = client.get("/bandit/snapshots/champion")
        assert res.status_code == 404
        assert "No champion snapshot found" in res.json()["detail"]


def test_bandit_drift_safety_route() -> None:
    fake_service = FakeService()
    with TestClient(api_main.app) as client:
        api_main.service = fake_service
        api_main.service_load_error = None

        # 1. Default payload
        res = client.post("/bandit/drift/safety")
        assert res.status_code == 200
        assert res.json()["is_safe"] is True
        assert "metrics" in res.json()

        # 2. Overridden thresholds
        res_override = client.post(
            "/bandit/drift/safety",
            json={
                "reference_snapshot": "ref.json",
                "max_l2_drift": 0.4,
                "min_cosine_similarity": 0.9,
                "max_reward_drop": 0.1,
                "allow_dominant_arm_change": False,
            },
        )
        assert res_override.status_code == 200
        assert res_override.json()["is_safe"] is True


def test_bandit_drift_safety_route_not_found() -> None:
    class MissingRefService(FakeService):
        def evaluate_bandit_drift_safety(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            raise FileNotFoundError("Reference snapshot missing")

    with TestClient(api_main.app) as client:
        api_main.service = MissingRefService()
        api_main.service_load_error = None

        res = client.post(
            "/bandit/drift/safety",
            json={"reference_snapshot": "missing.json"},
        )
        assert res.status_code == 404
        assert "Reference snapshot missing" in res.json()["detail"]


def test_bandit_rollback_route() -> None:
    fake_service = FakeService()
    with TestClient(api_main.app) as client:
        api_main.service = fake_service
        api_main.service_load_error = None

        res = client.post(
            "/bandit/rollback",
            json={
                "snapshot_path": "reports/champ.json",
                "update_active_policy": True,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["restored"] is True
        assert body["is_champion"] is True
        assert body["policy_updated"] is True


def test_bandit_rollback_route_not_found() -> None:
    class MissingSnapshotService(FakeService):
        def rollback_bandit_state(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            raise FileNotFoundError("Snapshot not found for rollback")

    with TestClient(api_main.app) as client:
        api_main.service = MissingSnapshotService()
        api_main.service_load_error = None

        res = client.post("/bandit/rollback")
        assert res.status_code == 404
        assert "Snapshot not found for rollback" in res.json()["detail"]


def test_health_route_includes_streaming_status() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["streaming_status"] == "healthy"


def test_health_streaming_route_healthy() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/health/streaming")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["reasons"] == []
        assert body["queue"]["running"] is True
        assert body["maintenance"]["total_sweeps"] == 1


def test_health_streaming_route_degraded() -> None:
    class DegradedService(FakeService):
        def streaming_health(self) -> dict[str, object]:
            return {
                "status": "degraded",
                "reasons": ["Queue saturation high: 85.0%"],
                "queue": {"running": True, "utilization_pct": 85.0},
                "maintenance": {"running": True},
                "drift": {"is_safe": True},
            }

    with TestClient(api_main.app) as client:
        api_main.service = DegradedService()
        api_main.service_load_error = None

        response = client.get("/health/streaming")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert len(body["reasons"]) == 1


def test_health_streaming_route_unhealthy_returns_503() -> None:
    class UnhealthyService(FakeService):
        def streaming_health(self) -> dict[str, object]:
            return {
                "status": "unhealthy",
                "reasons": ["Queue backpressure saturation (100.0%)"],
                "queue": {"running": False, "backpressure": True},
                "maintenance": {"running": False},
                "drift": {"is_safe": False},
            }

    with TestClient(api_main.app) as client:
        api_main.service = UnhealthyService()
        api_main.service_load_error = None

        response = client.get("/health/streaming")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"
        assert "Queue backpressure saturation" in body["reasons"][0]


def test_health_streaming_route_service_unavailable() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = None
        api_main.service_load_error = "Model artifacts not found."

        response = client.get("/health/streaming")
        assert response.status_code == 503
        assert "Model artifacts not found" in response.json()["detail"]


def test_bandit_observability_route() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/bandit/observability")
        assert response.status_code == 200
        body = response.json()
        assert body["health"]["status"] == "healthy"
        assert body["state"]["available"] is True
        assert body["journal"]["total_records"] == 10
        assert body["snapshots"]["total_count"] == 1
        assert body["off_policy_evaluation"] is None


def test_bandit_observability_route_include_ope() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/bandit/observability?include_ope=true")
        assert response.status_code == 200
        body = response.json()
        assert body["off_policy_evaluation"] is not None
        assert body["off_policy_evaluation"]["available"] is True
        assert "ci_95" in body["off_policy_evaluation"]["metrics"]["ips"]


def test_bandit_observability_route_service_unavailable() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = None
        api_main.service_load_error = "Model artifacts not found."

        response = client.get("/bandit/observability")
        assert response.status_code == 503


def test_bandit_observability_route_not_found() -> None:
    class MissingStateService(FakeService):
        def bandit_observability(
            self, *, include_ope: bool = False
        ) -> dict[str, object]:
            raise FileNotFoundError("State file not found")

    with TestClient(api_main.app) as client:
        api_main.service = MissingStateService()
        api_main.service_load_error = None

        response = client.get("/bandit/observability")
        assert response.status_code == 404
        assert "State file not found" in response.json()["detail"]




