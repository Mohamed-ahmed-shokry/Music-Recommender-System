import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

import api.main as api_main
from api.middleware import RequestSafetyMiddleware


class FakeService:
    def health(self) -> dict[str, object]:
        return {"status": "ok", "artifact_version": "4.0"}

    def metadata(self) -> dict[str, object]:
        return {"version": "4.0", "hybrid_config": {"default_content_weight": 0.25}}

    def popular_artists(self, top_k: int) -> dict[str, object]:
        return {
            "strategy": "popular_baseline",
            "recommendations": [
                {"artist_id": "artist_1", "artist_name": "A", "score": 10.0}
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
    ) -> dict[str, object]:
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
    ) -> dict[str, object]:
        if user_id == "ghost":
            raise ValueError(f"Unknown user_id: {user_id}")
        return {
            "user_id": user_id,
            "strategy": "track_similarity",
            "recommendations": [
                {
                    "track_id": "track_1",
                    "track_name": "Hit",
                    "artist_name": "Test Artist",
                    "score": 0.95,
                }
            ][:top_k],
            "include_listened": include_listened,
        }

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


def test_track_recommend_route_returns_422_for_unknown_user() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/tracks/recommend/ghost")

    assert response.status_code == 422
    assert "Unknown user_id" in response.json()["detail"]


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


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        ("get", "/popular-artists", {}),
        ("get", "/catalog/artists", {}),
        ("get", "/recommend/user/user_1", {}),
        ("get", "/tracks/recommend/user_1", {}),
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
