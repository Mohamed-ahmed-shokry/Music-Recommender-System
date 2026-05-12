from fastapi.testclient import TestClient

import api.main as api_main


class FakeService:
    def health(self) -> dict[str, object]:
        return {"status": "ok", "artifact_version": "3.0"}

    def metadata(self) -> dict[str, object]:
        return {"version": "3.0", "hybrid_config": {"default_content_weight": 0.25}}

    def popular_artists(self, top_k: int) -> dict[str, object]:
        return {
            "strategy": "popular_baseline",
            "recommendations": [
                {"artist_id": "artist_1", "artist_name": "A", "score": 10.0}
            ][:top_k],
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


def test_health_route_uses_loaded_service() -> None:
    with TestClient(api_main.app) as client:
        api_main.service = FakeService()
        api_main.service_load_error = None

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["artifact_version"] == "3.0"


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
