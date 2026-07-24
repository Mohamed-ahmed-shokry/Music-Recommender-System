from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from streamlit.testing.v1 import AppTest

from music_recommender.dashboard import (
    DASHBOARD_ARTIFACT_ENV_VAR,
    catalog_frame,
    load_dashboard_service,
    recommendation_frame,
    resolve_dashboard_artifact_path,
    split_metadata_terms,
)


class FakeDashboardService:
    def __init__(self) -> None:
        metadata = pd.DataFrame(
            {
                "artist_id": ["artist_1", "artist_2", "artist_3"],
                "artist_name": ["A", "B", "C"],
                "genres": ["pop", "pop;dance", "rock"],
                "mood_tags": ["bright", "bright;fun", "raw"],
                "country": ["Canada", "United States", "United Kingdom"],
                "era": ["2020s", "2020s", "2000s"],
            }
        )
        artist_stats = {
            "artist_1": {
                "artist_id": "artist_1",
                "artist_name": "A",
                "total_plays": 30,
                "listener_count": 3,
                "interaction_count": 3,
                "popularity_rank": 1,
            },
            "artist_2": {
                "artist_id": "artist_2",
                "artist_name": "B",
                "total_plays": 20,
                "listener_count": 2,
                "interaction_count": 2,
                "popularity_rank": 2,
            },
            "artist_3": {
                "artist_id": "artist_3",
                "artist_name": "C",
                "total_plays": 10,
                "listener_count": 1,
                "interaction_count": 1,
                "popularity_rank": 3,
            },
        }
        self.artifact = SimpleNamespace(
            mappings={
                "user_id_to_index": {"user_1": 0, "user_2": 1},
                "artist_id_to_name": {
                    "artist_1": "A",
                    "artist_2": "B",
                    "artist_3": "C",
                },
            },
            content_artifacts=SimpleNamespace(metadata=metadata),
            artist_stats=artist_stats,
            metadata={"training_device": "cpu"},
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "artifact_version": "4.0",
            "num_users": 2,
            "num_artists": 3,
            "num_interactions": 6,
            "content_features": 12,
        }

    def browse_artists(
        self,
        *,
        query: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        artists = []
        for metadata in self.artifact.content_artifacts.metadata.to_dict("records"):
            artist_id = metadata["artist_id"]
            artist = {
                **metadata,
                "genres": metadata["genres"].split(";"),
                "mood_tags": metadata["mood_tags"].split(";"),
                **self.artifact.artist_stats[artist_id],
            }
            if (
                query
                and query.casefold()
                not in " ".join(str(value) for value in artist.values()).casefold()
            ):
                continue
            artists.append(artist)
        page = artists[:limit]
        return {
            "total": len(artists),
            "has_more": len(page) < len(artists),
            "artists": page,
        }

    def recommend_user(self, **_: Any) -> dict[str, object]:
        return self._recommendation_response("hybrid_personalized")

    def recommend_profile(self, **_: Any) -> dict[str, object]:
        return self._recommendation_response("content_profile")

    def recommend_session(self, **_: Any) -> dict[str, object]:
        return self._recommendation_response("session_hybrid")

    def similar_artists(self, **_: Any) -> dict[str, object]:
        response = self._recommendation_response("hybrid_similarity")
        response["similar_artists"] = response.pop("recommendations")
        return response

    @staticmethod
    def _recommendation_response(strategy: str) -> dict[str, object]:
        return {
            "strategy": strategy,
            "recommendations": [
                {
                    "artist_id": "artist_2",
                    "artist_name": "B",
                    "score": 0.81234,
                    "popularity_rank": 2,
                    "reasons": ["Shares pop", "Matches bright"],
                }
            ],
        }


def dashboard_script(service) -> None:
    from music_recommender.dashboard import render_dashboard

    render_dashboard(service)


def test_split_metadata_terms_normalizes_and_sorts_values() -> None:
    values = ["pop; dance", ["bright", "pop"], None, "dance;rock"]

    assert split_metadata_terms(values) == ["bright", "dance", "pop", "rock"]


def test_recommendation_frame_supports_similarity_responses() -> None:
    frame = recommendation_frame(
        {
            "similar_artists": [
                {
                    "artist_id": "artist_2",
                    "artist_name": "B",
                    "score": 0.81234,
                    "popularity_rank": 2,
                    "reasons": ["Shares pop", "Matches bright"],
                }
            ]
        }
    )

    assert frame.loc[0, "Rank"] == 1
    assert frame.loc[0, "Artist"] == "B"
    assert frame.loc[0, "Score"] == 0.8123
    assert frame.loc[0, "Why"] == "Shares pop · Matches bright"


def test_catalog_frame_uses_service_search_and_formats_metadata() -> None:
    frame = catalog_frame(FakeDashboardService())
    filtered_frame = catalog_frame(FakeDashboardService(), query="canada")

    assert frame["artist_id"].tolist() == ["artist_1", "artist_2", "artist_3"]
    assert frame["total_plays"].tolist() == [30, 20, 10]
    assert frame.loc[1, "genres"] == "pop; dance"
    assert filtered_frame["artist_id"].tolist() == ["artist_1"]
    assert filtered_frame.attrs["total"] == 1


def test_dashboard_renders_all_product_workflows() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert app.title[0].value == "Music Recommender Studio"
    assert [tab.label for tab in app.tabs] == [
        "For You",
        "Taste Profile",
        "Session Mix",
        "Similar Artists",
        "Catalog",
    ]
    assert [metric.label for metric in app.metric] == [
        "Listeners",
        "Artists",
        "Interactions",
        "Content features",
    ]


def test_dashboard_personalized_form_displays_results() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.button[0].click().run()

    assert not app.exception
    assert any(
        caption.value == "Strategy: Hybrid Personalized" for caption in app.caption
    )
    assert any("Rank" in dataframe.value.columns for dataframe in app.dataframe)


def test_dashboard_entrypoint_explains_missing_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    missing_artifact = tmp_path / "missing.joblib"
    monkeypatch.setenv(DASHBOARD_ARTIFACT_ENV_VAR, str(missing_artifact))
    load_dashboard_service.clear()
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    app = AppTest.from_file(entrypoint, default_timeout=10).run()

    assert not app.exception
    assert app.error
    assert (
        app.error[0].value == "Recommender artifact not found. Train the model first."
    )
    assert app.code[0].value.endswith("train --no-use-gpu")


def test_dashboard_artifact_path_accepts_environment_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "custom.joblib"
    monkeypatch.setenv(DASHBOARD_ARTIFACT_ENV_VAR, str(artifact_path))

    assert resolve_dashboard_artifact_path() == artifact_path.resolve()
