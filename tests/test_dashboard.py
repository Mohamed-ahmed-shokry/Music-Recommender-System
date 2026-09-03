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
            ltr_model=None,
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


class MessageFakeService(FakeDashboardService):
    def recommend_user(self, **_: Any) -> dict[str, object]:
        response = self._recommendation_response("popular_fallback")
        response["message"] = "Unknown listener, returning popular artists."
        return response


class EmptyFakeService(FakeDashboardService):
    def recommend_user(self, **_: Any) -> dict[str, object]:
        return {"strategy": "hybrid_personalized", "recommendations": []}


class MoreCatalogService(FakeDashboardService):
    def browse_artists(
        self,
        *,
        query: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        del query, limit
        artist = self.artifact.content_artifacts.metadata.to_dict("records")[0]
        artist["genres"] = artist["genres"].split(";")
        artist["mood_tags"] = artist["mood_tags"].split(";")
        return {
            "total": 5,
            "has_more": True,
            "artists": [artist],
        }


class RaisingFakeService(FakeDashboardService):
    def recommend_user(self, **_: Any) -> dict[str, object]:
        raise ValueError("No artists match those controls.")


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
        "Ablation Summary",
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


def test_dashboard_artifact_path_defaults_to_bundle_path(
    monkeypatch,
) -> None:
    from music_recommender.config import ARTIFACT_BUNDLE_PATH

    monkeypatch.delenv(DASHBOARD_ARTIFACT_ENV_VAR, raising=False)

    assert resolve_dashboard_artifact_path() == ARTIFACT_BUNDLE_PATH


def test_dashboard_profile_tab_submits_preferences() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.button[1].click().run()

    assert not app.exception
    assert any(caption.value == "Strategy: Content Profile" for caption in app.caption)


def test_dashboard_session_tab_submits_mix() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.button[2].click().run()

    assert not app.exception
    assert any(caption.value == "Strategy: Session Hybrid" for caption in app.caption)


def test_dashboard_similarity_tab_submits() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.button[3].click().run()

    assert not app.exception
    assert any(
        caption.value == "Strategy: Hybrid Similarity" for caption in app.caption
    )


def test_dashboard_renders_fallback_message_in_results() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(MessageFakeService(),),
        default_timeout=10,
    ).run()

    app.button[0].click().run()

    assert not app.exception
    assert any(caption.value == "Strategy: Popular Fallback" for caption in app.caption)
    assert any(info.value.startswith("Unknown listener") for info in app.info)


def test_dashboard_warns_when_no_recommendations_match() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(EmptyFakeService(),),
        default_timeout=10,
    ).run()

    app.button[0].click().run()

    assert not app.exception
    assert any(
        warning.value == "No recommendations matched the selected controls."
        for warning in app.warning
    )


def test_dashboard_surfaces_service_errors_in_results() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(RaisingFakeService(),),
        default_timeout=10,
    ).run()

    app.button[0].click().run()

    assert not app.exception
    assert any(error.value == "No artists match those controls." for error in app.error)


def test_dashboard_catalog_caption_reports_truncated_results() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(MoreCatalogService(),),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(
        caption.value.startswith("Showing the first 1 of 5") for caption in app.caption
    )


def test_dashboard_entrypoint_renders_with_valid_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from music_recommender.artifacts import (
        build_recommender_artifact,
        save_artifact,
    )
    from music_recommender.content import build_content_artifacts
    from music_recommender.model import train_als_model
    from music_recommender.preprocessing import (
        build_user_item_matrix,
        create_id_mappings,
    )

    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2"],
            "artist_id": ["artist_1", "artist_2", "artist_2"],
            "artist_name": ["A", "B", "B"],
            "play_count": [10, 5, 7],
        }
    )
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(matrix, 4, 0.01, 1, 10.0, use_gpu=False)
    content = build_content_artifacts(
        pd.DataFrame(
            {
                "artist_id": ["artist_1", "artist_2"],
                "artist_name": ["A", "B"],
                "genres": ["pop", "rock"],
                "mood_tags": ["bright", "raw"],
                "country": ["Canada", "Canada"],
                "era": ["2020s", "2020s"],
            }
        ),
        ["artist_1", "artist_2"],
    )
    artifact = build_recommender_artifact(
        model=model,
        mappings=mappings,
        user_item_matrix=matrix,
        filtered_df=df,
        content_artifacts=content,
        raw_data_path=tmp_path / "missing.csv",
        metadata_path=tmp_path / "metadata.csv",
        training_config={
            "raw_data_path": str(tmp_path / "missing.csv"),
            "metadata_path": str(tmp_path / "metadata.csv"),
            "min_user_interactions": 1,
            "min_artist_interactions": 1,
            "factors": 4,
            "regularization": 0.01,
            "iterations": 1,
            "alpha": 10.0,
            "use_gpu": False,
            "content_weight": 0.25,
        },
        hybrid_config={"default_content_weight": 0.25},
        ranking_config={
            "include_listened": False,
            "popularity_penalty": 0.0,
            "diversity": 0.0,
        },
    )
    artifact_path = tmp_path / "artifact.joblib"
    save_artifact(artifact, artifact_path)
    monkeypatch.setenv(DASHBOARD_ARTIFACT_ENV_VAR, str(artifact_path))
    load_dashboard_service.clear()
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    app = AppTest.from_file(entrypoint, default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Music Recommender Studio"
    assert [tab.label for tab in app.tabs] == [
        "For You",
        "Taste Profile",
        "Session Mix",
        "Similar Artists",
        "Catalog",
        "Ablation Summary",
    ]
