from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from streamlit.testing.v1 import AppTest

from music_recommender.dashboard import (
    DASHBOARD_ARTIFACT_ENV_VAR,
    _ablation_ranking_rows,
    catalog_frame,
    load_dashboard_service,
    recommendation_frame,
    resolve_dashboard_artifact_path,
    split_metadata_terms,
    track_catalog_frame,
)


class FakeDashboardService:
    def __init__(self) -> None:
        self.context_features: tuple[str, ...] = (
            "log_plays",
            "log_unique_artists",
            "mean_popularity_rank",
        )
        self.snapshots: list[dict[str, object]] = []
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
            track_ltr_model=None,
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

    def bandit_status(self) -> dict[str, object]:
        status: dict[str, object] = {
            "available": True,
            "auto_sweep_threshold": 50,
            "snapshots_count": len(self.snapshots),
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
                    "long_tail": {
                        "selections": 1,
                        "total_reward": 0.4,
                        "mean_reward": 0.4,
                    },
                },
            },
            "policy": {"popular": 0.7, "long_tail": 0.3},
            "journal": {"length": 3, "pending": 1},
            "last_fold": {"offset": 2, "at": "2026-01-01T00:00:00+00:00"},
            "champion_snapshot": self.get_champion_snapshot(),
            "drift_guardrails": {
                "enabled": True,
                "auto_rollback": False,
                "thresholds": {
                    "max_l2_drift": 0.5,
                    "min_cosine_similarity": 0.7,
                },
            },
        }
        if getattr(self, "streaming_queue_enabled", False):
            status["streaming_queue"] = {
                "queue_depth": 5,
                "max_queue_size": 1000,
                "utilization_pct": 0.5,
                "backpressure": "drop_oldest",
                "flushed_count": 20,
                "total_flushed_batches": 2,
                "dropped_count": 0,
                "flush_errors": 0,
            }
        if getattr(self, "maintenance_worker_enabled", False):
            status["maintenance_worker"] = {
                "is_running": True,
                "interval_seconds": 60.0,
                "cycles_count": 4,
                "sweeps_count": 2,
                "records_folded_count": 15,
                "last_sweep_duration_seconds": 0.0125,
                "avg_sweep_duration_seconds": 0.015,
                "min_sweep_duration_seconds": 0.01,
                "max_sweep_duration_seconds": 0.02,
            }
        return status

    def flush_feedback(self, timeout: float | None = 5.0) -> None:
        self.last_flush_called = True

    def streaming_health(self) -> dict[str, Any]:
        queue_enabled = getattr(self, "streaming_queue_enabled", False)
        maint_enabled = getattr(self, "maintenance_worker_enabled", False)
        status_val = "healthy" if (queue_enabled or maint_enabled) else "disabled"
        return {
            "status": status_val,
            "healthy": True,
            "queue_enabled": queue_enabled,
            "maintenance_enabled": maint_enabled,
            "warnings": [],
            "timestamp": "2026-10-08T00:00:00Z",
        }

    def trigger_maintenance_sweep(self) -> dict[str, Any]:
        self.last_trigger_sweep_called = True
        return {"swept": True, "records_folded": 5}

    def tag_champion_snapshot(
        self, snapshot_path: str | Path, snapshot_dir: str | Path | None = None
    ) -> dict[str, Any]:
        self.champion_snapshot_filename = Path(snapshot_path).name
        return {
            "tagged": True,
            "filename": self.champion_snapshot_filename,
            "is_champion": True,
            "label": "champion",
            "total_selections": 10,
        }

    def get_champion_snapshot(
        self, snapshot_dir: str | Path | None = None
    ) -> dict[str, Any] | None:
        if getattr(self, "champion_snapshot_filename", None):
            return {
                "filename": self.champion_snapshot_filename,
                "label": "champion",
                "is_champion": True,
                "created_at": "2026-09-27T00:00:00Z",
                "total_selections": 10,
            }
        return None

    def rollback_bandit_state(
        self,
        snapshot_path: str | Path | None = None,
        target_state_path: str | Path | None = None,
        snapshot_dir: str | Path | None = None,
        *,
        update_active_policy: bool = True,
    ) -> dict[str, Any]:
        self.last_rollback_called = True
        self.last_rollback_target = str(snapshot_path) if snapshot_path else "latest"
        return {
            "rolled_back": True,
            "source_snapshot": self.last_rollback_target,
            "policy_updated": True,
            "policy": {"popular": 0.8, "long_tail": 0.2},
        }

    def evaluate_bandit_drift_safety(
        self,
        reference_path: str | Path | None = None,
        state_path: str | Path | None = None,
        snapshot_dir: str | Path | None = None,
        thresholds: Any = None,
    ) -> dict[str, Any]:
        return {
            "is_safe": True,
            "violations": [],
            "warnings": [],
            "reference_snapshot": str(reference_path or "ref.json"),
            "metrics": {"max_l2_drift": 0.05, "min_cosine_similarity": 0.99},
            "thresholds": {"max_l2_drift": 0.5, "min_cosine_similarity": 0.7},
        }

    def bandit_observability(
        self,
        *,
        include_ope: bool = False,
        ope_policy: Any = None,
        feedback_journal_path: Any = None,
        snapshot_dir: Any = None,
    ) -> dict[str, Any]:
        return {
            "timestamp": "2026-10-08T00:00:00Z",
            "health": self.streaming_health(),
            "streaming_queue": self.bandit_status().get("streaming_queue"),
            "maintenance_worker": self.bandit_status().get("maintenance_worker"),
            "sweep_timings": {
                "total_sweep_duration_seconds": 0.03,
                "avg_sweep_duration_seconds": 0.015,
                "min_sweep_duration_seconds": 0.01,
                "max_sweep_duration_seconds": 0.02,
                "last_sweep_duration_seconds": 0.0125,
            },
            "guardrails": {
                "enabled": True,
                "auto_rollback": False,
                "thresholds": {"max_l2_drift": 0.5},
            },
            "snapshots": {"total_count": len(self.snapshots), "champion": None},
            "off_policy_evaluation": (
                self.evaluate_bandit_off_policy() if include_ope else None
            ),
        }

    def list_bandit_snapshots(self) -> list[dict[str, object]]:
        return list(self.snapshots)

    def create_bandit_snapshot(self, label: str | None = None) -> Path:
        suffix = f"_{label}" if label else ""
        filename = f"bandit_state_20260927T000000Z{suffix}.json"
        path = f"reports/bandit_snapshots/{filename}"
        self.snapshots.append(
            {
                "filename": filename,
                "path": path,
                "timestamp": "20260927T000000Z",
                "label": label,
                "size_bytes": 1024,
            }
        )
        return Path(path)

    def compute_bandit_drift(
        self, reference_path: Path | str | None = None
    ) -> dict[str, object]:
        return {
            "current_state_path": "reports/bandit_state.json",
            "reference_state_path": str(
                reference_path or "reports/bandit_snapshots/ref.json"
            ),
            "has_drift": False,
            "max_l2_drift": 0.05,
            "mean_l2_drift": 0.025,
            "arms": {
                "popular": {
                    "theta_l2_drift": 0.05,
                    "theta_cosine_similarity": 0.99,
                    "selections_growth": 5,
                    "total_reward_diff": 3.5,
                    "mean_reward_diff": 0.1,
                },
                "long_tail": {
                    "theta_l2_drift": 0.0,
                    "theta_cosine_similarity": 1.0,
                    "selections_growth": 0,
                    "total_reward_diff": 0.0,
                    "mean_reward_diff": 0.0,
                },
            },
            "dominant_arm_a": "popular",
            "dominant_arm_b": "popular",
            "dominant_arm_changed": False,
        }

    def sweep_bandit_feedback(self, gamma: float | None = None) -> dict[str, object]:
        self.last_sweep_called = True
        self.last_gamma_passed = gamma
        status = self.bandit_status()
        status["journal"]["pending"] = 0
        return status

    def evaluate_bandit_off_policy(
        self,
        *,
        target_policy: Any = None,
        min_propensity: float = 0.01,
        ridge_lambda: float = 1.0,
    ) -> dict[str, Any]:
        return {
            "target_policy": str(target_policy or "linucb"),
            "summary": {
                "records_evaluated": 15,
                "match_rate": 0.85,
                "effective_sample_size": 12.4,
                "logging_mean_reward": 0.62,
            },
            "metrics": {
                "ips": {
                    "value": 0.72,
                    "standard_error": 0.05,
                    "ci_95": [0.62, 0.82],
                },
                "snips": {"value": 0.69},
                "direct_method": {
                    "value": 0.71,
                    "standard_error": 0.04,
                    "ci_95": [0.63, 0.79],
                },
                "doubly_robust": {
                    "value": 0.73,
                    "standard_error": 0.04,
                    "ci_95": [0.65, 0.81],
                },
            },
        }

    def compare_bandit_policies(
        self,
        *,
        policies: Any = None,
        rounds: int = 20,
    ) -> dict[str, Any]:
        return {
            "config": {"rounds": rounds},
            "policies": {
                "linucb": {
                    "cumulative_reward": 10.0,
                    "mean_reward": 0.5,
                    "regret": 2.0,
                },
                "thompson_sampling": {
                    "cumulative_reward": 12.0,
                    "mean_reward": 0.6,
                    "regret": 1.0,
                },
                "epsilon_greedy": {
                    "cumulative_reward": 9.5,
                    "mean_reward": 0.475,
                    "regret": 2.5,
                },
            },
            "summary": {
                "champion": "thompson_sampling",
                "leaderboard": [
                    {
                        "name": "thompson_sampling",
                        "mean_reward": 0.6,
                        "win_rate": 0.55,
                    },
                    {"name": "linucb", "mean_reward": 0.5, "win_rate": 0.30},
                    {
                        "name": "epsilon_greedy",
                        "mean_reward": 0.475,
                        "win_rate": 0.15,
                    },
                ],
            },
        }

    def derive_cold_start_policy_from_state(
        self,
        *,
        temperature: float = 1.0,
        temperature_decay: float = 0.0,
        min_temperature: float = 0.05,
        persist_path: Any = None,
        update_active_policy: bool = True,
    ) -> dict[str, float]:
        return {"popular": 0.65, "long_tail": 0.35}

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

    def recommend_tracks(self, **kwargs: Any) -> dict[str, object]:
        return {
            "strategy": "track_ltr" if kwargs.get("ltr") else "track_similarity",
            "recommendations": [
                {
                    "track_id": "track_1",
                    "track_name": "Hit",
                    "artist_name": "Test Artist",
                    "score": 0.95,
                }
            ],
        }

    def similar_tracks(self, **_: Any) -> dict[str, object]:
        return {
            "strategy": "track_similarity",
            "similar_tracks": [
                {
                    "track_id": "track_2",
                    "track_name": "Hit 2",
                    "artist_name": "Test Artist",
                    "score": 0.9,
                }
            ],
        }

    def browse_tracks(
        self,
        *,
        query: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        tracks = [
            {
                "track_id": "track_1",
                "track_name": "Hit",
                "artist_id": "artist_1",
                "artist_name": "Test Artist",
                "total_plays": 30,
                "popularity_rank": 1,
            },
            {
                "track_id": "track_2",
                "track_name": "Hit 2",
                "artist_id": "artist_1",
                "artist_name": "Test Artist",
                "total_plays": 20,
                "popularity_rank": 2,
            },
        ]
        if query:
            tracks = [
                track
                for track in tracks
                if query.casefold()
                in " ".join(str(value) for value in track.values()).casefold()
            ]
        page = tracks[:limit]
        return {
            "total": len(tracks),
            "has_more": len(page) < len(tracks),
            "tracks": page,
        }

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


def ablation_summary_script(summary) -> None:
    from music_recommender.dashboard import _render_ablation_summary_body

    _render_ablation_summary_body(summary)


class MessageFakeService(FakeDashboardService):
    def recommend_user(self, **_: Any) -> dict[str, object]:
        response = self._recommendation_response("popular_fallback")
        response["message"] = "Unknown listener, returning popular artists."
        return response


class MoreTracksCatalogService(FakeDashboardService):
    def browse_tracks(
        self,
        *,
        query: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        return {"total": 5, "has_more": True, "tracks": []}


def test_dashboard_tracks_catalog_caption_reports_truncated_results() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(MoreTracksCatalogService(),),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(
        caption.value.startswith("Showing the first 0 of 5") for caption in app.caption
    )


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
    assert frame.loc[0, "Score components"] == ""


def test_recommendation_frame_renders_hybrid_score_components() -> None:
    frame = recommendation_frame(
        {
            "recommendations": [
                {
                    "track_id": "track_1",
                    "track_name": "Song A",
                    "artist_name": "Artist A",
                    "score": 0.75,
                    "score_components": {
                        "content_score": 0.6,
                        "collaborative_score": 0.9,
                        "hybrid_score": 0.75,
                    },
                    "reasons": ["Because you listened to Song B"],
                }
            ]
        }
    )

    assert frame.loc[0, "Score components"] == "content 0.600 · collab 0.900"
    assert frame.loc[0, "Why"] == "Because you listened to Song B"


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
        "Tracks",
        "Catalog",
        "Ablation Summary",
        "Cold-Start Bandit",
    ]
    assert [metric.label for metric in app.metric] == [
        "Listeners",
        "Artists",
        "Interactions",
        "Content features",
        "Arms",
        "Context dim",
        "Alpha",
        "Total selections",
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


class LtrDashboardService(FakeDashboardService):
    def __init__(self) -> None:
        super().__init__()
        self.artifact.ltr_model = object()

    def recommend_user_ltr(self, **_: Any) -> dict[str, object]:
        return {
            "strategy": "ltr_ranked",
            "recommendations": [
                {
                    "artist_id": "artist_2",
                    "artist_name": "B",
                    "score": 0.9,
                }
            ],
        }


def test_dashboard_personalized_form_uses_ltr_when_enabled() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(LtrDashboardService(),),
        default_timeout=10,
    ).run()

    app.checkbox[2].check().run()
    app.button[0].click().run()

    assert not app.exception
    assert any(caption.value == "Strategy: Ltr Ranked" for caption in app.caption)


def test_dashboard_bandit_tab_renders_lifecycle_and_folds() -> None:
    service = FakeDashboardService()
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(
        caption.value.startswith("Served feedback is recorded")
        for caption in app.caption
    )
    assert any(metric.label == "Total selections" for metric in app.metric)
    assert any("Context features (3)" in caption.value for caption in app.caption)
    rendered = [
        *[caption.value for caption in app.caption],
        *[markdown.value for markdown in app.markdown],
    ]
    assert any("Journal" in text and "3" in text for text in rendered)

    fold_button = next(
        button for button in app.button if button.label == "Fold pending feedback"
    )
    fold_button.click().run()

    assert service.last_sweep_called is True
    assert any(
        "Online auto-sweep threshold: **50**" in caption.value
        for caption in app.caption
    )
    assert any(info.value.startswith("No snapshots found") for info in app.info)


class SubsetFeatureStateService(FakeDashboardService):
    def bandit_status(self) -> dict[str, object]:
        status = super().bandit_status()
        status["state"]["context_features"] = ["log_plays", "mean_popularity_rank"]
        return status


def test_dashboard_bandit_tab_shows_recorded_feature_subset() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(SubsetFeatureStateService(),),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any("Context features (2)" in caption.value for caption in app.caption)


class NoBanditService(FakeDashboardService):
    def bandit_status(self) -> dict[str, object]:
        return {
            "available": False,
            "state": None,
            "policy": None,
            "journal": {"length": 0, "pending": 0},
            "last_fold": None,
        }


def test_dashboard_bandit_tab_handles_missing_state() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(NoBanditService(),),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(info.value.startswith("No bandit state yet") for info in app.info)


def test_dashboard_bandit_tab_surfaces_fold_errors() -> None:
    class FailingSweepService(FakeDashboardService):
        def sweep_bandit_feedback(
            self, gamma: float | None = None
        ) -> dict[str, object]:
            raise ValueError("Failed to parse bandit state")

    app = AppTest.from_function(
        dashboard_script,
        args=(FailingSweepService(),),
        default_timeout=10,
    ).run()
    fold_button = next(
        button for button in app.button if button.label == "Fold pending feedback"
    )
    fold_button.click().run()

    assert not app.exception
    assert any(error.value.startswith("Fold failed:") for error in app.error)


def test_dashboard_bandit_tab_creates_snapshot() -> None:
    service = FakeDashboardService()
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    snap_button = next(
        button for button in app.button if button.label == "Create Snapshot"
    )
    snap_button.click().run()

    assert not app.exception
    assert len(service.snapshots) == 1
    assert any("Created snapshot" in success.value for success in app.success)


def test_dashboard_bandit_tab_displays_snapshots_and_drift() -> None:
    service = FakeDashboardService()
    snap_filename = "bandit_state_20260927T000000Z_baseline.json"
    service.snapshots.append(
        {
            "filename": snap_filename,
            "path": f"reports/bandit_snapshots/{snap_filename}",
            "timestamp": "20260927T000000Z",
            "label": "baseline",
            "size_bytes": 1024,
        }
    )
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any("Persisted snapshots (1)" in caption.value for caption in app.caption)
    assert any(metric.label == "Dominant Arm (Current)" for metric in app.metric)
    assert any(metric.label == "Max L2 Drift" for metric in app.metric)


def test_dashboard_bandit_tab_surfaces_snapshot_errors() -> None:
    class FailingSnapshotService(FakeDashboardService):
        def create_bandit_snapshot(self, label: str | None = None) -> Path:
            raise ValueError("Corrupt bandit state")

    app = AppTest.from_function(
        dashboard_script,
        args=(FailingSnapshotService(),),
        default_timeout=10,
    ).run()
    snap_button = next(
        button for button in app.button if button.label == "Create Snapshot"
    )
    snap_button.click().run()

    assert not app.exception
    assert any(
        error.value.startswith("Snapshot creation failed:") for error in app.error
    )


def test_dashboard_bandit_tab_derivation_ope_and_benchmark() -> None:
    service = FakeDashboardService()
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    assert not app.exception

    # 1. Derive policy
    derive_button = next(
        button for button in app.button if button.label == "Derive Policy"
    )
    derive_button.click().run()
    assert not app.exception
    assert any(
        "Derived cold-start serving weights successfully." in s.value
        for s in app.success
    )

    # 2. Run OPE
    ope_button = next(
        button
        for button in app.button
        if button.label == "Run Off-Policy Evaluation"
    )
    ope_button.click().run()
    assert not app.exception
    assert any(metric.label == "Records Evaluated" for metric in app.metric)
    assert any(metric.label == "Match Rate" for metric in app.metric)

    # 3. Run Benchmark
    bench_button = next(
        button
        for button in app.button
        if button.label == "Run Policy Benchmark"
    )
    bench_button.click().run()
    assert not app.exception
    assert any("Benchmark completed! Champion:" in s.value for s in app.success)


def test_ablation_ranking_rows_shapes_summary_data() -> None:
    summary = {
        "reports_loaded": 2,
        "ranking": [
            {"knob": "diversity", "mean_impact": 0.12345},
            {"knob": "popularity_penalty", "mean_impact": -0.05},
        ],
        "knobs": {
            "diversity": {"std_impact": 0.02, "count": 2},
            "popularity_penalty": {},
        },
    }

    frame = _ablation_ranking_rows(summary)

    assert frame.loc[0, "Knob"] == "diversity"
    assert frame.loc[0, "Mean Impact"] == 0.1235
    assert frame.loc[0, "Std Impact"] == 0.02
    assert frame.loc[0, "Runs"] == 2
    assert frame.loc[1, "Std Impact"] == 0.0
    assert frame.loc[1, "Runs"] == 0


def test_ablation_ranking_rows_returns_empty_frame_for_no_data() -> None:
    frame = _ablation_ranking_rows({"ranking": [], "knobs": {}})

    assert frame.empty


def test_ablation_summary_body_renders_ranking_table() -> None:
    summary = {
        "reports_loaded": 2,
        "ranking": [{"knob": "diversity", "mean_impact": 0.12345}],
        "knobs": {"diversity": {"std_impact": 0.02, "count": 2}},
    }
    app = AppTest.from_function(
        ablation_summary_script,
        args=(summary,),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert app.subheader[0].value == "Knob Importance by Mean Total Impact"
    assert app.dataframe[0].value.loc[0, "Knob"] == "diversity"
    assert any(
        "Aggregated from 2 ablation report(s)" in caption.value
        for caption in app.caption
    )


def test_ablation_summary_body_reports_empty_ranking() -> None:
    app = AppTest.from_function(
        ablation_summary_script,
        args=({"ranking": [], "knobs": {}, "reports_loaded": 0},),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(
        info.value == "No knob importance data in the summary." for info in app.info
    )


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


def test_dashboard_tracks_tab_recommends_tracks() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.button[4].click().run()

    assert not app.exception
    assert any(caption.value == "Strategy: Track Similarity" for caption in app.caption)


def test_dashboard_tracks_tab_supports_ltr() -> None:
    service = FakeDashboardService()
    service.artifact.track_ltr_model = "mock_track_ltr"
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    ltr_box = next((cb for cb in app.checkbox if cb.key == "tracks_use_ltr"), None)
    assert ltr_box is not None
    assert not ltr_box.disabled
    ltr_box.check().run()
    app.button[4].click().run()

    assert not app.exception
    assert any(caption.value == "Strategy: Track Ltr" for caption in app.caption)


def test_dashboard_tracks_tab_finds_similar_tracks() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.button[5].click().run()

    assert not app.exception
    assert any(caption.value == "Strategy: Track Similarity" for caption in app.caption)


def test_dashboard_tracks_tab_search_filters_tracks() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.text_input[0].set_value("track_1").run()

    assert not app.exception
    assert any(
        caption.value == "Showing 1 matching track(s)." for caption in app.caption
    )


def test_dashboard_tracks_tab_search_warns_without_matches() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    app.text_input[0].set_value("zzz-no-such-track").run()

    assert not app.exception
    assert any(
        warning.value == "No tracks match the current search."
        for warning in app.warning
    )


def test_track_catalog_frame_uses_service_search() -> None:
    frame = track_catalog_frame(FakeDashboardService())
    filtered_frame = track_catalog_frame(FakeDashboardService(), query="hit 2")

    assert frame["track_id"].tolist() == ["track_1", "track_2"]
    assert frame["total_plays"].tolist() == [30, 20]
    assert filtered_frame["track_id"].tolist() == ["track_2"]
    assert filtered_frame.attrs["total"] == 1


def test_dashboard_tracks_tab_renders_catalog_caption() -> None:
    app = AppTest.from_function(
        dashboard_script,
        args=(FakeDashboardService(),),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(caption.value == "Showing 2 matching tracks." for caption in app.caption)


def test_recommendation_frame_supports_track_responses() -> None:
    frame = recommendation_frame(
        {
            "similar_tracks": [
                {
                    "track_id": "track_2",
                    "track_name": "Hit 2",
                    "artist_name": "Test Artist",
                    "score": 0.9,
                }
            ]
        }
    )

    assert frame.loc[0, "Artist"] == "Hit 2 by Test Artist"
    assert frame.loc[0, "Artist ID"] == "track_2"


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
        "Tracks",
        "Catalog",
        "Ablation Summary",
        "Cold-Start Bandit",
    ]


def test_dashboard_bandit_tab_streaming_queue_metrics_and_flush() -> None:
    service = FakeDashboardService()
    service.streaming_queue_enabled = True
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any("Streaming Feedback Ingestion Queue" in m.value for m in app.markdown)
    assert any(metric.label == "Queue Depth" for metric in app.metric)
    assert any(metric.label == "Utilization" for metric in app.metric)
    assert any(metric.label == "Backpressure" for metric in app.metric)
    assert any(metric.label == "Flushed Batches" for metric in app.metric)

    flush_button = next(
        button for button in app.button if button.label == "Flush Feedback Queue"
    )
    flush_button.click().run()

    assert not app.exception
    assert service.last_flush_called is True
    assert any(
        "Streaming feedback queue flushed successfully." in s.value
        for s in app.success
    )


def test_dashboard_bandit_tab_maintenance_daemon_and_sweep() -> None:
    service = FakeDashboardService()
    service.maintenance_worker_enabled = True
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(
        "Asynchronous Maintenance Daemon" in m.value for m in app.markdown
    )
    assert any(metric.label == "Daemon Status" for metric in app.metric)
    assert any(metric.label == "Sweep Interval" for metric in app.metric)
    assert any(metric.label == "Cycles Executed" for metric in app.metric)
    assert any(metric.label == "Sweeps / Folded" for metric in app.metric)
    assert any(metric.label == "Last Sweep" for metric in app.metric)
    assert any(metric.label == "Avg Sweep" for metric in app.metric)

    trigger_button = next(
        button
        for button in app.button
        if button.label == "Trigger Maintenance Sweep"
    )
    trigger_button.click().run()

    assert not app.exception
    assert service.last_trigger_sweep_called is True
    assert any(
        "Maintenance sweep triggered:" in s.value
        and "record(s) folded." in s.value
        for s in app.success
    )


def test_dashboard_bandit_tab_champion_tagging_and_rollback() -> None:
    service = FakeDashboardService()
    snap_filename = "bandit_state_20260927T000000Z_baseline.json"
    service.snapshots.append(
        {
            "filename": snap_filename,
            "path": f"reports/bandit_snapshots/{snap_filename}",
            "timestamp": "20260927T000000Z",
            "label": "baseline",
            "size_bytes": 1024,
        }
    )
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any(
        "Champion Snapshot Management & State Rollback" in m.value
        for m in app.markdown
    )

    tag_button = next(
        button for button in app.button if button.label == "Tag as Champion"
    )
    tag_button.click().run()

    assert not app.exception
    assert service.champion_snapshot_filename == snap_filename
    assert any(
        f"Snapshot `{snap_filename}` tagged as champion." in s.value
        for s in app.success
    )

    rollback_button = next(
        button for button in app.button if button.label == "Roll Back State"
    )
    rollback_button.click().run()

    assert not app.exception
    assert service.last_rollback_called is True
    assert any(
        f"Rolled back active state to `{snap_filename}`." in s.value
        for s in app.success
    )


def test_dashboard_bandit_tab_drift_safety_evaluation_pass_and_breach() -> None:
    service = FakeDashboardService()
    snap_filename = "bandit_state_20260927T000000Z_baseline.json"
    service.snapshots.append(
        {
            "filename": snap_filename,
            "path": f"reports/bandit_snapshots/{snap_filename}",
            "timestamp": "20260927T000000Z",
            "label": "baseline",
            "size_bytes": 1024,
        }
    )
    app = AppTest.from_function(
        dashboard_script,
        args=(service,),
        default_timeout=10,
    ).run()

    verify_button = next(
        button for button in app.button if button.label == "Verify Drift Safety"
    )
    verify_button.click().run()

    assert not app.exception
    assert any(
        "Drift Safety Check: **PASSED**" in s.value for s in app.success
    )

    class BreachSafetyService(FakeDashboardService):
        def evaluate_bandit_drift_safety(
            self,
            reference_path: str | Path | None = None,
            state_path: str | Path | None = None,
            snapshot_dir: str | Path | None = None,
            thresholds: Any = None,
        ) -> dict[str, Any]:
            return {
                "is_safe": False,
                "violations": ["L2 drift 0.85 exceeds threshold 0.50"],
                "warnings": [],
                "reference_snapshot": str(reference_path or "ref.json"),
                "metrics": {"max_l2_drift": 0.85},
                "thresholds": {"max_l2_drift": 0.50},
            }

    breach_service = BreachSafetyService()
    breach_service.snapshots.append(
        {
            "filename": snap_filename,
            "path": f"reports/bandit_snapshots/{snap_filename}",
            "timestamp": "20260927T000000Z",
            "label": "baseline",
            "size_bytes": 1024,
        }
    )
    breach_app = AppTest.from_function(
        dashboard_script,
        args=(breach_service,),
        default_timeout=10,
    ).run()

    verify_button_breach = next(
        button
        for button in breach_app.button
        if button.label == "Verify Drift Safety"
    )
    verify_button_breach.click().run()

    assert not breach_app.exception
    assert any(
        "Drift Safety Check: **BREACH DETECTED**" in err.value
        for err in breach_app.error
    )
    assert any(
        "Violation: L2 drift 0.85 exceeds threshold 0.50" in w.value
        for w in breach_app.warning
    )




