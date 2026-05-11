from pathlib import Path

import pandas as pd

from music_recommender.artifacts import (
    build_artist_stats,
    build_recommender_artifact,
    create_dataset_fingerprint,
    load_artifact,
    save_artifact,
)
from music_recommender.model import train_als_model
from music_recommender.preprocessing import build_user_item_matrix, create_id_mappings


def artifact_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_2"],
            "artist_id": ["artist_1", "artist_2", "artist_2", "artist_3"],
            "artist_name": ["A", "B", "B", "C"],
            "play_count": [5, 3, 7, 2],
        }
    )


def test_artifact_bundle_saves_and_loads(tmp_path: Path) -> None:
    df = artifact_dataframe()
    mappings = create_id_mappings(df)
    matrix = build_user_item_matrix(
        df,
        mappings["user_id_to_index"],
        mappings["artist_id_to_index"],
    )
    model = train_als_model(matrix, 4, 0.01, 1, 10.0, use_gpu=False)
    artifact = build_recommender_artifact(
        model=model,
        mappings=mappings,
        user_item_matrix=matrix,
        filtered_df=df,
        raw_data_path=tmp_path / "missing.csv",
        training_config={"factors": 4},
    )
    artifact_path = tmp_path / "artifact.joblib"

    save_artifact(artifact, artifact_path)
    loaded_artifact = load_artifact(artifact_path)

    assert loaded_artifact.version == "2.0"
    assert loaded_artifact.user_item_matrix.shape == (2, 3)
    assert loaded_artifact.metadata["num_users"] == 2
    assert loaded_artifact.training_config["factors"] == 4
    assert "artist_2" in loaded_artifact.artist_stats


def test_artist_stats_include_popularity_rank() -> None:
    stats = build_artist_stats(artifact_dataframe())

    assert stats["artist_2"]["total_plays"] == 10
    assert stats["artist_2"]["listener_count"] == 2
    assert stats["artist_2"]["popularity_rank"] == 1


def test_dataset_fingerprint_changes_when_data_changes(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_path.write_text("user_id,artist_id,artist_name,play_count\nu,a,A,1\n")
    second_path.write_text("user_id,artist_id,artist_name,play_count\nu,a,A,2\n")

    first = create_dataset_fingerprint(first_path, artifact_dataframe())
    second = create_dataset_fingerprint(second_path, artifact_dataframe())

    assert first["sha256"] != second["sha256"]
