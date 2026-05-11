import pandas as pd
import pytest

from music_recommender.evaluate import (
    average_popularity,
    catalog_coverage,
    evaluate_repeated_holdout,
    intra_list_diversity,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    train_test_split_by_user,
)


def test_precision_at_k_works_on_known_example() -> None:
    score = precision_at_k(["a", "b", "c"], {"a", "c"}, k=3)

    assert score == pytest.approx(2 / 3)


def test_recall_at_k_works_on_known_example() -> None:
    score = recall_at_k(["a", "b", "c"], {"a", "c", "d"}, k=3)

    assert score == pytest.approx(2 / 3)


def test_map_at_k_works_on_known_example() -> None:
    score = map_at_k(
        [["a", "b", "c"], ["x", "y", "z"]],
        [{"a", "c"}, {"z"}],
        k=3,
    )

    assert score == pytest.approx((5 / 6 + 1 / 3) / 2)


def test_ndcg_at_k_works_on_known_example() -> None:
    score = ndcg_at_k(["a", "b", "c"], {"a", "c"}, k=3)

    assert score == pytest.approx(0.9197207891)


def test_train_test_split_by_user_keeps_users_in_train_when_possible() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_2", "user_3"],
            "artist_id": ["a", "b", "a", "c", "d"],
            "artist_name": ["A", "B", "A", "C", "D"],
            "play_count": [1, 2, 3, 4, 5],
        }
    )

    train_df, test_df = train_test_split_by_user(df, test_ratio=0.5, random_state=1)

    assert {"user_1", "user_2", "user_3"} <= set(train_df["user_id"])
    assert {"user_1", "user_2"} == set(test_df["user_id"])


def test_catalog_coverage_works_on_known_example() -> None:
    coverage = catalog_coverage([["a", "b"], ["b", "c"]], {"a", "b", "c", "d"})

    assert coverage == pytest.approx(0.75)


def test_average_popularity_works_on_known_example() -> None:
    popularity = average_popularity(
        [["a", "b"]],
        {
            "a": {"total_plays": 10},
            "b": {"total_plays": 30},
        },
    )

    assert popularity == pytest.approx(20.0)


def test_intra_list_diversity_works_on_known_example() -> None:
    diversity = intra_list_diversity(
        ["a", "b"],
        artist_factors=pd.DataFrame([[1.0, 0.0], [0.0, 1.0]]).to_numpy(),
        artist_id_to_index={"a": 0, "b": 1},
    )

    assert diversity == pytest.approx(1.0)


def test_repeated_holdout_returns_baseline_comparison() -> None:
    df = pd.DataFrame(
        {
            "user_id": [
                "user_1",
                "user_1",
                "user_1",
                "user_2",
                "user_2",
                "user_2",
                "user_3",
                "user_3",
                "user_3",
            ],
            "artist_id": [
                "artist_1",
                "artist_2",
                "artist_3",
                "artist_1",
                "artist_3",
                "artist_4",
                "artist_2",
                "artist_3",
                "artist_4",
            ],
            "artist_name": ["A", "B", "C", "A", "C", "D", "B", "C", "D"],
            "play_count": [5, 4, 3, 5, 4, 3, 5, 4, 3],
        }
    )

    metrics = evaluate_repeated_holdout(
        df,
        top_k=2,
        folds=2,
        use_gpu=False,
        compare_baseline=True,
    )

    assert "als" in metrics
    assert "popularity" in metrics
    assert "catalog_coverage" in metrics["als"]
