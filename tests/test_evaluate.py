import pandas as pd
import pytest

from music_recommender.evaluate import (
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
