import json
from pathlib import Path

import pandas as pd
import pytest

import music_recommender.evaluate as evaluate_module
from music_recommender.evaluate import (
    _average_metric_dicts,
    _summarize_recommendations,
    ablation_importances,
    aggregate_ablation_reports,
    average_popularity,
    average_precision_at_k,
    build_ablation_settings,
    catalog_coverage,
    compare_parameter_settings,
    evaluate_model,
    evaluate_repeated_holdout,
    explanation_coverage,
    intra_list_diversity,
    load_ablation_report,
    map_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    ranking_params_for_training,
    recall_at_k,
    select_winning_strategies,
    serendipity_at_k,
    strategy_leaderboard,
    train_test_split_by_user,
    unexpectedness_at_k,
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


def test_train_test_split_aggregates_duplicates_before_holdout() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_1"],
            "artist_id": ["a", "a", "b"],
            "artist_name": ["A", "A", "B"],
            "play_count": [1, 2, 3],
        }
    )

    train_df, test_df = train_test_split_by_user(
        df,
        test_ratio=0.5,
        random_state=1,
    )

    train_pairs = set(zip(train_df["user_id"], train_df["artist_id"], strict=True))
    test_pairs = set(zip(test_df["user_id"], test_df["artist_id"], strict=True))
    assert not train_pairs & test_pairs
    assert train_df["play_count"].sum() + test_df["play_count"].sum() == 6


@pytest.mark.parametrize(
    "test_ratio",
    [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), True],
)
def test_train_test_split_rejects_invalid_ratio(test_ratio: float) -> None:
    with pytest.raises(ValueError, match="test_ratio must be between 0 and 1"):
        train_test_split_by_user(
            pd.DataFrame(
                {
                    "user_id": ["user_1", "user_1"],
                    "artist_id": ["a", "b"],
                    "artist_name": ["A", "B"],
                    "play_count": [1, 2],
                }
            ),
            test_ratio=test_ratio,
        )


def test_train_test_split_rejects_invalid_random_state() -> None:
    with pytest.raises(ValueError, match="random_state"):
        train_test_split_by_user(
            pd.DataFrame(
                {
                    "user_id": ["user_1", "user_1"],
                    "artist_id": ["a", "b"],
                    "artist_name": ["A", "B"],
                    "play_count": [1, 2],
                }
            ),
            random_state=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": True}, "top_k"),
        ({"folds": 0}, "folds"),
        ({"folds": 1.5}, "folds"),
        ({"use_gpu": 1}, "use_gpu"),
        ({"compare_baseline": 1}, "compare_baseline"),
        ({"compare_all": "yes"}, "compare_all"),
    ],
)
def test_repeated_holdout_rejects_invalid_parameters(
    overrides: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "df": pd.DataFrame(),
        "top_k": 1,
        "folds": 1,
        "use_gpu": False,
        "compare_baseline": False,
        "compare_all": False,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        evaluate_repeated_holdout(**parameters)


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


def test_novelty_at_k_works_on_known_example() -> None:
    novelty = novelty_at_k(
        [["a", "c"]],
        {
            "a": {"popularity_rank": 1},
            "b": {"popularity_rank": 2},
            "c": {"popularity_rank": 3},
        },
    )

    assert novelty == pytest.approx(0.5)


def test_explanation_coverage_works_on_known_example() -> None:
    coverage = explanation_coverage(
        [
            [{"artist_id": "a", "reasons": ["shared pop"]}],
            [{"artist_id": "b", "reasons": []}],
        ]
    )

    assert coverage == pytest.approx(0.5)


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


def test_repeated_holdout_returns_all_model_comparison() -> None:
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
    metadata_df = pd.DataFrame(
        {
            "artist_id": ["artist_1", "artist_2", "artist_3", "artist_4"],
            "artist_name": ["A", "B", "C", "D"],
            "genres": ["pop", "pop", "rock", "rock"],
            "mood_tags": ["bright", "bright", "raw", "raw"],
            "country": ["US", "US", "UK", "UK"],
            "era": ["2020s", "2020s", "2000s", "2000s"],
        }
    )

    metrics = evaluate_repeated_holdout(
        df,
        top_k=2,
        folds=1,
        use_gpu=False,
        compare_all=True,
        metadata_df=metadata_df,
    )

    assert {"als", "popularity", "content", "hybrid"} <= set(metrics)
    assert "novelty_at_k" in metrics["hybrid"]
    assert "explanation_coverage" in metrics["content"]


def test_precision_at_k_returns_zero_for_non_positive_k() -> None:
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0


def test_recall_at_k_returns_zero_for_empty_relevant() -> None:
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_average_precision_at_k_returns_zero_for_empty_relevant() -> None:
    assert average_precision_at_k(["a"], set(), k=3) == 0.0


def test_map_at_k_returns_zero_for_empty_input() -> None:
    assert map_at_k([], [], k=3) == 0.0


def test_ndcg_at_k_returns_zero_for_empty_relevant() -> None:
    assert ndcg_at_k(["a"], set(), k=3) == 0.0


def test_catalog_coverage_returns_zero_for_empty_catalog() -> None:
    assert catalog_coverage([["a"]], set()) == 0.0


def test_average_popularity_returns_zero_without_known_artists() -> None:
    assert average_popularity([["a"]], {}) == 0.0


def test_novelty_at_k_returns_zero_for_empty_stats() -> None:
    assert novelty_at_k([["a"]], {}) == 0.0


def test_novelty_at_k_skips_artists_missing_from_stats() -> None:
    novelty = novelty_at_k(
        [["b", "unknown"]],
        {
            "a": {"popularity_rank": 1},
            "b": {"popularity_rank": 2},
        },
    )

    assert novelty == pytest.approx(1.0)


def test_unexpectedness_at_k_works_on_known_example() -> None:
    unexpectedness = unexpectedness_at_k(
        [["a", "b", "c", "d"]],
        {
            "a": {"popularity_rank": 1},
            "b": {"popularity_rank": 2},
            "c": {"popularity_rank": 3},
            "d": {"popularity_rank": 4},
        },
    )

    assert unexpectedness == pytest.approx(0.5)


def test_unexpectedness_at_k_returns_zero_for_empty_stats() -> None:
    assert unexpectedness_at_k([["a"]], {}) == 0.0


def test_unexpectedness_at_k_returns_zero_without_known_artists() -> None:
    assert (
        unexpectedness_at_k(
            [["unknown"]],
            {"a": {"popularity_rank": 1}, "b": {"popularity_rank": 2}},
        )
        == 0.0
    )


def test_serendipity_at_k_works_on_known_example() -> None:
    serendipity = serendipity_at_k(
        recommended_items=["a", "d"],
        relevant_items={"a", "d"},
        artist_stats={
            "a": {"popularity_rank": 1},
            "b": {"popularity_rank": 2},
            "c": {"popularity_rank": 3},
            "d": {"popularity_rank": 4},
        },
        k=4,
    )

    assert serendipity == pytest.approx(0.5)


def test_serendipity_at_k_returns_zero_without_relevant_hits() -> None:
    serendipity = serendipity_at_k(
        recommended_items=["a"],
        relevant_items={"unknown"},
        artist_stats={
            "a": {"popularity_rank": 1},
            "b": {"popularity_rank": 2},
        },
        k=1,
    )

    assert serendipity == 0.0


def test_serendipity_at_k_returns_zero_for_empty_inputs() -> None:
    stats = {"a": {"popularity_rank": 1}, "b": {"popularity_rank": 2}}

    assert serendipity_at_k(["a"], {"a"}, {}, k=1) == 0.0
    assert serendipity_at_k(["a"], {"a"}, stats, k=0) == 0.0
    assert serendipity_at_k(["a"], set(), stats, k=1) == 0.0


def test_explanation_coverage_returns_zero_for_empty_input() -> None:
    assert explanation_coverage([]) == 0.0


def test_intra_list_diversity_returns_zero_for_single_item() -> None:
    diversity = intra_list_diversity(
        ["a"],
        artist_factors=pd.DataFrame([[1.0, 0.0]]).to_numpy(),
        artist_id_to_index={"a": 0},
    )

    assert diversity == 0.0


def test_average_metric_dicts_returns_empty_for_empty_input() -> None:
    assert _average_metric_dicts([]) == {}


def test_summarize_recommendations_returns_zeros_for_empty_input() -> None:
    summary = _summarize_recommendations(
        list_of_recommended_items=[],
        list_of_relevant_items=[],
        catalog_items=set(),
        artist_stats={},
        artist_factors=pd.DataFrame().to_numpy(),
        artist_id_to_index={},
        top_k=3,
    )

    assert set(summary) == {
        "precision_at_k",
        "recall_at_k",
        "map_at_k",
        "ndcg_at_k",
        "catalog_coverage",
        "average_popularity",
        "intra_list_diversity",
        "novelty_at_k",
        "unexpectedness_at_k",
        "serendipity_at_k",
        "explanation_coverage",
    }
    assert all(value == 0.0 for value in summary.values())


def test_evaluate_model_delegates_to_repeated_holdout(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_repeated_holdout(**kwargs) -> dict:
        calls.append(kwargs)
        return {"precision_at_k": 0.5}

    monkeypatch.setattr(
        evaluate_module,
        "evaluate_repeated_holdout",
        fake_repeated_holdout,
    )
    df = pd.DataFrame()

    metrics = evaluate_model(df, top_k=3, use_gpu=False)

    assert metrics == {"precision_at_k": 0.5}
    assert calls == [
        {
            "df": df,
            "top_k": 3,
            "folds": 1,
            "use_gpu": False,
            "compare_baseline": False,
        }
    ]


def test_repeated_holdout_als_only_returns_flat_metrics() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_2"],
            "artist_id": ["artist_1", "artist_2", "artist_1", "artist_2"],
            "artist_name": ["A", "B", "A", "B"],
            "play_count": [5, 4, 5, 4],
        }
    )

    metrics = evaluate_repeated_holdout(df, top_k=1, folds=1, use_gpu=False)

    assert "als" not in metrics
    assert "popularity" not in metrics
    assert "precision_at_k" in metrics


def test_repeated_holdout_rejects_non_dict_recommend_kwargs() -> None:
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        evaluate_repeated_holdout(df, top_k=3, use_gpu=False, recommend_kwargs="x")


def test_repeated_holdout_rejects_non_bool_learn_to_rank() -> None:
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="learn_to_rank"):
        evaluate_repeated_holdout(df, top_k=3, use_gpu=False, learn_to_rank="yes")


def test_repeated_holdout_with_learn_to_rank_adds_ltr_arm(
    interactions_df,
) -> None:
    metrics = evaluate_repeated_holdout(
        interactions_df,
        top_k=2,
        folds=1,
        use_gpu=False,
        learn_to_rank=True,
    )

    assert "als" in metrics
    assert "ltr" in metrics
    assert "precision_at_k" in metrics["als"]
    assert "precision_at_k" in metrics["ltr"]


def test_evaluate_single_fold_forwards_learn_to_rank(monkeypatch) -> None:
    captured: dict = {}

    def fake_train_ltr(**kwargs) -> object:
        captured.update(kwargs)
        return object()

    def fake_rank_with_ltr(*args, **kwargs) -> list[dict]:
        return []

    monkeypatch.setattr(evaluate_module, "train_ltr_ranker", fake_train_ltr)
    monkeypatch.setattr(evaluate_module, "rank_with_ltr", fake_rank_with_ltr)
    df = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_2"],
            "artist_id": ["artist_1", "artist_2", "artist_1", "artist_2"],
            "artist_name": ["A", "B", "A", "B"],
            "play_count": [5, 4, 5, 4],
        }
    )

    evaluate_module._evaluate_single_fold(
        df=df,
        top_k=1,
        random_state=42,
        use_gpu=False,
        compare_baseline=False,
        compare_all=False,
        metadata_df=None,
        recommend_kwargs=None,
        learn_to_rank=True,
    )

    assert captured["random_state"] == 42


def test_compare_parameter_settings_returns_labeled_metrics(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_repeated_holdout(**kwargs) -> dict:
        calls.append(kwargs)
        return {"precision_at_k": 0.5}

    monkeypatch.setattr(
        evaluate_module,
        "evaluate_repeated_holdout",
        fake_repeated_holdout,
    )
    df = pd.DataFrame()

    metrics = compare_parameter_settings(
        df,
        top_k=3,
        parameter_sets={"control": {}, "diverse": {"diversity": 0.5}},
        use_gpu=False,
    )

    assert metrics == {
        "control": {"precision_at_k": 0.5},
        "diverse": {"precision_at_k": 0.5},
    }
    assert [call["recommend_kwargs"] for call in calls] == [{}, {"diversity": 0.5}]
    assert all(call["compare_baseline"] is False for call in calls)


def test_compare_parameter_settings_rejects_invalid_inputs() -> None:
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        compare_parameter_settings(df, top_k=3, parameter_sets={})
    with pytest.raises(ValueError):
        compare_parameter_settings(
            df,
            top_k=3,
            parameter_sets={"control": {}},
            folds=0,
        )
    with pytest.raises(ValueError):
        compare_parameter_settings(
            df,
            top_k=3,
            parameter_sets={"control": {}},
            use_gpu="x",
        )


def _comparison() -> dict[str, dict[str, float]]:
    control: dict[str, float] = {
        "precision_at_k": 0.5,
        "recall_at_k": 0.7,
        "map_at_k": 0.4,
        "ndcg_at_k": 0.45,
        "catalog_coverage": 0.8,
        "average_popularity": 90.0,
        "novelty_at_k": 0.6,
        "unexpectedness_at_k": 0.3,
        "serendipity_at_k": 0.2,
        "explanation_coverage": 1.0,
        "intra_list_diversity": 0.7,
    }
    diverse = dict(control)
    diverse.update(
        {
            "precision_at_k": 0.8,
            "recall_at_k": 0.9,
            "map_at_k": 0.6,
            "ndcg_at_k": 0.9,
            "novelty_at_k": 0.7,
            "intra_list_diversity": 0.9,
            "average_popularity": 95.0,
        }
    )
    return {"control": control, "diverse": diverse}


def test_select_winning_strategies_returns_per_metric_winner() -> None:
    winners = select_winning_strategies(_comparison())

    assert winners["precision_at_k"] == "diverse"
    assert winners["catalog_coverage"] == "control"
    assert winners["unexpectedness_at_k"] == "control"
    assert "average_popularity" not in winners


def test_select_winning_strategies_picks_first_label_on_ties() -> None:
    comparison = {"a": _comparison()["control"], "b": dict(_comparison()["control"])}

    winners = select_winning_strategies(comparison)

    assert all(label == "a" for label in winners.values())


def test_select_winning_strategies_rejects_empty_comparison() -> None:
    with pytest.raises(ValueError):
        select_winning_strategies({})


def test_select_winning_strategies_skips_metrics_absent_from_all() -> None:
    comparison = _comparison()
    for metrics in comparison.values():
        del metrics["serendipity_at_k"]

    winners = select_winning_strategies(comparison)

    assert "serendipity_at_k" not in winners
    assert len(winners) == 9


def test_strategy_leaderboard_ranks_by_wins_then_ndcg() -> None:
    comparison = _comparison()
    comparison["content"] = dict(_comparison()["control"])
    comparison["content"].update(
        {"ndcg_at_k": 0.3, "novelty_at_k": 0.9, "serendipity_at_k": 0.9}
    )

    leaderboard = strategy_leaderboard(comparison)

    assert leaderboard[0] == ("diverse", 5)
    assert leaderboard[1] == ("control", 3)
    assert leaderboard[2] == ("content", 2)


def test_strategy_leaderboard_breaks_ties_with_ndcg() -> None:
    comparison = {
        "a": dict(_comparison()["control"]),
        "b": dict(_comparison()["control"]),
    }
    for metric in (
        "precision_at_k",
        "recall_at_k",
        "map_at_k",
        "catalog_coverage",
        "explanation_coverage",
    ):
        comparison["a"][metric] = comparison["b"][metric] + 0.1
    for metric in (
        "ndcg_at_k",
        "novelty_at_k",
        "unexpectedness_at_k",
        "serendipity_at_k",
        "intra_list_diversity",
    ):
        comparison["b"][metric] = comparison["a"][metric] + 0.1

    leaderboard = strategy_leaderboard(comparison)

    assert leaderboard[0][0] == "b"
    assert leaderboard[1][0] == "a"
    assert all(wins == 5 for _, wins in leaderboard)


def test_ranking_params_for_training_extracts_ranking_keys_only() -> None:
    params = ranking_params_for_training(
        {
            "popularity_penalty": 0.2,
            "diversity": 0.5,
            "include_listened": True,
            "birds": "nope",
        }
    )

    assert params == {
        "popularity_penalty": 0.2,
        "diversity": 0.5,
        "include_listened": True,
    }


def test_ranking_params_for_training_returns_empty_for_unknown_keys() -> None:
    assert ranking_params_for_training({"content_weight": 0.4}) == {}


def test_repeated_holdout_builds_content_from_interactions_when_no_metadata() -> None:
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
            ],
            "artist_id": [
                "artist_1",
                "artist_2",
                "artist_3",
                "artist_1",
                "artist_3",
                "artist_4",
                "artist_2",
                "artist_4",
            ],
            "artist_name": ["A", "B", "C", "A", "C", "D", "B", "D"],
            "play_count": [5, 4, 3, 5, 4, 3, 5, 3],
        }
    )

    metrics = evaluate_repeated_holdout(
        df,
        top_k=1,
        folds=1,
        use_gpu=False,
        compare_all=True,
    )

    assert {"als", "popularity", "content", "hybrid"} <= set(metrics)
    assert "explanation_coverage" in metrics["content"]


def test_build_ablation_settings_keeps_champion_and_turns_active_knobs_off() -> None:
    settings = build_ablation_settings(
        {
            "popularity_penalty": 0.2,
            "diversity": 0.5,
            "include_listened": False,
        }
    )

    assert set(settings) == {
        "champion",
        "no_popularity_penalty",
        "no_diversity",
        "no_ranking",
    }
    assert settings["champion"] == {
        "popularity_penalty": 0.2,
        "diversity": 0.5,
        "include_listened": False,
    }
    assert settings["no_popularity_penalty"]["popularity_penalty"] == 0.0
    assert settings["no_diversity"]["diversity"] == 0.0
    assert settings["no_ranking"] == {
        "popularity_penalty": 0.0,
        "diversity": 0.0,
        "include_listened": False,
    }


def test_build_ablation_settings_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown ranking parameter 'birds'"):
        build_ablation_settings({"popularity_penalty": 0.2, "birds": 1})


def test_build_ablation_settings_rejects_already_neutral_champion() -> None:
    with pytest.raises(ValueError, match="already neutral"):
        build_ablation_settings(
            {
                "popularity_penalty": 0.0,
                "diversity": 0.0,
                "include_listened": False,
            }
        )


def test_ablation_importances_reports_signed_deltas_and_ranks_by_impact() -> None:
    champion = dict(_comparison()["control"])
    lowered = dict(champion)
    lowered["ndcg_at_k"] -= 0.1
    lowered["novelty_at_k"] -= 0.2
    lowered["intra_list_diversity"] -= 0.3
    comparison = {
        "champion": champion,
        "no_popularity_penalty": lowered,
        "no_ranking": lowered,
    }

    importance, ranking = ablation_importances(comparison)

    assert importance["popularity_penalty"]["ndcg_at_k"] == pytest.approx(0.1)
    assert importance["ranking_settings"]["ndcg_at_k"] == pytest.approx(0.1)
    assert importance["ranking_settings"]["novelty_at_k"] == pytest.approx(0.2)
    assert ranking[0][0] == "popularity_penalty"
    assert ranking[0][1] == pytest.approx(abs(-0.1) + abs(-0.2) + abs(-0.3))


def test_ablation_importances_requires_champion_arm() -> None:
    with pytest.raises(ValueError, match="must contain the 'champion' arm"):
        ablation_importances({"als": _comparison()["control"]})


def test_ablation_importances_ranks_single_knob_first_on_ties() -> None:
    champion = dict(_comparison()["control"])
    comparison = {
        "champion": champion,
        "no_popularity_penalty": champion,
        "no_ranking": champion,
    }

    _, ranking = ablation_importances(comparison)

    assert ranking == [("popularity_penalty", 0.0), ("ranking_settings", 0.0)]


def test_ablation_knob_name_passes_through_unknown_labels() -> None:
    assert evaluate_module._ablation_knob_name("champion_extra") == "champion_extra"


def test_write_ablation_report_persists_stable_json(tmp_path: Path) -> None:
    champion = dict(_comparison()["control"])
    lowered = dict(champion)
    lowered["ndcg_at_k"] -= 0.1
    comparison = {
        "champion": champion,
        "no_popularity_penalty": lowered,
    }

    report_path = evaluate_module.write_ablation_report(
        comparison,
        tmp_path,
        report_name="round-1",
    )

    assert report_path == tmp_path / "round-1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["champion_label"] == "champion"
    assert set(report["arms"]) == {"champion", "no_popularity_penalty"}
    assert report["importance"]["popularity_penalty"]["ndcg_at_k"] == pytest.approx(0.1)
    assert report["ranking"] == [
        {"knob": "popularity_penalty", "impact": pytest.approx(0.1)}
    ]
    assert "generated_at" in report


def test_write_ablation_report_creates_parent_and_uses_default_name(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "nested" / "output"
    champion = dict(_comparison()["control"])
    comparison = {
        "champion": champion,
        "no_popularity_penalty": champion,
    }
    report_path = evaluate_module.write_ablation_report(comparison, report_dir)

    assert report_path == report_dir / "ablation_importance.json"
    assert report_path.exists()
    json.loads(report_path.read_text(encoding="utf-8"))


def test_load_ablation_report_round_trips_written_report(tmp_path: Path) -> None:
    champion = dict(_comparison()["control"])
    comparison = {"champion": champion, "no_popularity_penalty": champion}
    written = evaluate_module.write_ablation_report(comparison, tmp_path)

    report = load_ablation_report(written)

    assert report["champion_label"] == "champion"
    assert set(report["importance"]) == {"popularity_penalty"}
    assert report["arms"]["champion"]["ndcg_at_k"] == champion["ndcg_at_k"]


def test_load_ablation_report_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_ablation_report(tmp_path / "missing.json")


def test_load_ablation_report_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse ablation report"):
        load_ablation_report(bad)


def test_load_ablation_report_rejects_non_ablation_schema(tmp_path: Path) -> None:
    unrelated = tmp_path / "other.json"
    unrelated.write_text(json.dumps({"foo": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid ablation report"):
        load_ablation_report(unrelated)


def test_aggregate_ablation_reports_averages_impacts_across_runs(
    tmp_path: Path,
) -> None:
    champion = dict(_comparison()["control"])
    first = {"champion": champion, "no_popularity_penalty": champion}
    second = {"champion": champion, "no_popularity_penalty": champion}
    evaluate_module.write_ablation_report(first, tmp_path, report_name="run-a")
    evaluate_module.write_ablation_report(second, tmp_path, report_name="run-b")

    summary = aggregate_ablation_reports(tmp_path)

    assert summary["reports_loaded"] == 2
    knob = summary["knobs"]["popularity_penalty"]
    assert knob["count"] == 2
    assert knob["mean_impact"] == pytest.approx(0.0)
    assert knob["median_impact"] == pytest.approx(0.0)
    assert knob["std_impact"] == pytest.approx(0.0)
    assert summary["ranking"] == [{"knob": "popularity_penalty", "mean_impact": 0.0}]


def test_aggregate_ablation_reports_combines_multiple_knobs_sorted_by_impact(
    tmp_path: Path,
) -> None:
    champion = dict(_comparison()["control"])
    lowered = dict(champion)
    lowered["ndcg_at_k"] -= 0.5
    lowered["novelty_at_k"] -= 0.3
    comparison = {
        "champion": champion,
        "no_popularity_penalty": lowered,
        "no_diversity": lowered,
    }
    evaluate_module.write_ablation_report(comparison, tmp_path, report_name="run")

    summary = aggregate_ablation_reports(tmp_path)

    impacts = {item["knob"]: item["mean_impact"] for item in summary["ranking"]}
    assert impacts["popularity_penalty"] == pytest.approx(0.8)
    assert impacts["diversity"] == pytest.approx(0.8)
    assert summary["ranking"][0]["knob"] == "diversity"


def test_aggregate_ablation_reports_raises_when_no_reports(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No ablation reports found"):
        aggregate_ablation_reports(tmp_path)
