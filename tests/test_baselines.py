import pytest

from music_recommender.baselines import popular_artists


def artist_stats_with_ranks() -> dict[str, dict[str, object]]:
    return {
        "artist_1": {
            "artist_id": "artist_1",
            "artist_name": "The Weeknd",
            "total_plays": 90,
            "popularity_rank": 1,
        },
        "artist_2": {
            "artist_id": "artist_2",
            "artist_name": "Drake",
            "total_plays": 70,
            "popularity_rank": 2,
        },
        "artist_3": {
            "artist_id": "artist_3",
            "artist_name": "Taylor Swift",
            "total_plays": 50,
            "popularity_rank": 3,
        },
    }


def test_popular_artists_returns_artists_in_popularity_order() -> None:
    recommendations = popular_artists(artist_stats_with_ranks(), top_k=3)

    assert [rec["artist_id"] for rec in recommendations] == [
        "artist_1",
        "artist_2",
        "artist_3",
    ]
    assert recommendations[0]["artist_name"] == "The Weeknd"
    assert recommendations[0]["score"] == 90
    assert recommendations[0]["popularity_rank"] == 1


def test_popular_artists_honors_top_k() -> None:
    recommendations = popular_artists(artist_stats_with_ranks(), top_k=2)

    assert [rec["artist_id"] for rec in recommendations] == ["artist_1", "artist_2"]


def test_popular_artists_excludes_requested_artists() -> None:
    recommendations = popular_artists(
        artist_stats_with_ranks(),
        top_k=3,
        exclude_artist_ids={"artist_1", "artist_3"},
    )

    assert [rec["artist_id"] for rec in recommendations] == ["artist_2"]


def test_popular_artists_without_exclusions_accepts_none() -> None:
    recommendations = popular_artists(artist_stats_with_ranks(), top_k=3)

    assert len(recommendations) == 3


def test_popular_artists_returns_empty_for_empty_stats() -> None:
    recommendations = popular_artists({}, top_k=3)

    assert recommendations == []


def test_popular_artists_returns_empty_when_all_artists_excluded() -> None:
    recommendations = popular_artists(
        artist_stats_with_ranks(),
        top_k=3,
        exclude_artist_ids={"artist_1", "artist_2", "artist_3"},
    )

    assert recommendations == []


def test_popular_artists_top_k_exceeds_catalog_returns_all() -> None:
    recommendations = popular_artists(artist_stats_with_ranks(), top_k=10)

    assert len(recommendations) == 3


def test_popular_artists_ties_broken_by_artist_id() -> None:
    stats = artist_stats_with_ranks()
    tied = {
        "artist_b": {
            **stats["artist_1"],
            "popularity_rank": 1,
            "artist_id": "artist_b",
        },
        "artist_a": {
            **stats["artist_2"],
            "popularity_rank": 1,
            "artist_id": "artist_a",
        },
    }

    recommendations = popular_artists(tied, top_k=2)

    assert [rec["artist_id"] for rec in recommendations] == ["artist_a", "artist_b"]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_popular_artists_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        popular_artists(artist_stats_with_ranks(), top_k=top_k)  # type: ignore[arg-type]