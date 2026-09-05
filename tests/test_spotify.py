from __future__ import annotations

from typing import Any

import pytest

from music_recommender import spotify as spotify_module
from music_recommender.spotify import (
    SpotifyAudioFeatures,
    SpotifyConfig,
    SpotifyTrack,
    build_track_metadata_frame,
    create_spotify_client,
    fetch_artist,
    fetch_artist_top_tracks,
    fetch_artists,
    fetch_audio_features,
    fetch_tracks,
    get_artist_related_artists,
    search_artists,
    search_tracks,
)


class FakeSpotifyClient:
    def artist(self, artist_id: str) -> dict[str, Any]:
        return {
            "id": artist_id,
            "name": "Test Artist",
            "genres": ["pop"],
            "popularity": 80,
            "followers": {"total": 1000},
            "external_urls": {"spotify": "https://open.spotify.com/artist/x"},
            "images": [],
        }

    def artists(self, ids: list[str]) -> dict[str, Any]:
        return {"artists": [self.artist(i) for i in ids]}

    def artist_top_tracks(self, artist_id: str, country: str = "US") -> dict[str, Any]:
        assert country
        return {
            "tracks": [
                {
                    "id": "track_1",
                    "name": "Hit",
                    "artists": [{"id": artist_id, "name": "Test Artist"}],
                    "album": {"id": "album_1", "name": "Album"},
                    "duration_ms": 200000,
                    "popularity": 90,
                    "explicit": False,
                    "external_urls": {},
                    "preview_url": None,
                }
            ]
        }

    def tracks(self, ids: list[str]) -> dict[str, Any]:
        return {
            "tracks": [
                {
                    "id": i,
                    "name": f"Song {i}",
                    "artists": [{"id": "artist_1", "name": "Test Artist"}],
                    "album": {"id": "album_1", "name": "Album"},
                    "duration_ms": 200000,
                    "popularity": 70,
                    "explicit": False,
                    "external_urls": {},
                    "preview_url": None,
                }
                for i in ids
            ]
        }

    def audio_features(self, ids: list[str]) -> list[dict[str, Any] | None]:
        out: list[dict[str, Any] | None] = []
        for i in ids:
            if i == "missing":
                out.append(None)
            else:
                out.append(
                    {
                        "id": i,
                        "danceability": 0.7,
                        "energy": 0.8,
                        "key": 5,
                        "loudness": -5.0,
                        "mode": 1,
                        "speechiness": 0.05,
                        "acousticness": 0.1,
                        "instrumentalness": 0.0,
                        "liveness": 0.1,
                        "valence": 0.9,
                        "tempo": 120.0,
                        "time_signature": 4,
                    }
                )
        return out

    def search(self, q: str, type: str, limit: int = 20) -> dict[str, Any]:
        assert q
        assert limit
        if type == "artist":
            return {"artists": {"items": [self.artist("artist_1")]}}
        return {
            "tracks": {
                "items": [
                    {
                        "id": "track_1",
                        "name": "Hit",
                        "artists": [{"id": "artist_1", "name": "Test Artist"}],
                        "album": {"id": "album_1", "name": "Album"},
                        "duration_ms": 200000,
                        "popularity": 90,
                        "explicit": False,
                        "external_urls": {},
                        "preview_url": None,
                    }
                ]
            }
        }

    def artist_related_artists(self, artist_id: str) -> dict[str, Any]:
        assert artist_id
        return {"artists": [self.artist("artist_2")]}


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    cfg = SpotifyConfig.from_env()
    assert cfg.client_id == "id"
    monkeypatch.delenv("SPOTIFY_CLIENT_ID")
    with pytest.raises(ValueError, match="must be set"):
        SpotifyConfig.from_env()


def test_create_client_raises_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spotify_module, "SPOTIPY_AVAILABLE", False)
    with pytest.raises(ImportError, match="spotipy is not installed"):
        create_spotify_client(SpotifyConfig("id", "secret"))


@pytest.mark.skipif(
    not spotify_module.SPOTIPY_AVAILABLE, reason="spotipy extra not installed"
)
def test_create_client_returns_spotify_client() -> None:
    import spotipy

    client = create_spotify_client(SpotifyConfig("id", "secret"))

    assert isinstance(client, spotipy.Spotify)


def test_fetch_helpers_with_fake_client() -> None:
    client: Any = FakeSpotifyClient()
    artist = fetch_artist(client, "artist_1")
    assert artist.name == "Test Artist"
    assert artist.followers == 1000

    artists = fetch_artists(client, ["artist_1", "artist_2"])
    assert len(artists) == 2

    tracks = fetch_artist_top_tracks(client, "artist_1")
    assert tracks[0].name == "Hit"

    tracks2 = fetch_tracks(client, ["track_1"])
    assert tracks2[0].album_name == "Album"

    feats = fetch_audio_features(client, ["track_1", "missing"])
    assert feats[0] is not None
    assert feats[0] is not None and feats[0].tempo == 120.0
    assert feats[1] is None

    assert len(search_artists(client, "test")) == 1
    assert len(search_tracks(client, "test")) == 1
    assert len(get_artist_related_artists(client, "artist_1")) == 1


def make_spotify_track(track_id: str = "track_1") -> SpotifyTrack:
    return SpotifyTrack(
        id=track_id,
        name="Hit",
        artist_ids=["artist_1"],
        artist_names=["Test Artist"],
        album_id="album_1",
        album_name="Album",
        duration_ms=200000,
        popularity=90,
        explicit=False,
        external_urls={},
        preview_url=None,
    )


def make_spotify_features(track_id: str = "track_1") -> SpotifyAudioFeatures:
    return SpotifyAudioFeatures(
        id=track_id,
        danceability=0.7,
        energy=0.8,
        key=5,
        loudness=-5.0,
        mode=1,
        speechiness=0.05,
        acousticness=0.1,
        instrumentalness=0.0,
        liveness=0.1,
        valence=0.9,
        tempo=120.0,
        time_signature=4,
    )


def test_build_track_metadata_frame_matches_contract() -> None:
    frame = build_track_metadata_frame(
        [make_spotify_track("track_1"), make_spotify_track("track_2")],
        [make_spotify_features("track_1"), make_spotify_features("track_2")],
    )

    assert list(frame["track_id"]) == ["track_1", "track_2"]
    assert frame.loc[0, "artist_name"] == "Test Artist"
    assert frame.loc[0, "tempo"] == 120.0


def test_build_track_metadata_frame_skips_missing_and_duplicates() -> None:
    frame = build_track_metadata_frame(
        [
            make_spotify_track("track_1"),
            make_spotify_track("track_1"),
            make_spotify_track("track_missing"),
        ],
        [make_spotify_features("track_1"), None],
    )

    assert list(frame["track_id"]) == ["track_1"]
