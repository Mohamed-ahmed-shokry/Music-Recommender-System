"""Spotify API integration for fetching artist and track data."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any

SPOTIPY_AVAILABLE = importlib.util.find_spec("spotipy") is not None


@dataclass
class SpotifyConfig:
    """Configuration for Spotify API client."""

    client_id: str
    client_secret: str

    @classmethod
    def from_env(cls) -> SpotifyConfig:
        """Load configuration from environment variables."""
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET "
                "must be set in environment."
            )
        return cls(client_id=client_id, client_secret=client_secret)


def create_spotify_client(config: SpotifyConfig) -> Any:
    """Create an authenticated Spotify client using client credentials flow."""
    if not SPOTIPY_AVAILABLE:
        raise ImportError(
            "spotipy is not installed. Install with: uv sync --extra spotify"
        )
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    auth_manager = SpotifyClientCredentials(
        client_id=config.client_id, client_secret=config.client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)


@dataclass
class SpotifyArtist:
    """Artist data from Spotify."""

    id: str
    name: str
    genres: list[str]
    popularity: int
    followers: int
    external_urls: dict[str, str]
    images: list[dict[str, Any]]


@dataclass
class SpotifyTrack:
    """Track data from Spotify."""

    id: str
    name: str
    artist_ids: list[str]
    artist_names: list[str]
    album_id: str
    album_name: str
    duration_ms: int
    popularity: int
    explicit: bool
    external_urls: dict[str, str]
    preview_url: str | None


@dataclass
class SpotifyAudioFeatures:
    """Audio features for a track from Spotify."""

    id: str
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int


def fetch_artist(client: Any, artist_id: str) -> SpotifyArtist:
    """Fetch a single artist by Spotify ID."""
    data = client.artist(artist_id)
    return SpotifyArtist(
        id=data["id"],
        name=data["name"],
        genres=data.get("genres", []),
        popularity=data.get("popularity", 0),
        followers=data.get("followers", {}).get("total", 0),
        external_urls=data.get("external_urls", {}),
        images=data.get("images", []),
    )


def fetch_artists(
    client: Any, artist_ids: list[str]
) -> list[SpotifyArtist]:
    """Fetch multiple artists by Spotify IDs (max 50 per request)."""
    artists = []
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i : i + 50]
        data = client.artists(batch)
        for artist_data in data["artists"]:
            if artist_data:
                artists.append(
                    SpotifyArtist(
                        id=artist_data["id"],
                        name=artist_data["name"],
                        genres=artist_data.get("genres", []),
                        popularity=artist_data.get("popularity", 0),
                        followers=artist_data.get("followers", {}).get("total", 0),
                        external_urls=artist_data.get("external_urls", {}),
                        images=artist_data.get("images", []),
                    )
                )
    return artists


def fetch_artist_top_tracks(
    client: Any, artist_id: str, country: str = "US"
) -> list[SpotifyTrack]:
    """Fetch an artist's top tracks."""
    data = client.artist_top_tracks(artist_id, country=country)
    tracks = []
    for track_data in data["tracks"]:
        tracks.append(
            SpotifyTrack(
                id=track_data["id"],
                name=track_data["name"],
                artist_ids=[a["id"] for a in track_data["artists"]],
                artist_names=[a["name"] for a in track_data["artists"]],
                album_id=track_data["album"]["id"],
                album_name=track_data["album"]["name"],
                duration_ms=track_data["duration_ms"],
                popularity=track_data.get("popularity", 0),
                explicit=track_data.get("explicit", False),
                external_urls=track_data.get("external_urls", {}),
                preview_url=track_data.get("preview_url"),
            )
        )
    return tracks


def fetch_tracks(client: Any, track_ids: list[str]) -> list[SpotifyTrack]:
    """Fetch multiple tracks by Spotify IDs (max 50 per request)."""
    tracks = []
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i : i + 50]
        data = client.tracks(batch)
        for track_data in data["tracks"]:
            if track_data:
                tracks.append(
                    SpotifyTrack(
                        id=track_data["id"],
                        name=track_data["name"],
                        artist_ids=[a["id"] for a in track_data["artists"]],
                        artist_names=[a["name"] for a in track_data["artists"]],
                        album_id=track_data["album"]["id"],
                        album_name=track_data["album"]["name"],
                        duration_ms=track_data["duration_ms"],
                        popularity=track_data.get("popularity", 0),
                        explicit=track_data.get("explicit", False),
                        external_urls=track_data.get("external_urls", {}),
                        preview_url=track_data.get("preview_url"),
                    )
                )
    return tracks


def fetch_audio_features(
    client: Any, track_ids: list[str]
) -> list[SpotifyAudioFeatures | None]:
    """Fetch audio features for multiple tracks (max 100 per request)."""
    features: list[SpotifyAudioFeatures | None] = []
    """Fetch audio features for multiple tracks (max 100 per request)."""
    features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        data = client.audio_features(batch)
        for feature_data in data:
            if feature_data:
                features.append(
                    SpotifyAudioFeatures(
                        id=feature_data["id"],
                        danceability=feature_data["danceability"],
                        energy=feature_data["energy"],
                        key=feature_data["key"],
                        loudness=feature_data["loudness"],
                        mode=feature_data["mode"],
                        speechiness=feature_data["speechiness"],
                        acousticness=feature_data["acousticness"],
                        instrumentalness=feature_data["instrumentalness"],
                        liveness=feature_data["liveness"],
                        valence=feature_data["valence"],
                        tempo=feature_data["tempo"],
                        time_signature=feature_data["time_signature"],
                    )
                )
            else:
                features.append(None)
    return features


def search_artists(
    client: Any, query: str, limit: int = 20
) -> list[SpotifyArtist]:
    """Search for artists by name."""
    data = client.search(q=query, type="artist", limit=limit)
    artists = []
    for artist_data in data["artists"]["items"]:
        artists.append(
            SpotifyArtist(
                id=artist_data["id"],
                name=artist_data["name"],
                genres=artist_data.get("genres", []),
                popularity=artist_data.get("popularity", 0),
                followers=artist_data.get("followers", {}).get("total", 0),
                external_urls=artist_data.get("external_urls", {}),
                images=artist_data.get("images", []),
            )
        )
    return artists


def search_tracks(
    client: Any, query: str, limit: int = 20
) -> list[SpotifyTrack]:
    """Search for tracks by name."""
    data = client.search(q=query, type="track", limit=limit)
    tracks = []
    for track_data in data["tracks"]["items"]:
        tracks.append(
            SpotifyTrack(
                id=track_data["id"],
                name=track_data["name"],
                artist_ids=[a["id"] for a in track_data["artists"]],
                artist_names=[a["name"] for a in track_data["artists"]],
                album_id=track_data["album"]["id"],
                album_name=track_data["album"]["name"],
                duration_ms=track_data["duration_ms"],
                popularity=track_data.get("popularity", 0),
                explicit=track_data.get("explicit", False),
                external_urls=track_data.get("external_urls", {}),
                preview_url=track_data.get("preview_url"),
            )
        )
    return tracks


def get_artist_related_artists(
    client: Any, artist_id: str
) -> list[SpotifyArtist]:
    """Get related artists for a given artist."""
    data = client.artist_related_artists(artist_id)
    artists = []
    for artist_data in data["artists"]:
        artists.append(
            SpotifyArtist(
                id=artist_data["id"],
                name=artist_data["name"],
                genres=artist_data.get("genres", []),
                popularity=artist_data.get("popularity", 0),
                followers=artist_data.get("followers", {}).get("total", 0),
                external_urls=artist_data.get("external_urls", {}),
                images=artist_data.get("images", []),
            )
        )
    return artists