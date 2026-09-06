"""Shared test fixtures for the music recommender test suite."""

from __future__ import annotations

import os

# Typer forces Rich terminal rendering when GITHUB_ACTIONS, FORCE_COLOR, or
# PY_COLORS is set (as on CI runners), which changes CLI error output and
# breaks output assertions. Disable it before any app module is imported so
# CLI snapshots render identically on every platform.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

import pandas as pd
import pytest


@pytest.fixture()
def interactions_df() -> pd.DataFrame:
    """Standard interactions dataframe used across test modules."""
    return pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_2", "user_3", "user_3"],
            "artist_id": [
                "artist_1",
                "artist_2",
                "artist_2",
                "artist_3",
                "artist_3",
                "artist_4",
            ],
            "artist_name": ["A", "B", "B", "C", "C", "D"],
            "play_count": [10, 8, 9, 7, 8, 6],
        }
    )


@pytest.fixture()
def metadata_df() -> pd.DataFrame:
    """Standard artist metadata dataframe used across test modules."""
    return pd.DataFrame(
        {
            "artist_id": ["artist_1", "artist_2", "artist_3", "artist_4"],
            "artist_name": ["A", "B", "C", "D"],
            "genres": ["pop", "pop;dance", "rock", "soul"],
            "mood_tags": ["bright", "bright;fun", "raw", "warm"],
            "country": [
                "United States",
                "United States",
                "United Kingdom",
                "Canada",
            ],
            "era": ["2020s", "2020s", "2000s", "2010s"],
        }
    )
