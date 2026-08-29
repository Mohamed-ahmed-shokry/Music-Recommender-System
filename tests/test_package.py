import importlib
import importlib.metadata

import music_recommender


def test_version_falls_back_when_distribution_metadata_is_missing(
    monkeypatch,
) -> None:
    def raise_not_found(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError("distribution missing")

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)

    reloaded = importlib.reload(music_recommender)

    assert reloaded.__version__ == "0.0.0"
