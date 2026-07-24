from pathlib import Path
from typing import Any

import joblib
import pytest

import music_recommender.utils as utils
from music_recommender.utils import atomic_joblib_dump


def test_atomic_joblib_dump_replaces_target_without_temporary_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.joblib"
    joblib.dump({"version": "old"}, target)

    atomic_joblib_dump({"version": "new"}, target)

    assert joblib.load(target) == {"version": "new"}
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_joblib_dump_preserves_target_when_serialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.joblib"
    original_content = b"healthy artifact"
    target.write_bytes(original_content)

    def fail_dump(_: Any, path: str | Path) -> None:
        Path(path).write_bytes(b"partial artifact")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(utils.joblib, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        atomic_joblib_dump({"version": "new"}, target)

    assert target.read_bytes() == original_content
    assert list(tmp_path.iterdir()) == [target]
