from pathlib import Path

from music_recommender.config import PROJECT_ROOT_ENV_VAR, resolve_project_root


def test_resolve_project_root_defaults_to_repository(monkeypatch) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV_VAR, raising=False)

    expected_root = Path(__file__).resolve().parents[1]

    assert resolve_project_root() == expected_root


def test_resolve_project_root_accepts_environment_override(
    monkeypatch,
    tmp_path,
) -> None:
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(runtime_root))

    assert resolve_project_root() == runtime_root.resolve()
