from pathlib import Path

import pytest

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


@pytest.mark.parametrize("blank_value", ["", "   ", "\t", " \n "])
def test_resolve_project_root_ignores_blank_environment(
    monkeypatch, blank_value
) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, blank_value)

    expected_root = Path(__file__).resolve().parents[1]

    assert resolve_project_root() == expected_root


def test_resolve_project_root_strips_surrounding_whitespace(
    monkeypatch,
    tmp_path,
) -> None:
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, f"  {runtime_root}  ")

    assert resolve_project_root() == runtime_root.resolve()


def test_resolve_project_root_expands_home_directories(monkeypatch) -> None:
    home = Path.home()
    monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, "~/runtime-root")

    resolved = resolve_project_root()

    assert resolved == (home / "runtime-root").resolve()


def test_bandit_policy_config_constants() -> None:
    from music_recommender.config import (
        BANDIT_ALPHA_DECAY_ENV_VAR,
        BANDIT_GAMMA_ENV_VAR,
        BANDIT_POLICY_TYPE_ENV_VAR,
        DEFAULT_BANDIT_ALPHA_DECAY,
        DEFAULT_BANDIT_GAMMA,
        DEFAULT_BANDIT_POLICY_TYPE,
    )

    assert BANDIT_POLICY_TYPE_ENV_VAR == "MUSIC_RECOMMENDER_BANDIT_POLICY_TYPE"
    assert BANDIT_ALPHA_DECAY_ENV_VAR == "MUSIC_RECOMMENDER_BANDIT_ALPHA_DECAY"
    assert BANDIT_GAMMA_ENV_VAR == "MUSIC_RECOMMENDER_BANDIT_GAMMA"
    assert DEFAULT_BANDIT_POLICY_TYPE == "linucb"
    assert DEFAULT_BANDIT_ALPHA_DECAY == 0.0
    assert DEFAULT_BANDIT_GAMMA == 1.0
