# Contributing

Thanks for considering a contribution!

## Quick start

```bash
uv sync --locked --extra dashboard --extra tracking --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run --extra dashboard --extra tracking pytest --cov
```

Or use the Makefile:

```bash
make install
make ci
```

## Workflow

1. Fork and create a feature branch.
2. Keep changes small and focused; one logical change per commit.
3. Run `make ci` locally before opening a PR.
4. Train and smoke-test artifacts when touching data/model code:

```bash
uv run python -m music_recommender.cli train --no-use-gpu
uv run python -m music_recommender.cli artifact-info
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 5 --explain
```

5. Update `CHANGELOG.md` under `Unreleased` and add tests for new behavior.

## Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Commit style

- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `style:`, `test:`, `refactor:`.
- Prefer imperative, lower-case subjects without trailing period.
