.PHONY: install train demo api dashboard test lint format typecheck coverage ci clean

install:
	uv sync --locked --extra dashboard --extra tracking --extra spotify --dev

train:
	uv run python -m music_recommender.cli train --no-use-gpu

demo:
	uv run python -m music_recommender.cli demo

api:
	uv run uvicorn api.main:app --reload

dashboard:
	uv run --extra dashboard streamlit run streamlit_app.py

test:
	uv run --extra dashboard --extra tracking --extra spotify pytest --cov

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

coverage:
	uv run --extra dashboard --extra tracking --extra spotify pytest --cov --cov-report=term

ci: lint typecheck test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist htmlcov
	rm -rf artifacts/*.joblib artifacts/models/* artifacts/mappings/*
