# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS base

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY api ./api
RUN uv sync --locked --no-dev --no-editable

COPY data ./data
RUN mkdir -p artifacts/models artifacts/mappings

ENV MUSIC_RECOMMENDER_ROOT=/app
RUN .venv/bin/python -m music_recommender.cli train --no-use-gpu


FROM base AS runtime

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /app app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/artifacts /app/artifacts

ENV MUSIC_RECOMMENDER_ROOT=/app \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
