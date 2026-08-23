# syntax=docker/dockerfile:1

FROM python:3.14-slim-bookworm AS base

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


FROM builder AS dashboard-builder

COPY streamlit_app.py ./
RUN uv sync --locked --no-dev --no-editable --extra dashboard


FROM base AS runtime-common

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /app app

WORKDIR /app

ENV MUSIC_RECOMMENDER_ROOT=/app \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app


FROM runtime-common AS dashboard-runtime

COPY --from=dashboard-builder --chown=app:app /app/.venv /app/.venv
COPY --from=dashboard-builder --chown=app:app /app/artifacts /app/artifacts
COPY --from=dashboard-builder --chown=app:app \
    /app/streamlit_app.py /app/streamlit_app.py

ENV HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "streamlit_app.py"]


FROM runtime-common AS runtime

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/artifacts /app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
