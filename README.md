# Music Recommender System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-4B32C3)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![ALS](https://img.shields.io/badge/recommender-implicit%20ALS-111827)](https://github.com/benfred/implicit)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)](https://docs.pytest.org/)
[![Lint](https://img.shields.io/badge/lint-ruff-D7FF64)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A production-style artist recommendation system that turns implicit listening
signals into personalized music recommendations.

This project uses collaborative filtering with Alternating Least Squares, ALS,
from the `implicit` library. It includes a reusable Python package, versioned
serving artifacts, cold-start handling, ranking controls, baseline comparison,
a Typer CLI, a FastAPI API, tests, linting, and a portfolio-ready architecture.

## Contents

- [Overview](#overview)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [System Architecture](#system-architecture)
- [Recommendation Strategies](#recommendation-strategies)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [CLI Reference](#cli-reference)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Artifact Bundle](#artifact-bundle)
- [Model Card](#model-card)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

Music platforms usually do not wait for users to rate songs. They infer taste
from behavior: plays, repeats, skips, saves, follows, and listening frequency.
This project models that idea with artist-level play counts.

The current system:

- trains an ALS model from implicit feedback;
- recommends unseen artists for known users;
- serves popular fallback recommendations for unknown users;
- finds similar artists from learned latent factors;
- stores everything needed for serving in a versioned artifact bundle;
- compares ALS against a popularity baseline;
- exposes both CLI and API workflows.

The included dataset is intentionally small so the whole project can run quickly
on a laptop. The same pipeline can be pointed at a larger Last.fm-style dataset
with the same columns.

## What This Project Demonstrates

| Area | Implementation |
| --- | --- |
| Recommendation modeling | ALS collaborative filtering with implicit play-count feedback |
| Production structure | `src/` package layout, CLI, API, tests, docs, ignored artifacts |
| Serving design | `RecommenderService` loads artifacts once for CLI/API use |
| Cold start | Unknown users receive popular artist fallback recommendations |
| Ranking controls | Optional listened-item inclusion, popularity penalty, diversity reranking |
| Evaluation | ALS vs popularity baseline with ranking and catalog metrics |
| Reproducibility | `uv`, `pyproject.toml`, `uv.lock`, deterministic sample data |
| Portfolio polish | README, MIT license, clean commands, model card, roadmap |

## System Architecture

```text
data/raw/sample_interactions.csv
        |
        v
Data validation
        |
        v
User and artist filtering
        |
        v
ID mappings + sparse user-item matrix
        |
        v
ALS model training with implicit
        |
        v
Versioned recommender artifact bundle
        |
        v
RecommenderService
        |
        +--> Typer CLI
        |
        +--> FastAPI API
```

The API uses `RecommenderService`, so model artifacts are loaded once instead of
rebuilding matrices or reloading raw CSV data for every request.

## Recommendation Strategies

| Strategy | When it is used | Behavior |
| --- | --- | --- |
| `als_personalized` | Known user ID | Scores artists using ALS user and artist factors |
| `popular_fallback` | Unknown user ID | Returns globally popular artists from training data |
| `popular_baseline` | Evaluation and CLI baseline | Ranks artists by total plays and listener count |
| Similar artists | Known artist ID | Uses cosine similarity between artist factor vectors |

Ranking controls are optional and beginner-friendly by default:

- listened artists are excluded by default;
- popularity penalty is `0.0` by default;
- diversity reranking is `0.0` by default.

## Project Structure

```text
music-recommender-system/
|-- api/
|   |-- __init__.py
|   `-- main.py
|-- artifacts/
|   |-- mappings/
|   `-- models/
|-- data/
|   |-- raw/
|   |   `-- sample_interactions.csv
|   |-- processed/
|   `-- README.md
|-- notebooks/
|   `-- 01_exploration.ipynb
|-- src/
|   `-- music_recommender/
|       |-- artifacts.py
|       |-- baselines.py
|       |-- cli.py
|       |-- config.py
|       |-- data.py
|       |-- evaluate.py
|       |-- model.py
|       |-- preprocessing.py
|       |-- ranking.py
|       |-- recommend.py
|       |-- service.py
|       `-- utils.py
|-- tests/
|-- LICENSE
|-- README.md
|-- pyproject.toml
`-- uv.lock
```

Generated model artifacts are ignored by Git.

## Quickstart

Clone and install with `uv`:

```bash
git clone https://github.com/Mohamed-ahmed-shokry/music-recommender-system.git
cd music-recommender-system
uv sync
```

Train the model:

```bash
uv run python -m music_recommender.cli train
```

Run the demo:

```bash
uv run python -m music_recommender.cli demo
```

Start the API:

```bash
uv run uvicorn api.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## GPU Support

Optional GPU dependencies can be installed with:

```bash
uv sync --extra gpu
```

GPU training also requires compatible NVIDIA CUDA runtime libraries. If the GPU
path is unavailable, training falls back to CPU and records the reason in the
artifact metadata.

CPU-only training:

```bash
uv run python -m music_recommender.cli train --no-use-gpu
```

## CLI Reference

Prepare data:

```bash
uv run python -m music_recommender.cli prepare-data
```

Train with the default sample dataset:

```bash
uv run python -m music_recommender.cli train
```

Train from a specific CSV:

```bash
uv run python -m music_recommender.cli train --data-path data/raw/sample_interactions.csv
```

Inspect the saved artifact bundle:

```bash
uv run python -m music_recommender.cli artifact-info
```

Recommend artists:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10
```

Recommend with reranking controls:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10 --diversity 0.2 --popularity-penalty 0.1
```

Include artists the user already listened to:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --include-listened
```

Show popular artists:

```bash
uv run python -m music_recommender.cli popular-artists --top-k 10
```

Find similar artists:

```bash
uv run python -m music_recommender.cli similar-artists --artist-id artist_2 --top-k 10
```

Evaluate ALS:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10
```

Compare ALS with a popularity baseline:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 --folds 5 --compare-baseline
```

## API Reference

Train before starting the API:

```bash
uv run python -m music_recommender.cli train
```

Run FastAPI:

```bash
uv run uvicorn api.main:app --reload
```

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic API message |
| `GET` | `/health` | Artifact and service health |
| `GET` | `/metadata` | Training config, dataset fingerprint, artifact metadata |
| `GET` | `/popular-artists?top_k=10` | Popular artist recommendations |
| `GET` | `/recommend/user/{user_id}?top_k=10` | Personalized or fallback recommendations |
| `GET` | `/similar-artists/{artist_id}?top_k=10` | Similar artists |

Example requests:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
curl "http://127.0.0.1:8000/popular-artists?top_k=10"
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10"
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10&diversity=0.2&popularity_penalty=0.1"
curl "http://127.0.0.1:8000/similar-artists/artist_2?top_k=10"
```

Known-user response:

```json
{
  "user_id": "user_1",
  "strategy": "als_personalized",
  "recommendations": [
    {
      "artist_id": "artist_10",
      "artist_name": "Coldplay",
      "score": 0.4845,
      "popularity_rank": 6
    }
  ]
}
```

Unknown-user response:

```json
{
  "user_id": "new_user",
  "strategy": "popular_fallback",
  "message": "Unknown user_id 'new_user'. Returning popular artists.",
  "recommendations": [
    {
      "artist_id": "artist_2",
      "artist_name": "Drake",
      "score": 174.0,
      "popularity_rank": 1
    }
  ]
}
```

## Evaluation

The project reports ranking quality, catalog behavior, and popularity bias:

| Metric | Meaning |
| --- | --- |
| `Precision@K` | Share of recommended artists that are relevant |
| `Recall@K` | Share of relevant artists recovered by recommendations |
| `MAP@K` | Ranking-sensitive precision across users |
| `NDCG@K` | Ranking quality with higher weight for top positions |
| Catalog coverage | Share of the artist catalog recommended at least once |
| Average popularity | Average total plays of recommended artists |
| Intra-list diversity | Average dissimilarity within each recommendation list |

Run a baseline comparison:

```bash
uv run python -m music_recommender.cli evaluate --top-k 5 --folds 2 --compare-baseline --no-use-gpu
```

Example output:

```text
Evaluation over 2 fold(s):
ALS:
  Precision@5: 0.2167
  Recall@5: 0.5417
  MAP@5: 0.3132
  NDCG@5: 0.4013
  Catalog coverage: 0.9444
  Average popularity: 93.2417
  Intra-list diversity: 0.5004
Popularity:
  Precision@5: 0.1500
  Recall@5: 0.3750
  MAP@5: 0.2295
  NDCG@5: 0.3089
  Catalog coverage: 0.5833
  Average popularity: 110.9250
  Intra-list diversity: 0.4357
```

The comparison is useful because a recommender can look strong by recommending
only globally popular artists. The baseline makes that tradeoff visible.

## Artifact Bundle

Training creates a versioned artifact bundle at:

```text
artifacts/recommender_artifact.joblib
```

The bundle contains:

| Field | Purpose |
| --- | --- |
| model | Trained ALS model |
| mappings | User and artist ID mappings |
| user-item matrix | Filtered sparse interaction matrix for serving |
| artist stats | Total plays, listener count, interaction count, popularity rank |
| metadata | Created time, training device, dataset fingerprint, dimensions |
| training config | ALS factors, regularization, iterations, alpha, GPU flag |

Inspect it with:

```bash
uv run python -m music_recommender.cli artifact-info
```

## Data Contract

Input CSV files must include:

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | Original user identifier |
| `artist_id` | string | Original artist identifier |
| `artist_name` | string | Display name for the artist |
| `play_count` | numeric | Positive implicit feedback signal |

Validation rejects empty data, missing required columns, missing IDs or names,
non-numeric play counts, and non-positive play counts.

## Development

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Format code:

```bash
uv run ruff format .
```

Check formatting:

```bash
uv run ruff format --check .
```

Current coverage focus:

- data validation;
- preprocessing and sparse matrix creation;
- ALS training and persistence;
- recommendation behavior;
- artifact bundles;
- service-layer behavior;
- ranking controls;
- evaluation metrics.

## Model Card

| Section | Details |
| --- | --- |
| Intended use | Learning, portfolio demonstration, small-scale artist recommendation experiments |
| Model type | Implicit-feedback ALS collaborative filtering |
| Training signal | Positive play counts |
| Prediction target | Artist-level recommendations |
| Cold start | Unknown users receive popular artists; unknown artists cannot get similarity results |
| Serving | Local artifact bundle loaded by `RecommenderService` |
| Bias controls | Popularity baseline, popularity penalty, catalog coverage, diversity metric |
| Main limitation | Small synthetic sample dataset and no content metadata yet |

## Roadmap

- Add Spotify API integration.
- Add track-level recommendations.
- Add genre and audio-feature content similarity.
- Build a hybrid ALS plus content-based recommender.
- Add a learning-to-rank model after candidate generation.
- Improve cold-start with onboarding preferences.
- Add novelty and serendipity metrics.
- Add a Streamlit dashboard.
- Deploy the API with Docker.
- Add MLflow experiment tracking.
- Add GitHub Actions CI.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
