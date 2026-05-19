# Music Recommender System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-4B32C3)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![ALS](https://img.shields.io/badge/recommender-implicit%20ALS-111827)](https://github.com/benfred/implicit)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)](https://docs.pytest.org/)
[![Lint](https://img.shields.io/badge/lint-ruff-D7FF64)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A production-style artist recommendation system that turns implicit listening
signals and artist metadata into personalized, explainable music
recommendations.

This project uses collaborative filtering with Alternating Least Squares, ALS,
from the `implicit` library, then blends ALS scores with content-based artist
metadata similarity. It includes a reusable Python package, versioned serving
artifacts, cold-start onboarding, session-aware recommendations,
recommendation explanations, ranking controls, baseline comparison, a Typer CLI,
a FastAPI API, tests, linting, and a portfolio-ready architecture.

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
- blends collaborative and content-based scores with a configurable weight;
- recommends artists for new users from favorite artists, genres, or mood tags;
- builds short-term session recommendations from a known user plus seed artists,
  genres, moods, and explicit exclusions;
- explains recommendations with score components and matched metadata;
- serves popular fallback recommendations for unknown users;
- finds similar artists from ALS factors, metadata, or a hybrid of both;
- stores everything needed for serving in a versioned artifact bundle;
- compares ALS, popularity, content-only, and hybrid strategies;
- exposes both CLI and API workflows.

The included dataset is intentionally small so the whole project can run quickly
on a laptop. The same pipeline can be pointed at a larger Last.fm-style dataset
with the same columns.

## What This Project Demonstrates

| Area | Implementation |
| --- | --- |
| Recommendation modeling | ALS collaborative filtering with implicit play-count feedback |
| Content modeling | TF-IDF artist metadata vectors for genre, mood, country, and era |
| Hybrid ranking | Configurable ALS plus content scoring with score explanations |
| Session ranking | Short-term seed artists, genre and mood intent, exclusions, and user taste blending |
| Production structure | `src/` package layout, CLI, API, tests, docs, ignored artifacts |
| Serving design | `RecommenderService` loads artifacts once for CLI/API use |
| Cold start | Unknown users receive popular fallback or profile/session-based recommendations |
| Ranking controls | Optional listened-item inclusion, popularity penalty, diversity reranking |
| Evaluation | ALS, popularity, content, and hybrid metrics with novelty and explanations |
| Reproducibility | `uv`, `pyproject.toml`, `uv.lock`, deterministic sample data |
| Portfolio polish | README, MIT license, clean commands, model card, roadmap |

## System Architecture

```text
data/raw/sample_interactions.csv      data/raw/sample_artist_metadata.csv
        |
        v
Data validation + metadata validation
        |
        v
User and artist filtering
        |
        v
ID mappings + sparse user-item matrix + content vectors
        |
        v
ALS model training with implicit + hybrid scoring inputs
        |
        v
Versioned v4 recommender artifact bundle
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
| `hybrid_personalized` | Known user ID | Blends ALS score and content profile score |
| `content_profile` | New user onboarding | Scores artists from favorite artists, genres, and mood tags |
| `session_hybrid` | Known user plus session seeds | Blends long-term ALS taste with short-term artist, genre, and mood intent |
| `session_content` | Session seeds without a known user | Scores artists from short-term artist, genre, and mood intent |
| `content_similarity` | Metadata artist similarity | Finds artists with similar genres, moods, country, and era |
| `hybrid_similarity` | Hybrid artist similarity | Blends ALS factor similarity and content similarity |
| `als_similarity` | ALS artist similarity | Uses cosine similarity between artist factor vectors |
| `popular_fallback` | Unknown user ID | Returns globally popular artists from training data |
| `popular_baseline` | Evaluation and CLI baseline | Ranks artists by total plays and listener count |


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
|   |   |-- sample_artist_metadata.csv
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
|       |-- content.py
|       |-- data.py
|       |-- evaluate.py
|       |-- metadata.py
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

Validate artist metadata:

```bash
uv run python -m music_recommender.cli prepare-metadata
```

Train with the default sample dataset:

```bash
uv run python -m music_recommender.cli train
```

Train from a specific CSV:

```bash
uv run python -m music_recommender.cli train --data-path data/raw/sample_interactions.csv
```

Train with explicit interaction and metadata files:

```bash
uv run python -m music_recommender.cli train --data-path data/raw/sample_interactions.csv --metadata-path data/raw/sample_artist_metadata.csv
```

Inspect the saved artifact bundle:

```bash
uv run python -m music_recommender.cli artifact-info
```

Recommend artists:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10
```

Recommend with hybrid score explanations:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10 --content-weight 0.25 --explain
```

Recommend with reranking controls:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10 --content-weight 0.25 --diversity 0.2 --popularity-penalty 0.1
```

Include artists the user already listened to:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --include-listened
```

Show popular artists:

```bash
uv run python -m music_recommender.cli popular-artists --top-k 10
```

Recommend from onboarding preferences:

```bash
uv run python -m music_recommender.cli recommend-profile --artist-ids artist_1,artist_6 --genres pop,electronic --top-k 10 --explain
```

Build a short-term session mix:

```bash
uv run python -m music_recommender.cli recommend-session --user-id user_1 --artist-ids artist_1,artist_6 --genres pop,electronic --mood-tags bright,dancefloor --exclude-artist-ids artist_2 --top-k 10 --content-weight 0.35 --explain
```

Find similar artists with ALS, content, or hybrid similarity:

```bash
uv run python -m music_recommender.cli similar-artists --artist-id artist_2 --method hybrid --top-k 10 --explain
```

Find metadata-similar artists:

```bash
uv run python -m music_recommender.cli content-similar-artists --artist-id artist_2 --top-k 10 --explain
```

Evaluate ALS:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10
```

Compare ALS with a popularity baseline:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 --folds 5 --compare-baseline
```

Compare ALS, popularity, content-only, and hybrid strategies:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 --folds 5 --compare-all --no-use-gpu
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
| `GET` | `/recommend/user/{user_id}?top_k=10&content_weight=0.25&explain=true` | Hybrid personalized or fallback recommendations |
| `POST` | `/recommend/profile` | Onboarding recommendations from artists, genres, and moods |
| `POST` | `/recommend/session` | Short-term session recommendations from seeds, exclusions, and optional user taste |
| `GET` | `/similar-artists/{artist_id}?method=hybrid&top_k=10` | ALS, content, or hybrid similar artists |
| `GET` | `/content-similar-artists/{artist_id}?top_k=10` | Metadata-only similar artists |

Example requests:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
curl "http://127.0.0.1:8000/popular-artists?top_k=10"
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10&content_weight=0.25&explain=true"
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10&diversity=0.2&popularity_penalty=0.1"
curl "http://127.0.0.1:8000/similar-artists/artist_2?method=hybrid&top_k=10&explain=true"
curl "http://127.0.0.1:8000/content-similar-artists/artist_2?top_k=10&explain=true"
curl -X POST http://127.0.0.1:8000/recommend/profile \
  -H "Content-Type: application/json" \
  -d '{"artist_ids":["artist_1","artist_6"],"genres":["pop","electronic"],"top_k":10,"explain":true}'
curl -X POST http://127.0.0.1:8000/recommend/session \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_1","artist_ids":["artist_1","artist_6"],"genres":["pop","electronic"],"mood_tags":["bright"],"exclude_artist_ids":["artist_2"],"top_k":10,"content_weight":0.35,"explain":true}'
```

Known-user response:

```json
{
  "user_id": "user_1",
  "strategy": "hybrid_personalized",
  "content_weight": 0.25,
  "recommendations": [
    {
      "artist_id": "artist_7",
      "artist_name": "Taylor Swift",
      "score": 0.3357,
      "popularity_rank": 5,
      "score_components": {
        "collaborative_score": 0.4287,
        "content_score": 0.1543,
        "hybrid_score": 0.3357
      },
      "matched_metadata": {
        "genres": ["pop", "singer-songwriter"],
        "mood_tags": ["bright", "romantic", "anthemic"]
      },
      "reasons": [
        "Shares 2010s, pop with The Weeknd"
      ]
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

Session response:

```json
{
  "user_id": "user_1",
  "strategy": "session_hybrid",
  "content_weight": 0.35,
  "seed_artist_ids": ["artist_1", "artist_6"],
  "genres": ["pop", "electronic"],
  "mood_tags": ["bright"],
  "excluded_artist_ids": ["artist_1", "artist_2", "artist_6"],
  "recommendations": [
    {
      "artist_id": "artist_7",
      "artist_name": "Taylor Swift",
      "score": 0.4421,
      "score_components": {
        "collaborative_score": 0.4287,
        "session_content_score": 0.6242,
        "hybrid_score": 0.4421
      },
      "reasons": [
        "Matches your selected preferences: bright, pop"
      ]
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
| Novelty@K | Average inverse popularity rank of recommended artists |
| Explanation coverage | Share of recommendations with non-empty reasons |
| Intra-list diversity | Average dissimilarity within each recommendation list |

Run the full v4 comparison:

```bash
uv run python -m music_recommender.cli evaluate --top-k 5 --folds 2 --compare-all --no-use-gpu
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
  Novelty@5: 0.4520
  Explanation coverage: 0.0000
  Intra-list diversity: 0.5004
Popularity:
  Precision@5: 0.1500
  Recall@5: 0.3750
  MAP@5: 0.2295
  NDCG@5: 0.3089
  Catalog coverage: 0.5833
  Average popularity: 110.9250
  Novelty@5: 0.2387
  Explanation coverage: 0.0000
  Intra-list diversity: 0.4357
Content:
  Precision@5: 0.3000
  Recall@5: 0.7500
  MAP@5: 0.6236
  NDCG@5: 0.7000
  Catalog coverage: 0.9722
  Average popularity: 86.1667
  Novelty@5: 0.5265
  Explanation coverage: 1.0000
  Intra-list diversity: 0.9009
Hybrid:
  Precision@5: 0.2417
  Recall@5: 0.6042
  MAP@5: 0.3726
  NDCG@5: 0.4651
  Catalog coverage: 0.9444
  Average popularity: 91.9250
  Novelty@5: 0.4647
  Explanation coverage: 1.0000
  Intra-list diversity: 0.9196
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
| content artifacts | Metadata dataframe, TF-IDF vectorizer, content matrix, feature names |
| metadata | Created time, training device, dataset fingerprints, dimensions |
| training config | ALS factors, regularization, iterations, alpha, GPU flag |
| hybrid config | Default content weight and serving-time hybrid settings |

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

Artist metadata CSV files must include:

| Column | Type | Description |
| --- | --- | --- |
| `artist_id` | string | Must match interaction artist IDs |
| `artist_name` | string | Display name |
| `genres` | string | Semicolon-separated genre labels |
| `mood_tags` | string | Semicolon-separated mood or style tags |
| `country` | string | Artist country or market |
| `era` | string | Main listening or release era |

Metadata validation rejects missing columns, duplicate artist IDs, empty genre or
mood fields, and interaction artists that are not covered by metadata.

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
- session recommendation behavior;
- content vectorization and content recommendations;
- artifact bundles;
- service-layer behavior;
- FastAPI route behavior;
- ranking controls;
- evaluation metrics.

## Model Card

| Section | Details |
| --- | --- |
| Intended use | Learning, portfolio demonstration, small-scale artist recommendation experiments |
| Model type | Hybrid implicit-feedback ALS plus content-based metadata similarity |
| Training signal | Positive play counts and artist metadata |
| Prediction target | Artist-level recommendations |
| Cold start | Unknown users receive popular artists or profile/session-based recommendations |
| Serving | Local artifact bundle loaded by `RecommenderService` |
| Bias controls | Popularity baseline, popularity penalty, catalog coverage, diversity and novelty metrics |
| Explainability | Score components, matched metadata, and human-readable reasons |
| Main limitation | Small synthetic sample dataset; no real streaming events or external catalog integration yet |

## Roadmap

- Add Spotify API integration.
- Add track-level recommendations.
- Add audio-feature content similarity.
- Add a learning-to-rank model after candidate generation.
- Add advanced serendipity metrics.
- Add a Streamlit dashboard.
- Deploy the API with Docker.
- Add MLflow experiment tracking.
- Add GitHub Actions CI.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
