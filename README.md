# Music Recommender System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-4B32C3)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![ALS](https://img.shields.io/badge/recommender-implicit%20ALS-111827)](https://github.com/benfred/implicit)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)](https://docs.pytest.org/)
[![Lint](https://img.shields.io/badge/lint-ruff-D7FF64)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/Mohamed-ahmed-shokry/music-recommender-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed-ahmed-shokry/music-recommender-system/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A production-style artist recommendation system that turns implicit listening
signals and artist metadata into personalized, explainable music
recommendations.

This project uses collaborative filtering with Alternating Least Squares, ALS,
from the `implicit` library, then blends ALS scores with content-based artist
metadata similarity. It includes a reusable Python package, versioned serving
artifacts, cold-start onboarding, session-aware recommendations,
recommendation explanations, ranking controls, baseline comparison, a Typer CLI,
a FastAPI API, an interactive Streamlit dashboard, Docker deployment, continuous
integration, optional MLflow experiment tracking, tests, linting, and a
portfolio-ready architecture.

## Contents

- [Overview](#overview)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [System Architecture](#system-architecture)
- [Recommendation Strategies](#recommendation-strategies)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [CLI Reference](#cli-reference)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Docker Deployment](#docker-deployment)
- [Evaluation](#evaluation)
- [Experiment Tracking](#experiment-tracking)
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
- exposes searchable, filterable artist catalog discovery;
- stores everything needed for serving in a versioned artifact bundle;
- compares ALS, popularity, content-only, and hybrid strategies;
- re-ranks candidates with an optional learning-to-rank model;
- tracks reproducible training and evaluation runs in MLflow;
- recommends tracks with audio-feature similarity from sample track data;
- fetches live artist/track/audio-feature data via the optional Spotify extras;
- exposes CLI, API, and interactive dashboard workflows.

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
| Serving design | `RecommenderService` loads artifacts once for CLI, API, and dashboard use |
| Deployment | Multi-stage, non-root Docker image with a baked model artifact and health check |
| Continuous integration | Locked install, formatting, lint, tests, package build, and container smoke test |
| Interactive product | Streamlit studio for personalized, profile, session, similarity, and catalog exploration |
| Cold start | Unknown users receive popular fallback or profile/session-based recommendations |
| Ranking controls | Optional listened-item inclusion, popularity penalty, diversity reranking |
| Evaluation | ALS, popularity, content, and hybrid metrics with novelty and explanations |
| Learning to rank | Ridge re-ranker over collaborative/popularity/user features, served via CLI, API, and dashboard |
| Quality gates | A/B compare-settings with `--promote-winner` and `--min-quality-threshold` CI gate |
| Track recommendations | Audio-feature similarity with sample track fixtures and CLI workflows |
| External data | Optional Spotify integration (search, top tracks, related artists, audio features) |
| Experiment tracking | Optional MLflow training parameters, evaluation metrics, tags, and artifacts |
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
        |
        +--> Streamlit dashboard
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
| `track_similarity` | Track recommendations and similarity | Ranks tracks by audio-feature cosine similarity |
| `popular_fallback` | Unknown user ID | Returns globally popular artists from training data |
| `popular_baseline` | Evaluation and CLI baseline | Ranks artists by total plays and listener count |


## Project Structure

```text
music-recommender-system/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- api/
|   |-- __init__.py
|   |-- main.py
|   `-- middleware.py
|-- artifacts/
|   |-- mappings/
|   `-- models/
|-- data/
|   |-- raw/
|   |   |-- sample_artist_metadata.csv
|   |   |-- sample_interactions.csv
|   |   |-- sample_track_interactions.csv
|   |   `-- sample_track_metadata.csv
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
|       |-- dashboard.py
|       |-- data.py
|       |-- evaluate.py
|       |-- metadata.py
|       |-- model.py
|       |-- preprocessing.py
|       |-- ranking.py
|       |-- recommend.py
|       |-- service.py
|       |-- spotify.py
|       |-- tracking.py
|       |-- tracks.py
|       `-- utils.py
|-- tests/
|-- .dockerignore
|-- CHANGELOG.md
|-- PLAN.md
|-- compose.yaml
|-- Dockerfile
|-- LICENSE
|-- README.md
|-- streamlit_app.py
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

CPU training:

```bash
uv run python -m music_recommender.cli train --no-use-gpu
```

## CLI Reference

Check the installed release:

```bash
uv run music-recommender --version
```

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

Train with champion reranking settings (served as defaults unless overridden):

```bash
uv run python -m music_recommender.cli train --no-use-gpu --popularity-penalty 0.2 --diversity 0.5 --include-listened
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

Re-rank the recommendations with the bundled learning-to-rank model (trained
during `train` and persisted on the artifact):

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10 --ltr
```

`--ltr` asks the service for the `ltr_personalized` strategy, which re-orders the
ALS candidates with the ridge re-ranker described below. If the artifact has no
bundled ranker it falls back to the standard hybrid list. Champion ranking
settings still apply to the underlying candidates.

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

A/B test reranking settings of the same ALS pipeline on identical holdouts:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 --folds 5 \
  --compare-settings "control:;penalty:popularity_penalty=0.2"
```

Add a learning-to-rank arm that re-ranks the ALS candidates with a lightweight
pointwise model trained on the training fold:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 --folds 5 --learn-to-rank
```

`--learn-to-rank` reports an additional `ltr` strategy alongside `als` (and any
baselines requested with `--compare-baseline`/`--compare-all`). The re-ranker is
a ridge regressor over interpretable per-candidate features — the ALS
collaborative score, the artist's log popularity and normalized popularity rank,
and the user's interaction count — trained on positive interactions paired with
sampled negatives, then used to re-order the served list. It only re-ranks
candidates the collaborative model already surfaced, so serving stays cheap.

Ablate each active knob of a champion ranking configuration to see its
per-metric contribution:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 --folds 5 \
  --ablations "popularity_penalty=0.2,diversity=0.5"
```

`--ablations` evaluates the champion config plus a `no_<knob>` arm for every
active knob and a fully neutral `no_ranking` arm on the same holdout, then
reports each arm's signed per-metric delta versus the champion and a knob
importance ranking by total absolute impact. It is an alternative to
`--compare-settings` (and the other compare/`--promote-winner` flags) and
cannot be combined with them.

The ablation result is also persisted as a JSON report (with the arm metrics,
signed deltas, and importance ranking under a stable schema) so runs over
different datasets can be compared side by side. It is written to `reports/`
(override with `--report-dir`):

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 \
  --ablations "popularity_penalty=0.2,diversity=0.5" \
  --report-dir reports/round-1
```

Aggregate many persisted reports (e.g. one per dataset) to surface which knobs
stay important across runs:

```bash
uv run python -m music_recommender.cli ablation-summary --report-dir reports/
```

`ablation-summary` loads every generated report in the directory and prints,
per knob, the mean and standard deviation of the total impact across runs plus
how many reports contributed, ranked by mean impact. A small standard deviation
relative to the mean flags a stable knob, while a large one suggests the knob's
effect depends on the dataset.

The aggregated result is also written as a JSON report (default
`reports/ablation_summary.json`, override with `--summary-path`) so dashboards
and CI can consume the stable knob-importance summary deterministically:

```bash
uv run python -m music_recommender.cli ablation-summary \
  --report-dir reports/ --summary-path reports/ablation_summary.json
```

Spotify integration (optional):

```bash
uv sync --extra spotify
export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
uv run python -m music_recommender.cli spotify-search-artists "Daft Punk"
uv run python -m music_recommender.cli spotify-artist <spotify_artist_id>
uv run python -m music_recommender.cli spotify-artist-top-tracks <spotify_artist_id>
uv run python -m music_recommender.cli spotify-related-artists <spotify_artist_id>
uv run python -m music_recommender.cli spotify-audio-features --ids <track_id_1>,<track_id_2>
```

Import a real track catalog from Spotify top tracks plus audio features into
the track metadata contract:

```bash
uv run python -m music_recommender.cli spotify-import-catalog \
  --artist-ids <spotify_artist_id_1>,<spotify_artist_id_2> \
  --output data/raw/spotify_track_metadata.csv
```

Track-level recommendations with audio-feature similarity:

```bash
uv run python -m music_recommender.cli prepare-track-data
uv run python -m music_recommender.cli track-recommendations --user-id user_1 --top-k 10
uv run python -m music_recommender.cli similar-tracks --track-id track_1 --top-k 10
```

Evaluate track similarity with repeated per-user holdouts:

```bash
uv run python -m music_recommender.cli evaluate-tracks --top-k 10 --folds 2
```

Track evaluation reports precision, recall, MAP, NDCG, catalog coverage,
average popularity, and novelty over the held-out tracks.

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
| `GET` | `/evaluation/ablation-summary` | Persisted aggregated knob-importance summary |
| `GET` | `/catalog/artists?query=pop&limit=25` | Search and page through artists and metadata |
| `GET` | `/popular-artists?top_k=10` | Popular artist recommendations |
| `GET` | `/recommend/user/{user_id}?top_k=10&content_weight=0.25&explain=true` | Hybrid personalized or fallback recommendations |
| `GET` | `/recommend/user/{user_id}/ltr?top_k=10&diversity=0.2&popularity_penalty=0.1` | LTR re-ranked personalized recommendations |
| `GET` | `/tracks/recommend/{user_id}?top_k=10` | Track recommendations with audio-feature similarity |
| `GET` | `/tracks/similar/{track_id}?top_k=10` | Tracks similar to a selected track by audio features |
| `POST` | `/recommend/profile` | Onboarding recommendations from artists, genres, and moods |
| `POST` | `/recommend/session` | Short-term session recommendations from seeds, exclusions, and optional user taste |
| `GET` | `/similar-artists/{artist_id}?method=hybrid&top_k=10` | ALS, content, or hybrid similar artists |
| `GET` | `/content-similar-artists/{artist_id}?top_k=10` | Metadata-only similar artists |

Example requests:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
curl http://127.0.0.1:8000/evaluation/ablation-summary
curl "http://127.0.0.1:8000/catalog/artists?genre=pop&country=Canada&limit=25"
curl "http://127.0.0.1:8000/popular-artists?top_k=10"
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10&content_weight=0.25&explain=true"
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10&diversity=0.2&popularity_penalty=0.1"
curl "http://127.0.0.1:8000/recommend/user/user_1/ltr?top_k=10&diversity=0.2&popularity_penalty=0.1"
curl "http://127.0.0.1:8000/tracks/recommend/user_1?top_k=10"
curl "http://127.0.0.1:8000/tracks/similar/track_1?top_k=10"
curl "http://127.0.0.1:8000/similar-artists/artist_2?method=hybrid&top_k=10&explain=true"
curl "http://127.0.0.1:8000/content-similar-artists/artist_2?top_k=10&explain=true"
curl -X POST http://127.0.0.1:8000/recommend/profile \
  -H "Content-Type: application/json" \
  -d '{"artist_ids":["artist_1","artist_6"],"genres":["pop","electronic"],"top_k":10,"explain":true}'
curl -X POST http://127.0.0.1:8000/recommend/session \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_1","artist_ids":["artist_1","artist_6"],"genres":["pop","electronic"],"mood_tags":["bright"],"exclude_artist_ids":["artist_2"],"top_k":10,"content_weight":0.35,"explain":true}'
```

Catalog discovery accepts a free-text `query`, exact `genre`, `mood_tag`,
`country`, and `era` filters, plus `offset` and `limit` pagination. API result
counts are capped at 100 items per request. Request bodies are limited to 64
KiB before JSON parsing to bound server resource use. Every API response
includes an `X-Request-ID` correlation header and an `X-Process-Time` header in
seconds; a valid client-supplied `X-Request-ID` is echoed back.

## Dashboard

Install the optional dashboard dependencies:

```bash
uv sync --extra dashboard
```

Train the model if an artifact is not already available, then launch Streamlit:

```bash
uv run python -m music_recommender.cli train --no-use-gpu
uv run --extra dashboard streamlit run streamlit_app.py
```

Open the dashboard at:

```text
http://127.0.0.1:8501
```

The dashboard provides seven workflows:

- personalized hybrid recommendations for known listeners (with optional LTR re-ranking);
- cold-start recommendations from favorite artists, genres, and moods;
- session mixes that blend long-term taste with short-term intent;
- ALS, metadata, and hybrid artist similarity;
- track recommendations, audio-feature track similarity, and track catalog search;
- responsive catalog search over artist metadata and popularity statistics;
- aggregated ablation-importance summary for ranking knob analysis.

The trained `RecommenderService` is cached as a shared Streamlit resource, so
widget reruns do not reload the model artifact. Catalog search uses the same
service contract as the API and bounds rendered results for larger datasets.

## Docker Deployment

Build and run both the API and dashboard with Docker Compose:

```bash
docker compose up --build
```

The images install only their required runtime dependencies, train the bundled
sample dataset on CPU during the build, and run as a non-root user. Compose also
enables read-only root filesystems and temporary `/tmp` mounts.

Verify the running services:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8501/_stcore/health
```

The API is available at `http://127.0.0.1:8000` and the dashboard at
`http://127.0.0.1:8501`.

Stop the service:

```bash
docker compose down
```

To build and run without Compose:

```bash
docker build --tag music-recommender-api .
docker run --rm --publish 8000:8000 music-recommender-api
```

Build and run only the dashboard:

```bash
docker build --target dashboard-runtime --tag music-recommender-dashboard .
docker run --rm --publish 8501:8501 music-recommender-dashboard
```

Version tags that match `pyproject.toml`, for example `v0.6.0`, run the complete
quality gate and publish both images to GitHub Container Registry:

```bash
docker pull ghcr.io/mohamed-ahmed-shokry/music-recommender-api:0.6.0
docker pull ghcr.io/mohamed-ahmed-shokry/music-recommender-dashboard:0.6.0
```

Published images receive full, major/minor, major, and stable `latest` tags.
Each image also includes OCI metadata and a GitHub artifact provenance
attestation. Release actions are pinned to immutable commits and publishing
uses only tag-scoped source plus the repository `GITHUB_TOKEN`.

For non-editable or packaged deployments, set `MUSIC_RECOMMENDER_ROOT` to the
directory that contains the `data/` and `artifacts/` directories. The container
sets this to `/app` automatically.

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

LTR re-ranked response:

```json
{
  "user_id": "user_1",
  "strategy": "ltr_personalized",
  "recommendations": [
    {
      "artist_id": "artist_7",
      "artist_name": "Taylor Swift",
      "score": 0.8921,
      "popularity_rank": 5,
      "score_components": {
        "collaborative_score": 0.4287,
        "content_score": 0.1543,
        "hybrid_score": 0.3357,
        "ltr_score": 0.8921
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
| Unexpectedness@K | Share of recommended artists from the popularity long tail (serendipity proxy) |
| Serendipity@K | Share of relevant top-K recommendations that are also from the popularity long tail |
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
  Unexpectedness@5: 0.3611
  Serendipity@5: 0.2333
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
  Unexpectedness@5: 0.0833
  Serendipity@5: 0.0500
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
  Unexpectedness@5: 0.7222
  Serendipity@5: 0.6667
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
  Unexpectedness@5: 0.5833
  Serendipity@5: 0.4833
  Explanation coverage: 1.0000
  Intra-list diversity: 0.9196
```

The comparison is useful because a recommender can look strong by recommending
only globally popular artists. The baseline makes that tradeoff visible, and the
long-tail unexpectedness metric makes the tradeoff explicit: content-only and
hybrid strategies surface more hard-to-discover artists, while the popularity
baseline almost never does. Serendipity@K adds the relevance dimension, so a
strategy earns credit only when it is surprising and useful at the same time.

### A/B testing reranking settings

The same ALS pipeline can be A/B tested under different reranking settings on
identical holdouts, so tuning `popularity_penalty`, `diversity`, and
`include_listened` happens against a control in one command:

```bash
uv run python -m music_recommender.cli evaluate \
  --top-k 5 --folds 2 --no-use-gpu \
  --compare-settings "control:;penalty:popularity_penalty=0.2;diverse:popularity_penalty=0.2,diversity=0.5"
```

Each semicolon-separated setting is `label:key=value,...`; an empty value list
is the control. Values are parsed as booleans, integers, or floats. The harness
prints one full metric row per label, then a batting-average style summary that
names the winner of each quality metric and ranks the settings by metrics won
(ties broken by NDCG@K):

```text
Winners by metric:
  precision_at_k: penalty
  recall_at_k: control
  ...
Overall: penalty won 6 of 10 metrics.
```

The labeled metrics are written to `evaluation/metrics.json`, and the tracking
run is tagged with the labels being compared.

### Promoting the winner into serving

Reranking knobs are training-time settings, so the winning A/B configuration can
be packaged with the model. Ask the compare command to promote the winner
directly — it retrains with the winning setting's ranking parameters and saves
the new artifact:

```bash
uv run python -m music_recommender.cli evaluate --no-use-gpu \
  --compare-settings "control:;diversity:popularity_penalty=0.2,diversity=0.5" \
  --top-k 10 --promote-winner
```

After ranking the labels by metrics won, the command retrains the model, stores
the winning `popularity_penalty`, `diversity`, and `include_listened` on the
artifact, and prints which setting was promoted. (Promotion can also be done by
hand — see below.) `inspect-artifacts` shows the stored settings, and the serving
service uses them as defaults: `recommend_user`, `recommend_user_als`, and
`recommend_session` fall back to the champion settings whenever a reranking knob
is not explicitly provided, so a fresh deployment immediately serves the
configuration that won the comparison. Callers that pass their own knobs (the
dashboard sliders, for example) still override it. Legacy artifact bundles
without a ranking configuration default to neutral settings (`penalty=0`,
`diversity=0`, no listened items).

To promote manually, train with the champion settings and the artifact records
them as its ranking configuration:

```bash
uv run python -m music_recommender.cli train --no-use-gpu \
  --popularity-penalty 0.2 --diversity 0.5 --include-listened
```

### CI quality gate for auto-promotion

For continuous integration pipelines, you can add a quality gate that only promotes
the winning setting if it meets minimum metric thresholds. This prevents promoting
a setting that wins the A/B test but doesn't meet your quality bar:

```bash
uv run python -m music_recommender.cli evaluate --no-use-gpu \
  --compare-settings "control:;diversity:popularity_penalty=0.2,diversity=0.5" \
  --top-k 10 --promote-winner \
  --min-quality-threshold "ndcg_at_k=0.4,precision_at_k=0.2"
```

The `--min-quality-threshold` option accepts comma-separated `metric=value` pairs.
All specified thresholds must be met by the winning setting for promotion to proceed.
If any threshold is not met, the promotion is skipped with a warning. Available
metrics include `precision_at_k`, `recall_at_k`, `map_at_k`, `ndcg_at_k`,
`catalog_coverage`, `average_popularity`, `novelty_at_k`, `unexpectedness_at_k`,
`serendipity_at_k`, `explanation_coverage`, and `intra_list_diversity`.

This is useful in CI/CD pipelines where you want to automatically promote winning
configurations only when they meet your quality standards.

For a hard gate that fails the pipeline when the winner misses the bar, add
`--fail-on-quality-gate` (exits non-zero instead of skipping promotion with a
warning):

```bash
uv run python -m music_recommender.cli evaluate --no-use-gpu \
  --compare-settings "control:;diversity:popularity_penalty=0.2,diversity=0.5" \
  --top-k 10 --promote-winner \
  --min-quality-threshold "ndcg_at_k=0.4,precision_at_k=0.2" \
  --fail-on-quality-gate
```

The scheduled `quality-gate` workflow (`.github/workflows/quality-gate.yml`)
runs this gated A/B promotion plus a knob ablation every Monday and on demand,
uploading the ablation reports as a workflow artifact for review.

## Experiment Tracking

The optional tracking integration uses `mlflow-skinny` in the project and a
separate MLflow server process. This keeps the recommender's modern data stack
isolated from the full server dependency set.

Install the tracking client:

```bash
uv sync --extra tracking
```

Start a local MLflow server in another terminal:

```bash
uvx --from "mlflow==3.14.0" mlflow server --host 127.0.0.1 --port 5000
```

Track a training run:

```bash
uv run --extra tracking music-recommender train --no-use-gpu --track --tracking-uri http://127.0.0.1:5000 --run-name als-baseline
```

Training records the input and model configuration, dataset dimensions and
density, the actual training device, GPU fallback details, and the versioned
serving artifact.

Track a full strategy comparison:

```bash
uv run --extra tracking music-recommender evaluate --top-k 5 --folds 2 --compare-all --no-use-gpu --track --tracking-uri http://127.0.0.1:5000 --run-name v4-comparison
```

Evaluation records the run configuration, every nested ranking metric, strategy
tags, and the complete metrics document as a JSON artifact. Open
`http://127.0.0.1:5000` to compare runs.

`--tracking-uri` can be omitted when `MLFLOW_TRACKING_URI` is set. Tracking is
opt-in, so the default train and evaluate workflows do not require MLflow.

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
| track bundle | Track interactions, audio-feature matrix, similarity, and lookup (optional; legacy bundles load without it and fall back to CSVs) |

Inspect it with:

```bash
uv run python -m music_recommender.cli artifact-info
```

Model, mapping, and serving-bundle writes use atomic replacement, so an
interrupted retraining run does not overwrite the last healthy file with a
partial artifact. On load, the service verifies the artifact version, mappings,
matrix and factor dimensions, content alignment, statistics, and metadata. An
invalid bundle leaves the API liveness route available while `/health` returns
`503` with an actionable retraining message.

## Data Contract

Input CSV files must include:

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | Original user identifier |
| `artist_id` | string | Original artist identifier |
| `artist_name` | string | Display name for the artist |
| `play_count` | numeric | Positive implicit feedback signal |

Interaction identifiers are loaded as text, so values such as `001` retain
leading zeroes. Boundary whitespace is trimmed and repeated rows for the same
user and artist are combined by summing `play_count` before filtering or
training. Validation rejects empty data, missing required columns, blank IDs or
names, conflicting names for one artist ID, and non-numeric, non-finite, or
non-positive play counts.

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

Use the Makefile for common workflows:

```bash
make install   # uv sync with dashboard + tracking + dev
make ci        # lint + typecheck + tests
make coverage  # run tests with the statement coverage report
make train     # train on CPU
make api       # run FastAPI with reload
make dashboard # run Streamlit
```

Or run the underlying commands directly:

Run tests:

```bash
uv run --extra dashboard --extra tracking --extra spotify pytest --cov
```

Run linting:

```bash
uv run ruff check .
```

Run strict static type checks:

```bash
uv run mypy
```

Format code:

```bash
uv run ruff format .
```

Check formatting:

```bash
uv run ruff format --check .
```

Enable pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) for workflow and disclosure details.

Every push to `main` and every pull request runs the same locked dashboard,
tracking, Spotify, and development install, formatting, lint, test, and
package-build checks in GitHub Actions. CI also builds both Docker targets,
starts each container, and verifies the API and Streamlit health endpoints. A
weekly `quality-gate` workflow re-runs the gated A/B promotion and knob
ablation, uploading the reports as workflow artifacts.

Release tags repeat the quality gate without reusable dependency caches before
publishing the API and dashboard images, then attach registry provenance.

The pytest configuration treats warnings as errors and enforces at least 75%
statement coverage across the application package and API. The current suite
reaches 100% statement coverage; the two `__main__` entry points and one
defensive guard are marked as deliberate exclusions because they only execute
under `python -m` or the Streamlit runtime.

Current coverage focus:

- data validation;
- preprocessing and sparse matrix creation;
- ALS training and persistence;
- recommendation behavior;
- session recommendation behavior;
- content vectorization and content recommendations;
- artifact bundles;
- service-layer behavior;
- artist catalog search, filters, and pagination;
- FastAPI route behavior;
- Streamlit dashboard rendering and interaction;
- MLflow tracking lifecycle and CLI integration;
- ranking controls;
- evaluation metrics;
- track validation, audio-feature similarity, and holdout evaluation;
- Spotify integration and catalog import.

## Model Card

| Section | Details |
| --- | --- |
| Intended use | Learning, portfolio demonstration, small-scale artist recommendation experiments |
| Model type | Hybrid implicit-feedback ALS plus content-based metadata similarity |
| Training signal | Positive play counts and artist metadata |
| Prediction target | Artist- and track-level recommendations |
| Cold start | Unknown users receive popular artists or profile/session-based recommendations |
| Serving | Local artifact bundle loaded by the CLI, API, or Streamlit dashboard |
| Bias controls | Popularity baseline, popularity penalty, catalog coverage, diversity and novelty metrics |
| Explainability | Score components, matched metadata, and human-readable reasons |
| Reproducibility | Versioned artifacts, locked dependencies, deterministic splits, and optional MLflow run tracking |
| Main limitation | Small synthetic sample dataset; no real streaming events yet (Spotify catalog import available) |

## Roadmap

See [PLAN.md](PLAN.md) for the full phased plan.

- Add Spotify API integration. ✓ (module + CLI, optional `spotify` extra)
- Add track-level recommendations. ✓ (CLI + sample data; API serving next)
- Add audio-feature content similarity. ✓ (track content matrix; artist
  audio-feature blending next)
- Expose the learning-to-rank re-ranked recommendations through the API and
  dashboard with a configurable toggle. ✓
- Add a CI quality gate that auto-promotes the winning setting when the A/B run
  passes a minimum quality threshold. ✓
- Render the aggregated ablation summary in the Streamlit dashboard. ✓
- Next: 0.5.0 release (track API, bundled track artifacts, Spotify import
  pipeline, and track evaluation are all served).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
Release notes are maintained in [CHANGELOG.md](CHANGELOG.md).
