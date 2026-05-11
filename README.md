# Music Recommender System

A production-style, artist-level music recommendation project built with Python,
`uv`, FastAPI, and Alternating Least Squares collaborative filtering from the
`implicit` library.

The project turns implicit listening behavior, such as play counts, into artist
recommendations. It includes a reusable package, versioned model artifact bundle,
cold-start fallback strategy, configurable ranking controls, a Typer CLI, a
FastAPI API, evaluation against a popularity baseline, tests, Ruff linting, and a
clean portfolio-ready structure.

## Why This Project Matters

Real music recommendation systems rarely depend on explicit ratings. They infer
preference from behavior: plays, saves, skips, repeats, and follows. This project
models that idea with collaborative filtering and keeps the implementation small
enough to study while still including the engineering pieces expected in a real
ML service:

- repeatable training;
- saved serving artifacts;
- fast API startup and request handling;
- cold-start behavior;
- baseline comparison;
- ranking quality metrics;
- clear CLI workflows.

The current version recommends artists, not individual tracks.

## Features

- Validate user-artist interaction data.
- Build a sparse user-item matrix from play counts.
- Train ALS with the production-ready `implicit` package.
- Prefer GPU training when supported, with graceful CPU fallback.
- Save a versioned artifact bundle for serving.
- Recommend artists for known users.
- Return popular fallback recommendations for unknown users.
- Find artists similar to a selected artist.
- Apply optional popularity penalty and diversity reranking.
- Compare ALS against a popularity baseline.
- Serve recommendations with FastAPI.
- Run all workflows through `uv`.

## Tech Stack

- Python 3.11
- uv
- pandas
- numpy
- scipy
- implicit
- scikit-learn
- typer
- joblib
- fastapi
- uvicorn
- pytest
- ruff

## How It Works

1. Load `data/raw/sample_interactions.csv`.
2. Validate `user_id`, `artist_id`, `artist_name`, and `play_count`.
3. Filter users and artists with too few interactions.
4. Build stable user and artist ID mappings.
5. Build a sparse user-artist matrix.
6. Train ALS on the item-user matrix expected by `implicit`.
7. Save a versioned artifact bundle containing:
   - trained model;
   - ID mappings;
   - filtered user-item matrix;
   - artist popularity stats;
   - training config;
   - dataset fingerprint;
   - training timestamp.
8. Load the artifact once through `RecommenderService`.
9. Serve personalized, similar-artist, and popularity fallback recommendations.
10. Evaluate ALS against a popularity baseline.

## Architecture

```text
Raw CSV
  -> validation
  -> filtering
  -> ID mappings
  -> sparse matrix
  -> ALS training
  -> artifact bundle
  -> RecommenderService
  -> CLI / FastAPI
```

`RecommenderService` is the main serving layer. The API uses it so requests do
not rebuild matrices or reload raw data every time.

## Project Structure

```text
music-recommender-system/
├── api/
│   ├── __init__.py
│   └── main.py
├── artifacts/
│   ├── mappings/
│   └── models/
├── data/
│   ├── raw/
│   │   └── sample_interactions.csv
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   └── music_recommender/
│       ├── artifacts.py
│       ├── baselines.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── preprocessing.py
│       ├── ranking.py
│       ├── recommend.py
│       ├── service.py
│       └── utils.py
├── tests/
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

Generated model artifacts are ignored by Git.

## Installation

```bash
git clone https://github.com/Mohamed-ahmed-shokry/music-recommender-system.git
cd music-recommender-system
uv sync
```

Optional GPU dependencies:

```bash
uv sync --extra gpu
```

GPU training still requires compatible NVIDIA CUDA runtime libraries. If GPU
training is unavailable, the project falls back to CPU training and prints the
reason.

## CLI Usage

Prepare data:

```bash
uv run python -m music_recommender.cli prepare-data
```

Train the model:

```bash
uv run python -m music_recommender.cli train
```

Train from a specific data path:

```bash
uv run python -m music_recommender.cli train --data-path data/raw/sample_interactions.csv
```

Train on CPU only:

```bash
uv run python -m music_recommender.cli train --no-use-gpu
```

Inspect saved artifacts:

```bash
uv run python -m music_recommender.cli artifact-info
```

Recommend artists for a user:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10
```

Recommend with ranking controls:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10 --diversity 0.2 --popularity-penalty 0.1
```

Include artists the user already listened to:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --include-listened
```

Show popular fallback recommendations:

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

Compare ALS with a popularity baseline over repeated holdout folds:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10 --folds 5 --compare-baseline
```

Run a full demo:

```bash
uv run python -m music_recommender.cli demo
```

## API Usage

Train first:

```bash
uv run python -m music_recommender.cli train
```

Start the API:

```bash
uv run uvicorn api.main:app --reload
```

Base endpoint:

```bash
curl http://127.0.0.1:8000/
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Metadata:

```bash
curl http://127.0.0.1:8000/metadata
```

Recommend artists:

```bash
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10"
```

Recommend with ranking controls:

```bash
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10&diversity=0.2&popularity_penalty=0.1"
```

Popular artists:

```bash
curl "http://127.0.0.1:8000/popular-artists?top_k=10"
```

Similar artists:

```bash
curl "http://127.0.0.1:8000/similar-artists/artist_2?top_k=10"
```

## Response Strategies

Known users use ALS:

```json
{
  "user_id": "user_1",
  "strategy": "als_personalized",
  "recommendations": [
    {
      "artist_id": "artist_15",
      "artist_name": "Calvin Harris",
      "score": 0.86,
      "popularity_rank": 9
    }
  ]
}
```

Unknown users receive a popularity fallback:

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

## Evaluation Metrics

The project reports:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- catalog coverage
- average recommendation popularity
- intra-list diversity

The popularity baseline is useful because many recommenders look strong only
because they recommend globally popular items. Comparing ALS to that baseline
makes the evaluation more honest.

Example:

```bash
uv run python -m music_recommender.cli evaluate --top-k 5 --folds 2 --compare-baseline --no-use-gpu
```

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

## Tests and Linting

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

## Model Card

Intended use:
This project is intended for learning, portfolio demonstration, and small-scale
artist recommendation experiments.

Training data:
The included dataset is a synthetic sample with overlapping listening patterns
across hip-hop, pop, rock, electronic, and mixed-taste users. Users can replace
it with a larger Last.fm-style dataset that uses the same columns.

Model:
The recommender uses implicit-feedback ALS. It learns user and artist factors
from play counts and recommends artists with high predicted affinity.

Cold start:
Unknown users receive popular artists. Unknown artists cannot receive similarity
results because they do not have trained factor vectors.

Known limitations:
The sample dataset is small, artist metadata is minimal, and the system does not
yet include real-time updates, track-level modeling, or content-based features.

Fairness and bias:
Popularity can dominate recommendation systems. This project includes a
popularity baseline, popularity penalty, catalog coverage metric, and diversity
metric to make that tradeoff visible.

## Future Improvements

- Add Spotify API integration.
- Add track-level recommendations.
- Add genre and audio-feature content similarity.
- Build a hybrid ALS plus content-based recommender.
- Add a ranking model after candidate generation.
- Improve cold-start with onboarding preferences.
- Add novelty and serendipity metrics.
- Add a Streamlit dashboard.
- Deploy the API with Docker.
- Add MLflow experiment tracking.
- Add GitHub Actions CI.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
