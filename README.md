# Music Recommender System

A clean, beginner-friendly music artist recommendation system built with Python,
`uv`, and ALS collaborative filtering from the production-ready `implicit`
library.

The project recommends artists from implicit feedback data such as play counts.
It includes a reusable Python package, sample dataset, command line interface,
FastAPI API, tests, linting, and documentation suitable for a GitHub portfolio.

## Project Idea

Music platforms often infer taste from behavior instead of ratings. This project
uses artist play counts as implicit feedback, builds a sparse user-artist matrix,
and trains an Alternating Least Squares model to recommend artists a user has not
listened to yet.

The first version works at the artist level, not the track level.

## Features

- Load and validate user-artist play count data.
- Filter sparse users and artists.
- Build a SciPy sparse user-item matrix.
- Train an ALS collaborative filtering model with `implicit`.
- Prefer GPU training when CUDA support is available, with CPU fallback.
- Recommend artists for a selected user.
- Find artists similar to a selected artist.
- Evaluate recommendations with ranking metrics.
- Use a Typer CLI for local workflows.
- Serve recommendations with a FastAPI API.
- Test core behavior with pytest.
- Format and lint with Ruff.

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

1. The system loads `data/raw/sample_interactions.csv`.
2. It validates required columns: `user_id`, `artist_id`, `artist_name`, and
   `play_count`.
3. It filters users and artists with too few interactions.
4. It creates integer ID mappings for users and artists.
5. It builds a sparse user-artist matrix with play counts as values.
6. It trains ALS using the item-user matrix expected by `implicit`.
7. It recommends unseen artists by scoring artist and user factor vectors.
8. It evaluates recommendation quality with ranking metrics.

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
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── preprocessing.py
│       ├── recommend.py
│       └── utils.py
├── tests/
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

## Installation

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/Mohamed-ahmed-shokry/music-recommender-system.git
cd music-recommender-system
uv sync
```

Optional GPU dependencies for `implicit` can be installed with:

```bash
uv sync --extra gpu
```

GPU training also requires compatible NVIDIA CUDA runtime libraries. If GPU
support is unavailable, the project falls back to CPU training and prints the
fallback reason.

## CLI Usage

Prepare data:

```bash
uv run python -m music_recommender.cli prepare-data
```

Train the model:

```bash
uv run python -m music_recommender.cli train
```

Train without GPU fallback attempts:

```bash
uv run python -m music_recommender.cli train --no-use-gpu
```

Recommend artists for a user:

```bash
uv run python -m music_recommender.cli recommend-user --user-id user_1 --top-k 10
```

Find similar artists:

```bash
uv run python -m music_recommender.cli similar-artists --artist-id artist_2 --top-k 10
```

Evaluate the model:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10
```

Run the demo:

```bash
uv run python -m music_recommender.cli demo
```

## API Usage

Train the model first:

```bash
uv run python -m music_recommender.cli train
```

Start the API:

```bash
uv run uvicorn api.main:app --reload
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/
```

Recommend artists:

```bash
curl "http://127.0.0.1:8000/recommend/user/user_1?top_k=10"
```

Find similar artists:

```bash
curl "http://127.0.0.1:8000/similar-artists/artist_2?top_k=10"
```

## Evaluation Metrics

The evaluation module uses a per-user train/test split and reports:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K

Run:

```bash
uv run python -m music_recommender.cli evaluate --top-k 10
```

## Example Output

```text
Recommendations for user_1:
1. Taylor Swift (artist_7) - score: 1.0156
2. Billie Eilish (artist_9) - score: 0.9896
3. Ariana Grande (artist_8) - score: 0.9565

Artists similar to artist_2:
1. SZA (artist_17) - score: 0.9925
2. The Weeknd (artist_1) - score: 0.9814
3. J. Cole (artist_5) - score: 0.9773
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

## Future Improvements

- Add Spotify API integration.
- Add track-level recommendations.
- Add content-based recommendation using genre and audio features.
- Build a hybrid recommender combining ALS and content similarity.
- Add a ranking model after candidate generation.
- Add cold-start handling for new users and new artists.
- Add popularity bias control.
- Add recommendation diversity and novelty metrics.
- Add a Streamlit dashboard.
- Deploy the API using Docker.
- Add MLflow experiment tracking.

## Limitations

- The sample dataset is intentionally small for demos and tests.
- The first version recommends artists, not individual tracks.
- New users and new artists need additional cold-start logic.
- GPU training depends on local CUDA and `implicit` GPU support.
- The API loads artifacts from local disk and is not optimized for high traffic.

## Learning Goals

- Understand implicit-feedback collaborative filtering.
- Learn how ALS factorizes sparse interaction data.
- Practice clean Python package structure with a `src` layout.
- Use `uv` as the dependency and execution workflow.
- Build CLI, API, tests, linting, and documentation around an ML project.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
