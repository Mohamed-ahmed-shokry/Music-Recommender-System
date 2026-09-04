# Project Plan — Music Recommender System

This plan tracks completed phases, the current phase, and next steps.
It is updated incrementally as phases land.

## Completed

- Phase 1 — Core ALS + hybrid artist recommendations, artifact bundle, CLI.
- Phase 2 — FastAPI service, Streamlit dashboard, Docker, CI.
- Phase 3 — Ranking controls (penalty/diversity), evaluation + baselines.
- Phase 4 — LTR re-ranker in training/eval (`ltr_personalized`, `--ltr`).
- Phase 5 — A/B compare-settings, promote-winner, ablation reports + summary.
- Phase 6 — LTR via API (`GET /recommend/user/{user_id}/ltr`) + dashboard
  toggle, ablation-summary dashboard tab.
- Phase 7 — CI quality gate (`--min-quality-threshold`) for auto-promotion.
- Phase 8 — Spotify integration module + CLI (search, artist, top tracks,
  related, audio features) with optional `spotify` extra.
- Phase 9 — Track-level recommendations with audio-feature similarity,
  sample track CSVs, `prepare-track-data`, `track-recommendations`,
  `similar-tracks`.

## In progress

- Phase 10 — Docs refresh (README, CHANGELOG, data README, PLAN).
- Phase 11 — Coverage for new modules (tracks, spotify) — done, keep ≥75%.
- Phase 12 — Track recommendations via API + dashboard tab — done.

## Next steps

1. Persist track artifacts (mappings, similarity) in versioned bundle.
2. Add `evaluate --compare-all-tracks` or track holdout metrics.
3. Add Spotify import pipeline: fetch → `data/raw/sample_track_*` format.
4. CI: quality-gate job example + ablation-summary artifact upload.
5. Release 0.5.0 (track API + dashboard now served; bundle persistence left).

## Quality gates (every change)

- `uv run pytest -q` — 468+ tests must pass.
- `uv run ruff check .`
- `uv run mypy`
- Coverage ≥75% (`pytest --cov`).
- Small atomic commits, push after each green gate.
