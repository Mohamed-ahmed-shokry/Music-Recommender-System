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
- Phase 13 — Spotify import pipeline (`spotify-import-catalog`) — done.
- Phase 14 — Strict quality gate (`--fail-on-quality-gate`) + scheduled
  `quality-gate` workflow with ablation report upload — done.
- Phase 15 — Track bundle persistence in the versioned artifact (optional
  field, legacy fallback, train + serve wiring) — done.
- Phase 16 — Track holdout evaluation (`track_evaluate`, `evaluate-tracks`)
  reusing the shared ranking metrics — done.
- Phase 17 — Coverage sweep to 96% (track validators, CSV fallback, track
  CLI paths, Spotify client) — done.
- Release 0.5.0 — shipped (tag `v0.5.0`, containers publishing via release
  workflow).

## Next steps

1. Monitor the `v0.5.0` release workflow and GHCR publication.
2. Backfill track popularity/novelty metrics in `evaluate-tracks`.
3. Dashboard track catalog search.
4. Plan 0.6.0 scope.

## Quality gates (every change)

- `uv run pytest -q` — 468+ tests must pass.
- `uv run ruff check .`
- `uv run mypy`
- Coverage ≥75% (`pytest --cov`).
- Small atomic commits, push after each green gate.
