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
- Release 0.5.0 — shipped code-wise (tag `v0.5.0`); the release workflow's
  verify job is red for a pre-existing reason (see Phase 18), so GHCR
  publication did NOT happen.
- Phase 18 — CI triage: bisected the Linux red to pre-existing
  `FORCE_COLOR`-sensitive CLI output assertions (red since Aug 31, before
  this work); fixed via hermetic `_TYPER_FORCE_DISABLE_TERMINAL` test env,
  fixed nondeterministic ablation arm order, and added pinned JUnit failure
  annotations for future diagnosis.
- Phase 19 — Track popularity statistics + novelty/average-popularity in
  `evaluate-tracks` — done.
- Phase 20 — Dashboard track catalog search — done.
- Release 0.6.0 — shipped and published (tag `v0.6.0`; release run #2
  succeeded, both GHCR images live). CI green on main for the first time
  since Aug 30.

Note: the `v0.5.0` tag never published images (its release run failed
verify on the pre-existing CI red). Treat 0.6.0 as the published release.

- Phase 21 — Dependabot #5 triaged by bumping the upload-artifact pin to
  verified v7.0.1 in-repo (supersedes the PR).
- Phase 22 — Track catalog parity (`browse_tracks` + `GET /tracks/catalog`).

## Next steps

1. Release 0.7.0.

## Quality gates (every change)

- `uv run pytest -q` — 468+ tests must pass.
- `uv run ruff check .`
- `uv run mypy`
- Coverage ≥75% (`pytest --cov`).
- Small atomic commits, push after each green gate.
