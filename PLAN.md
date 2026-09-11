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

- Release 0.7.0 — shipped (tag `v0.7.0`).
- Phase 23 — Dependabot triage by bumping remaining action pins
  (`setup-uv` v10.0.1, `login-action` v4.6.0, `attest` v4.2.2; verified
  SHAs), closing PRs #1, #3, and #4 (PR #5 already superseded).
- Phase 24 — Dashboard track catalog table on the shared service contract.
- Release 0.8.0 — shipped and published (tag `v0.8.0`, release run #4
  succeeded, both GHCR images live).
- Phase 25 — Track popularity baseline (`recommend_popular_tracks`) with
  `evaluate-tracks --compare-baseline` reporting both arms.
- Release 0.9.0 — shipped and published (tag `v0.9.0`, release run #5
  succeeded, both GHCR images live).
- Phase 26 — Track ranking knobs (popularity penalty + audio-feature
  diversity) exposed across service, API, CLI, and dashboard, with the
  audio-feature matrix persisted on `TrackServingResources` and validated on
  artifact bundles.
- Phase 27 — Tunable track evaluation: `evaluate_track_holdout` and
  `evaluate-tracks` accept the same penalty/diversity knobs.
- Phase 28 — Track evaluation reports: `write_track_report` /
  `load_track_report` and `evaluate-tracks --report-path` write stable JSON
  run reports to `reports/`.
- Phase 29 — Top tracks baseline surfaced across surfaces: `popular_tracks`
  helper, `RecommenderService.popular_tracks`, `GET /tracks/popular`, and a
  `popular-tracks` CLI command.
- Release 0.10.0 — shipped and published (tag `v0.10.0`, release run #6
  succeeded, both GHCR images live).
- Phase 30 — Track recommendation explanations (`explain` parity): reasons
  citing listened tracks on the function, service, API, CLI, and dashboard.
- Phase 31 — Cold-start fallback: users without track listening history get
  popular tracks (`popular_fallback`) instead of an empty list.
- Phase 32 — Track recommendations honor the artifact champion ranking
  config via `_ranking_overrides`, matching the artist surfaces.
- Phase 33 — Track evaluation reports `explanation_coverage` and, with it,
  keeps `evaluate-tracks` output parity with the artist evaluator.
- Phase 34 — Track evaluation also reports `serendipity_at_k`, surfacing
  whether relevant recommendations come from the popularity long tail.

## Next steps

1. Release 0.11.0 (tag `v0.11.0`, verify release workflow publishes both
   GHCR images).
2. Plan 0.12.0 scope — candidates: hybrid taste-driven track
   recommendations (ALS artist affinities mapped to tracks), dashboard
   track-explanation coverage display, and Jupyter walkthrough notebooks.

## Quality gates (every change)

- `uv run pytest -q` — 568+ tests must pass.
- `uv run ruff check .`
- `uv run mypy`
- Coverage ≥75% (`pytest --cov`).
- Small atomic commits, push after each green gate.
