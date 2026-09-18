# Project Plan — Music Recommender System

This plan tracks completed phases, the current phase, and next steps.
It is updated incrementally as phases land.

## Current milestone (0.14.0) — shipped

- Phase 41 — Track evaluation parity metrics: added `unexpectedness_at_k`
  and `intra_list_diversity` to the track evaluator, closing the last two
  metric gaps with artist-side evaluation. (commit `8031656`)
- Phase 42 — Track compare-all mode: `evaluate_track_holdout` and the
  `evaluate-tracks` CLI accept `--compare-all`, running similarity,
  popularity, and hybrid strategies in a single pass for side-by-side
  comparison. Mutually exclusive with `--compare-baseline`. (commit `b3f2625`)
- Phase 43 — Notebook walkthroughs: three notebooks in `notebooks/` covering
  data exploration, API serving, and evaluation with `compare_all`.
  (commit `529bd81`)
- Phase 44 — Docs and release: refreshed PLAN, README, CHANGELOG, bumped to
  0.14.0.
- Release 0.14.0 — shipped and published (tag `v0.14.0`).

## Completed (0.13.0 era)

- Phase 37 — Hybrid artist-taste explainability: track recommendations in
  hybrid mode explain the collaborative driver ("Artist affinity: X") next
  to the content reasons, and the dashboard renders `score_components` for
  hybrid hits so the blended score is visible where README promises it.
  (commit `4a896bd`)
- Phase 38 — Structured logging: a shared `configure_logging` helper, request
  logging in the API middleware, and INFO logs for training, artifact
  load/build, and the CLI train/eval commands close the observability gap.
  (commit `c30bfa7`)
- Phase 39 — Track-surface quality sweep: fix audit-found bugs (spotify
  duplicate init, redundant CLI except tuple, `--report-path` semantics),
  remove dead config, and add tests for the uncovered behavior (LTR API +
  dashboard, ablation summary render, quality-threshold errors, track
  validators, truncated track-catalog caption). (commit `18326f3`)
- Phase 40 — Docs and release: refreshed README roadmap/TOC/Logging section,
  CHANGELOG, bumped to 0.13.0. (commit `06e9606`)
- Release 0.13.0 — shipped and published (tag `v0.13.0`, release run
  34660611573 succeeded, both GHCR images live).

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

## Completed (track era, 0.9→0.12)

- Phase 10 — Docs refresh (README, CHANGELOG, data README, PLAN) — ongoing
  habit, refreshed with each release.
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
- Release 0.10.0 — shipped and published (tag `v0.10.0`, release run
  34545220536 succeeded, both GHCR images live).
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
- Release 0.11.0 — shipped and published (tag `v0.11.0`, release run
  34546595097 succeeded, both GHCR images live).
- Phase 35 — Hybrid taste-driven track recommendations: the `track_hybrid`
  strategy blends collaborative artist taste (a small ALS fit on artist
  plays) with audio-feature similarity via `content.py::hybrid_scores`,
  exposed as `method`/`content_weight` on the track recommendations
  function, `RecommenderService`, the API, CLI, and dashboard. Hybrid hits
  carry `score_components` (content, collaborative, and hybrid scores).
- Phase 36 — Hybrid track evaluation: `evaluate_track_holdout` and the
  `evaluate-tracks` CLI accept `method`/`content_weight`, training the
  artist-taste model per fold on the held-in interactions for a costed
  side-by-side against the pure content arm.
- Release 0.12.0 — shipped and published (tag `v0.12.0`, release run
  34652949643 succeeded, both GHCR images live).

## Historical (0.5→0.8 era, later superseded in scope by the track era)

- Release 0.5.0 — shipped code-wise (tag `v0.5.0`); the release workflow's
  verify job was red for a pre-existing reason, so GHCR publication did NOT
  happen. Treat 0.6.0 as the published release.
- Phase 18 — CI triage: fixed pre-existing `FORCE_COLOR`-sensitive CLI output
  assertions via a hermetic `_TYPER_FORCE_DISABLE_TERMINAL` test env, fixed
  nondeterministic ablation arm order, pinned JUnit failure annotations.
- Release 0.6.0 — shipped and published (tag `v0.6.0`; first release whose
  CI was green on main since Aug 30).
- Phase 21 — Dependabot #5 triaged by pinning upload-artifact v7.0.1 in-repo.
- Release 0.7.0 — shipped (tag `v0.7.0`).
- Phase 23 — Dependabot triage: `setup-uv` v10.0.1, `login-action` v4.6.0,
  `attest` v4.2.2 pinned (verified SHAs), closing PRs #1/#3/#4/#5.
- Release 0.8.0 — shipped and published (tag `v0.8.0`, release run #4
  succeeded, both GHCR images live).

## Completed (0.15.0 era)

- Phase 45 — Replace bare `assert` in production code with runtime guards
  that raise `ValueError` (tracks.py `artist_taste_per_track` check).
  (commit `3618412`)
- Phase 46 — Narrow broad `except Exception` in taste-model training
  (model.py) to `(ValueError, RuntimeError)` and log at `warning` level
  instead of `debug`. (commit `4661ea4`)
- Phase 47 — Move top-level `numpy` import in `cli.py` to local scope for
  faster CLI startup. (commit `09ce619`)
- Phase 48 — Remove `noqa: E501` suppressions in `ltr.py` by reformatting
  long lines with an intermediate variable. (commit `b800b00`)
- Phase 49 — Add dedicated unit tests for `baselines.py` (`popular_artists`
  with exclusions, empty stats, tie-breaking, top_k boundary).
  (commit `bb3e62a`)
- Phase 50 — Expand `test_config.py` edge-case coverage (blank env var,
  whitespace stripping, home-expansion). (commit `f660deb`)
- Phase 52 — Track-side A/B parity: `compare_track_parameter_settings` plus
  `evaluate-tracks --compare-settings` mirroring `evaluate-artists`, with
  labeled metric rows, winners-by-metric, and the overall leaderboard.
  Mutually exclusive with `--compare-baseline` / `--compare-all`.
  (commits `f0df062`, `6f1720b`, `cdc6396`)
- Phase 51 — Docs and release: refreshed PLAN, README, CHANGELOG, bumped to
  0.15.0.

## Completed (0.16.0 era)

- Phase 53 — Track-side ablation suite: `ablate_track_parameter_settings`
  helper in `track_evaluate.py` plus `evaluate-tracks --ablations` and
  `--report-dir` in `cli.py`, reporting champion row, ablation arm deltas,
  knob importance rankings, and persisting standardized JSON ablation reports
  that integrate directly with `ablation-summary` and the dashboard.
- Phase 54 — Docs and release: refreshed PLAN, README, CHANGELOG, bumped to
  0.16.0.

## Current milestone (0.17.0) — shipped

- Phase 55 — Track LTR re-ranker: `train_track_ltr_ranker` and `rank_tracks_with_ltr`
  in `ltr.py`, training a pointwise Ridge model using content similarity, log plays,
  normalized popularity rank, and user interaction count. (commit `8eb4098`)
- Phase 56 — Track holdout evaluation LTR arm: `evaluate_track_holdout` accepts
  `learn_to_rank: bool = False`, fitting the ranker per fold on held-in interactions
  and evaluating the re-ranked candidates under the `ltr` arm. (commit `8fbb5e0`)
- Phase 57 — CLI support for track LTR: `evaluate-tracks --learn-to-rank` option
  in `cli.py`, supporting single, baseline, and all-arm comparisons with mutual
  exclusion enforcement. (commit `92c724f`)
- Phase 58 — Cross-surface evaluation parity reporting: `surfaces.py` helper module
  and `evaluate-surfaces` CLI command comparing artist ALS and track audio-feature
  similarity side by side on equivalent holdouts with per-metric deltas and persistent
  JSON reports. (commit `c3cc8ac`)
- Phase 59 — Docs and release: refreshed PLAN, README, CHANGELOG, bumped to 0.17.0.

## Next steps (after 0.17.0)

1. Multi-objective candidate re-ranking (balancing accuracy, diversity, and novelty Pareto frontiers).
2. Online contextual bandit simulation for cold-start exploration.

## Quality gates (every change)

- `uv run pytest -q` — tests must pass.
- `uv run ruff check .`
- `uv run mypy`
- Coverage ≥75% (`pytest --cov`).
- Small atomic commits, push after each green gate.
