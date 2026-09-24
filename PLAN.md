# Project Plan — Music Recommender System

This plan tracks completed phases, the current phase, and next steps.
It is updated incrementally as phases land.

## Current milestone (0.22.0) — in progress

Online serve-feedback loop: the bandit feedback loop becomes usable from the
live serving surface. `recommend-user` (unknown-user branch) can record the
served feedback — the neutral cold-start context, the dominant arm of the
active policy, and an engagement reward computed from the blended serving
response — straight into the feedback journal that `record-bandit-feedback`
writes and `bandit-update` folds. This closes the loop described in 0.21.0
end-to-end without any new dependency.

- Phase 81 — Serve-feedback helpers:
  - `dominant_policy_arm`: deterministic pick of the highest-weight policy arm.
  - `feedback_from_bandit_serve`: given a policy, artist stats, and top-k,
    produce a validated `{context, arm, reward}` record whose reward is the
    precision@k of the blended served response against the dominant arm's own
    ranking — an engagement proxy computed entirely from the serve.
- Phase 82 — Service integration: `RecommenderService.recommend_user` gains
  `record_feedback` and `feedback_journal_path`; the `bandit_fallback` branch
  appends the serve-feedback to the journal (`reports/bandit_feedback.json` by
  default) and reports it in the response. Known users and the pure
  `popular_fallback` branch never record.
- Phase 83 — CLI wiring: `recommend-user --record-feedback` (plus
  `--feedback-path`), printing where feedback was recorded.
- Phase 84 — API wiring: `GET /recommend/user/{user_id}?record_feedback=true`
  records the serve-feedback for the bandit branch.
- Phase 85 — Tests: helper determinism/validation (dominant arm selection,
  pure-popular reduction to reward 1.0, neutral context), service recording
  on/off and error paths, CLI and API wiring.
- Phase 86 — Docs and release: PLAN, README (online feedback), CHANGELOG,
  version bump to 0.22.0.

## Previous milestone (0.14.0) — shipped

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

## Completed (0.17.0 era)

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

## Previous milestone (0.18.0) — shipped

- Phase 60 — Track LTR model serving parity: include track LTR ranker in artifact
  bundle and serve via `RecommenderService.recommend_tracks_ltr`. (commit `4afbd34`)
- Phase 61 — Track LTR API, CLI, and Dashboard: `GET /recommend/tracks/{user_id}/ltr`,
  `track-recommendations --ltr`, and Streamlit UI toggle. (commit `92fd72e`)
- Phase 62 — Multi-objective candidate re-ranking & Pareto frontier engine:
  `multi_objective.py` implementing scalarized re-ranking, dominance checks, and
  Pareto frontier computation. (commit `0978d99`)
- Phase 63 — Global novelty weight controls across all recommendation surfaces:
  service, API, CLI (`--novelty-weight`), and dashboard sliders. (commit `1855c5c`)
- Phase 64 — Multi-objective Pareto frontier evaluation and CLI: `evaluate_pareto_frontier`,
  `evaluate --pareto-frontier` command, and track evaluation parity. (commit `ee6ecd0`)
- Phase 65 — Docs and release: refreshed PLAN, README, CHANGELOG, bumped to 0.18.0.

## Previous milestone (0.21.0) — shipped

Online bandit updates from live serving feedback: the LinUCB engine's learned
state (ridge-statistics A/b, selection counts, cumulative rewards) becomes a
persisted, mergeable prior. Simulation resumes from a prior state, a feedback
journal captures reward observed per served request, and an update step folds
new observations into the prior so the next simulation (and the policy derived
from its report) starts from accumulated experience instead of scratch.

- Phase 75 — LinUCB state snapshot/restore:
  - `snapshot_bandit_state` / `LinUCBContextualBandit.from_state`: export and
    restore the engine's sufficient statistics (per-arm `a`, `b`,
    `selections`, `rewards`) plus config (arms, context_dim, alpha), with
    full validation so a restored engine is byte-identical in behavior.
  - `write_bandit_state` / `load_bandit_state`: persistent JSON state files
    mirroring the report/policy I/O pattern; default `reports/bandit_state.json`.
    (commit `17642e8`)
- Phase 76 — Feedback journal and additive fold:
  - `append_bandit_feedback` / `load_bandit_feedback`: append/read served-
    request observations `{context, arm, reward}` at `reports/bandit_feedback.json`.
  - `fold_bandit_state`: apply a batch of feedback records to a prior state by
    replaying the engine's additive ridge regression (`A += x xᵀ`,
    `b += r x`, tally increments), returning an updated prior.
  - `feedback_from_report`: extract round observations from a bandit report so
    offline batches can be folded too. (commit `96db512`)
- Phase 77 — Simulation from a prior:
  - `simulate_cold_start_exploration` gains `initial_state`; round records now
    carry the `context` vector so reports are replayable as feedback. Report
    config records `prior` selections/rewards and arms statistics accumulate
    prior + new learning. (commit `d54a919`)
- Phase 78 — CLI wiring:
  - `simulate-bandit --from-state` (resume learning from a persisted state), and
    `--write-state` writes the trained state to `reports/bandit_state.json`.
  - New `bandit-update` command folding a report or journal into a prior state
    (defaults to the persisted state), printing per-arm stats.
  - New `record-bandit-feedback` command appending a served-request observation
    to the feedback journal. (commits `8fe81c1`, `9e19f5b`)
- Phase 79 — Tests: state snapshot/restore roundtrips and validation, additive
  fold correctness vs. direct engine updates, feedback journal append/load,
  report-context extraction, prior-seeded simulation determinism and cumulative
  stats, CLI wiring and error paths. (796 tests total)
- Phase 80 — Docs and release: PLAN, README (feedback loop), CHANGELOG, version
  bump to 0.21.0.

## Previous milestone (0.20.0) — shipped

Live serving integration of the cold-start bandit: the `popular_fallback` path is
rewritten so an unknown user is served by the bandit's learned policy (arm weights
from a `simulate-bandit` report) instead of pure popularity, while the old behavior
is preserved when no policy file exists.

- Phase 70 — Bandit policy derivation and serving ranking:
  - `derive_cold_start_policy(report)`: read a bandit report and compute per-arm
    serving weights (softmax over `mean_reward`), deterministic.
  - `rank_cold_start_bandit(policy, artist_stats, top_k)`: serve top-k by a
    positional weighted blend of the arm rankings (Borda-style), deterministic,
    reducing to `popular_artists` when the policy is `{"popular": 1.0}`.
  - `write_cold_start_policy` / `load_cold_start_policy`: persist/validate the
    policy JSON, mirroring the report helpers. (commits `fd9018f`, `1234ca2`)
- Phase 71 — Service integration: `RecommenderService` gains a `cold_start_policy`
  loaded by `from_artifacts` (from `COLD_START_POLICY_PATH` when present);
  `recommend_user` unknown-user branch serves `bandit_fallback` when a policy is
  set and keeps `popular_fallback` otherwise; `metadata()` reports the active
  cold-start strategy. (commit `0bfde00`)
- Phase 72 — CLI: `bandit-policy` command (derive weights from a bandit report and
  write the policy JSON) and `recommend-user --cold-start-policy-path` to serve
  with the learned policy. (commits `410f9a1`, `1619571`)
- Phase 73 — Tests: policy derivation determinism/validation, blended ranking
  behavior, policy report roundtrip, service `bandit_fallback`/`popular_fallback`
  paths, `metadata` strategy, and CLI end-to-end + error paths.
- Phase 74 — Docs and release: PLAN, README (bandit serving), CHANGELOG, version
  bump to 0.20.0.

## Previous milestone (0.19.0) — shipped

- Phase 66 — Cold-start exploration bandit simulation:
  - `bandit.py` with a LinUCB contextual bandit engine (`LinUCBContextualBandit`)
    that learns, per context, which cold-start arm (strategy) serves new users
    best. (commit `d4c1093`)
  - Cold-start arm registry (`DEFAULT_COLD_START_ARMS`):
    `popular` (existing `popular_artists` baseline), `balanced` and `long_tail`
    (increasingly aggressive popularity-penalty strategies).
  - `simulate_cold_start_exploration`: offline harness that holds out users,
    derives a bootstrap context from each cold user's earliest interactions,
    serves an arm per round, rewards precision@k against held-out plays, and
    tracks selection counts, cumulative reward, and regret vs. the in-hindsight
    best arm — alongside the static `popular` control.
  - `write_bandit_report` / `load_bandit_report`: persistent JSON reports
    mirroring the `surfaces.py` pattern.
- Phase 67 — CLI wiring: `simulate-bandit` command exposing `--top-k`,
  `--rounds`, `--seed`, `--arms`, `--holdout-ratio`, `--alpha`, and
  `--report-name` / `--report-dir`.
- Phase 68 — Tests: engine update/select, arm ordering, reward computation,
  simulation determinism and regret, report roundtrip/schema validation, and
  CLI output + error paths. (`d4c1093`, `08608d7`)
- Phase 69 — Docs and release: PLAN, README (evaluation + CLI), CHANGELOG,
  version bump to 0.19.0.

## Next steps (after 0.22.0)

1. Two-tower neural candidate retrieval (PyTorch / ONNX runtime) — deferred
   until a real-scale catalog is available: the current sample dataset (18
   artists, 36 tracks) cannot meaningfully train or validate a neural
   retrieval model, and the phase would add heavy new dependencies.
2. Per-request context capture: feed real user observation windows into
   `recommend-user` so the recorded feedback context is user-specific instead
   of the neutral vector, and fold per-arm attribution.
3. Online bandit state snapshot and scheduled `bandit-update` folding of live
   serves back into the simulation prior, surfaced in the dashboard/API
   metadata.

## Quality gates (every change)

- `uv run pytest -q` — tests must pass.
- `uv run ruff check .`
- `uv run mypy`
- Coverage ≥75% (`pytest --cov`).
- Small atomic commits, push after each green gate.
