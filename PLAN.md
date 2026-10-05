# Project Plan — Music Recommender System

This plan tracks completed phases, the current phase, and next steps.
It is updated incrementally as phases land.

## Current milestone (0.29.0) — planned

Real-Time Streaming Feedback Ingestion and Asynchronous Policy Maintenance Worker.
High-throughput production serving benefits from decoupling feedback journaling and policy update sweeps from synchronous request-response threads:
1. Buffered in-memory feedback queue with non-blocking async flush worker to disk/stream.
2. Background daemon maintenance thread within `RecommenderService` for automatic periodic sweeping and snapshot rotation without external CLI dependencies.
3. Sliding-window drift alerts and automatic rollback to champion snapshots when policy divergence exceeds safety thresholds.
4. Health & observability endpoints exposing real-time queue depth, sweep latency, and OPE confidence intervals.

- Phase 130 — Streaming feedback ingestion queue & worker:
  `StreamingFeedbackQueue` with non-blocking enqueue, batch worker, and backpressure handling. ✓
  - Objective: Decouple feedback collection from request-response serving latency by buffering feedback records in a thread-safe, bounded memory queue and asynchronously flushing them in atomic batches to disk.
  - Acceptance Criteria:
    - AC-1: `append_bandit_feedback_batch` atomically persists batches of validated feedback records to the feedback journal. ✓
    - AC-2: `StreamingFeedbackQueue` provides thread-safe non-blocking ingestion with bounded memory (`max_queue_size`). ✓
    - AC-3: Backpressure strategies (`drop_oldest`, `reject`, `block`) prevent runaway memory usage under burst load. ✓
    - AC-4: Background worker flushes on batch size threshold or elapsed flush interval, handling I/O errors gracefully. ✓
    - AC-5: Manual `flush()` and `close()` ensure zero record loss on planned shutdown or service teardown. ✓
    - AC-6: `RecommenderService` supports optional `feedback_queue` to route cold-start bandit feedback non-blockingly, reporting queue metrics in `bandit_status()` and `metadata()`. ✓
    - AC-7: Comprehensive test suite for queue concurrency, backpressure, batch writing, and service integration. ✓
  - Implementation Tasks:
    - Task 1: Atomic batch journal append helper (`append_bandit_feedback_batch`) in `bandit.py`. ✓
    - Task 2: Core `StreamingFeedbackQueue` class with worker daemon and backpressure policies in `bandit.py`. ✓
    - Task 3: Unit tests for `append_bandit_feedback_batch` and `StreamingFeedbackQueue` in `tests/test_bandit.py`. ✓
    - Task 4: Service integration in `RecommenderService` (`service.py`) and status exposure. ✓
    - Task 5: Integration tests in `tests/test_service.py`. ✓
    - Task 6: Documentation updates in `README.md` and `PLAN.md`. ✓
- Phase 131 — Service async maintenance daemon:
  Integrated background maintenance thread in `RecommenderService` with graceful shutdown. ✓
  - Objective: Automate background bandit state maintenance and policy hot-reloading within `RecommenderService` using an asynchronous daemon worker that periodically sweeps feedback without blocking serving threads, coordinates with `StreamingFeedbackQueue`, supports snapshot rotation, and ensures clean lifecycle teardown.
  - Acceptance Criteria:
    - AC-1: `BanditMaintenanceWorker` provides a dedicated background daemon thread executing periodic sweeps on configurable intervals (`interval_seconds`) when pending feedback reaches `min_pending_records`. ✓
    - AC-2: Worker coordinates with `StreamingFeedbackQueue` by flushing buffered in-memory feedback prior to checking pending feedback counts. ✓
    - AC-3: Snapshot rotation support (`snapshot_on_sweep`) automatically creates state snapshots upon successful folding and prunes historical snapshots according to retention limits. ✓
    - AC-4: Policy hot-reloading (`auto_update_policy`) derives fresh cold-start policy weights and dynamically refreshes `RecommenderService.cold_start_policy` in-memory. ✓
    - AC-5: Thread synchronization locks prevent race conditions between background sweeps, explicit service operations, and request-serving paths. ✓
    - AC-6: Maintenance telemetry and health metrics (`cycles_count`, `sweeps_count`, `records_folded_count`, `last_sweep_at`, `last_sweep_duration_seconds`, `last_error`) are exposed in `BanditMaintenanceWorker.get_metrics()`, `service.maintenance_status()`, and `service.bandit_status()`. ✓
    - AC-7: Clean lifecycle management (`start`, `stop`, `trigger_sweep`, `close`) integrated into `RecommenderService.close()`, service context manager, and FastAPI `lifespan`. ✓
    - AC-8: Environment variable and constructor configuration support across `service.py` and `config.py`. ✓
    - AC-9: Comprehensive unit and integration test coverage for worker loops, flushing, hot-reloading, snapshot rotation, and teardown. ✓
  - Implementation Tasks:
    - Task 1: Add configuration constants and environment variables in `config.py`. ✓
    - Task 2: Implement `MaintenanceWorkerMetrics` and `BanditMaintenanceWorker` in `bandit.py`. ✓
    - Task 3: Unit tests for `BanditMaintenanceWorker` in `tests/test_bandit.py`. ✓
    - Task 4: Integrate maintenance daemon into `RecommenderService` in `service.py`. ✓
    - Task 5: Integrate service teardown in `api/main.py` lifespan. ✓
    - Task 6: Integration tests in `tests/test_service.py` and `tests/test_api.py`. ✓
    - Task 7: Documentation in `README.md` and mark Phase 131 complete in `PLAN.md`. ✓
- Phase 132 — Automated drift guardrails & rollback:
  `evaluate_drift_safety` and automatic fallback/restore to known good snapshot.
- Phase 133 — Observability tooling & CLI/API instrumentation:
  Metrics for queue depth, sweep timing, and streaming health endpoints.
- Phase 134 — Dashboard, testing, and 0.29.0 release.

## Previous milestone (0.28.0) — shipped

Contextual Bandit Off-Policy Evaluation (OPE), Epsilon-Greedy Exploration, and Multi-Policy Comparative Benchmarking.
Production cold-start recommendation requires evaluating and benchmarking bandit policies safely before live deployment. Previously, operators could only simulate one policy at a time in isolation, had no mechanism to evaluate candidate policies on historical logged feedback without live traffic risk, could only derive serving policies from offline simulation reports rather than production state snapshots, and lacked the classic epsilon-greedy contextual exploration baseline. This milestone delivered:
1. Epsilon-Greedy contextual bandit policy (`EpsilonGreedyContextualBandit`) with dynamic epsilon cooling schedule $\epsilon(t) = \epsilon_0 / (1 + \lambda t)$ and reproducible seeded exploration.
2. State-driven policy derivation with dynamic temperature annealing: `derive_cold_start_policy` accepts either simulation reports or persisted bandit states / snapshots, applying temperature annealing $\tau(t) = \max(\tau_{min}, \tau_0 / (1 + \lambda_{temp} t))$.
3. Off-Policy Evaluation (OPE) engine: Inverse Propensity Scoring (IPS), Self-Normalized IPS (SnIPS), Direct Method (DM), and Doubly Robust (DR) estimation with standard errors and effective sample sizes on logged feedback.
4. Multi-policy comparative simulation benchmarking: `compare_bandit_simulation_policies` runs candidate policies across identical cold-start holdout splits, reporting side-by-side rewards, regrets, win rates, and arm distributions.
5. End-to-end integration across Service, CLI (`simulate-bandit --compare-policies`, `bandit-eval-offline`, `bandit-policy --from-state`), FastAPI endpoints (`POST /bandit/evaluate/off-policy`, `POST /bandit/evaluate/compare`, `POST /bandit/policy/derive`), and Streamlit dashboard controls.

- Phase 124 — Epsilon-Greedy contextual bandit engine & dynamic temperature annealing:
  `EpsilonGreedyContextualBandit` extending `BaseContextualBandit`, state snapshot/restore support,
  and state/snapshot-driven policy derivation in `derive_cold_start_policy` with temperature annealing. ✓
- Phase 125 — Off-Policy Evaluation (OPE) & multi-policy benchmarking engine:
  `compute_off_policy_evaluation` (IPS, SnIPS, DM, DR), `compare_bandit_simulation_policies`,
  report serializers `write_bandit_comparison_report` and `write_ope_report`. ✓
- Phase 126 — Service integration:
  `RecommenderService.evaluate_bandit_off_policy`, `compare_bandit_policies`, and
  `derive_cold_start_policy_from_state`, with `bandit_status()` and `metadata()` exposure. ✓
- Phase 127 — CLI tooling for OPE and multi-policy comparison:
  `simulate-bandit --compare-policies`, `bandit-eval-offline`, and `bandit-policy --from-state`. ✓
- Phase 128 — API endpoints for OPE and policy derivation:
  `POST /bandit/evaluate/off-policy`, `POST /bandit/evaluate/compare`, and `POST /bandit/policy/derive`. ✓
- Phase 129 — Dashboard, testing, documentation, and 0.28.0 release:
  Cold-Start Bandit tab OPE and comparison views, full test suite pass (994 tests, 95.02% coverage), version bump to 0.28.0 in
  `pyproject.toml`, updated `README.md`, `CHANGELOG.md`, `PLAN.md`. ✓

## Previous milestone (0.27.0) — shipped

Bandit Exponential Reward Discounting and Thompson Sampling Arm Policies.
Over time, user tastes and song popularity shift, causing early cold-start
feedback to dominate ridge regression estimates without adapting to seasonal or
trend changes. In addition, LinUCB with fixed $\alpha$ can over-explore suboptimal
arms once estimates stabilize. This milestone introduces:
1. Exponential reward discounting / recency weighting: customizable decay factor
   $\gamma \in (0, 1]$ applied during feedback folding ($A \leftarrow \gamma A + x x^T$,
   $b \leftarrow \gamma b + r x$) so that recent impressions carry higher weight.
2. Thompson Sampling bandit policy: Bayesian posterior sampling for contextual
   bandit arm selection alongside LinUCB.
3. Exploration schedule / decay: dynamic $\alpha(t) = \alpha_0 / (1 + \lambda t)$
   cooling for stable production exploitation.
4. Comprehensive CLI, service, API, and dashboard policy controls.

- Phase 118 — Discounted fold engine & Thompson Sampling implementation:
  `ThompsonSamplingContextualBandit` alongside `LinUCBContextualBandit`,
  exponential recency discounting factor $\gamma \in (0, 1]$, dynamic alpha cooling schedule.
- Phase 119 — Service policy configuration and dynamic exploration scheduling:
  `bandit_policy_type`, `bandit_alpha_decay`, `bandit_gamma` in `RecommenderService`,
  recency-weighted feedback sweeps, and policy parameter exposure in `bandit_status()`.
- Phase 120 — CLI tooling for decay tuning and policy evaluation:
  `--policy-type`, `--alpha-decay`, and `--gamma` in `simulate-bandit`, `--gamma` in
  `bandit-update` and `bandit-sweep`, policy model display in `bandit-status`.
- Phase 121 — API endpoints and request-level policy selection:
  `policy_type`, `alpha_decay`, and `gamma` exposed in `GET /bandit/status`,
  `POST /bandit/update` payload accepts optional `gamma`.
- Phase 122 — Dashboard policy switching and recency-weighted reward visualization:
  Cold-Start Bandit tab displays policy model and dynamic alpha schedule with interactive
  `gamma` discount slider when folding.
- Phase 123 — Documentation, testing, and 0.27.0 release:
  Full test suite pass, version bump to 0.27.0 in `pyproject.toml`, updated `README.md`, `CHANGELOG.md`, `PLAN.md`.

## Previous milestone (0.26.0) — shipped

Automated online bandit sweeping, state snapshotting, and policy drift
tracking. Live serving appends feedback to the journal, but folding previously
required manual operator intervention via `bandit-update` or `POST /bandit/update`.
Furthermore, `bandit_state.json` was updated in place with no historical audit
trail, snapshot retention, or parameter drift tracking. This milestone delivers:
1. Automated online sweeping: configurable auto-sweep threshold in the service
   (`auto_sweep_threshold`) that folds pending feedback automatically during serving
   when pending count reaches the threshold, plus a daemon/worker CLI command
   (`bandit-sweep --loop --interval`).
2. Bandit state snapshotting: timestamped snapshots (`save_bandit_snapshot`,
   `list_bandit_snapshots`, `load_bandit_snapshot`, `restore_bandit_snapshot`,
   `prune_bandit_snapshots`) stored in `reports/bandit_snapshots/`.
3. Policy parameter drift tracking: `compute_bandit_drift` measuring ridge
   coefficient drift ($\Delta \theta$, $L_2$ norm, cosine similarity), selection
   growth, reward changes, and dominant arm shifts between any two states or snapshots.
4. End-to-end integration: CLI commands (`bandit-snapshot`, `bandit-drift`,
   `bandit-sweep`), API endpoints (`GET /bandit/snapshots`, `POST /bandit/snapshots`,
   `GET /bandit/drift`), and Streamlit dashboard controls on the Cold-Start Bandit tab.

- Phase 112 — Bandit state snapshotting & drift engine: `compute_arm_thetas`,
  `compute_bandit_drift`, `save_bandit_snapshot`, `list_bandit_snapshots`,
  `load_bandit_snapshot`, `restore_bandit_snapshot`, `prune_bandit_snapshots`
  in `bandit.py`.
- Phase 113 — Service auto-sweep & snapshot integration: `auto_sweep_threshold`
  in `RecommenderService`, automatic sweep execution in `recommend_user`,
  snapshot/drift methods, and metadata/status exposure.
- Phase 114 — CLI commands: `bandit-snapshot` (create/list/restore/prune),
  `bandit-drift` (formatted drift table), and `bandit-sweep` (interval loop or threshold).
- Phase 115 — API endpoints: `GET/POST /bandit/snapshots`, `GET /bandit/drift`,
  and auto-sweep threshold reporting in `/bandit/status`.
- Phase 116 — Dashboard: snapshot management and drift comparison view on the
  Cold-Start Bandit tab.
- Phase 117 — Docs and release: PLAN, README, CHANGELOG, version bump to 0.26.0.

## Previous milestone (0.25.0) — shipped

Configurable bandit context features. The cold-start bandit's context
feature set is a hard-coded tuple (`DEFAULT_CONTEXT_FEATURES`) threaded through
default arguments across simulation, serving, feedback recording, folding, and
observability. A team cannot choose or reorder the observation-window features
without editing source, unknown feature names fail with a raw `KeyError`, and
dimension mismatches surface only as scattered runtime checks that derive from
the default tuple. This milestone makes the active feature set a single
validated source of truth — explicit argument, environment variable, or
persisted JSON config — and derives every context dimension from it, so custom
feature sets work end to end and mismatches fail fast with actionable errors.

- Phase 102 — Feature registry: `CONTEXT_FEATURE_REGISTRY` maps feature names
  to value extractors; `resolve_context_features` validates a requested set
  (non-empty, known names, no duplicates) with a clear error listing the
  supported features.
- Phase 103 — Persisted config: `save_bandit_context_features` /
  `load_bandit_context_features` (default `reports/bandit_context_features.json`)
  plus a `MUSIC_RECOMMENDER_CONTEXT_FEATURES` environment override.
- Phase 104 — Registry-driven context building: `build_cold_start_context`
  resolves feature values through the registry (subsets and reordering
  supported) instead of a hard-coded dict lookup.
- Phase 105 — CLI simulation wiring: `simulate-bandit --context-features a,b,c`
  resolves the feature set and the report records it; validation rejects
  unknown names up front.
- Phase 106 — State/fold consistency: snapshots record `context_features`;
  `fold_bandit_state` and `sweep_bandit_journal` report dimension mismatches
  with an actionable message; `bandit-update --context-features` checks the
  state's recorded dimension.
- Phase 107 — Service integration: `RecommenderService` resolves the active
  features once, `recommend_user` validate/injects caller contexts against
  them, and `metadata()` / `bandit_status()` expose the feature set.
- Phase 108 — CLI observability: new `bandit-context` command to view/save the
  active set; `bandit-status` prints the resolved features.
- Phase 109 — API: `/bandit/status` includes the resolved features; optional
  `context_features` on `/bandit/update` validates against the state.
- Phase 110 — Dashboard: the Cold-Start Bandit tab shows the resolved feature
  set alongside the lifecycle.
- Phase 111 — Docs and release: PLAN, README, CHANGELOG, version bump to 0.25.0.

## Previous milestone (0.24.0) — shipped

Online bandit state lifecycle and observability. Live serves append to the
feedback journal but nothing folds those observations into the persisted
bandit state automatically: `bandit-update` re-folds the *entire* journal on
every run, so a repeated or scheduled invocation double-counts previously
folded records, and the bandit state / journal statistics are only visible in
raw JSON files. This milestone makes the online fold idempotent (safe to
schedule) and surfaces the bandit lifecycle in the service, API, CLI, and
dashboard.

- Phase 94 — Idempotent journal sweep: `sweep_bandit_journal` folds only the
  pending (not-yet-folded) records into a bandit state and records a fold
  watermark (`journal_fold_offset`, `journal_folded_at`) in the state config,
  so repeated sweeps are safe; a journal that shrank below the offset resets
  the watermark.
- Phase 95 — Status summary: `pending_feedback_count` and
  `summarize_bandit_lifecycle` build a readable snapshot (per-arm
  selections/rewards/mean reward, active policy weights, journal record and
  pending counts, last fold time) from the persisted files.
- Phase 96 — Service integration: `RecommenderService.bandit_status()` and
  `sweep_bandit_feedback()`; `metadata()` gains a top-level `bandit` state
  summary.
- Phase 97 — CLI: `bandit-status` command (read-only) and `bandit-update`
  folds the journal through the idempotent sweep.
- Phase 98 — API: `GET /bandit/status` and `POST /bandit/update` (the online
  sweep returning the updated status).
- Phase 99 — Dashboard: a "Cold-Start Bandit" tab renders the lifecycle status
  with a "fold pending feedback" action.
- Phase 100 — Tests: sweep idempotency/watermark/reset, status summary with and
  without files, and service/CLI/API/dashboard wiring. 858 tests.
- Phase 101 — Docs and release: PLAN, README (online sweep and
  observability), CHANGELOG, version bump to 0.24.0.

## Previous milestone (0.23.0) — shipped

Contextual, per-arm served bandit feedback. 0.22.0 records served feedback but
the context is always the neutral zero vector (the folded ridge update is a
no-op on `A`/`b` for live serves) and only the dominant policy arm receives
reward credit. This milestone makes live feedback genuinely train the
contextual prior and credit every arm of the policy.

- Phase 87 — Per-arm serve feedback records: `feedback_records_from_bandit_serve`
  returns one validated record per policy arm, each rewarded by the precision@k
  of the blended serve against that arm's own ranking; `feedback_from_bandit_serve`
  delegates to it (dominant-arm record, unchanged contract).
- Phase 88 — Serve-context validation: `validate_serve_context` enforces a
  finite, non-empty numeric vector of the default feature dimension so caller
  supplied contexts can be folded safely; wired into the serve-feedback path.
- Phase 89 — Service integration: `recommend_user` accepts an optional
  `context` and records the full per-arm feedback batch when `record_feedback`
  is set, echoing the dominant arm plus the list of credited arms.
- Phase 90 — CLI wiring: `recommend-user --context "a,b,c"` forwards a real
  observation window (rejected without `--record-feedback`).
- Phase 91 — API wiring: `GET /recommend/user/{user_id}?context=a,b,c` forwards
  the observation context.
- Phase 92 — Tests: per-arm record attribution/determinism, context validation,
  service/CLI/API wiring and error paths.
- Phase 93 — Docs and release: PLAN, README (contextual feedback), CHANGELOG,
  version bump to 0.23.0.

## Previous milestone (0.22.0) — shipped

Online serve-feedback loop. `recommend-user` (unknown-user branch) records the
served feedback — the neutral cold-start context, the dominant arm of the
active policy, and an engagement reward computed from the blended serving
response — into the feedback journal that `record-bandit-feedback` writes and
`bandit-update` folds. This closes the loop described in 0.21.0 end-to-end
without any new dependency.

- Phase 81 — Serve-feedback helpers: `dominant_policy_arm` (deterministic
  highest-weight arm, lexicographic tie-break) and
  `feedback_from_bandit_serve` (validated `{context, arm, reward}` record whose
  reward is the precision@k of the blended serve against the dominant arm's own
  ranking); `neutral_serve_context` exposes the brand-new-user zero context.
- Phase 82 — Service integration: `RecommenderService.recommend_user` gained
  `record_feedback` and `feedback_journal_path`; the `bandit_fallback` branch
  appends to the journal (`reports/bandit_feedback.json` by default) and reports
  arm/reward/path in its response. Known users and `popular_fallback` never
  record.
- Phase 83 — CLI: `recommend-user --record-feedback` (plus `--feedback-path`),
  printing where feedback was recorded; rejected together with `--ltr`.
- Phase 84 — API: `GET /recommend/user/{user_id}?record_feedback=true`.
- Phase 85 — Tests: helper determinism/validation, pure-popular reward-1.0,
  neutral context, fold of a serve record into a prior, service on/off and
  known-user exclusion, CLI and API wiring. 811 tests.
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

## Next steps (after 0.25.0)

1. Two-tower neural candidate retrieval (PyTorch / ONNX runtime) — deferred
   until a real-scale catalog is available: the current sample dataset (18
   artists, 36 tracks) cannot meaningfully train or validate a neural
   retrieval model, and the phase would add heavy new dependencies.
2. Scheduled online fold: run the idempotent sweep automatically (cron /
   internal scheduler) so live served feedback folds back into the bandit
   prior without manual `bandit-update` (the 0.24.0 watermark makes this safe).
3. New context feature extractors (beyond the registry's shipped three) —
   requires a catalog with the corresponding signals; the 0.25.0 registry
   makes this a config-only change.

## Quality gates (every change)

- `uv run pytest -q` — tests must pass.
- `uv run ruff check .`
- `uv run mypy`
- Coverage ≥75% (`pytest --cov`).
- Small atomic commits, push after each green gate.
