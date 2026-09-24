# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.22.0] - 2026-09-24

### Added

- Online serving closes the bandit feedback loop end-to-end: the live
  `recommend-user` cold-start branch can journal its own serve feedback so
  served traffic folds back into the next simulation prior.
  - `dominant_policy_arm`: deterministic highest-weight policy arm selection
    (lexicographic tie-break).
  - `feedback_from_bandit_serve`: builds a validated `{context, arm, reward}`
    record directly from a blended bandit serve — the reward is the precision@k
    of the served response against the dominant arm's own ranking, an
    engagement proxy computed entirely from the serve.
  - `neutral_serve_context`: the zero cold-start context of a brand-new user.
  - `RecommenderService.recommend_user(record_feedback, feedback_journal_path)`:
    the `bandit_fallback` branch appends the serve feedback to the journal
    (default `reports/bandit_feedback.json`) and reports arm/reward/path in its
    response; known users and the `popular_fallback` branch never record.
  - CLI: `recommend-user --record-feedback` (with `--feedback-path`) journals
    and prints where the serve feedback was recorded.
  - API: `GET /recommend/user/{user_id}?record_feedback=true` records the
    serve feedback for the bandit branch.

### Tests

- 811 tests; added coverage for serve-feedback record determinism and
  validation, the pure-popular reward-1.0 reduction, neutral context, folding a
  serve feedback record into a prior state, service recording on/off and
  known-user exclusion, CLI pass-through and `--record-feedback` + `--ltr`
  rejection, and the API query-param wiring.

### Documentation

- README documents the served-feedback flow
  (`recommend-user --record-feedback` → `bandit-update`) and the API equivalent;
  PLAN and CHANGELOG updated for the 0.22.0 milestone.

## [0.21.0] - 2026-09-23

### Added

- Online bandit feedback loop: the LinUCB cold-start engine's learned state is a
  persisted, mergeable prior so experience accumulates across runs:
  - `snapshot_bandit_state` / `LinUCBContextualBandit.from_state`: export and
    restore the engine's sufficient statistics (per-arm `a`, `b`, `selections`,
    `rewards`) plus config, so a restored engine behaves identically.
  - `write_bandit_state` / `load_bandit_state`: persistent JSON state files
    (default `reports/bandit_state.json`).
  - `append_bandit_feedback` / `load_bandit_feedback`: a JSON feedback journal
    (default `reports/bandit_feedback.json`) accumulating `{context, arm, reward}`
    observations per served request.
  - `fold_bandit_state`: replays feedback batches through the engine's additive
    ridge update (`A += x xᵀ`, `b += r x`, tally increments), returning an
    updated prior; `feedback_from_report` extracts round observations from a
    report so offline batches fold too.
  - `simulate_cold_start_exploration(initial_state=...)`: resumes learning from
    a persisted prior, validated against arms/context_features/alpha; round
    records now carry the served `context` vector and the report config records
    the prior's selections and total reward.
  - CLI: `simulate-bandit --from-state` (resume from a prior) and `--write-state`
    (persist the trained state); `bandit-update` (fold a report or journal into
    a prior state) and `record-bandit-feedback` (journal a served-request
    observation).

### Tests

- 796 tests, ~96.6% statement coverage; added unit tests for state
  snapshot/restore roundtrips and validation, additive fold correctness vs.
  direct engine updates, feedback journal append/load and error paths, report
  round-context extraction, prior-seeded simulation determinism and cumulative
  prior stats, and CLI wiring/error paths for the new and extended commands.

## [0.20.0] - 2026-09-22

### Added

- Live cold-start bandit serving integration: bandit simulation results now drive
  production cold-start rankings instead of a static popularity fallback:
  - `derive_cold_start_policy` in `bandit.py`: converts a bandit report into per-arm
    serving weights through a deterministic softmax over the learned mean rewards.
  - `rank_cold_start_bandit`: serves top-k cold-start artists via a Borda-style
    policy-weighted blend across arms, reducing exactly to `popular_artists` when
    the policy is `{"popular": 1.0}`.
  - `write_cold_start_policy` / `load_cold_start_policy`: persist and validate the
    serving policy JSON (default `reports/cold_start_policy.json`).
  - `RecommenderService.from_artifacts` auto-loads the policy file when present;
    `recommend_user` serves unknown users with strategy `bandit_fallback`
    (policy-weighted blend) when a policy is set and keeps `popular_fallback`
    otherwise. `metadata()` reports the active cold-start strategy and weights.
  - CLI: `bandit-policy` command derives and persists serving weights from a bandit
    report; `recommend-user --cold-start-policy-path` serves with an explicit
    policy file, while the default auto-loads the project policy.

### Tests

- 767 tests, ~96.8% statement coverage; added unit tests for policy derivation
  (determinism, temperature, softmax ordering, validation), blended serving
  ranking (pure-popular equivalence, top-k/validation, determinism), policy JSON
  roundtrip/validation, service `bandit_fallback`/`popular_fallback` paths and
  policy auto-load, and CLI `bandit-policy` plus `recommend-user` policy wiring.

## [0.19.0] - 2026-09-22

### Added

- Cold-start exploration bandit simulation: `bandit.py` module simulating an online
  contextual multi-armed bandit (LinUCB) that decides which cold-start strategy to serve
  per user context, learning precision@k-style engagement rewards online:
  - `LinUCBContextualBandit`: deterministic UCB arm selection with per-arm ridge
    regression over context features.
  - `rank_cold_start_arm`: `popular` (training-set popularity, the current production
    fallback), `balanced` (mild mid-tail penalty), and `long_tail` (strong long-tail
    exploration) arms.
  - `simulate_cold_start_exploration`: warm/cold user splits, bootstrap-context
    windows, per-round precision@k rewards, and regret against best-in-hindsight and
    the always-popular control; deterministic for a given seed.
  - `write_bandit_report` / `load_bandit_report`: standardized JSON report persistence.
- CLI command: `simulate-bandit` mirroring `evaluate-surfaces`, with `--top-k`,
  `--rounds`, `--seed`, `--arms`, `--holdout-ratio`, `--alpha`, `--report-name`,
  and `--report-dir`, rendering per-arm selection/reward tables and writing reports
  to `reports/bandit_simulation.json` by default.

### Tests

- 745 tests, 96.8% statement coverage; added comprehensive unit and CLI test suites for
  bandit engine selection/updates, arm ranking, context building, simulation determinism,
  report roundtrips/schema validation, and the `simulate-bandit` CLI including error paths.

## [0.18.0] - 2026-09-19

### Added

- Track-side Learning-to-Rank (LTR) Serving Parity:
  - Artifact bundle persistence: `train_track_ltr_ranker` output is bundled into `ArtifactBundle` (`track_ltr_ranker` / `track_ltr_ranker.joblib`).
  - Serving service integration: `RecommenderService.recommend_tracks_ltr` generates candidate recommendations and re-ranks them using the persisted track LTR model, falling back gracefully to similarity or popular tracks.
  - API endpoint: `GET /recommend/tracks/{user_id}/ltr` (and alias `GET /tracks/recommend/{user_id}/ltr`) with full parameter validation and error handling.
  - CLI flag: `track-recommendations --ltr` to request LTR re-ranked track recommendations directly from the command line.
  - Streamlit dashboard: interactive toggle in the Track Recommendations tab enabling LTR re-ranking alongside existing ranking controls.
- Multi-Objective Re-Ranking & Pareto Frontier Engine:
  - `multi_objective.py` engine implementing scalarized multi-objective re-ranking (`re_rank_multi_objective`) balancing relevance, diversity, and novelty.
  - Dominance analysis: `dominates` and `compute_pareto_frontier` algorithms identifying the non-dominated Pareto frontier and finding the best balanced configuration across multi-dimensional objectives.
- Global Novelty Weight Control:
  - `novelty_weight` parameter added across all recommendation methods in `RecommenderService` (`recommend_user`, `recommend_user_als`, `recommend_user_ltr`, `recommend_session`, `recommend_tracks`, `recommend_tracks_ltr`).
  - FastAPI query parameter `novelty_weight: UnitInterval = 0.0` exposed across artist and track recommendation endpoints.
  - CLI flags: `--novelty-weight` added to `recommend-user` and `track-recommendations`.
  - Streamlit dashboard: interactive Novelty Weight sliders added to Personalized and Track recommendation interfaces.
- Multi-Objective Pareto Frontier Evaluation:
  - `evaluate_pareto_frontier`, `write_pareto_report`, and `load_pareto_report` in `evaluate.py` performing multi-dimensional grid sweeps over relevance, diversity, and novelty, extracting the Pareto frontier and identifying the best balanced trade-off.
  - CLI command: `evaluate --pareto-frontier` rendering detailed configuration evaluation tables, annotating Pareto-optimal configs, and persisting standardized JSON reports to `reports/pareto_frontier.json`.
  - Track holdout evaluation parity: `evaluate_track_holdout` accepts `novelty_weight`, propagating it to similarity and hybrid candidate recommendation arms.

### Tests

- 723 tests, 96.8% statement coverage; added comprehensive unit, service, API, and CLI test suites for track LTR serving, scalarized multi-objective re-ranking, Pareto frontier extraction, novelty weighting, and Pareto evaluation sweeps.

## [0.17.0] - 2026-09-18

### Added

- Track-side Learning-to-Rank (LTR): `train_track_ltr_ranker` and `rank_tracks_with_ltr`
  in `ltr.py`, fitting a pointwise Ridge regressor over content similarity,
  log1p total plays, normalized popularity rank, and user interaction count.
- Track holdout evaluation LTR arm: `evaluate_track_holdout` accepts `learn_to_rank: bool = False`,
  training the ranker on held-in interactions and reporting the re-ranked candidates under the `ltr` arm.
- CLI `--learn-to-rank`: `evaluate-tracks --learn-to-rank` option in `cli.py`, reporting
  `Similarity` and `LTR` arms (or alongside `Popularity` and `Hybrid` when combined with
  `--compare-baseline` or `--compare-all`), with `--ablations` mutual exclusion.
- Cross-surface evaluation parity reporting: `surfaces.py` module (`compare_surface_metrics`,
  `write_surface_comparison_report`, `load_surface_comparison_report`) and new CLI command
  `evaluate-surfaces`, evaluating artist and track recommendation systems side by side on
  equivalent holdout splits, computing per-metric deltas, and persisting standardized JSON reports.

### Tests

- 680 tests, 96.8% statement coverage; added comprehensive unit and CLI tests for
  track LTR ranker training/inference, track holdout LTR evaluation, `evaluate-tracks --learn-to-rank`,
  cross-surface comparison calculations, report serialization roundtrips, and `evaluate-surfaces` CLI.

## [0.16.0] - 2026-09-17

### Added

- Track-side ablation suite: `ablate_track_parameter_settings` helper in
  `track_evaluate.py` plus `evaluate-tracks --ablations` and `--report-dir` in
  `cli.py`. Systematically ablates active ranking parameters
  (`popularity_penalty`, `diversity`, `include_listened`) across holdouts,
  printing champion metrics, ablation arm rows, and knob importance rankings.
- Standardized track ablation reports: persisted to `reports/` as JSON
  (`track_ablation_importance.json` by default or customized with `--report-name`
  and `--report-dir`), matching the schema used by artist ablation reports.
- Full end-to-end integration: track ablation reports are automatically
  aggregated by `ablation-summary` alongside artist reports and rendered in the
  Streamlit dashboard's ablation summary tab.

### Tests

- 653 tests, 96.8% statement coverage; added comprehensive unit and CLI tests for
  track ablations, flag mutual exclusion, neutral champion rejection, report
  persistence round-trip, and aggregation via `ablation-summary`.

## [0.15.0] - 2026-09-16

### Added

- Track-side A/B parity: `compare_track_parameter_settings` and the
  `evaluate-tracks --compare-settings` option now mirror the artist-side
  harness. Each semicolon-separated `label:key=value,...` setting runs the
  same holdout with its reranking kwargs, printing one metric row per label
  plus winners-by-metric and the overall leaderboard. Mutually exclusive
  with `--compare-baseline` and `--compare-all`.
- Dedicated unit tests for `baselines.popular_artists` covering popularity
  ordering, exclusions, empty stats, tie-breaking, and invalid `top_k`.
- Expanded `resolve_project_root` edge-case tests (blank environment value,
  surrounding whitespace stripping, and home-directory expansion).

### Fixed

- Replaced the bare `assert artist_taste_per_track is not None` in
  `recommend_tracks_for_user` with an explicit `ValueError` runtime guard
  that survives `python -O` (optimized) execution.
- Narrowed the taste-model training fallback in `train_and_save_model` from
  a bare `except Exception` to `(ValueError, RuntimeError)` and elevated the
  skip log from debug to warning so operators can see when the optional
  artist-taste model is skipped.

### Changed

- `cli.py` now imports numpy locally inside `track-recommendations` instead
  of at module load, reducing startup cost for every other CLI command.
- Refactored the `ltr.py` grouped play-count feature build to remove both
  `noqa: E501` suppressions using an intermediate variable.

### Tests

- 641 tests, ~96.6% statement coverage (up from 611 in 0.13.0); added
  `test_baselines.py`, expanded `test_config.py`, and covered
  `compare_track_parameter_settings` plus the new CLI option and its mutual
  exclusion error.

## [0.14.0] - 2026-09-14

### Added

- Track evaluation now reports `unexpectedness_at_k` and
  `intra_list_diversity`, closing the last two metric gaps with artist-side
  evaluation. The CLI `evaluate-tracks` prints both metrics.
- `evaluate_track_holdout` and the `evaluate-tracks` CLI accept
  `--compare-all`, running similarity, popularity, and hybrid strategies in a
  single pass and returning all three arms as a labelled dict. Mutually
  exclusive with `--compare-baseline`.
- Three notebook walkthroughs in `notebooks/`: data exploration (01),
  API serving (02), and evaluation with compare_all (03).

## [0.13.0] - 2026-09-12

### Added

- Hybrid track explanations now name the collaborative driver: with
  `explain=true`, hybrid hits carry an `Artist affinity: <artist>` reason
  whenever the recommended track's artist contributed positively on the
  collaborative side (`track_artist_lookup`, wired through the service, CLI,
  and track evaluation).
- The dashboard `recommendation_frame` renders a `Score components` column
  (`content … · collab …`) for hybrid responses instead of discarding the
  blended score breakdown.
- Structured logging: `music_recommender.logging_setup.configure_logging`
  (idempotent) is configured by the CLI app callback and at API startup; the
  API middleware logs one line per request (`method`, `path`, `status`,
  `request_id`, `duration`), and the training pipeline, model fit, artifact
  loading, and API service load emit summary records. Documented under
  README "Logging".

### Fixed

- Removed a duplicated docstring/initializer in Spotify `fetch_audio_features`
  and collapsed redundant `except (ValueError, Exception)` tuples in the CLI.
- Renamed `evaluate-tracks --report-path` to `--report-name`: the value names
  a file stored under the reports directory, not a path.
- Removed the unused `config.PROCESSED_DATA_DIR`.
- Extracted the dashboard ablation-summary table shaping
  (`_ablation_ranking_rows`) and body rendering
  (`_render_ablation_summary_body`) into testable units.

### Tests

- Covered the LTR API route (success + 422), the LTR dashboard branch, the
  ablation-summary rendering, quality-threshold parsing edge cases, track
  interaction/metadata validator branches, and the truncated track-catalog
  caption; added logging tests for `configure_logging`, middleware request
  lines, ALS training logs, and artifact-load logs. 611 tests, ~96.8%
  statement coverage.

## [0.12.0] - 2026-09-12

### Added

- Hybrid taste-driven track recommendations (`method=hybrid`): a small ALS
  model fit on artist plays captures collaborative artist taste, which is
  mapped to per-track affinities and blended with the audio-feature
  ("content") similarity via `content.py::hybrid_scores`. `content_weight`
  balances the two signals (0.0 = pure artist taste, 1.0 = pure content),
  matching the artist-side hybrid math. Hybrid recommendations carry
  `score_components` with content, collaborative, and hybrid scores.
- Hybrid support across surfaces: `recommend_tracks_for_user`,
  `RecommenderService.recommend_tracks` (`method`, `content_weight`),
  `GET /tracks/recommend/{user_id}?method=hybrid&content_weight=0.5`, the
  `track-recommendations --method --content-weight` CLI, and a Method
  selector + Artist-taste slider on the dashboard Tracks tab.
- Hybrid track evaluation: `evaluate_track_holdout` and the
  `evaluate-tracks` CLI accept `method`/`content_weight`, training the
  artist-taste model on each fold's held-in interactions so the hybrid arm
  can be costed side-by-side with the pure content arm.
- New track helpers for reuse: `train_track_artist_taste`,
  `artist_taste_scores_for_user`, and `artist_affinity_to_track_scores`.

## [0.11.0] - 2026-09-11

### Added

- Track recommendation explanations (`explain`), matching the artist side.
  `recommend_tracks_for_user` attaches reasons citing the listened tracks
  that drove each recommendation; the service, API
  (`GET /tracks/recommend/{user_id}?explain=true`), CLI
  (`track-recommendations --explain`), and dashboard Tracks tab (Why column)
  all expose them.
- Cold-start fallback for track recommendations: users with a listening
  profile but no track history receive popular tracks
  (`strategy: popular_fallback`) instead of an empty list.
- Track recommendations honor the artifact champion ranking config via
  `_ranking_overrides` when callers omit `popularity_penalty`, `diversity`,
  or `include_listened`, matching the artist surfaces.
- `explanation_coverage` and `serendipity_at_k` metrics in track holdout
  evaluation and `evaluate-tracks` output, for parity with the artist
  evaluator.

## [0.10.0] - 2026-09-11

### Added

- Popularity penalty and diversity knobs for track recommendations, matched to
  the artist side. `recommend_tracks_for_user` applies an MMR-style diversity
  pass over the audio-feature matrix and a play-ranked popularity penalty; the
  service (`popularity_penalty`, `diversity`), API
  (`GET /tracks/recommend/{user_id}?popularity_penalty=&diversity=`), CLI
  (`track-recommendations --popularity-penalty --diversity`), and dashboard
  Tracks tab all expose them.
- Audio-feature matrix persisted on `TrackServingResources` (`feature_matrix`)
  and validated on artifact bundles so diversity reranking works for both
  live-CSV and bundled track resources.
- Tunable track evaluation: `evaluate_track_holdout` and `evaluate-tracks`
  forward the same popularity-penalty and diversity knobs into the holdout
  recommender for tuning experiments.
- Persistent track evaluation reports: `write_track_report` /
  `load_track_report` and `evaluate-tracks --report-path`, storing the run
  configuration and per-arm metrics under a stable schema in `reports/`.
- Top tracks baseline surfaced across surfaces: the `popular_tracks` helper,
  `RecommenderService.popular_tracks`, `GET /tracks/popular`, and a
  `popular-tracks` CLI command.

## [0.9.0] - 2026-09-08

### Added

- Popularity-baseline comparison for track evaluation
  (`evaluate-tracks --compare-baseline`), reporting similarity and popularity
  arms side by side on identical holdouts.

## [0.8.0] - 2026-09-08

### Added

- Browsable track catalog table in the dashboard Tracks tab, backed by the
  shared `browse_tracks` service contract.
- Bumped pinned GitHub Actions (`setup-uv` v10.0.1, `docker/login-action`
  v4.6.0, `actions/attest` v4.2.2; verified SHAs), closing dependabot PRs
  #1, #3, and #4.

## [0.7.0] - 2026-09-07

### Added

- Track catalog browsing: `RecommenderService.browse_tracks` with free-text
  search, artist filter, and pagination, served via `GET /tracks/catalog`.
- Bumped the `actions/upload-artifact` pin to v7.0.1 (verified SHA),
  superseding dependabot PR #5.

## [0.6.0] - 2026-09-06

### Added

- Average popularity and novelty in track holdout evaluation, backed by
  bundled track popularity statistics (`build_track_stats`).
- Track catalog search in the dashboard Tracks tab.
- JUnit failure annotations in CI (`mikepenz/action-junit-report`, pinned) so
  failing tests are visible per-test without log access.

### Fixed

- Pinned plain CLI rendering in tests (`_TYPER_FORCE_DISABLE_TERMINAL`), fixing
  Linux CI failures where Typer's forced terminal output dropped usage-error
  messages under `FORCE_COLOR`/`GITHUB_ACTIONS`.
- Sorted ablation arms for deterministic knob ordering across runs.

## [0.5.0] - 2026-09-05

### Added

- Strict mypy checks and a 75% coverage regression gate in local, CI, and
  release verification.
- Corruption tests for artifact mappings, numeric model data, content
  alignment, artist statistics, provenance, and training configuration.
- API request correlation and processing-time response headers.
- OpenAPI metadata, CORS support with exposed request headers, and enriched
  FastAPI documentation.
- Makefile, `.editorconfig`, `.pre-commit-config.yaml`, Dependabot, and
  GitHub issue/PR templates for smoother contributor workflows.
- Type-safe method parameter constrained to `Literal["als", "content", "hybrid"]`.
- Custom `__repr__` for `RecommenderArtifact` and `ContentArtifacts`.
- `py.typed` marker for downstream mypy support.
- `CONTRIBUTING.md` and `SECURITY.md`.
- Locked pre-commit tooling in the development dependency group.
- Regression coverage for metadata text preservation, API metadata/CORS headers,
  sparse recommendation serving, and path validation.
- CLI command tests covering every training, evaluation, serving, and catalog
  command, including artifact inspection, onboarding profiles, and session mixes.
- An end-to-end `train_and_save_model` happy-path test that trains on temporary
  CSVs and verifies model, mappings, and artifact bundle persistence.
- Regression tests confirming a failed training run leaves no partial model or
  artifact bundle on disk.
- Missing-value validation coverage for every interaction and artist metadata
  column, closing the remaining schema-validation gaps.
- Regression coverage for every FastAPI route happy path and service-error
  response, raising the API module to full coverage.
- Edge-case coverage for scoring, ranking, model, preprocessing, content, and
  tracking modules, including zero-score matrices, degenerate profiles,
  corrupt artifact loads, and failed experiment-tracking wraps.
- Regression tests for request-body-limit middleware, the Streamlit dashboard
  submit flows and fallback/error states, and artifact validation error paths
  (version, mapping, factor, content, metadata, and fingerprint mismatches).
- CLI regression tests for metadata preparation, GPU fallback echoes, tracked
  artifact logging, and per-command service-error exits, plus evaluation
  empty-input and metadata-fallback paths.
- GPU fallback coverage for ALS training, including failed GPU model creation,
  failed GPU fitting, and `to_cpu` conversion on CPU-only hardware.
- Regression tests for the legacy artifacts loader success path, the POSIX
  parent-directory fsync branch, and the package version fallback when the
  distribution metadata is unavailable.
- A long-tail `unexpectedness_at_k` metric that reports the share of
  recommendations surfacing from the bottom half of the popularity ranking.
  It is included in evaluation summaries, the CLI `evaluate` output, and the
  tracked metrics document, making the popularity bias tradeoff explicit.
- A relevance-weighted `serendipity_at_k` metric that measures the share of
  relevant top-K recommendations also coming from the popularity long tail,
  completing the serendipity dimension of the evaluation summary.
- Full 100% statement coverage, with the two `__main__` entry points and one
  defensive guard documented as deliberate exclusions, plus a `make coverage`
  target for the report.
- An A/B parameter-settings comparison harness (`compare_parameter_settings`)
  that evaluates the same ALS pipeline under different reranking settings on
  identical holdouts, exposed through `evaluate --compare-settings` with
  `label:key=value,...` parsing and labeled metric rows in the CLI output and
  the tracked metrics document.
- Batting-average style strategy selection for A/B results:
  `select_winning_strategies` names the per-metric winner between settings and
  `strategy_leaderboard` ranks them by metrics won (ties broken by NDCG@K). The
  `evaluate --compare-settings` output ends with the winner-by-metric list and
  the overall best setting.
- Champion ranking settings promoted into serving: `train` records
  `popularity_penalty`, `diversity`, and `include_listened` as the artifact's
  ranking configuration, and `recommend_user`, `recommend_user_als`, and
  `recommend_session` serve them as defaults whenever the caller does not pass
  explicit knobs. Legacy bundles without the configuration fall back to neutral
  settings.
- `evaluate --promote-winner`: after an `--compare-settings` A/B run, retrain
  the model with the overall winner's ranking parameters and save the new
  artifact, closing the loop from comparison to serving.
- `evaluate --learn-to-rank`: a pointwise learning-to-rank re-ranker built on a
  ridge regressor over collaborative/popularity/user features. It is trained on
  each fold and used to re-rank the ALS candidates, reported as an additional
  `ltr` arm in the holdout metrics. The new `music_recommender.ltr` module
  exposes `train_ltr_ranker` and `rank_with_ltr`.
- Learning-to-rank in serving: `train` trains the re-ranker and persists it on
  the artifact bundle alongside the collaborative model. `recommend_user_ltr`
  serves it, re-ranking the ALS candidates with the `ltr_personalized` strategy
  (falling back to the standard hybrid list when no ranker is bundled), and it
  is exposed through `recommend-user --ltr`. Champion ranking settings still
  apply to the underlying candidates.
- `evaluate --ablations`: ablation / feature-importance reporting for the
  champion ranking configuration. `build_ablation_settings` derives a
  `no_<knob>` arm for every active knob plus a fully neutral `no_ranking` arm,
  `ablation_importances` reports each arm's signed per-metric delta versus the
  champion, and the CLI prints a knob importance ranking by total absolute
  impact. It is an alternative to `--compare-settings`/`--promote-winner` and
  cannot be combined with them.
- Persistent ablation reports: `evaluate --ablations` also writes a JSON report
  (arm metrics, signed deltas, and knob importance ranking under a stable
  schema) to `reports/`, overridden with `--report-dir`, so runs over different
  datasets produce comparable importance signals.
- `ablation-summary`: aggregates every persisted ablation report in a directory
  and reports, per knob, the mean/median/std of the total impact across runs
  plus the run count, ranked by mean impact. This surfaces which knobs stay
  important across datasets versus whose effect depends on the data. The
  aggregated result is also persisted as a JSON report (default
  `reports/ablation_summary.json`, overridden with `--summary-path`) for
  downstream dashboards and CI.
- API endpoint `GET /evaluation/ablation-summary` serves the persisted
  aggregated knob-importance summary (HTTP 404 when no summary has been
  generated, HTTP 422 if the file is unparsable), so dashboards and CI can pull
  the aggregated evaluation deterministically.
- API endpoint `GET /recommend/user/{user_id}/ltr` serves LTR re-ranked
  recommendations; dashboard adds an LTR toggle and an Ablation Summary tab.
- `evaluate --min-quality-threshold`: CI quality gate for `--promote-winner`
  that only promotes when all `metric=value` thresholds are met.
- Spotify integration (`music_recommender.spotify`, optional `spotify` extra)
  with artist/track search, top tracks, related artists, and audio features,
  plus `spotify-*` CLI commands.
- `spotify-import-catalog`: imports Spotify top tracks plus audio features for
  a list of artists into a track-metadata CSV matching the track contract.
- `evaluate --fail-on-quality-gate`: strict mode for `--min-quality-threshold`
  that exits non-zero when the winning setting misses the bar, for CI gates.
- Scheduled `quality-gate` workflow: weekly/on-demand gated A/B promotion plus
  knob ablation, uploading the reports via a pinned `actions/upload-artifact`.
- Optional `track_bundle` on the versioned artifact: `train` persists track
  interactions, the audio-feature matrix, similarity, and lookup; serving
  prefers the bundle and falls back to CSVs for legacy bundles.
- Track holdout evaluation (`music_recommender.track_evaluate`,
  `evaluate-tracks` CLI) reporting precision, recall, MAP, NDCG, and catalog
  coverage over repeated per-user splits.
- Track-level recommendations (`music_recommender.tracks`) with audio-feature
  content similarity, sample `sample_track_interactions.csv` /
  `sample_track_metadata.csv`, and `prepare-track-data`,
  `track-recommendations`, `similar-tracks` CLI commands.
- Track serving in `RecommenderService` (`recommend_tracks`, `similar_tracks`
  with enriched track metadata), API endpoints `GET /tracks/recommend/{user_id}`
  and `GET /tracks/similar/{track_id}`, and a dashboard Tracks tab with
  track-aware result rendering.
- `PLAN.md` with completed phases, in-progress work, and next steps.

### Fixed

- Model persistence is deferred until metadata validation, content build, and
  artifact construction all succeed, so a failed retraining run no longer leaves
  a new model file next to an older artifact bundle.
- CLI demo now surfaces genuine runtime training failures as clean errors with a
  non-zero exit code, matching the training command.
- ALS training now rejects complex-valued interaction matrices instead of
  letting them slip through numeric validation.
- Artifact age formatting treats naive (timezone-less) timestamps as UTC rather
  than crashing on the timestamp subtraction.
- CLI training now surfaces genuine runtime training failures as clean errors
  with a non-zero exit code instead of dumping a raw traceback.
- String dtype loading to preserve leading zeroes in artist identifiers.
- `MUSIC_RECOMMENDER_ROOT` and MLflow tracking URI resolution now strips
  surrounding whitespace.
- `atomic_joblib_dump` fsyncs temporary files and parent directory for durability.
- Training and evaluation CLI commands report errors without tracebacks.
- Large request bodies are rejected above 64 KiB before unbounded parsing.
- API user and artist path identifiers now reject blank or oversized values.
- `load_model` and `load_mappings` now raise actionable errors on corrupt files.
- Dataset fingerprinting gracefully handles unreadable source files.
- Shared `is_finite_number` utility and `_weighted_profile` empty-array guard.
- `zip()` calls in evaluation now use `strict=True` to catch length mismatches.
- Eliminated loop variable shadowing in `get_similar_artists`.

### Changed

- CORS origins now configurable via `CORS_ORIGINS` environment variable.
- Pre-commit uses the project's locked environment for strict mypy checks.
- CI validates `compose.yaml` before building containers.
- Ruff lint expanded with `SIM` and `C4` simplifications.
- Docker images now use `STOPSIGNAL SIGTERM` for graceful shutdown.
- Dockerfiles declare explicit `STOPSIGNAL SIGTERM` for graceful shutdown.
- Consistent `from __future__ import annotations` across all source modules.
- Recommendation request bodies now reject unknown fields, implicit type
  coercion, blank values, oversized strings, and unbounded preference lists.
- Training, filtering, ranking, and evaluation parameters now fail fast with
  actionable errors before expensive processing or artifact writes.
- Artifact loading now verifies mapping bijections, finite matrices and factors,
  content metadata alignment, popularity statistics, dataset fingerprints, and
  recorded training settings.
- Sparse content recommendations avoid dense matrix conversion when diversity
  reranking is disabled.
- MIME type for content metadata is validated before parsing.
- Clarified implicit library inverted naming for latent factors.
- Deduplicated `Recommendation` type alias into `recommend` module.
- Removed duplicate `METADATA_TEXT_COLUMNS` constant.
- Extracted magic explanation limits into named constants in `content.py`.
- Removed redundant `(?u)` regex flag from TF-IDF pattern.

## [0.4.0] - 2026-07-24

### Added

- Hybrid ALS and metadata recommendations with score explanations.
- Cold-start taste profiles and session-aware recommendation mixes.
- Popularity penalty, diversity reranking, novelty, coverage, and baseline
  evaluation metrics.
- Searchable, filterable, paginated artist catalog shared by the API and
  dashboard.
- Interactive Streamlit recommendation studio.
- Optional MLflow tracking for training and evaluation parameters, metrics,
  tags, and artifacts.
- Versioned serving bundles, CLI entry point, FastAPI service, container health
  checks, and Docker Compose deployment.
- Tag-gated GitHub Container Registry publication for API and dashboard images
  with OCI metadata and provenance attestations.

### Changed

- Interaction ingestion now preserves text identifiers, trims whitespace, and
  aggregates duplicate user-artist signals before training.
- Evaluation normalizes event rows before holdout splitting to prevent
  duplicate-pair leakage.
- Model, mapping, and serving artifacts are written atomically and validated
  structurally on load.
- API result sizes are bounded and invalid artifacts produce readiness failures
  without taking down the liveness route.
- Python package metadata now includes license, authorship, classifiers,
  keywords, project links, and an installed version command.

### Security

- Upgraded FastAPI, Starlette, and idna to patched dependency floors.
- Migrated API tests to HTTPX2 and made the full suite warning-free.
- Pinned release actions to immutable commits and disabled reusable caches and
  persisted checkout credentials in artifact-publishing jobs.

[Unreleased]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.13.0
[0.12.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.12.0
[0.11.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.11.0
[0.10.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.10.0
[0.9.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.9.0
[0.8.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.8.0
[0.7.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.7.0
[0.6.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.6.0
[0.5.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.5.0
[0.4.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.4.0
