# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Mohamed-ahmed-shokry/Music-Recommender-System/releases/tag/v0.4.0
