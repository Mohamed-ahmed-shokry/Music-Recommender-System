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
- `CONTRIBUTING.md` and `SECURITY.md`.
- Locked pre-commit tooling in the development dependency group.
- Regression coverage for metadata text preservation, API metadata/CORS headers,
  sparse recommendation serving, and path validation.

### Changed

- Recommendation request bodies now reject unknown fields, implicit type
  coercion, blank values, oversized strings, and unbounded preference lists.
- Training, filtering, ranking, and evaluation parameters now fail fast with
  actionable errors before expensive processing or artifact writes.
- Artifact loading now verifies mapping bijections, finite matrices and factors,
  content metadata alignment, popularity statistics, dataset fingerprints, and
  recorded training settings.
- CLI training and evaluation workflows now report validation and missing-file
  failures without exposing internal tracebacks and route errors to stderr.
- API request bodies are rejected above 64 KiB before unbounded parsing.
- Ruff lint now includes `SIM` and `C4` simplifications; `.dockerignore` and
  `.gitignore` aligned and compose services include healthchecks and resource
  limits.
- CI validates `compose.yaml` before building containers.
- Pre-commit uses the project’s locked environment for strict mypy checks.

### Fixed

- Metadata CSV loader now preserves text types for whitespace handling parity
  with interaction loading.
- `MUSIC_RECOMMENDER_ROOT` and MLflow tracking URI resolution now strips
  surrounding whitespace.
- Atomic artifact writes now fsync file and parent directory for durability.
- Serving no longer densifies content vectors unless diversity reranking is
  requested.
- API user and artist path identifiers now reject blank or oversized values.

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
