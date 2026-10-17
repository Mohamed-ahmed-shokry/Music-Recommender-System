# Milestone 0.30.0 Roadmap — Phase 138

## Phase 138: Observability Dashboard Templates, Telemetry Integration Suite, and CI Green Parity (0.30.0)

### 1. Objective
Complete Milestone 0.30.0 by repairing broken CI formatting across all affected modules, delivering production-grade Grafana dashboard provisioning and templates visualizing all Prometheus telemetry, building a validator tool and end-to-end integration tests for telemetry and alerting pipelines, updating release documentation, and tagging 0.30.0 readiness with 100% green CI.

### 2. Scope
1. **CI Repair & Formatting Compliance**:
   - Resolve repository-wide `ruff format --check .` failures across 13 source and test files.
   - Commit formatting micro-updates per file or logical module.
   - Ensure local parity for all CI steps (`ruff format --check`, `ruff check`, `mypy`, `pytest`, `uv build`).
2. **Grafana Dashboard Templates & Provisioning**:
   - `configs/grafana/dashboards/music-recommender-overview.json`: Complete Grafana dashboard definition with panels for:
     - Service health & uptime (`music_recommender_up`)
     - Streaming feedback queue (depth, capacity, utilization ratio, enqueue/flush/drop counters)
     - Maintenance daemon (worker running, sweep counts, records folded, sweep latencies, drift evaluations/violations, rollbacks)
     - HTTP request rate, latency distribution (p50, p95, p99), endpoint traffic breakdown, error status codes
     - Cold-start bandit arm weights and champion status
     - Artifact inventory gauges (users, artists, interactions)
     - Alert status summary panel
   - `configs/grafana/provisioning/dashboards/dashboards.yaml`: Standard Grafana provider configuration.
   - `configs/grafana/provisioning/datasources/datasources.yaml`: Default Prometheus datasource configuration.
   - `compose.yaml`: Observability service definitions for Prometheus and Grafana.
3. **Telemetry & Dashboard Validator**:
   - `validate_dashboard_metrics` helper in `src/music_recommender/telemetry.py` verifying dashboard panels against known Prometheus metric families.
   - `music-recommender validate-dashboard` CLI command in `src/music_recommender/cli.py`.
4. **End-to-End Observability Integration Suite**:
   - `tests/test_observability_integration.py` verifying full telemetry lifecycle: Prometheus scrape format, alert rule thresholds, dashboard metric bindings, and CLI validation.
5. **Milestone 0.30.0 Release Packaging & Documentation**:
   - Bump version to `0.30.0` in `pyproject.toml`.
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, and `docs/PROGRESS.md`.
   - Ensure 100% green CI locally and remotely on GitHub Actions.

### 3. Acceptance Criteria
- **AC-1**: All 13 repository source and test files pass `uv run ruff format --check .` with zero formatting differences.
- **AC-2**: Production Grafana dashboard template `configs/grafana/dashboards/music-recommender-overview.json` is valid JSON and covers all 7 telemetry subsystems.
- **AC-3**: Grafana provisioning configurations (`configs/grafana/provisioning/dashboards/dashboards.yaml` and `configs/grafana/provisioning/datasources/datasources.yaml`) are valid YAML and correctly configured for Prometheus integration.
- **AC-4**: `compose.yaml` defines optional observability services (`prometheus` and `grafana`) with valid Docker Compose syntax and passes `docker compose config --quiet`.
- **AC-5**: Dashboard validator (`validate_dashboard_metrics` in `src/music_recommender/telemetry.py` and `music-recommender validate-dashboard` CLI command) verifies metric references against known metric families, reporting invalid metric names or missing panels with clear errors and exit codes.
- **AC-6**: Integration test suite `tests/test_observability_integration.py` verifies Prometheus scrape compatibility, Grafana dashboard metric references, and alert rule alignment with 100% passing tests.
- **AC-7**: Milestone version bumped to `0.30.0` in `pyproject.toml`, distribution builds cleanly (`uv build`), and `README.md`, `CHANGELOG.md`, `PLAN.md`, and `docs/PROGRESS.md` document Phase 138 and Milestone 0.30.0.
- **AC-8 (Mandatory)**: All local CI-parity checks (`ruff format --check .`, `ruff check .`, `mypy`, `pytest`, `docker compose config --quiet`, `uv build`) and remote CI workflow runs pass 100% green.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run pytest tests/test_observability_integration.py`
- `uv run pytest -q --no-cov`
- `docker compose config --quiet`
- `uv build`
- GitHub Actions CI workflow on `phase/138-dashboards-and-release` passing 100%.

### 5. Exclusions
- Kubernetes Helm charts or Operator CRDs (deferred to Milestone 0.31.0).
- External cloud monitoring integrations like Datadog or CloudWatch agent configs (out of scope for pure Prometheus/Grafana stack).

### 6. Micro-Task Breakdown
- [x] Task 1: Initialize Phase 138 roadmap and update `PLAN.md`.
- [x] Task 2: Format existing source files (`bandit.py`, `cli.py`, `dashboard.py`, `service.py`, `telemetry.py`).
- [x] Task 3: Format existing test files (`test_api.py`, `test_bandit.py`, `test_bandit_snapshots.py`, `test_cli.py`, `test_config.py`, `test_dashboard.py`, `test_service.py`, `test_telemetry.py`).
- [x] Task 4: Verify formatting and CI parity locally (`ruff format --check .`, `ruff check .`).
- [x] Task 5: Create Grafana dashboard template `configs/grafana/dashboards/music-recommender-overview.json`.
- [x] Task 6: Create Grafana provisioning configurations `dashboards.yaml` and `datasources.yaml`.
- [x] Task 7: Update `compose.yaml` with Prometheus and Grafana observability services and verify with `docker compose config --quiet`.
- [x] Task 8: Implement dashboard metric validator helper in `src/music_recommender/telemetry.py`.
- [x] Task 9: Implement `validate-dashboard` CLI command in `src/music_recommender/cli.py`.
- [x] Task 10: Implement unit tests for dashboard validation in `tests/test_telemetry.py` and `tests/test_cli.py`.
- [x] Task 11: Implement end-to-end observability integration test suite `tests/test_observability_integration.py`.
- [x] Task 12: Bump package version to `0.30.0` in `pyproject.toml` and test `uv build`.
- [x] Task 13: Update `README.md` and `CHANGELOG.md` with 0.30.0 observability guides and release notes.
- [ ] Task 14: Update `docs/PROGRESS.md` and `PLAN.md` marking Phase 138 and Milestone 0.30.0 complete.
- [ ] Task 15: Run full CI parity suite locally, push micro-commits, monitor remote GitHub Actions run to green, and open draft PR.

