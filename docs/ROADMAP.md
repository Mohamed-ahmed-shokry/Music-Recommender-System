# Milestone 0.32.0 Roadmap — Phase 144

## Phase 144: Maintenance Daemon Operational Hardening (0.32.0)

### 1. Objective
Make the 0.31.0 `maintenance-daemon` deployable and self-monitoring end to end:
the daemon writes a fresh JSON heartbeat the Docker healthcheck can validate
(the previous compose healthcheck polled an HTTP endpoint the daemon never
serves, so the `maintenance` container was permanently unhealthy), the
dashboard renders live maintenance worker status, and milestone 0.32.0 is
released with 100% green CI.

### 2. Scope
1. **Daemon heartbeat (`src/music_recommender/cli.py`)**:
   - New `--heartbeat-path` option on `maintenance-daemon` (default: disabled).
   - On start and after every daemon cycle, write a JSON heartbeat file:
     `{"timestamp": <epoch>, "interval_seconds": ..., "running": true, ...
     worker metric snapshot}` honoring the configured parent directories.
   - Hardened `while True: time.sleep(interval)` loop replaced by the same
     beat-then-sleep cycle so the heartbeat reflects loop liveness.
2. **Dashboard visibility (`src/music_recommender/dashboard.py`)**:
   - "Cold-Start Bandit" tab renders a maintenance worker summary block
     (running, cycles, sweeps, records folded, errors) from
     `service.maintenance_status()` when a worker is active; silent when none.
3. **Compose hardening (`compose.yaml`)**:
   - `maintenance` service healthcheck reads `/tmp/maintenance_heartbeat.json`
     and fails when the heartbeat is older than 2x interval (a util-only
     python snippet); daemon command passes `--heartbeat-path /tmp/...`.
4. **Tests & docs**:
   - Heartbeat unit tests in `tests/test_cli.py`; dashboard tests in
     `tests/test_dashboard.py` (worker present / absent); YAML parse check.
   - README maintenance section updated (heartbeat flag + healthcheck notes);
     CHANGELOG 0.32.0; `docs/PROGRESS.md` decision log.
5. **Release 0.32.0**:
   - Version bump in `pyproject.toml`, `uv lock`, `uv build`, full CI-parity
     suite, push, draft PR, green remote CI.

### 3. Acceptance Criteria
- **AC-1**: `maintenance-daemon --heartbeat-path <p>` creates and refreshes a
  parseable JSON heartbeat with `timestamp`, `interval_seconds`, and worker
  metrics; verified by unit tests.
- **AC-2**: The compose `maintenance` healthcheck validates heartbeat
  freshness (no HTTP endpoint dependency) and `docker compose config` remains
  valid; verified via YAML parse and remote CI `Validate Compose configuration`.
- **AC-3**: Dashboard "Cold-Start Bandit" tab shows the maintenance worker
  summary when `maintenance_status()` returns metrics and stays quiet without;
  verified by `tests/test_dashboard.py`.
- **AC-4**: Full suite green: `uv run pytest -q` (≥1172 tests), `ruff check .`,
  `ruff format --check .`, `mypy`, coverage ≥75%.
- **AC-5**: Version is `0.32.0`, distributions build via `uv build`.
- **AC-6 (mandatory)**: All local CI-parity checks and remote CI workflow runs
  pass 100%.

### 4. Validation & CI Plan
- Local parity (matching `.github/workflows/ci.yml`):
  `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy`,
  `uv run --extra dashboard --extra tracking --extra spotify pytest --cov`,
  `uv build`; compose YAML parse via `python -c "import yaml..."` (docker
  unavailable locally; remote CI covers `docker compose config`).
- End-to-end: run `maintenance-daemon --heartbeat-path` briefly and inspect
  the heartbeat file; render the dashboard tab against the artifact.

### 5. Exclusions
- Cron-style scheduler expressions (fixed `--interval` retained; reversible
  follow-up) — no codebase evidence of a scheduler abstraction to extend.
- Alertmanager routing (logged in `PLAN.md` next-steps; 0.30.0 webhook
  dispatch already covers notification delivery).

### 6. Micro-Task Breakdown
1. `docs(roadmap)` — this file + PROGRESS reset (this commit).
2. `feat(cli)` — heartbeat helper `write_maintenance_heartbeat`.
3. `test(cli)` — heartbeat unit tests.
4. `feat(cli)` — wire `--heartbeat-path` into `maintenance_daemon` loop.
5. `test(cli)` — daemon end-to-end heartbeat test (short interval, interrupt).
6. `fix(compose)` — heartbeat-based healthcheck + `--heartbeat-path` command.
7. `docs(readme)` — heartbeat & healthcheck documentation.
8. `feat(dashboard)` — maintenance worker summary block in bandit tab.
9. `test(dashboard)` — render tests (worker present / absent).
10. `test(compose-yard)` — YAML parse validation test if not already covered.
11. `docs(readme)/docs(changelog)/chore(release)` — 0.32.0 release packaging.
12. `docs(progress)` — finalize decision log; push; verify remote CI green.

---

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

