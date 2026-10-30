# Milestone 0.35.0 Roadmap — Phase 159 & 160

## Phase 159 & 160: Alert Dead-Letter Queue (DLQ), Persistent Spillover Storage, Incident Sequence Replay, and Operator Audit Tooling (0.35.0)

### 1. Objective
Harden the production alerting and webhook architecture by implementing persistent disk spillover for alert events evicted from the in-memory ring buffer, a dedicated Dead-Letter Queue (DLQ) for malformed, unparseable, or rejected alert notifications with audit metadata, a sequential alert replay engine for incident postmortems and staging drills, operator CLI inspection commands (`alert-dlq` and `replay-alerts`), and Streamlit dashboard visibility, releasing Milestone 0.35.0 with 100% green CI.

### 2. Scope
1. **Dead-Letter & Spillover Data Structures (`src/music_recommender/alerting.py`)**:
   - `DeadLetterAlertRecord`: dataclass with timestamp, reason, raw_payload, channel, and error details.
   - `AlertSpilloverStorage`: thread-safe append-only file storage with JSON Lines format, reading with limits and filters, and DLQ purge support.
   - `AlertHistoryBuffer` eviction integration: automatic spillover to disk storage when configured.
   - `AlertReplayEngine`: reads past alert events or DLQ records from JSON/JSONL, dry-run validation, and timed sequential dispatch against webhook endpoints.
2. **Service Integration (`src/music_recommender/service.py`)**:
   - Support `alert_storage_dir`, `spill_on_evict`, DLQ recording, combined history queries (`include_spillover=True`), and alert replay methods.
3. **FastAPI Webhook & DLQ Endpoints (`api/main.py`)**:
   - Capture unparseable / invalid webhook payloads directly into the DLQ file (`POST /alerts/webhook`).
   - `GET /alerts/dlq`: inspect captured dead-letter items.
   - `DELETE /alerts/dlq`: clear dead-letter records.
   - `POST /alerts/replay`: trigger alert replay payloads for staging integration tests.
   - Support `?include_spillover=true` query parameter on `GET /alerts/history`.
4. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender alert-dlq`: inspect and purge dead-letter items in table or JSON format.
   - `music-recommender replay-alerts`: replay serialized alert notifications against a webhook URL with configurable interval and dry-run flag.
   - Support `--include-spillover` flag in `music-recommender alert-history`.
5. **Dashboard Visibility (`src/music_recommender/dashboard.py`)**:
   - Surface DLQ status badge and spillover count in the sidebar alerting expander.
6. **Comprehensive Tests**:
   - Unit tests in `tests/test_alerting.py` for storage, DLQ capture, eviction spillover, and replay engine.
   - Integration tests in `tests/test_api.py` for DLQ capture, purge, replay endpoint, and combined history.
   - CLI tests in `tests/test_cli.py` for `alert-dlq` and `replay-alerts`.
   - Dashboard tests in `tests/test_dashboard.py` for DLQ badge rendering.
7. **Documentation & Release 0.35.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.35.0` in `pyproject.toml`, refresh `uv.lock`, verify `uv build`.

### 3. Acceptance Criteria
- [ ] AC-1: `DeadLetterAlertRecord` and `AlertSpilloverStorage` persist spilled alert events and unparseable dead-letter payloads in append-only JSONL files.
- [ ] AC-2: `AlertHistoryBuffer` evicts to disk spillover when `spill_on_evict` is enabled without data loss.
- [ ] AC-3: FastAPI endpoints `GET /alerts/dlq`, `DELETE /alerts/dlq`, `POST /alerts/replay`, and `GET /alerts/history?include_spillover=true` function correctly.
- [ ] AC-4: CLI commands `music-recommender alert-dlq` (list/purge) and `music-recommender replay-alerts` (live/dry-run) operate cleanly.
- [ ] AC-5: Streamlit dashboard sidebar surfaces DLQ warning indicators when dead-letter records exist.
- [ ] AC-6: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Exclusions
- S3 / remote cloud object store alert backup (local disk JSONL fits container and persistent volume needs without adding external cloud SDK dependencies).

### 6. Micro-Task Breakdown
- [x] Task 1: Define Phase 159 roadmap, plan, and initialize progress record.
- [x] Task 2: Implement `DeadLetterAlertRecord` and `AlertSpilloverStorage` in `src/music_recommender/alerting.py`.
- [x] Task 3: Implement `AlertReplayEngine` and replay report models in `src/music_recommender/alerting.py`.
- [x] Task 4: Integrate eviction spillover in `AlertHistoryBuffer` in `src/music_recommender/alerting.py`.
- [x] Task 5: Add unit tests for storage, DLQ, and replay in `tests/test_alerting.py`.
- [x] Task 6: Integrate spillover storage, DLQ, and replay in `RecommenderService` in `src/music_recommender/service.py`.
- [x] Task 7: Implement DLQ ingestion, `GET /alerts/dlq`, `DELETE /alerts/dlq`, and `POST /alerts/replay` in `api/main.py`.
- [x] Task 8: Add API integration tests in `tests/test_api.py`.
- [x] Task 9: Implement `alert-dlq` and `replay-alerts` CLI commands in `src/music_recommender/cli.py`.
- [x] Task 10: Add unit tests for `alert-dlq` and `replay-alerts` in `tests/test_cli.py`.
- [x] Task 11: Add DLQ badge and spillover visibility in `src/music_recommender/dashboard.py`.
- [x] Task 12: Add dashboard DLQ tests in `tests/test_dashboard.py`.
- [x] Task 13: Update `README.md` and `CHANGELOG.md` with 0.35.0 documentation.
- [x] Task 14: Bump package version to `0.35.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 15: Final CI parity verification, push, open draft PR, monitor remote CI to green.

---

# Milestone 0.34.0 Roadmap — Phase 154

## Phase 154: Alertmanager Webhook Ingestion, Alert History, and Operator Audit Tooling (0.34.0)

### 1. Objective
Complete the production alerting lifecycle by implementing the webhook receiver endpoint `POST /alerts/webhook` and history inspection endpoint `GET /alerts/history` in FastAPI (`api/main.py`), supporting both standard Prometheus Alertmanager webhook payloads and direct transition payloads, maintaining a thread-safe in-memory ring-buffer audit history of received alert transitions, providing operator CLI tooling (`alert-history`) and Streamlit dashboard alert history visibility, and releasing Milestone 0.34.0 with 100% green CI.

### 2. Scope
1. **Webhook Ingestion & History Data Structures (`src/music_recommender/alerting.py`)**:
   - `ReceivedAlertEvent`: dataclass representing a parsed alert notification with timestamp, alertname, status, severity, channel, summary, and labels.
   - `parse_alert_payload`: robust parser supporting both Alertmanager v4 JSON format and direct single-transition payloads.
   - `AlertHistoryBuffer`: thread-safe ring buffer storing the latest $N$ alert events with query filtering (`limit`, `channel`, `status`).
2. **FastAPI Webhook & History Endpoints (`api/main.py`)**:
   - `POST /alerts/webhook`: handles incoming Alertmanager notifications and direct test alerts, logs events, records them in the history buffer, and returns status.
   - `GET /alerts/history`: returns recorded alert events with optional filtering (`limit`, `channel`, `status`).
3. **Service Integration (`src/music_recommender/service.py`)**:
   - Attach `alert_history` to `RecommenderService` and expose `get_alert_history()`.
4. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender alert-history`: view recorded alert events from CLI in table or JSON format.
5. **Dashboard Visibility (`src/music_recommender/dashboard.py`)**:
   - Render recent alert history in the dashboard sidebar expander.
6. **Comprehensive Tests**:
   - Unit tests in `tests/test_alerting.py` for payload parsing and history buffer.
   - API tests in `tests/test_api.py` for `POST /alerts/webhook` and `GET /alerts/history`.
   - CLI tests in `tests/test_cli.py` for `alert-history`.
   - Dashboard tests in `tests/test_dashboard.py`.
7. **Documentation & Release 0.34.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.34.0` in `pyproject.toml`, refresh `uv.lock`, verify `uv build`.

### 3. Acceptance Criteria
- **AC-1**: `parse_alert_payload` correctly parses standard Prometheus Alertmanager v4 payloads and direct transition payloads into structured `ReceivedAlertEvent` objects.
- **AC-2**: `AlertHistoryBuffer` is thread-safe, enforces capacity bounds, and supports filtering by channel, status, and limit.
- **AC-3**: `POST /alerts/webhook` accepts alert payloads with optional `?channel=` query param, returns HTTP 200, logs received alerts, and records to history.
- **AC-4**: `GET /alerts/history` returns recent alert events as JSON, honoring `limit`, `channel`, and `status` query parameters.
- **AC-5**: `music-recommender alert-history` CLI command outputs recorded alert events in table or JSON format with filtering options.
- **AC-6**: Streamlit dashboard sidebar renders recent alert events when history is available.
- **AC-7**: All unit and integration tests pass (≥1189 tests), `ruff check .` clean, `ruff format --check .` clean, `mypy` clean, coverage ≥75%.
- **AC-8**: Milestone version bumped to `0.34.0`, distributions build via `uv build`.
- **AC-9 (Mandatory)**: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- End-to-end API test: `POST /alerts/webhook` followed by `GET /alerts/history` using TestClient.
- End-to-end CLI test: `music-recommender test-alert` against local API, then `music-recommender alert-history`.
- Remote GitHub Actions CI run on `phase/154-alertmanager-webhook-receiver` passing 100%.

### 5. Exclusions
- External database persistence for alert history (in-memory ring buffer satisfies single-instance container requirements without adding external database dependencies).

### 6. Micro-Task Breakdown
- [x] Task 1: Initialize Phase 154 roadmap, plan, and progress record.
- [x] Task 2: Implement `ReceivedAlertEvent`, `parse_alert_payload`, and `AlertHistoryBuffer` in `src/music_recommender/alerting.py`.
- [x] Task 3: Unit tests for alert payload parser and history buffer in `tests/test_alerting.py`.
- [x] Task 4: Integrate `alert_history` into `RecommenderService` in `src/music_recommender/service.py`.
- [x] Task 5: Implement `POST /alerts/webhook` and `GET /alerts/history` endpoints in `api/main.py`.
- [x] Task 6: Add API tests for alert webhook and history endpoints in `tests/test_api.py`.
- [x] Task 7: Implement `alert-history` command in `src/music_recommender/cli.py`.
- [x] Task 8: Add unit tests for `alert-history` CLI command in `tests/test_cli.py`.
- [x] Task 9: Add recent alert event rendering to dashboard sidebar in `src/music_recommender/dashboard.py`.
- [x] Task 10: Add dashboard alert history tests in `tests/test_dashboard.py`.
- [x] Task 11: Update `README.md` and `CHANGELOG.md` with 0.34.0 webhook and history docs.
- [x] Task 12: Bump version to `0.34.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 13: Full CI-parity verification, push, open draft PR, monitor remote CI to green.

---

# Milestone 0.33.0 Roadmap — Phase 149

## Phase 149: Alertmanager Multi-Tier Routing, Operator Alert CLI, and Dashboard Alert Monitoring (0.33.0)

### 1. Objective
Complete the production alerting pipeline by delivering multi-tier Alertmanager routing configuration and generator, wiring Alertmanager into the Prometheus and Docker Compose observability stack with storm-prevention inhibition rules, providing operator CLI commands (`check-alerts` and `test-alert`) for alert inspection and synthetic webhook testing, surfacing live alert health in the Streamlit dashboard, and releasing Milestone 0.33.0 with 100% green CI.

### 2. Scope
1. **Alertmanager Configuration & Routing Generator (`src/music_recommender/alerting.py`)**:
   - Multi-tier alert routing data structures and generator `render_alertmanager_config`.
   - Tiered receivers: `critical` (immediate notification, short repeat interval), `warning` (batched notification, standard repeat interval), and default fallback.
   - Storm inhibition rules: inhibit warning alerts when `ServiceDown` is firing.
   - Shipped configuration in `configs/alertmanager.yml` matching the generator output.
   - CLI generator mode: `python -m music_recommender.alerting --alertmanager [PATH]`.
2. **Prometheus Alertmanager Wiring & Compose Integration (`configs/prometheus.yml`, `compose.yaml`)**:
   - `configs/prometheus.yml`: add `alerting: alertmanagers:` targeting `alertmanager:9093`.
   - `compose.yaml`: add `alertmanager` service under `observability` profile with healthcheck and volume mount for `configs/alertmanager.yml`.
3. **Operator Alert CLI (`src/music_recommender/cli.py`)**:
   - `music-recommender check-alerts`: evaluates alert rules against current service metrics or offline artifact, outputs formatted table or JSON, supports `--firing-only`, `--severity`, `--fail-on-critical`.
   - `music-recommender test-alert`: dispatches a synthetic alert notification to verify webhook endpoint connectivity.
4. **Dashboard Alert Visibility (`src/music_recommender/dashboard.py`)**:
   - Sidebar alert health badge and status indicators (e.g. "Alerts: Normal" or firing alert count).
   - Dedicated "Alerts & Health" expander or tab displaying active rules, thresholds, severity, and current evaluation status.
5. **Comprehensive Integration & Unit Tests**:
   - Tests in `tests/test_alerting.py` for Alertmanager config generation, inhibition rules, and generator CLI.
   - Tests in `tests/test_cli.py` for `check-alerts` and `test-alert`.
   - Tests in `tests/test_dashboard.py` for alert health rendering.
   - Observability integration tests in `tests/test_observability_integration.py` for Alertmanager YAML validity and Prometheus configuration.
6. **Documentation & Release 0.33.0**:
   - README alert operations section.
   - CHANGELOG 0.33.0 release notes.
   - Version bump to 0.33.0 in `pyproject.toml`, `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- **AC-1**: `render_alertmanager_config()` outputs valid Alertmanager YAML with multi-tier routing (critical vs warning), receivers, and inhibition rules; byte-identical to `configs/alertmanager.yml`.
- **AC-2**: `python -m music_recommender.alerting --alertmanager` writes valid Alertmanager config to path or stdout; tested with exit code 0.
- **AC-3**: `configs/prometheus.yml` defines Alertmanager target `alertmanager:9093` and `compose.yaml` defines `alertmanager` service with valid healthcheck and schema passing YAML verification.
- **AC-4**: `music-recommender check-alerts` evaluates live/offline metrics, reports rule statuses in table or JSON format, respects `--severity` and `--firing-only`, and exits with code 1 under `--fail-on-critical` when critical alerts fire.
- **AC-5**: `music-recommender test-alert` dispatches synthetic notification to the specified or configured webhook URL with error isolation and status reporting.
- **AC-6**: Dashboard renders alert health indicators and status view reflecting rule states cleanly.
- **AC-7**: All unit and integration tests pass (≥1176 tests), `ruff check .` clean, `ruff format --check .` clean, `mypy` clean, coverage ≥75%.
- **AC-8**: Milestone version is `0.33.0`, distribution builds via `uv build`.
- **AC-9 (Mandatory)**: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- YAML validation of `configs/alertmanager.yml`, `configs/prometheus.yml`, and `compose.yaml`.
- CLI manual execution: `music-recommender check-alerts --format json`, `music-recommender test-alert --dry-run`.
- Remote GitHub Actions CI run on `phase/149-alertmanager-routing` passing 100%.

### 5. Exclusions
- Third-party PagerDuty / Opsgenie API credentials or live paid notification services (handled via standard Alertmanager webhook receivers).
- Custom Alertmanager clustering mesh across multi-node networks (out of single-node compose scope).

### 6. Micro-Task Breakdown
- [x] Task 1: Initialize Phase 149 roadmap, plan, and progress record.
- [x] Task 2: Implement Alertmanager configuration generator and inhibition models in `src/music_recommender/alerting.py`.
- [x] Task 3: Add unit tests for Alertmanager config generator and CLI in `tests/test_alerting.py`.
- [x] Task 4: Ship `configs/alertmanager.yml` and wire Alertmanager into `configs/prometheus.yml` and `compose.yaml`.
- [x] Task 5: Implement `check-alerts` command in `src/music_recommender/cli.py`.
- [x] Task 6: Implement `test-alert` command in `src/music_recommender/cli.py`.
- [x] Task 7: Add unit tests for `check-alerts` and `test-alert` in `tests/test_cli.py`.
- [x] Task 8: Add alert health status and overview in `src/music_recommender/dashboard.py`.
- [x] Task 9: Add dashboard alert tests in `tests/test_dashboard.py`.
- [x] Task 10: Add Alertmanager integration tests in `tests/test_observability_integration.py`.
- [x] Task 11: Update documentation in `README.md` and `CHANGELOG.md`.
- [x] Task 12: Bump package version to `0.33.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 13: Final CI-parity check, push, open draft PR, monitor remote CI to 100% green.

---

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

