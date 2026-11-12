# Milestone 0.42.0 Roadmap — Phase 177 & 178

## Phase 177 & 178: Distributed Trace Context Propagation, OpenTelemetry Span Ingestion, and Alert Trace Correlator (0.42.0)

### 1. Objective
Equip the platform with enterprise distributed tracing and alert-to-trace deep linking:
1. A **Distributed Tracing Engine** (`TraceContext`, `SpanRecord`, `DistributedTracer`, `SpanKind`, `SpanStatus`) supporting the W3C Trace Context standard (`traceparent`, `tracestate`), thread/async context management via `contextvars`, and bounded in-memory ring buffering with OTLP JSON export capability.
2. An **Alert-to-Trace Correlator** (`AlertTraceCorrelator`) linking firing/resolved alert incidents directly with candidate traces exhibiting errors or high latency within the anomaly window.
3. Integration with HTTP middleware (`TraceContextMiddleware` in `api/middleware.py`), `RecommenderService` internal pipeline spans (`recommender.recommend`, `bandit.select_arm`, `alert.evaluate`), FastAPI endpoints (`/traces/*`), operator CLI tooling (`trace-inspect`, `list-traces`), and Streamlit dashboard timeline visualization, releasing Milestone 0.42.0 with 100% green CI.

### 2. Scope
1. **Tracing Models & Engine (`src/music_recommender/tracing.py`)**:
   - `TraceContext`: W3C `traceparent` (`00-{trace_id}-{span_id}-{flags}`) parser, generator, and context propagation.
   - `SpanRecord`: OpenTelemetry-compatible span with start/end time, duration, status, attributes, events, and `to_otlp_dict()` serializer.
   - `DistributedTracer`: Thread-safe context manager for span lifecycles, active context propagation, and queryable ring buffer.
   - `AlertTraceCorrelator`: Automated correlation between alert incidents and root-cause traces within time windows.
2. **HTTP Middleware & Header Propagation (`api/middleware.py`)**:
   - `TraceContextMiddleware`: Extract incoming `traceparent` / `tracestate` headers (or generate root trace), record HTTP server span, and inject `traceparent` and `x-trace-id` in response headers.
3. **Service Layer Instrumentation (`src/music_recommender/service.py`)**:
   - Instrument `recommend_user`, `recommend_similar_artists`, `recommend_tracks`, `evaluate_alerts`, and `sweep_bandit_feedback` with child spans.
   - Expose service methods: `get_trace()`, `search_spans()`, `correlate_alert_traces()`.
4. **FastAPI Endpoints (`api/main.py`)**:
   - `GET /traces`: list/search recent spans with filtering by name, status, min duration.
   - `GET /traces/{trace_id}`: retrieve complete span tree for a trace.
   - `POST /traces/correlate/alert`: correlate an alert rule/incident with candidate traces.
5. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender trace-inspect <trace_id>`: inspect hierarchical span execution tree.
   - `music-recommender list-traces`: list and filter recorded spans with table and JSON formatting.
6. **Dashboard Visualization (`src/music_recommender/dashboard.py`)**:
   - Streamlit tracing tab / sidebar inspector rendering span execution timelines and latency breakdowns.
7. **Comprehensive Tests**:
   - Unit tests in `tests/test_tracing.py`.
   - Middleware and API tests in `tests/test_api.py`.
   - Service integration tests in `tests/test_service.py`.
   - CLI tests in `tests/test_cli.py`.
   - Dashboard tests in `tests/test_dashboard.py`.
8. **Documentation & Release 0.42.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.42.0` in `pyproject.toml`, run `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- [ ] AC-1: `TraceContext` parses and formats W3C `traceparent` and `tracestate` headers correctly.
- [ ] AC-2: `DistributedTracer` manages thread/task context, records spans with accurate duration and status, and converts to OTLP JSON format.
- [ ] AC-3: `AlertTraceCorrelator` correlates alert incidents with candidate error/latency traces during anomaly windows.
- [ ] AC-4: `TraceContextMiddleware` propagates trace context in HTTP request/response headers.
- [ ] AC-5: `RecommenderService` creates child spans for core recommendation and evaluation operations and exposes trace query APIs.
- [ ] AC-6: FastAPI endpoints `/traces/*` return validated schemas for trace trees, span search, and alert correlation.
- [ ] AC-7: Operator CLI commands `trace-inspect` and `list-traces` function cleanly in human-readable and JSON modes.
- [ ] AC-8: Streamlit dashboard renders live trace inspection and timeline breakdowns.
- [ ] AC-9: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Micro-Task Breakdown
- [ ] Task 1: Define Phase 177 & 178 roadmap, update plan, and initialize progress record.
- [ ] Task 2: Implement `TraceContext` with W3C `traceparent`/`tracestate` in `src/music_recommender/tracing.py`.
- [ ] Task 3: Add unit tests for `TraceContext` in `tests/test_tracing.py`.
- [ ] Task 4: Implement `SpanRecord`, `SpanKind`, and `SpanStatus` in `src/music_recommender/tracing.py`.
- [ ] Task 5: Add unit tests for `SpanRecord` and OTLP format conversion in `tests/test_tracing.py`.
- [ ] Task 6: Implement `DistributedTracer` with context propagation and span buffer in `src/music_recommender/tracing.py`.
- [ ] Task 7: Add unit tests for `DistributedTracer` in `tests/test_tracing.py`.
- [ ] Task 8: Implement `AlertTraceCorrelator` in `src/music_recommender/tracing.py`.
- [ ] Task 9: Add unit tests for `AlertTraceCorrelator` in `tests/test_tracing.py`.
- [ ] Task 10: Implement `TraceContextMiddleware` in `api/middleware.py`.
- [ ] Task 11: Add unit tests for `TraceContextMiddleware` in `tests/test_api.py`.
- [ ] Task 12: Instrument `RecommenderService` with distributed tracer and span queries in `src/music_recommender/service.py`.
- [ ] Task 13: Add service integration tests for tracing in `tests/test_service.py`.
- [ ] Task 14: Implement `/traces/*` endpoints in `api/main.py`.
- [ ] Task 15: Add API integration tests for `/traces/*` in `tests/test_api.py`.
- [ ] Task 16: Implement CLI commands `trace-inspect` and `list-traces` in `src/music_recommender/cli.py`.
- [ ] Task 17: Add CLI tests in `tests/test_cli.py`.
- [ ] Task 18: Add trace viewer to Streamlit dashboard in `src/music_recommender/dashboard.py` and test in `tests/test_dashboard.py`.
- [ ] Task 19: Update `README.md` and `CHANGELOG.md` with 0.42.0 documentation.
- [ ] Task 20: Bump package version to `0.42.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [ ] Task 21: Verify local CI parity, push to remote, open draft PR, monitor remote CI to 100% green.

---

# Milestone 0.41.0 Roadmap — Phase 175 & 176

## Phase 175 & 176: Alert Incident Analytics, MTTD/MTTR Telemetry, and Prometheus Alert Exporter (0.41.0)

### 1. Objective
Harden production observability with incident lifecycle tracking and quantitative reliability metrics:
1. An **Alert Incident Analytics Engine** (`IncidentLifecycleRecord`, `RuleIncidentStats`, `AlertIncidentTracker`, `AlertIncidentAnalytics`) that tracks full incident lifecycles (started_at, resolved_at, duration_seconds), computing quantitative reliability metrics including Mean Time to Detect (MTTD), Mean Time to Resolve (MTTR), Mean Time Between Failures (MTBF), and Service Level Objective (SLO) compliance percentages per alert rule.
2. A **Prometheus Alert Metrics Exporter** exposing native OpenMetrics/Prometheus gauge and histogram families for live alert status, MTTD, MTTR, active silence counts, maintenance window states, and circuit breaker health (`music_recommender_alert_active`, `music_recommender_alert_mttd_seconds`, `music_recommender_alert_mttr_seconds`, `music_recommender_alert_silences_active`, `music_recommender_alert_maintenance_active`, `music_recommender_alert_circuit_breaker_state`).
3. Full integration in `RecommenderService`, FastAPI endpoints (`/alerts/analytics/*`), operator CLI commands (`alert-analytics`, `incident-history`), and Streamlit dashboard telemetry visualizations, releasing Milestone 0.41.0 with 100% green CI.

### 2. Scope
1. **Incident Lifecycle Models & Analytics Engine (`src/music_recommender/alerting.py`)**:
   - `IncidentLifecycleRecord`: dataclass capturing `incident_id`, `rule_name`, `severity`, `started_at`, `resolved_at`, `duration_seconds`, `remediations_triggered`, `silenced`, `root_causes`.
   - `RuleIncidentStats`: dataclass computing aggregate metrics: total incidents, active incidents, MTTD (seconds), MTTR (seconds), MTBF (seconds), longest incident duration, and SLO availability compliance percentage (e.g., 99.9% target).
   - `AlertIncidentTracker`: thread-safe incident tracker listening to alert state transitions (`AlertTransition`), correlating firing and resolution events, maintaining active incidents and rolling historical incident records with disk persistence.
   - `AlertIncidentAnalytics`: analytical evaluator generating summary reports, per-rule telemetry breakdown, and SLO availability audits.
2. **Prometheus Alert Metrics Exporter (`src/music_recommender/telemetry.py`, `src/music_recommender/alerting.py`)**:
   - Export metric families:
     - `music_recommender_alert_active`: gauge labeled by `rule` and `severity`.
     - `music_recommender_alert_mttd_seconds`: gauge of mean time to detect per `rule`.
     - `music_recommender_alert_mttr_seconds`: gauge of mean time to resolve per `rule`.
     - `music_recommender_alert_silences_active`: gauge of currently active silence rules.
     - `music_recommender_alert_maintenance_active`: gauge of currently active maintenance windows.
     - `music_recommender_alert_circuit_breaker_state`: gauge labeled by `channel` (0=closed, 1=half_open, 2=open).
   - Integrate with `export_prometheus_metrics()` and `/metrics`.
3. **Service Layer Integration (`src/music_recommender/service.py`)**:
   - Attach `AlertIncidentTracker` to `RecommenderService.evaluate_alerts()`.
   - Provide service methods `get_incident_analytics()`, `get_incident_history()`, `get_slo_compliance_report()`.
4. **FastAPI Endpoints (`api/main.py`)**:
   - `GET /alerts/analytics`: summary MTTD, MTTR, MTBF, and incident statistics.
   - `GET /alerts/analytics/incidents`: paginated historical incident lifecycle records.
   - `GET /alerts/analytics/slo`: SLO availability compliance report per rule.
5. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender alert-analytics`: tabular and JSON inspection of MTTD, MTTR, and incident frequencies.
   - `music-recommender incident-history`: audit log of completed and ongoing incident lifecycles.
6. **Dashboard Telemetry (`src/music_recommender/dashboard.py`)**:
   - Streamlit alerting sidebar: incident analytics metrics (MTTD/MTTR, total incidents, SLO compliance gauge).
7. **Comprehensive Tests**:
   - Unit tests in `tests/test_alerting.py` for lifecycle tracking, MTTD/MTTR math, and SLO compliance calculations.
   - Telemetry tests in `tests/test_telemetry.py` for Prometheus alert metric families.
   - Integration tests in `tests/test_service.py` for service incident tracking.
   - API integration tests in `tests/test_api.py`.
   - CLI tests in `tests/test_cli.py`.
   - Dashboard tests in `tests/test_dashboard.py`.
8. **Documentation & Release 0.41.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.41.0` in `pyproject.toml`, run `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- [ ] AC-1: `AlertIncidentTracker` tracks incident lifecycles (`started_at`, `resolved_at`, `duration_seconds`) on firing and resolved transitions.
- [ ] AC-2: `AlertIncidentAnalytics` computes accurate MTTD, MTTR, MTBF, and SLO compliance percentages per alert rule.
- [ ] AC-3: Prometheus exposition exports `music_recommender_alert_active`, `mttd_seconds`, `mttr_seconds`, `silences_active`, and `circuit_breaker_state`.
- [ ] AC-4: `RecommenderService` coordinates incident lifecycle tracking on alert evaluation and exposes query methods.
- [ ] AC-5: FastAPI endpoints `/alerts/analytics/*` return validated schemas for summary, incident history, and SLO reports.
- [ ] AC-6: Operator CLI commands `alert-analytics` and `incident-history` support table and JSON formatting.
- [ ] AC-7: Streamlit dashboard renders live MTTD/MTTR indicators and SLO availability metrics.
- [ ] AC-8: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Exclusions
- External long-term time-series databases (e.g. InfluxDB/Thanos) for incident rollups are optional remote sinks; core tracking operates in-process with persistent JSON storage.

### 6. Micro-Task Breakdown
- [ ] Task 1: Define Phase 175 & 176 roadmap, update plan, and initialize progress record.
- [ ] Task 2: Implement `IncidentLifecycleRecord`, `RuleIncidentStats`, and `AlertIncidentAnalytics` in `src/music_recommender/alerting.py`.
- [ ] Task 3: Add unit tests for `IncidentLifecycleRecord` and `AlertIncidentAnalytics` in `tests/test_alerting.py`.
- [ ] Task 4: Implement `AlertIncidentTracker` in `src/music_recommender/alerting.py`.
- [ ] Task 5: Add unit tests for `AlertIncidentTracker` in `tests/test_alerting.py`.
- [ ] Task 6: Implement Prometheus alerting metrics export in `src/music_recommender/telemetry.py` and `alerting.py`.
- [ ] Task 7: Add unit tests for Prometheus alert metrics in `tests/test_telemetry.py` and `tests/test_alerting.py`.
- [ ] Task 8: Integrate `AlertIncidentTracker` into `RecommenderService` in `src/music_recommender/service.py`.
- [ ] Task 9: Add service integration tests in `tests/test_service.py`.
- [ ] Task 10: Implement endpoints `/alerts/analytics/*` in `api/main.py`.
- [ ] Task 11: Add API integration tests in `tests/test_api.py`.
- [ ] Task 12: Implement CLI commands `alert-analytics` and `incident-history` in `src/music_recommender/cli.py`.
- [ ] Task 13: Add CLI tests in `tests/test_cli.py`.
- [ ] Task 14: Add incident analytics and MTTD/MTTR status to Streamlit dashboard in `src/music_recommender/dashboard.py`.
- [ ] Task 15: Add dashboard tests in `tests/test_dashboard.py`.
- [ ] Task 16: Update `README.md` and `CHANGELOG.md` with 0.41.0 documentation.
- [ ] Task 17: Bump package version to `0.41.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [ ] Task 18: Verify local CI parity, push to remote, open draft PR, monitor remote CI to 100% green.

---

# Milestone 0.40.0 Roadmap — Phase 173 & 174

## Phase 173 & 174: Alert Silencing, Maintenance Window Schedules, and Slack/Teams Webhook Adapters (0.40.0)

### 1. Objective
Enhance incident management and operations by providing dynamic alert suppression and rich team collaboration integrations:
1. An **Alert Silencing & Maintenance Window Engine** (`AlertSilenceRule`, `SilenceMatcher`, `MaintenanceWindow`, `AlertSilencingEngine`) that matches active alerts against targeted silence criteria (rule name, severity, labels) and scheduled maintenance windows, suppressing notification dispatch while logging audit records.
2. **Slack & Microsoft Teams Rich Card Adapters** (`SlackWebhookAdapter`, `TeamsWebhookAdapter`) delivering interactive, structured notification cards (Slack Block Kit with color-coded severity bars and action links; Microsoft Teams Adaptive Cards with FactSets and action buttons) integrated into `MultiChannelDispatcher` alongside PagerDuty and Opsgenie.
3. Full integration in `RecommenderService`, FastAPI endpoints (`/alerts/silences/*`, `/alerts/maintenance/*`), operator CLI commands (`silence-alert`, `list-silences`, `expire-silence`, `schedule-maintenance`, `list-maintenance`), and Streamlit dashboard controls, releasing Milestone 0.40.0 with 100% green CI.

### 2. Scope
1. **Alert Silencing & Maintenance Engine (`src/music_recommender/alerting.py`)**:
   - `SilenceMatcher`: matcher configuration targeting alert name, severity, and label key-value pairs.
   - `AlertSilenceRule`: rule configuring silence window (`starts_at`, `ends_at`), author, comment, matchers, and enabled flag.
   - `MaintenanceWindow`: maintenance schedule (`window_id`, `name`, `starts_at`, `ends_at`, `suppress_all`, `affected_rules`).
   - `SilenceDecision`: enum (`ACTIVE`, `SUPPRESSED_SILENCE_RULE`, `SUPPRESSED_MAINTENANCE_WINDOW`).
   - `AlertSilencingEngine`: thread-safe engine storing active silences and maintenance windows, matching alert transitions, and generating suppression decisions.
2. **Slack & Microsoft Teams Webhook Adapters (`src/music_recommender/alerting.py`)**:
   - `SlackWebhookAdapter`: Slack Block Kit payload formatter with header, markdown section, colored attachments (`#E01E5A` danger, `#ECB22E` warning, `#2EB886` resolved), and button elements.
   - `TeamsWebhookAdapter`: Microsoft Teams Adaptive Card / Office 365 Connector payload formatter with title, themeColor, and facts table.
3. **Dispatcher Integration (`src/music_recommender/alerting.py`)**:
   - Integrate `AlertSilencingEngine`, `SlackWebhookAdapter`, and `TeamsWebhookAdapter` into `MultiChannelDispatcher`.
4. **Service Integration (`src/music_recommender/service.py`)**:
   - Attach `AlertSilencingEngine` to `RecommenderService`.
   - Provide service methods: `add_silence_rule()`, `remove_silence_rule()`, `get_silence_rules()`, `schedule_maintenance_window()`, `cancel_maintenance_window()`, `get_maintenance_windows()`.
5. **FastAPI Endpoints (`api/main.py`)**:
   - `GET /alerts/silences`: list active and historical silence rules.
   - `POST /alerts/silences`: create new alert silence rule.
   - `DELETE /alerts/silences/{silence_id}`: expire/delete silence rule.
   - `GET /alerts/maintenance`: list scheduled maintenance windows.
   - `POST /alerts/maintenance`: schedule maintenance window.
   - `DELETE /alerts/maintenance/{window_id}`: cancel maintenance window.
6. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender silence-alert`: silence an alert rule with duration or timestamps.
   - `music-recommender list-silences`: list active and expired silences.
   - `music-recommender expire-silence`: expire a silence immediately.
   - `music-recommender schedule-maintenance`: schedule a maintenance window.
   - `music-recommender list-maintenance`: list maintenance windows.
7. **Dashboard Visibility (`src/music_recommender/dashboard.py`)**:
   - Streamlit alerting tab: add Alert Silences management UI and Active Maintenance Windows display.
8. **Comprehensive Tests**:
   - Unit tests in `tests/test_alerting.py` for silence matching, maintenance windows, Slack and Teams formatting.
   - Integration tests in `tests/test_service.py` for service silencing workflows.
   - API tests in `tests/test_api.py`.
   - CLI tests in `tests/test_cli.py`.
   - Dashboard tests in `tests/test_dashboard.py`.
9. **Documentation & Release 0.40.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.40.0` in `pyproject.toml`, run `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- [x] AC-1: `AlertSilencingEngine` evaluates `AlertSilenceRule` matchers against alert transitions and suppresses notification dispatch during active windows.
- [x] AC-2: `MaintenanceWindow` suppresses notifications for all or targeted alert rules during active maintenance intervals.
- [x] AC-3: `SlackWebhookAdapter` formats Slack Block Kit payloads with severity colors and action buttons.
- [x] AC-4: `TeamsWebhookAdapter` formats Adaptive Card payloads with facts and details.
- [x] AC-5: `MultiChannelDispatcher` dispatches to Slack and Teams channels and honors silencing suppression.
- [x] AC-6: `RecommenderService` provides complete management of silences and maintenance windows.
- [x] AC-7: FastAPI endpoints `/alerts/silences/*` and `/alerts/maintenance/*` return validated schema responses.
- [x] AC-8: Operator CLI commands `silence-alert`, `list-silences`, `expire-silence`, `schedule-maintenance`, and `list-maintenance` operate cleanly.
- [x] AC-9: Streamlit dashboard renders silence rules and maintenance window controls.
- [x] AC-10: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Exclusions
- External internet delivery during automated test runs is simulated with local HTTP test servers and mocks. Real outbound webhooks require configured webhook URLs.

### 6. Micro-Task Breakdown
- [x] Task 1: Define Phase 173 & 174 roadmap, update plan, and initialize progress record.
- [x] Task 2: Implement `SilenceMatcher`, `AlertSilenceRule`, `MaintenanceWindow`, and `SilenceDecision` in `src/music_recommender/alerting.py`.
- [x] Task 3: Implement `AlertSilencingEngine` in `src/music_recommender/alerting.py`.
- [x] Task 4: Add unit tests for `AlertSilenceRule`, `MaintenanceWindow`, and `AlertSilencingEngine` in `tests/test_alerting.py`.
- [x] Task 5: Implement `SlackWebhookAdapter` Block Kit formatter and dispatcher in `src/music_recommender/alerting.py`.
- [x] Task 6: Implement `TeamsWebhookAdapter` Adaptive Card formatter and dispatcher in `src/music_recommender/alerting.py`.
- [x] Task 7: Add unit tests for `SlackWebhookAdapter` and `TeamsWebhookAdapter` in `tests/test_alerting.py`.
- [x] Task 8: Integrate silencing engine and chat webhook adapters into `MultiChannelDispatcher` in `src/music_recommender/alerting.py`.
- [x] Task 9: Add unit tests for dispatcher silencing and chat adapters in `tests/test_alerting.py`.
- [x] Task 10: Integrate silencing and maintenance windows into `RecommenderService` in `src/music_recommender/service.py`.
- [x] Task 11: Add service integration tests in `tests/test_service.py`.
- [x] Task 12: Implement endpoints `/alerts/silences/*` and `/alerts/maintenance/*` in `api/main.py`.
- [x] Task 13: Add API integration tests in `tests/test_api.py`.
- [x] Task 14: Implement CLI commands `silence-alert`, `list-silences`, `expire-silence`, `schedule-maintenance`, and `list-maintenance` in `src/music_recommender/cli.py`.
- [x] Task 15: Add CLI tests in `tests/test_cli.py`.
- [x] Task 16: Add silencing management and maintenance window status to Streamlit dashboard in `src/music_recommender/dashboard.py`.
- [x] Task 17: Add dashboard tests in `tests/test_dashboard.py`.
- [x] Task 18: Update `README.md` and `CHANGELOG.md` with 0.40.0 documentation.
- [x] Task 19: Bump package version to `0.40.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 20: Verify local CI parity, push to remote, monitor remote CI to 100% green.

---

# Milestone 0.39.0 Roadmap — Phase 171 & 172

## Phase 171 & 172: Automated Incident Remediation and Outbound Webhook Circuit Breaking (0.39.0)

### 1. Objective
Harden production operations with autonomous self-healing and fault-tolerant notification dispatch:
1. An **Automated Incident Remediation Runner** (`IncidentRemediationRunner`, `RemediationRule`, `RemediationActionType`) that executes targeted self-healing operations (flushing queues, restarting workers, rotating logs, triggering bandit folds) when specific alert conditions fire, bounded by configurable cooldown windows and execution limits.
2. An **Outbound Webhook Circuit Breaker** (`CircuitBreaker`, `CircuitBreakerConfig`, `CircuitBreakerState`) integrated into `WebhookDispatcher` and `MultiChannelDispatcher` to fail fast and shed traffic during downstream gateway outages, automatically probing and recovering when endpoints return to health.
3. Full service integration in `RecommenderService`, REST endpoints (`/alerts/remediation/*`, `/alerts/circuit-breaker/stats`), operator CLI commands (`remediation-history`, `trigger-action`, `circuit-stats`), and Streamlit dashboard status indicators, releasing Milestone 0.39.0 with 100% green CI.

### 2. Scope
1. **Automated Incident Remediation Runner (`src/music_recommender/alerting.py`)**:
   - `RemediationActionType`: enum defining supported self-healing actions (`FLUSH_FEEDBACK_QUEUE`, `RESTART_MAINTENANCE_WORKER`, `ROTATE_ALERT_STORAGE`, `PURGE_DLQ`, `TRIGGER_ONLINE_FOLD`, `CUSTOM_CALLBACK`).
   - `RemediationRule`: rule configuration mapping alert rule name to remediation actions with `cooldown_seconds`, `max_attempts`, and `enabled` toggles.
   - `RemediationActionResult`: record tracking execution status (`EXECUTED`, `COOLDOWN_SUPPRESSED`, `MAX_ATTEMPTS_EXCEEDED`, `FAILED`, `DRY_RUN`), output summary, error details, and timestamp.
   - `IncidentRemediationRunner`: thread-safe runner maintaining rule registrations, action handlers, execution history, and cooldown state.
2. **Outbound Webhook Circuit Breaker (`src/music_recommender/alerting.py`)**:
   - `CircuitBreakerState`: enum (`CLOSED`, `OPEN`, `HALF_OPEN`).
   - `CircuitBreakerConfig`: failure threshold, recovery timeout seconds, and half-open probe success threshold.
   - `CircuitBreaker`: thread-safe state machine monitoring delivery outcomes, opening to fail fast upon threshold breaches, and transitioning through half-open back to closed.
3. **Dispatcher Circuit Protection (`src/music_recommender/alerting.py`)**:
   - Integrate circuit breaker into `WebhookDispatcher` and `MultiChannelDispatcher` to prevent thread pool exhaustion and cascading network latency.
4. **Service Integration (`src/music_recommender/service.py`)**:
   - Attach `IncidentRemediationRunner` with service handlers to `RecommenderService`.
   - Provide `get_remediation_stats()`, `get_remediation_history()`, `trigger_remediation_action()`, and `get_circuit_breaker_stats()`.
5. **FastAPI Endpoints (`api/main.py`)**:
   - `GET /alerts/remediation/rules`: query active remediation rules.
   - `GET /alerts/remediation/history`: inspect self-healing execution audit trail.
   - `POST /alerts/remediation/trigger`: manually trigger an action with dry-run support.
   - `GET /alerts/circuit-breaker/stats`: inspect circuit breaker status per channel.
6. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender remediation-history`: view automated remediation actions and results.
   - `music-recommender trigger-action`: invoke a remediation action on demand.
   - `music-recommender circuit-stats`: inspect circuit breaker health and trips.
7. **Dashboard Visibility (`src/music_recommender/dashboard.py`)**:
   - Surface self-healing remediation activity and circuit breaker indicators in sidebar.
8. **Comprehensive Tests**:
   - Unit tests in `tests/test_alerting.py` for remediation runner, action cooldowns, circuit breaker transitions, and protected dispatch.
   - Integration tests in `tests/test_service.py` for service self-healing handlers.
   - API integration tests in `tests/test_api.py`.
   - CLI tests in `tests/test_cli.py`.
   - Dashboard tests in `tests/test_dashboard.py`.
9. **Documentation & Release 0.39.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.39.0` in `pyproject.toml`, run `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- [x] AC-1: `IncidentRemediationRunner` executes bound self-healing actions on alert firing, enforcing cooldowns and execution limits.
- [x] AC-2: `CircuitBreaker` correctly transitions across CLOSED, OPEN, and HALF_OPEN states based on delivery failures and recovery timeouts.
- [x] AC-3: `WebhookDispatcher` and `MultiChannelDispatcher` reject traffic immediately when circuit is OPEN without invoking network layer.
- [x] AC-4: `RecommenderService` binds core operational healing handlers (queue flush, worker restart, storage rotation, online fold).
- [x] AC-5: FastAPI endpoints `/alerts/remediation/*` and `/alerts/circuit-breaker/stats` return valid schema responses.
- [x] AC-6: Operator CLI commands `remediation-history`, `trigger-action`, and `circuit-stats` operate cleanly in table and JSON formats.
- [x] AC-7: Streamlit dashboard sidebar surfaces self-healing activity and circuit breaker status.
- [x] AC-8: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Exclusions
- Arbitrary shell script execution from remediation rules is excluded for security reasons; only statically registered typed Python handlers are permitted.

### 6. Micro-Task Breakdown
- [x] Task 1: Define Phase 171 & 172 roadmap, update plan, and initialize progress record.
- [x] Task 2: Implement `RemediationActionType`, `RemediationRule`, `RemediationActionResult`, and `IncidentRemediationRunner` in `src/music_recommender/alerting.py`.
- [x] Task 3: Add unit tests for `IncidentRemediationRunner` in `tests/test_alerting.py`.
- [x] Task 4: Implement `CircuitBreakerState`, `CircuitBreakerConfig`, and `CircuitBreaker` in `src/music_recommender/alerting.py`.
- [x] Task 5: Add unit tests for `CircuitBreaker` in `tests/test_alerting.py`.
- [x] Task 6: Integrate circuit breaking into `WebhookDispatcher` and `MultiChannelDispatcher` in `src/music_recommender/alerting.py`.
- [x] Task 7: Add unit tests for circuit-breaker protected dispatching in `tests/test_alerting.py`.
- [x] Task 8: Integrate `IncidentRemediationRunner` into `RecommenderService` in `src/music_recommender/service.py`.
- [x] Task 9: Add service integration tests in `tests/test_service.py`.
- [x] Task 10: Implement endpoints `/alerts/remediation/rules`, `/alerts/remediation/history`, `/alerts/remediation/trigger`, `/alerts/circuit-breaker/stats` in `api/main.py`.
- [x] Task 11: Add API integration tests in `tests/test_api.py`.
- [x] Task 12: Implement CLI commands `remediation-history`, `trigger-action`, and `circuit-stats` in `src/music_recommender/cli.py`.
- [x] Task 13: Add CLI tests in `tests/test_cli.py`.
- [x] Task 14: Add self-healing remediation and circuit-breaker status indicators to Streamlit sidebar in `src/music_recommender/dashboard.py`.
- [x] Task 15: Add dashboard tests in `tests/test_dashboard.py`.
- [x] Task 16: Update `README.md` and `CHANGELOG.md` with 0.39.0 documentation.
- [x] Task 17: Bump package version to `0.39.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 18: Verify local CI parity, push to remote, open draft PR, monitor remote CI to 100% green.

---

# Milestone 0.38.0 Roadmap — Phase 169 & 170

## Phase 169 & 170: Alert Notification Throttling, Escalation Policies, and Multi-Channel Incident Gateways (0.38.0)

### 1. Objective
Enhance operational alerting resilience and prevent notification storms by implementing a thread-safe notification throttler and deduplication engine (`AlertNotificationThrottler`), tiered time-based escalation policies (`AlertEscalationPolicy`, `AlertEscalationEngine`), outbound multi-channel incident gateway integrations for PagerDuty Events API v2 and Opsgenie Alert API v2 (`PagerDutyGateway`, `OpsgenieGateway`, `MultiChannelDispatcher`), REST API observability endpoints (`GET /alerts/throttling/stats`, `POST /alerts/gateway/test`), operator CLI tooling (`throttle-stats`, `test-gateway`), and Streamlit dashboard alerting controls, releasing Milestone 0.38.0 with 100% green CI.

### 2. Scope
1. **Notification Throttling & Deduplication Engine (`src/music_recommender/alerting.py`)**:
   - `AlertThrottleConfig`: dataclass configuring deduplication window (`dedup_window_seconds`), rate limits (`max_alerts_per_window`, `window_seconds`), and repeat interval (`repeat_interval_seconds`).
   - `ThrottleDecision`: enum indicating `ALLOWED`, `SUPPRESSED_DUPLICATE`, `RATE_LIMITED`, `REPEAT_SUPPRESSED`.
   - `AlertNotificationThrottler`: thread-safe in-memory sliding-window throttler/deduplicator that filters alert transitions.
2. **Tiered Incident Escalation Engine (`src/music_recommender/alerting.py`)**:
   - `AlertEscalationTier`: dataclass defining continuous firing duration thresholds and assigned destination channels.
   - `AlertEscalationPolicy`: policy mapping alert rules and severities to escalation tiers.
   - `AlertEscalationEngine`: tracks ongoing firing duration and transitions alerts to higher-tier channels when escalation intervals elapse.
3. **Outbound Multi-Channel Incident Gateways (`src/music_recommender/alerting.py`)**:
   - `PagerDutyGateway`: formats alert transitions into PagerDuty Events API v2 payloads (`trigger`, `acknowledge`, `resolve`) with standard severity mapping and custom details, dispatching via standard library HTTP client.
   - `OpsgenieGateway`: formats alert transitions into Opsgenie Alert API v2 payloads with priority mapping (P1..P5), alias, tags, and description.
   - `MultiChannelDispatcher`: orchestrates dispatch across multiple gateways with `MultiChannelDispatchReport`.
4. **Service Integration (`src/music_recommender/service.py`)**:
   - Attach `AlertNotificationThrottler` and `MultiChannelDispatcher` to `RecommenderService`.
   - Methods: `get_alert_throttling_stats()`, `reset_alert_throttling()`, `dispatch_multichannel_alerts()`.
5. **FastAPI Endpoints (`api/main.py`)**:
   - `GET /alerts/throttling/stats`: returns current throttling, deduplication, and suppression statistics.
   - `POST /alerts/throttling/reset`: resets the active throttler state.
   - `POST /alerts/gateway/test`: tests outbound gateway formatting and dispatch (PagerDuty, Opsgenie, Webhook) with dry-run support.
6. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender throttle-stats`: displays throttler metrics and suppressed alert counts in table or JSON format.
   - `music-recommender test-gateway`: dispatches synthetic test alert to a target gateway (PagerDuty, Opsgenie, Webhook) with `--channel`, `--dry-run`, `--severity`.
7. **Dashboard Visibility (`src/music_recommender/dashboard.py`)**:
   - Surface active notification throttling and suppressed alerts in sidebar alerting expander.
8. **Comprehensive Tests**:
   - Unit tests in `tests/test_alerting.py` for throttling, escalation, PagerDuty gateway, Opsgenie gateway, and multi-channel dispatcher.
   - Integration tests in `tests/test_service.py` for service multi-channel dispatch and throttler management.
   - API integration tests in `tests/test_api.py` for `/alerts/throttling/stats`, `/alerts/throttling/reset`, `/alerts/gateway/test`.
   - CLI tests in `tests/test_cli.py` for `throttle-stats` and `test-gateway`.
   - Dashboard tests in `tests/test_dashboard.py`.
9. **Documentation & Release 0.38.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.38.0` in `pyproject.toml`, run `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- [x] AC-1: `AlertNotificationThrottler` enforces deduplication windows, rate limits, and repeat intervals with `ThrottleDecision` tracking.
- [x] AC-2: `AlertEscalationEngine` computes ongoing firing duration and resolves tiered channel escalation.
- [x] AC-3: `PagerDutyGateway` and `OpsgenieGateway` serialize and dispatch compliant API v2 payloads with dry-run and mock support.
- [x] AC-4: FastAPI endpoints `GET /alerts/throttling/stats`, `POST /alerts/throttling/reset`, and `POST /alerts/gateway/test` respond with valid schemas.
- [x] AC-5: Operator CLI commands `throttle-stats` and `test-gateway` operate cleanly in table and JSON formats.
- [x] AC-6: Streamlit dashboard sidebar surfaces notification throttling indicators.
- [x] AC-7: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Exclusions
- Third-party SDK client packages (e.g. `pdpyras` or `opsgenie-sdk`) are excluded in favor of standard library `urllib.request` integrations to maintain zero additional runtime dependencies.

### 6. Micro-Task Breakdown
- [x] Task 1: Define Phase 169 & 170 roadmap, update plan, and initialize progress record.
- [x] Task 2: Implement `AlertThrottleConfig`, `ThrottleDecision`, and `AlertNotificationThrottler` in `src/music_recommender/alerting.py`.
- [x] Task 3: Add unit tests for `AlertNotificationThrottler` in `tests/test_alerting.py`.
- [x] Task 4: Implement `AlertEscalationTier`, `AlertEscalationPolicy`, and `AlertEscalationEngine` in `src/music_recommender/alerting.py`.
- [x] Task 5: Add unit tests for alert escalation engine in `tests/test_alerting.py`.
- [x] Task 6: Implement `PagerDutyGateway`, `OpsgenieGateway`, and `MultiChannelDispatcher` in `src/music_recommender/alerting.py`.
- [x] Task 7: Add unit tests for PagerDuty and Opsgenie gateways in `tests/test_alerting.py`.
- [x] Task 8: Integrate throttler and multi-channel dispatcher in `RecommenderService` in `src/music_recommender/service.py`.
- [x] Task 9: Add service tests for throttling and gateway dispatch in `tests/test_service.py`.
- [x] Task 10: Implement `/alerts/throttling/stats`, `/alerts/throttling/reset`, and `/alerts/gateway/test` in `api/main.py`.
- [x] Task 11: Add API integration tests in `tests/test_api.py`.
- [x] Task 12: Implement `throttle-stats` and `test-gateway` CLI commands in `src/music_recommender/cli.py`.
- [x] Task 13: Add CLI tests in `tests/test_cli.py`.
- [x] Task 14: Add throttling indicators to sidebar in `src/music_recommender/dashboard.py`.
- [x] Task 15: Add dashboard tests in `tests/test_dashboard.py`.
- [x] Task 16: Update `README.md` and `CHANGELOG.md` with 0.38.0 documentation.
- [x] Task 17: Bump package version to `0.38.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 18: Verify local CI parity, push to remote, monitor CI to 100% green.
- [ ] Task 18: CI parity verification, push, open draft PR, monitor remote CI to green.

---

# Milestone 0.37.0 Roadmap — Phase 167 & 168

## Phase 167 & 168: Size-Capped Log Rotation, Retention Pruning, DLQ Exponential Retry, and Archive Snapshots (0.37.0)

### 1. Objective
Harden persistent alert storage and recovery operations by implementing size-capped automatic and manual file rotation for alert spillover and DLQ logs, age-based retention pruning, automated DLQ retry execution with exponential backoff and jitter, compressed archive bundle export/restore (`.tar.gz`/`.zip`) for disaster recovery, REST API endpoints (`POST /alerts/rotate`, `POST /alerts/dlq/retry`, `GET /alerts/archive`), operator CLI commands (`rotate-alerts`, `retry-dlq`, `archive-alerts`), and Streamlit dashboard storage management visibility, delivering Milestone 0.37.0 with 100% green CI.

### 2. Scope
1. **Size-Capped Rotation & Retention Pruning (`src/music_recommender/alerting.py`)**:
   - `AlertLogRotationConfig`: dataclass configuring `max_file_size_bytes`, `max_backups`, and `retention_days`.
   - File rotation logic in `AlertSpilloverStorage`: rotate current file to `.1`, `.1` to `.2`, etc. when file exceeds threshold or upon explicit call to `rotate_storage(max_backups)`.
   - Retention policy enforcement: `enforce_retention(max_age_days, max_backups)` to purge outdated rotated logs.
2. **Disaster Recovery Archive Bundling (`src/music_recommender/alerting.py`)**:
   - `create_archive_bundle(output_path, format="tar.gz", include_spillover=True, include_dlq=True)`: packages alert spillover, rotated logs, DLQ records, and metadata into a gzip tarball or zip bundle.
   - `restore_archive_bundle(archive_path, target_dir=None)`: inspects or extracts alert records from an archive bundle.
3. **Automated DLQ Retry Engine with Exponential Backoff (`src/music_recommender/alerting.py`)**:
   - `DLQRetryPolicy`: dataclass with `max_retries`, `base_delay_seconds`, `backoff_multiplier`, `max_delay_seconds`.
   - `DLQRetryResult` and `DLQRetryReport`: detailed per-record and aggregated retry results.
   - `retry_dead_letters(...)`: attempts re-dispatch of dead-letter records to a webhook or internal buffer, with backoff calculation, retry count tracking, and auto-removal of resolved items.
4. **Service Integration (`src/music_recommender/service.py`)**:
   - Add `rotate_alert_storage(max_backups)`, `enforce_alert_retention(max_age_days, max_backups)`, `retry_dead_letters(policy, target_url, ...)`, `archive_alert_storage(output_path, format)`.
   - Update `get_alert_storage_stats()` with rotation and retention metadata.
5. **FastAPI Endpoints (`api/main.py`)**:
   - `POST /alerts/rotate`: triggers storage rotation and retention pruning, returning status and purged/rotated counts.
   - `POST /alerts/dlq/retry`: retries dead-letter queue records with exponential backoff.
   - `GET /alerts/archive`: exports and streams/returns a compressed tarball/zip archive bundle of persistent storage.
   - `GET /alerts/storage/stats`: returns enhanced storage stats including rotated generations, disk usage, and retention limits.
6. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender rotate-alerts`: rotates logs and enforces retention policies (`--max-backups`, `--max-age-days`, `--dry-run`).
   - `music-recommender retry-dlq`: retries dead-letter records with exponential backoff (`--max-retries`, `--target-url`, `--channel`, `--dry-run`).
   - `music-recommender archive-alerts`: exports storage bundle to tar.gz or zip (`--output`, `--format`).
7. **Dashboard Visibility (`src/music_recommender/dashboard.py`)**:
   - Add storage rotation status, rotated backups count, and DLQ retry status display in sidebar alert expander.
8. **Comprehensive Tests**:
   - Unit tests in `tests/test_alerting.py` for rotation, retention, archive bundles, and DLQ retry.
   - Integration tests in `tests/test_service.py` for service storage rotation, retry, and archiving.
   - API tests in `tests/test_api.py` for `/alerts/rotate`, `/alerts/dlq/retry`, `/alerts/archive`.
   - CLI tests in `tests/test_cli.py` for `rotate-alerts`, `retry-dlq`, `archive-alerts`.
   - Dashboard tests in `tests/test_dashboard.py`.
9. **Documentation & Release 0.37.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.37.0` in `pyproject.toml`, run `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- [x] AC-1: `AlertSpilloverStorage` supports size-capped log rotation (`rotate_storage`) and age-based retention pruning (`enforce_retention`).
- [x] AC-2: `create_archive_bundle` produces valid `.tar.gz` and `.zip` archives containing spillover, DLQ logs, and manifest metadata.
- [x] AC-3: `retry_dead_letters` executes exponential backoff retries on DLQ records with `DLQRetryReport` tracking and purged-on-success updates.
- [x] AC-4: FastAPI endpoints `POST /alerts/rotate`, `POST /alerts/dlq/retry`, `GET /alerts/archive`, and `GET /alerts/storage/stats` return correct schemas.
- [x] AC-5: Operator CLI commands `rotate-alerts`, `retry-dlq`, and `archive-alerts` function cleanly with table and JSON outputs.
- [x] AC-6: Streamlit dashboard sidebar surfaces storage rotation and DLQ retry indicators.
- [x] AC-7: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Exclusions
- External cloud object storage (S3/GCS) uploading is excluded in favor of standard tarball/zip archive bundles and HTTP file streaming.

### 6. Micro-Task Breakdown
- [x] Task 1: Define Phase 167 & 168 roadmap, update plan, and initialize progress record.
- [x] Task 2: Implement `AlertLogRotationConfig` and size-capped rotation in `AlertSpilloverStorage` (`src/music_recommender/alerting.py`).
- [x] Task 3: Implement age-based retention pruning in `AlertSpilloverStorage` (`src/music_recommender/alerting.py`).
- [x] Task 4: Add unit tests for log rotation and retention pruning in `tests/test_alerting.py`.
- [x] Task 5: Implement `create_archive_bundle` and `restore_archive_bundle` in `src/music_recommender/alerting.py`.
- [x] Task 6: Add unit tests for archive bundle creation and extraction in `tests/test_alerting.py`.
- [x] Task 7: Implement `DLQRetryPolicy`, `DLQRetryResult`, `DLQRetryReport`, and `retry_dead_letters` in `src/music_recommender/alerting.py`.
- [x] Task 8: Add unit tests for DLQ retry engine in `tests/test_alerting.py`.
- [x] Task 9: Integrate storage rotation, retention, archiving, and DLQ retry into `RecommenderService` in `src/music_recommender/service.py`.
- [x] Task 10: Add service tests in `tests/test_service.py`.
- [x] Task 11: Implement `/alerts/rotate`, `/alerts/dlq/retry`, `/alerts/archive`, `/alerts/storage/stats` in `api/main.py`.
- [x] Task 12: Add API integration tests in `tests/test_api.py`.
- [x] Task 13: Implement `rotate-alerts`, `retry-dlq`, and `archive-alerts` in `src/music_recommender/cli.py`.
- [x] Task 14: Add CLI unit tests in `tests/test_cli.py`.
- [x] Task 15: Add storage rotation and retry indicators in `src/music_recommender/dashboard.py`.
- [x] Task 16: Add dashboard tests in `tests/test_dashboard.py`.
- [x] Task 17: Update `README.md` and `CHANGELOG.md` with 0.37.0 documentation.
- [x] Task 18: Bump package version to `0.37.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 19: CI parity verification, push, open draft PR, monitor remote CI to green.

---

# Milestone 0.36.0 Roadmap — Phase 163 & 164

## Phase 163 & 164: InfluxDB Metric & Alert Exporter, Prometheus Remote-Write Adapter, and Alert Correlation Graph (0.36.0)

### 1. Objective
Provide push-based telemetry integration and advanced incident root-cause analysis by implementing an InfluxDB Line Protocol 2.0 metrics formatter and push adapter (`POST /metrics/push`, `music-recommender push-metrics`), Prometheus remote-write compatibility, a directed acyclic alert root-cause correlation graph (`AlertCorrelationGraph`) with causal chain detection and symptom filtering (`GET /alerts/correlation`, `music-recommender alert-correlate`), and Streamlit dashboard visibility, releasing Milestone 0.36.0 with 100% green CI.

### 2. Scope
1. **Push Telemetry & InfluxDB Line Protocol (`src/music_recommender/telemetry.py`)**:
   - `format_influxdb_line_protocol`: converts `MetricFamily` samples to InfluxDB Line Protocol 2.0 with escaping for measurement, tag keys/values, field values, and nanosecond timestamps.
   - `TelemetryPushConfig`: dataclass specifying endpoint URL, protocol (`"influxdb"`, `"prometheus_json"`, `"prometheus_text"`), custom headers, timeout, and batch size.
   - `TelemetryPushResult`: dataclass recording success, sample count, status code, duration, and error.
   - `TelemetryPushAdapter`: sends serialized metric batches via standard library HTTP client.
2. **Alert Root-Cause Correlation Engine (`src/music_recommender/alerting.py`)**:
   - `AlertCausalRule`: defines directed causal dependency (`parent_alert` -> `child_alert`, e.g. `StreamingQueueHighUtilization` -> `StreamingQueueDroppedRecords`).
   - `DEFAULT_ALERT_CORRELATION_RULES`: baseline domain causal dependencies between streaming queue, maintenance worker, latency, and drift alert rules.
   - `AlertCorrelationGraph`: constructs causal DAG, identifies root-cause alerts vs downstream symptoms, and computes incident causal chains within configurable time windows.
   - `AlertCorrelationReport`: dataclass summarizing root causes, symptoms, unlinked alerts, and causal paths.
3. **Service Integration (`src/music_recommender/service.py`)**:
   - Expose `export_influxdb_metrics(custom_labels, default_measurement)`.
   - Expose `push_telemetry(config)`.
   - Expose `correlate_alert_history(window_seconds, custom_rules)`.
4. **FastAPI Endpoints (`api/main.py`)**:
   - `GET /metrics/influxdb`: exposes real-time service metrics formatted in InfluxDB Line Protocol (`text/plain; charset=utf-8`).
   - `POST /metrics/push`: pushes current service metrics to an external telemetry collector URL.
   - `GET /alerts/correlation`: evaluates active alert history from buffer and returns root-cause analysis report.
   - `POST /alerts/correlate`: accepts an external batch of alert events and returns correlation graph report.
5. **Operator CLI Tooling (`src/music_recommender/cli.py`)**:
   - `music-recommender export-metrics --format influxdb`: outputs InfluxDB Line Protocol to stdout or file.
   - `music-recommender push-metrics`: pushes live telemetry to remote HTTP endpoint.
   - `music-recommender alert-correlate`: evaluates active alert history or file, printing root causes and causal chains in tabular/ASCII/JSON formats.
6. **Dashboard Visibility (`src/music_recommender/dashboard.py`)**:
   - Show root-cause analysis and symptom indicators in sidebar alerting expander.
7. **Comprehensive Tests**:
   - Unit tests in `tests/test_telemetry.py` for InfluxDB formatting and push adapter.
   - Unit tests in `tests/test_alerting.py` for causal rules, DAG construction, and root-cause detection.
   - Integration tests in `tests/test_api.py` for `/metrics/influxdb`, `/metrics/push`, `/alerts/correlation`, `/alerts/correlate`.
   - CLI tests in `tests/test_cli.py` for `export-metrics --format influxdb`, `push-metrics`, `alert-correlate`.
   - Dashboard tests in `tests/test_dashboard.py` for root-cause badge rendering.
8. **Documentation & Release 0.36.0**:
   - Update `README.md`, `CHANGELOG.md`, `PLAN.md`, `docs/PROGRESS.md`.
   - Bump version to `0.36.0` in `pyproject.toml`, run `uv lock`, and `uv build`.

### 3. Acceptance Criteria
- [x] AC-1: `format_influxdb_line_protocol` renders `MetricFamily` samples into valid InfluxDB Line Protocol 2.0 with proper escaping and nanosecond timestamps.
- [x] AC-2: `TelemetryPushAdapter` dispatches metric batches via HTTP to remote collectors with `TelemetryPushResult` tracking.
- [x] AC-3: `AlertCorrelationGraph` identifies root-cause alerts, isolates downstream symptoms, and traces causal chains within time windows.
- [x] AC-4: FastAPI endpoints `GET /metrics/influxdb`, `POST /metrics/push`, `GET /alerts/correlation`, and `POST /alerts/correlate` respond with valid schemas.
- [x] AC-5: CLI commands `export-metrics --format influxdb`, `push-metrics`, and `alert-correlate` operate cleanly in terminal and headless modes.
- [x] AC-6: Streamlit dashboard sidebar surfaces alert root-cause analysis and symptom counts.
- [x] AC-7: All local CI-parity checks and remote CI workflow runs pass 100%.

### 4. Validation & CI Plan
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy`
- `uv run --extra dashboard --extra tracking --extra spotify pytest -q`
- `uv build`
- Remote GitHub Actions CI on PR branch 100% green.

### 5. Exclusions
- External InfluxDB server provisioning or Docker compose service dependency (pure Python exporter and push adapter allows connecting to any remote InfluxDB/Prometheus endpoint without requiring a local InfluxDB container).

### 6. Micro-Task Breakdown
- [x] Task 1: Define Phase 163 & 164 roadmap, plan, and initialize progress record.
- [x] Task 2: Implement InfluxDB Line Protocol 2.0 formatter in `src/music_recommender/telemetry.py`.
- [x] Task 3: Implement `TelemetryPushConfig`, `TelemetryPushResult`, and `TelemetryPushAdapter` in `src/music_recommender/telemetry.py`.
- [x] Task 4: Add unit tests for InfluxDB formatting and push adapter in `tests/test_telemetry.py`.
- [x] Task 5: Implement `AlertCausalRule`, `AlertCorrelationGraph`, and `AlertCorrelationReport` in `src/music_recommender/alerting.py`.
- [x] Task 6: Add unit tests for alert correlation graph and root-cause analysis in `tests/test_alerting.py`.
- [x] Task 7: Integrate InfluxDB export, push telemetry, and alert correlation in `RecommenderService` in `src/music_recommender/service.py`.
- [x] Task 8: Implement `GET /metrics/influxdb`, `POST /metrics/push`, `GET /alerts/correlation`, and `POST /alerts/correlate` in `api/main.py`.
- [x] Task 9: Add API integration tests in `tests/test_api.py`.
- [x] Task 10: Implement `push-metrics` and `alert-correlate` CLI commands, and extend `export-metrics --format influxdb` in `src/music_recommender/cli.py`.
- [x] Task 11: Add unit tests for new CLI commands in `tests/test_cli.py`.
- [x] Task 12: Add alert correlation indicator to sidebar in `src/music_recommender/dashboard.py`.
- [x] Task 13: Add dashboard alert correlation tests in `tests/test_dashboard.py`.
- [x] Task 14: Update `README.md` and `CHANGELOG.md` with 0.36.0 documentation.
- [x] Task 15: Bump package version to `0.36.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [x] Task 16: Final CI parity verification, push, open draft PR, monitor remote CI to green.

---

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

