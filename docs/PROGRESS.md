# Phase 163 & 164 Progress Record & Decision Log

## Phase 163 & 164: InfluxDB Metric & Alert Exporter, Prometheus Remote-Write Adapter, and Alert Correlation Graph (0.36.0)

- **Milestone**: 0.36.0
- **Status**: In Progress
- **Branch**: `phase/163-remote-write-and-correlation`

---

## 1. Task Status

| Task ID | Task Description | Status | Commit |
|---|---|---|---|
| Task 1 | Roadmap, Plan & Progress initialization | Completed | `4af65ca`, `11fcc8b` |
| Task 2 | InfluxDB Line Protocol 2.0 formatter in `telemetry.py` | Pending | |
| Task 3 | `TelemetryPushConfig`, `TelemetryPushResult`, and `TelemetryPushAdapter` in `telemetry.py` | Pending | |
| Task 4 | Unit tests for InfluxDB formatting & push adapter in `tests/test_telemetry.py` | Pending | |
| Task 5 | `AlertCausalRule`, `AlertCorrelationGraph`, and incident report models in `alerting.py` | Pending | |
| Task 6 | Unit tests for alert correlation graph and root-cause analysis in `tests/test_alerting.py` | Pending | |
| Task 7 | Service integration in `src/music_recommender/service.py` | Pending | |
| Task 8 | FastAPI endpoints (`/metrics/influxdb`, `/metrics/push`, `/alerts/correlation`, `/alerts/correlate`) in `api/main.py` | Pending | |
| Task 9 | API integration tests in `tests/test_api.py` | Pending | |
| Task 10 | CLI commands (`push-metrics`, `alert-correlate`, `export-metrics --format influxdb`) in `cli.py` | Pending | |
| Task 11 | Unit tests for new CLI commands in `tests/test_cli.py` | Pending | |
| Task 12 | Dashboard alert correlation visibility in `dashboard.py` | Pending | |
| Task 13 | Dashboard alert correlation tests in `tests/test_dashboard.py` | Pending | |
| Task 14 | Documentation updates in `README.md` and `CHANGELOG.md` | Pending | |
| Task 15 | Version bump to 0.36.0 in `pyproject.toml`, `uv lock`, and `uv build` | Pending | |
| Task 16 | CI parity verification, remote CI push, and green parity confirmation | Pending | |

---

## 2. Decision Log

| Decision | Options Considered | Rationale & Evidence |
|---|---|---|
| InfluxDB Line Protocol format | (a) InfluxDB 1.x line protocol, (b) InfluxDB 2.0 / Telegraf compatible line protocol with nanosecond precision | (b) InfluxDB 2.0 format is universally supported across InfluxDB v2/v3, Telegraf, QuestDB, VictoriaMetrics, and Prometheus Remote-Write ingestion bridges. |
| Remote push transport | (a) Third-party `requests` / `httpx` / `influxdb_client`, (b) Python standard library `urllib.request` | (b) Pure standard library `urllib.request` keeps deployment zero-dependency, lightweight, and robust across all Python runtime environments. |
| Alert root-cause graph algorithm | (a) Full Bayesian network, (b) Directed Acyclic Graph (DAG) with causal rule evaluation over timestamped windows | (b) Explicit deterministic causal DAG with parent->child dependency matching enables transparent, predictable root-cause attribution without training data or probabilistic uncertainty. |

---

## 3. CI & Validation Log

| Command | Result |
|---|---|
| `uv run ruff format --check .` | ✅ Baseline clean |
| `uv run ruff check .` | ✅ Baseline clean |
| `uv run mypy` | ✅ Baseline clean |
| `uv run --extra dashboard --extra tracking --extra spotify pytest -q` | ✅ Baseline 1,218 passed |
| `uv build` | ✅ Baseline built cleanly |

---

## 4. Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-1 InfluxDB Line Protocol 2.0 formatter | Pending | |
| AC-2 Push telemetry adapter & result tracking | Pending | |
| AC-3 Alert correlation graph & root-cause detection | Pending | |
| AC-4 FastAPI metrics push & correlation endpoints | Pending | |
| AC-5 Operator CLI export, push, and correlation commands | Pending | |
| AC-6 Streamlit dashboard correlation indicators | Pending | |
| AC-7 100% green CI locally and remotely | Pending | |

---

## 5. Blockers

None.
