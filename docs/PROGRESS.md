# Milestone 0.41.0 Progress Record — Phase 175 & 176

## 1. Micro-Task Checklist
- [x] Task 1: Define Phase 175 & 176 roadmap, update plan, and initialize progress record.
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

## 2. Acceptance Criteria Tracking
| Criterion | Status | Verification Evidence |
| :--- | :---: | :--- |
| AC-1: AlertIncidentTracker tracks incident lifecycles on firing and resolved transitions | `Pending` | `tests/test_alerting.py` |
| AC-2: AlertIncidentAnalytics computes accurate MTTD, MTTR, MTBF, and SLO compliance | `Pending` | `tests/test_alerting.py` |
| AC-3: Prometheus exposition exports alert gauge and metric families | `Pending` | `tests/test_telemetry.py` |
| AC-4: RecommenderService coordinates incident lifecycle tracking and query methods | `Pending` | `tests/test_service.py` |
| AC-5: FastAPI endpoints /alerts/analytics/* return validated schemas | `Pending` | `tests/test_api.py` |
| AC-6: Operator CLI commands alert-analytics and incident-history function cleanly | `Pending` | `tests/test_cli.py` |
| AC-7: Streamlit dashboard renders live MTTD/MTTR indicators and SLO availability | `Pending` | `tests/test_dashboard.py` |
| AC-8: 100% green CI parity locally and remotely | `Pending` | Local checks + GitHub Actions runs |

## 3. Autonomous Decisions Log
- **Decision 1 (Incident Lifecycle Correlation)**: Correlate incidents by `rule_name` and severity transition. When an alert transitions from OK -> FIRING, create a new `IncidentLifecycleRecord`. If remediations or repeat transitions occur while active, append events. When it transitions FIRING -> RESOLVED, compute final `duration_seconds` and close the record.
- **Decision 2 (Telemetry Export Standard)**: OpenMetrics/Prometheus conventions: metric names `music_recommender_alert_active`, `music_recommender_alert_mttd_seconds`, `music_recommender_alert_mttr_seconds` labeled by `rule`.
