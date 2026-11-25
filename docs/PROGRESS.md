# Milestone 0.49.0 Progress Record — Phase 191 & 192

## 1. Micro-Task Checklist
- [x] Task 1: Define Phase 191 & 192 roadmap, update `PLAN.md`, and initialize `docs/PROGRESS.md`.
- [x] Task 2: Implement `IncidentSeverity`, `SLATarget`, `RunbookEntry`, `PostmortemActionItem`, `IncidentPostmortem`, and `SLAComplianceReport` in `src/music_recommender/postmortem.py`.
- [x] Task 3: Add unit tests for postmortem/SLA models in `tests/test_postmortem.py`.
- [x] Task 4: Implement `PostmortemEngine` and global accessors in `src/music_recommender/postmortem.py`.
- [x] Task 5: Add unit tests for runbook matching, postmortem generation, SLA compliance, and state persistence in `tests/test_postmortem.py`.
- [x] Task 6: Integrate `PostmortemEngine` into `RecommenderService` (`src/music_recommender/service.py`) and Prometheus metrics (`src/music_recommender/telemetry.py`).
- [x] Task 7: Add service and telemetry integration tests in `tests/test_service.py`.
- [ ] Task 8: Implement `/postmortem/*` endpoints in `api/main.py`.
- [ ] Task 9: Add API integration tests for `/postmortem/*` in `tests/test_api.py`.
- [ ] Task 10: Implement CLI commands `postmortem-generate` and `sla-report` in `src/music_recommender/cli.py`.
- [ ] Task 11: Add CLI tests in `tests/test_cli.py`.
- [ ] Task 12: Add `"Postmortems & SLA"` tab to `src/music_recommender/dashboard.py` and test in `tests/test_dashboard.py`.
- [ ] Task 13: Update `README.md`, `CHANGELOG.md`, `PLAN.md`, and `docs/PROGRESS.md` for `0.49.0`.
- [ ] Task 14: Bump version to `0.49.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [ ] Task 15: Run full local CI-parity checks, push to remote, monitor Draft PR CI to 100% green.

## 2. Acceptance Criteria Tracking
| Criterion | Status | Verification Evidence |
| :--- | :---: | :--- |
| AC-1: PostmortemEngine matches root-cause runbooks and generates structured postmortems with Markdown export | `Passed` | `tests/test_postmortem.py` |
| AC-2: PostmortemEngine tracks downtime events, computes SLA compliance/error-budget burn, and supports JSON persistence | `Passed` | `tests/test_postmortem.py` |
| AC-3: RecommenderService exposes postmortem/SLA methods and Prometheus exports postmortem and SLA metrics | `Passed` | `tests/test_service.py` |
| AC-4: FastAPI /postmortem/* endpoints validate payloads and serve postmortems, SLA reports, and runbook matching | `Pending` | `tests/test_api.py` |
| AC-5: CLI commands postmortem-generate and sla-report execute cleanly in text and --json modes | `Pending` | `tests/test_cli.py` |
| AC-6: Streamlit dashboard renders Postmortems & SLA tab cleanly without errors | `Pending` | `tests/test_dashboard.py` |
| AC-7: All local CI-parity checks and remote CI workflow runs pass 100% | `Pending` | Local checks + GitHub Actions runs |

## 3. Autonomous Decisions Log
- **Decision 1 (Built-in Root-Cause Runbook Registry)**: Pre-populate `PostmortemEngine` with domain-specific runbooks (latency spike, feature drift / stale embeddings, rate-limit surge, regional failover, privacy budget exhaustion) and match by normalized symptom/tag overlap so operators get immediate remediation steps without external dependencies.
- **Decision 2 (Deterministic SLA & Error-Budget Accounting)**: Compute availability percentage, allowed downtime minutes (`window_minutes * (1 - target_availability_pct / 100)`), error-budget consumed/remaining percentages, and burn rate directly from recorded downtime events and postmortem durations.
- **Decision 3 (Structured Markdown Export & JSON Persistence)**: Support `to_markdown()` on `IncidentPostmortem` for incident review docs and `export_state()` / `import_state()` / `save_state()` / `load_state()` on `PostmortemEngine` consistent with the repository's persistence pattern.
