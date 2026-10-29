# Phase 159 & 160 Progress Record & Decision Log

## Phase 159 & 160: Alert Dead-Letter Queue (DLQ), Persistent Spillover Storage, Incident Sequence Replay, and Operator Audit Tooling (0.35.0)

- **Milestone**: 0.35.0
- **Status**: In Progress
- **Branch**: `phase/159-alert-dlq-and-replay`

---

## 1. Task Status

| Task ID | Task Description | Status | Commit |
|---|---|---|---|
| Task 1 | Roadmap, Plan & Progress initialization | Completed | `010bdf2`, `12622ba` |
| Task 2 | `DeadLetterAlertRecord` and `AlertSpilloverStorage` in `alerting.py` | In Progress | |
| Task 3 | `AlertReplayEngine` and replay report models in `alerting.py` | Planned | |
| Task 4 | Eviction spillover in `AlertHistoryBuffer` in `alerting.py` | Planned | |
| Task 5 | Unit tests for storage, DLQ, and replay in `tests/test_alerting.py` | Planned | |
| Task 6 | Service integration in `src/music_recommender/service.py` | Planned | |
| Task 7 | DLQ ingestion, endpoints (`/alerts/dlq`, `/alerts/replay`) in `api/main.py` | Planned | |
| Task 8 | API integration tests in `tests/test_api.py` | Planned | |
| Task 9 | `alert-dlq` and `replay-alerts` CLI commands in `src/music_recommender/cli.py` | Planned | |
| Task 10 | Unit tests for `alert-dlq` and `replay-alerts` in `tests/test_cli.py` | Planned | |
| Task 11 | Dashboard DLQ badge & spillover visibility in `dashboard.py` | Planned | |
| Task 12 | Dashboard DLQ tests in `tests/test_dashboard.py` | Planned | |
| Task 13 | Documentation updates in `README.md` and `CHANGELOG.md` | Planned | |
| Task 14 | Version bump to 0.35.0 in `pyproject.toml`, `uv lock`, and `uv build` | Planned | |
| Task 15 | CI parity verification, remote CI push, and green parity confirmation | Planned | |

---

## 2. Decision Log

| Decision | Options Considered | Rationale & Evidence |
|---|---|---|
| Disk storage format | (a) SQLite database, (b) JSON Lines (`.jsonl`), (c) Raw text files | (b) JSON Lines wins: append-only zero-lock crash-safe writes, streamable line-by-line inspection with standard CLI utilities (`grep`, `jq`), human-readable, and zero additional dependency overhead. |
| In-memory vs. disk spillover | (a) Drop oldest events on eviction, (b) Spill evicted events to disk before dropping from RAM | (b) Spilling evicted events guarantees bounded memory consumption (default 500 in RAM) without losing long-term historical incident records. |
| Dead-Letter Queue handling | (a) Reject with 400 and discard, (b) Record unparseable/rejected payloads to dedicated DLQ file with error context | (b) Preserving malformed alert notifications in DLQ enables forensic diagnosis of misconfigured external webhooks or payload syntax mismatches without service crashing. |

---

## 3. CI & Validation Log

| Command | Result |
|---|---|
| `uv run ruff format --check .` (baseline) | ✅ 65 files already formatted |
| `uv run ruff check .` (baseline) | ✅ All checks passed |
| `uv run mypy` (baseline) | ✅ Success: no issues found in 30 source files |
| `uv build` (baseline) | ✅ Distribution packages built cleanly |

---

## 4. Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-1 `DeadLetterAlertRecord` & `AlertSpilloverStorage` | Pending | |
| AC-2 Eviction spillover in `AlertHistoryBuffer` | Pending | |
| AC-3 DLQ & replay FastAPI endpoints | Pending | |
| AC-4 `alert-dlq` & `replay-alerts` CLI commands | Pending | |
| AC-5 Dashboard DLQ warning indicator | Pending | |
| AC-6 100% green CI locally and remotely | Pending | |

---

## 5. Blockers

None.
