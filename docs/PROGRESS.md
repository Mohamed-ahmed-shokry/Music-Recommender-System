# Phase 159 & 160 Progress Record & Decision Log

## Phase 159 & 160: Alert Dead-Letter Queue (DLQ), Persistent Spillover Storage, Incident Sequence Replay, and Operator Audit Tooling (0.35.0)

- **Milestone**: 0.35.0
- **Status**: Completed
- **Branch**: `phase/159-alert-dlq-and-replay`

---

## 1. Task Status

| Task ID | Task Description | Status | Commit |
|---|---|---|---|
| Task 1 | Roadmap, Plan & Progress initialization | Completed | `010bdf2`, `12622ba`, `77cff86` |
| Task 2 | `DeadLetterAlertRecord` and `AlertSpilloverStorage` in `alerting.py` | Completed | `5c839a1` |
| Task 3 | `AlertReplayEngine` and replay report models in `alerting.py` | Completed | `ac1fbf5` |
| Task 4 | Eviction spillover in `AlertHistoryBuffer` in `alerting.py` | Completed | `b59de66` |
| Task 5 | Unit tests for storage, DLQ, and replay in `tests/test_alerting.py` | Completed | `d585b8b` |
| Task 6 | Service integration in `src/music_recommender/service.py` | Completed | `4137956`, `c693066` |
| Task 7 | DLQ ingestion, endpoints (`/alerts/dlq`, `/alerts/replay`) in `api/main.py` | Completed | `fd0cf14` |
| Task 8 | API integration tests in `tests/test_api.py` | Completed | `9455763` |
| Task 9 | `alert-dlq` and `replay-alerts` CLI commands in `src/music_recommender/cli.py` | Completed | `dcaed90` |
| Task 10 | Unit tests for `alert-dlq` and `replay-alerts` in `tests/test_cli.py` | Completed | `a0f5035` |
| Task 11 | Dashboard DLQ badge & spillover visibility in `dashboard.py` | Completed | `4fb7e64` |
| Task 12 | Dashboard DLQ tests in `tests/test_dashboard.py` | Completed | `a7ad875` |
| Task 13 | Documentation updates in `README.md` and `CHANGELOG.md` | Completed | `3307c64`, `b00ac0e` |
| Task 14 | Version bump to 0.35.0 in `pyproject.toml`, `uv lock`, and `uv build` | Completed | `9816fb8`, `6595a7f` |
| Task 15 | CI parity verification, remote CI push, and green parity confirmation | Completed | `0ba3bdc`, `dedd6f4` |

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
| `uv run ruff format --check .` | ✅ All files formatted |
| `uv run ruff check .` | ✅ All checks passed |
| `uv run mypy` | ✅ Success: no issues found in 30 source files |
| `uv run --extra dashboard --extra tracking --extra spotify pytest -q` | ✅ 1,223 passed in ~100s |
| `uv build` | ✅ Successfully built distributions for v0.35.0 |

---

## 4. Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-1 `DeadLetterAlertRecord` & `AlertSpilloverStorage` | Met | Implemented in `alerting.py` and validated with full CRUD and corruption tests in `tests/test_alerting.py`. |
| AC-2 Eviction spillover in `AlertHistoryBuffer` | Met | Verified in `tests/test_alerting.py`: buffer evicts oldest events to JSONL disk spillover and merges on combined queries. |
| AC-3 DLQ & replay FastAPI endpoints | Met | Verified in `tests/test_api.py`: `GET /alerts/dlq`, `DELETE /alerts/dlq`, `POST /alerts/replay`, and `GET /alerts/history?include_spillover=true`. |
| AC-4 `alert-dlq` & `replay-alerts` CLI commands | Met | Verified in `tests/test_cli.py`: table and JSON formats, purge option, dry-run replay, and input validation. |
| AC-5 Dashboard DLQ warning indicator | Met | Verified in `tests/test_dashboard.py`: Streamlit AppTest validates sidebar badge and warning for pending DLQ items. |
| AC-6 100% green CI locally and remotely | Met | All checks pass locally and GitHub Actions CI workflow runs 100% green on PR #25. |

---

## 5. Blockers

None.
