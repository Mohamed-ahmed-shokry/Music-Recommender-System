# Phase 167 & 168 Progress Record & Decision Log

## Phase 167 & 168: Size-Capped Log Rotation, Retention Pruning, DLQ Exponential Retry, and Archive Snapshots (0.37.0)

- **Milestone**: 0.37.0
- **Status**: In Progress
- **Branch**: `phase/167-alert-rotation-and-retention`

---

## 1. Task Status

| Task ID | Task Description | Status | Commit |
|---|---|---|---|
| Task 1 | Roadmap, Plan & Progress initialization | Completed | `d057d67`, `3b38839` |
| Task 2 | `AlertLogRotationConfig` and size-capped rotation in `AlertSpilloverStorage` | Completed | `b1242f2`, `32a7a3a` |
| Task 3 | Age-based retention pruning in `AlertSpilloverStorage` | Completed | `741aafe` |
| Task 4 | Unit tests for log rotation and retention pruning in `tests/test_alerting.py` | Completed | `9772aea` |
| Task 5 | `create_archive_bundle` and `restore_archive_bundle` in `alerting.py` | Completed | `d351523` |
| Task 6 | Unit tests for archive bundle creation and extraction in `tests/test_alerting.py` | Completed | `6e55ac3` |
| Task 7 | `DLQRetryPolicy`, `DLQRetryResult`, `DLQRetryReport`, and `retry_dead_letters` in `alerting.py` | Completed | `1f20869`, `bf951dd`, `5d624ba` |
| Task 8 | Unit tests for DLQ retry engine in `tests/test_alerting.py` | Completed | `1d4fbe7` |
| Task 9 | Integration in `RecommenderService` in `src/music_recommender/service.py` | In Progress | |
| Task 10 | Service tests in `tests/test_service.py` | Pending | |
| Task 11 | Endpoints `/alerts/rotate`, `/alerts/dlq/retry`, `/alerts/archive`, `/alerts/storage/stats` in `api/main.py` | Pending | |
| Task 12 | API integration tests in `tests/test_api.py` | Pending | |
| Task 13 | CLI commands `rotate-alerts`, `retry-dlq`, and `archive-alerts` in `cli.py` | Pending | |
| Task 14 | CLI unit tests in `tests/test_cli.py` | Pending | |
| Task 15 | Storage rotation and retry indicators in `src/music_recommender/dashboard.py` | Pending | |
| Task 16 | Dashboard tests in `tests/test_dashboard.py` | Pending | |
| Task 17 | Documentation updates in `README.md` and `CHANGELOG.md` | Pending | |
| Task 18 | Package version bump to `0.37.0` in `pyproject.toml`, `uv lock`, and `uv build` | Pending | |
| Task 19 | CI parity verification, push, open draft PR, monitor remote CI to green | Pending | |

---

## 2. Decision Log

| Decision | Options Considered | Rationale & Evidence |
|---|---|---|
| Log rotation naming convention | (a) Standard `.1`, `.2`, ..., `.N` suffix, (b) Date-timestamped filenames | (a) Standard numeric backup suffixes (`.1`, `.2`, etc.) maintain deterministic generation indexing, fixed disk space bounding, and compatibility with standard logrotate tools. |
| Archive bundle formats | (a) Tarball (`.tar.gz`) only, (b) Zip (`.zip`) only, (c) Both `.tar.gz` and `.zip` supported via Python standard library `tarfile` and `zipfile` | (c) Supporting both `.tar.gz` and `.zip` with standard library modules provides cross-platform flexibility (Linux/Docker and Windows operators) without third-party dependencies. |
| DLQ retry backoff algorithm | (a) Fixed constant interval, (b) Truncated exponential backoff with jitter | (b) Exponential backoff (`min(max_delay, base_delay * (multiplier ** retry_count))`) with optional jitter prevents thundering herd against recovering downstream webhooks. |

---

## 3. CI & Validation Log

| Command | Result |
|---|---|
| `uv run ruff format --check .` | Pending |
| `uv run ruff check .` | Pending |
| `uv run mypy` | Pending |
| `uv run --extra dashboard --extra tracking --extra spotify pytest -q` | Pending |
| `uv build` | Pending |

---

## 4. Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-1 Size-capped log rotation & retention pruning | Pending | |
| AC-2 Disaster recovery archive bundling (`.tar.gz`/`.zip`) | Pending | |
| AC-3 DLQ exponential retry engine | Pending | |
| AC-4 FastAPI rotation, retry, and archive endpoints | Pending | |
| AC-5 Operator CLI rotation, retry, and archive commands | Pending | |
| AC-6 Streamlit dashboard storage management visibility | Pending | |
| AC-7 100% green CI locally and remotely | Pending | |

---

## 5. Blockers

None.
