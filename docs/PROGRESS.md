# Milestone 0.44.0 Progress Record — Phase 181 & 182

## 1. Micro-Task Checklist
- [x] Task 1: Define Phase 181 & 182 roadmap, update `PLAN.md`, and initialize `docs/PROGRESS.md`.
- [x] Task 2: Implement `DeploymentMode`, `CanaryPolicy`, `ABGuardrailConfig`, and `RoutingDecision` in `src/music_recommender/canary.py`.
- [x] Task 3: Add unit tests for canary policy validation and `RoutingDecision` in `tests/test_canary.py`.
- [x] Task 4: Implement `ArmMetrics`, `GuardrailEvaluation`, `ShadowCanaryRouter`, and global accessors in `src/music_recommender/canary.py`.
- [x] Task 5: Add unit tests for sticky bucketing, shadow comparisons, two-proportion z-test guardrails, auto-rollback, and state sync in `tests/test_canary.py`.
- [x] Task 6: Integrate `ShadowCanaryRouter` into `RecommenderService` (`src/music_recommender/service.py`) and Prometheus metrics (`src/music_recommender/telemetry.py`).
- [x] Task 7: Add service and telemetry integration tests in `tests/test_service.py`.
- [x] Task 8: Implement `/canary/*` endpoints in `api/main.py`.
- [x] Task 9: Add API integration tests for `/canary/*` in `tests/test_api.py`.
- [x] Task 10: Implement CLI commands `canary-status` and `canary-configure` in `src/music_recommender/cli.py`.
- [x] Task 11: Add CLI tests in `tests/test_cli.py`.
- [x] Task 12: Add `"Shadow & Canary"` tab to `src/music_recommender/dashboard.py` and test in `tests/test_dashboard.py`.
- [ ] Task 13: Update `README.md`, `CHANGELOG.md`, `PLAN.md`, and `docs/PROGRESS.md` for `0.44.0`.
- [ ] Task 14: Bump version to `0.44.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [ ] Task 15: Run full local CI-parity checks, push to remote, monitor Draft PR CI to 100% green.

## 2. Acceptance Criteria Tracking
| Criterion | Status | Verification Evidence |
| :--- | :---: | :--- |
| AC-1: ShadowCanaryRouter performs deterministic sticky bucketing, canary splitting, and shadow comparison recording | `Passed` | `tests/test_canary.py` |
| AC-2: ABGuardrailConfig and evaluate_guardrails() detect error/latency/CTR regressions and trigger automated rollback | `Passed` | `tests/test_canary.py` |
| AC-3: RecommenderService exposes canary governance methods and Prometheus exports canary metrics | `Passed` | `tests/test_service.py` |
| AC-4: FastAPI /canary/* endpoints validate payloads and manage canary config, evaluation, rollback, and promotion | `Passed` | `tests/test_api.py` |
| AC-5: CLI commands canary-status and canary-configure work in human-readable and --json modes | `Passed` | `tests/test_cli.py` |
| AC-6: Streamlit dashboard renders Shadow & Canary tab cleanly without errors | `Passed` | `tests/test_dashboard.py` |
| AC-7: All local CI-parity checks and remote CI workflow runs pass 100% | `Pending` | Local checks + GitHub Actions runs |

## 3. Autonomous Decisions Log
- **Decision 1 (Deterministic Sticky Bucketing)**: Use `hashlib.sha256` hashing over `f"{salt}:{entity_id}"` mapped to `[0.0, 1.0)` so traffic splitting is reproducible across restarts and nodes without external state stores.
- **Decision 2 (Standard-Library Statistical Significance)**: Implement the two-proportion z-test (`p_value = math.erfc(abs(z) / math.sqrt(2.0))`) using Python's standard library `math.erfc` instead of adding `scipy` as a dependency, keeping packaging minimal and deterministic.
- **Decision 3 (Non-Breaking Default Mode)**: Default `ShadowCanaryRouter` to `DeploymentMode.CHAMPION_ONLY` (`canary_weight=0.0`, `shadow_enabled=False`) so existing serving paths and tests remain untouched unless explicitly configured.
