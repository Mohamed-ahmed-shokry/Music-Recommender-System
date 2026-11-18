# Milestone 0.47.0 Progress Record — Phase 187 & 188

## 1. Micro-Task Checklist
- [x] Task 1: Define Phase 187 & 188 roadmap, update `PLAN.md`, and initialize `docs/PROGRESS.md`.
- [ ] Task 2: Implement `ExplanationNode`, `ExplanationEdge`, `CounterfactualSuggestion`, and `UserPreferenceProfile` in `src/music_recommender/explainability.py`.
- [ ] Task 3: Add unit tests for graph primitives, counterfactuals, and preference profiles in `tests/test_explainability.py`.
- [ ] Task 4: Implement `ExplainabilityEngine` and global accessors in `src/music_recommender/explainability.py`.
- [ ] Task 5: Add unit tests for preference filtering/re-ranking, graph construction, counterfactual recourse, and state persistence in `tests/test_explainability.py`.
- [ ] Task 6: Integrate `ExplainabilityEngine` into `RecommenderService` (`src/music_recommender/service.py`) and Prometheus metrics (`src/music_recommender/telemetry.py`).
- [ ] Task 7: Add service and telemetry integration tests in `tests/test_service.py`.
- [ ] Task 8: Implement `/explain/*` endpoints in `api/main.py`.
- [ ] Task 9: Add API integration tests for `/explain/*` in `tests/test_api.py`.
- [ ] Task 10: Implement CLI commands `explain-graph` and `user-preferences` in `src/music_recommender/cli.py`.
- [ ] Task 11: Add CLI tests in `tests/test_cli.py`.
- [ ] Task 12: Add `"Explainability & Controls"` tab to `src/music_recommender/dashboard.py` and test in `tests/test_dashboard.py`.
- [ ] Task 13: Update `README.md`, `CHANGELOG.md`, `PLAN.md`, and `docs/PROGRESS.md` for `0.47.0`.
- [ ] Task 14: Bump version to `0.47.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [ ] Task 15: Run full local CI-parity checks, push to remote, monitor Draft PR CI to 100% green.

## 2. Acceptance Criteria Tracking
| Criterion | Status | Verification Evidence |
| :--- | :---: | :--- |
| AC-1: ExplainabilityEngine constructs valid user-to-factor-to-artist explanation graphs and counterfactual recourse suggestions | `Pending` | `tests/test_explainability.py` |
| AC-2: UserPreferenceProfile and apply_preferences() enforce genre boosts/mutes, mood boosts, and artist blocks with JSON persistence | `Pending` | `tests/test_explainability.py` |
| AC-3: RecommenderService exposes explainability/preference methods and Prometheus exports explainability metrics | `Pending` | `tests/test_service.py` |
| AC-4: FastAPI /explain/* endpoints validate payloads and manage preferences, graphs, and counterfactuals | `Pending` | `tests/test_api.py` |
| AC-5: CLI commands explain-graph and user-preferences execute cleanly in text and --json modes | `Pending` | `tests/test_cli.py` |
| AC-6: Streamlit dashboard renders Explainability & Controls tab cleanly without errors | `Pending` | `tests/test_dashboard.py` |
| AC-7: All local CI-parity checks and remote CI workflow runs pass 100% | `Pending` | Local checks + GitHub Actions runs |

## 3. Autonomous Decisions Log
- **Decision 1 (Bipartite Factor Explanation Graph)**: Represent recommendations as a 3-layer directed graph (`user` -> `genre`/`mood`/`factor` -> `artist`) with normalized edge weights so UI and API consumers can render interactive explanations directly without Neo4j.
- **Decision 2 (Additive + Multiplicative Preference Re-Ranking)**: Filter `blocked_artists`, apply multiplicative suppression (`0.25x`) for `muted_genres`, and additive score boosts for `boosted_genres` and `boosted_moods`, keeping preference overrides predictable and reversible.
- **Decision 3 (Deterministic JSON Snapshot Persistence)**: Implement `export_state()` / `import_state()` / `save_state()` / `load_state()` on `ExplainabilityEngine` consistent with `ratelimit.py`, `canary.py`, `feature_drift.py`, and `replication.py`.
