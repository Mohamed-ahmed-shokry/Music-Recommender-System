# Milestone 0.50.0 Progress Record — Phase 193 & 194

## 1. Micro-Task Checklist
- [x] Task 1: Define Phase 193 & 194 roadmap, update `PLAN.md`, and initialize `docs/PROGRESS.md`.
- [x] Task 2: Implement `OnboardingStatus`, `ExternalArtistProfile`, `TransferNeighbor`, and `OnboardedArtistRecord` in `src/music_recommender/cold_start_transfer.py`.
- [x] Task 3: Add unit tests for cold-start onboarding models in `tests/test_cold_start_transfer.py`.
- [x] Task 4: Implement `ColdStartTransferEngine` and global accessors in `src/music_recommender/cold_start_transfer.py`.
- [x] Task 5: Add unit tests for metadata transfer projection, artifact anchor syncing, and JSON persistence in `tests/test_cold_start_transfer.py`.
- [x] Task 6: Integrate `ColdStartTransferEngine` into `RecommenderService` (`src/music_recommender/service.py`) and Prometheus metrics (`src/music_recommender/telemetry.py`).
- [x] Task 7: Add service and telemetry integration tests in `tests/test_service.py`.
- [x] Task 8: Implement `/onboarding/*` endpoints in `api/main.py`.
- [x] Task 9: Add API integration tests for `/onboarding/*` in `tests/test_api.py`.
- [ ] Task 10: Implement CLI commands `onboarding-status` and `onboard-artist` in `src/music_recommender/cli.py`.
- [ ] Task 11: Add CLI tests in `tests/test_cli.py`.
- [ ] Task 12: Add `"Cold-Start Onboarding"` tab to `src/music_recommender/dashboard.py` and test in `tests/test_dashboard.py`.
- [ ] Task 13: Update `README.md`, `CHANGELOG.md`, `PLAN.md`, and `docs/PROGRESS.md` for `0.50.0`.
- [ ] Task 14: Bump version to `0.50.0` in `pyproject.toml`, run `uv lock`, and `uv build`.
- [ ] Task 15: Run full local CI-parity checks, push to remote, monitor Draft PR CI to 100% green.

## 2. Acceptance Criteria Tracking
| Criterion | Status | Verification Evidence |
| :--- | :---: | :--- |
| AC-1: ColdStartTransferEngine projects external artist metadata into normalized anchor-aligned embeddings with confidence scores | `Passed` | `tests/test_cold_start_transfer.py` |
| AC-2: ColdStartTransferEngine syncs catalog anchors from RecommenderArtifact and supports JSON state persistence | `Passed` | `tests/test_cold_start_transfer.py` |
| AC-3: RecommenderService exposes cold-start onboarding methods and Prometheus exports cold-start onboarding metrics | `Passed` | `tests/test_service.py` |
| AC-4: FastAPI /onboarding/* endpoints validate payloads and manage artist onboarding and status inspection | `Passed` | `tests/test_api.py` |
| AC-5: CLI commands onboarding-status and onboard-artist execute cleanly in text and --json modes | `Pending` | `tests/test_cli.py` |
| AC-6: Streamlit dashboard renders Cold-Start Onboarding tab cleanly without errors | `Pending` | `tests/test_dashboard.py` |
| AC-7: All local CI-parity checks and remote CI workflow runs pass 100% | `Pending` | Local checks + GitHub Actions runs |

## 3. Autonomous Decisions Log
- **Decision 1 (Weighted Metadata Jaccard Transfer Projection)**: Compute semantic similarity between incoming external artist profiles and catalog anchors using weighted tag overlap (genres 2.0, moods 1.5, country 0.5, era 0.5) so zero-interaction artists receive deterministic, unit-normalized embeddings aligned with the collaborative latent space.
- **Decision 2 (Deterministic Hash Fallback Embedding)**: When no catalog anchor shares metadata with an incoming artist, generate a deterministic SHA-256-seeded unit vector matching the anchor embedding dimension and flag low confidence so onboarding never crashes on novel genres.
- **Decision 3 (Automatic Artifact Anchor Sync & JSON Persistence)**: Support `sync_from_artifact(artifact)` on `ColdStartTransferEngine` plus `export_state()` / `import_state()` / `save_state()` / `load_state()` consistent with the repository's persistence pattern.
