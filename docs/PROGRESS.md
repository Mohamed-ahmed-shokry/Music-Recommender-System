# Milestone 0.42.0 Progress Record — Phase 177 & 178

## 1. Micro-Task Checklist
- [x] Task 1: Define Phase 177 & 178 roadmap, update plan, and initialize progress record.
- [x] Task 2: Implement `TraceContext` with W3C `traceparent`/`tracestate` in `src/music_recommender/tracing.py`.
- [x] Task 3: Add unit tests for `TraceContext` in `tests/test_tracing.py`.
- [x] Task 4: Implement `SpanRecord`, `SpanKind`, and `SpanStatus` in `src/music_recommender/tracing.py`.
- [x] Task 5: Add unit tests for `SpanRecord` and OTLP format conversion in `tests/test_tracing.py`.
- [x] Task 6: Implement `DistributedTracer` with context propagation and span buffer in `src/music_recommender/tracing.py`.
- [x] Task 7: Add unit tests for `DistributedTracer` in `tests/test_tracing.py`.
- [x] Task 8: Implement `AlertTraceCorrelator` in `src/music_recommender/tracing.py`.
- [x] Task 9: Add unit tests for `AlertTraceCorrelator` in `tests/test_tracing.py`.
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

## 2. Acceptance Criteria Tracking
| Criterion | Status | Verification Evidence |
| :--- | :---: | :--- |
| AC-1: TraceContext parses and formats W3C traceparent and tracestate headers | `Passed` | `tests/test_tracing.py` (5 tests passing) |
| AC-2: DistributedTracer manages thread/task context, records spans, converts to OTLP JSON | `Passed` | `tests/test_tracing.py` (15 tests passing) |
| AC-3: AlertTraceCorrelator correlates alert incidents with candidate traces | `Passed` | `tests/test_tracing.py` (17 tests passing) |
| AC-4: TraceContextMiddleware propagates trace context in HTTP headers | `Pending` | `tests/test_api.py` |
| AC-5: RecommenderService instruments internal pipeline operations and exposes queries | `Pending` | `tests/test_service.py` |
| AC-6: FastAPI endpoints /traces/* return validated schemas | `Pending` | `tests/test_api.py` |
| AC-7: Operator CLI commands trace-inspect and list-traces operate cleanly | `Pending` | `tests/test_cli.py` |
| AC-8: Streamlit dashboard renders live trace inspection and timeline breakdowns | `Pending` | `tests/test_dashboard.py` |
| AC-9: All local CI-parity checks and remote CI workflow runs pass 100% | `Pending` | Local checks + GitHub Actions runs |

## 3. Autonomous Decisions Log
- **Decision 1 (W3C Trace Context Standard)**: Implement canonical W3C `traceparent` specification (`version-trace_id-span_id-trace_flags` formatted as `00-{32hex}-{16hex}-{01|00}`) and optional `tracestate` comma-separated vendor pairs.
- **Decision 2 (OpenTelemetry Span Compatibility)**: Provide `to_otlp_dict()` serializer on `SpanRecord` mapping nanosecond timestamps (`startTimeUnixNano`, `endTimeUnixNano`), numeric span kinds, and typed attributes for OTLP/HTTP collector interoperability.
- **Decision 3 (Context Propagation)**: Use Python `contextvars.ContextVar` to achieve seamless async and multithreaded trace context propagation without explicit variable passing across function call boundaries.
