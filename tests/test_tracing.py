"""Unit tests for distributed tracing models and W3C trace context."""

from __future__ import annotations

import pytest

from music_recommender.tracing import (
    DistributedTracer,
    SpanKind,
    SpanRecord,
    SpanStatus,
    TraceContext,
    generate_span_id,
    generate_trace_id,
)


def test_trace_id_and_span_id_generation() -> None:
    tid1 = generate_trace_id()
    tid2 = generate_trace_id()
    sid1 = generate_span_id()
    sid2 = generate_span_id()

    assert len(tid1) == 32
    assert len(sid1) == 16
    assert tid1 != tid2
    assert sid1 != sid2
    assert int(tid1, 16) > 0
    assert int(sid1, 16) > 0


def test_trace_context_validation() -> None:
    # Valid context
    ctx = TraceContext(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        trace_flags="01",
        trace_state="vendor=sample",
    )
    assert ctx.is_sampled is True
    assert (
        ctx.to_traceparent()
        == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )

    # Invalid trace_id length
    with pytest.raises(ValueError, match="Invalid W3C trace_id"):
        TraceContext(trace_id="short", span_id="00f067aa0ba902b7")

    # Invalid trace_id all zeros
    with pytest.raises(ValueError, match="Invalid W3C trace_id"):
        TraceContext(trace_id="0" * 32, span_id="00f067aa0ba902b7")

    # Invalid span_id all zeros
    with pytest.raises(ValueError, match="Invalid W3C span_id"):
        TraceContext(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="0" * 16,
        )

    # Invalid parent_span_id
    with pytest.raises(ValueError, match="Invalid W3C parent_span_id"):
        TraceContext(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            parent_span_id="bad_id",
        )


def test_w3c_traceparent_parsing() -> None:
    header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    ctx = TraceContext.from_traceparent(header, trace_state="rojo=1")
    assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert ctx.span_id == "00f067aa0ba902b7"
    assert ctx.trace_flags == "01"
    assert ctx.trace_state == "rojo=1"
    assert ctx.is_sampled is True
    assert ctx.to_traceparent() == header

    # Malformed headers
    with pytest.raises(ValueError, match="Malformed W3C traceparent"):
        TraceContext.from_traceparent("invalid-header")

    with pytest.raises(ValueError, match="Unsupported W3C traceparent version"):
        TraceContext.from_traceparent(
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        )


def test_new_root_and_new_child() -> None:
    root = TraceContext.new_root(sampled=True, trace_state="v=1")
    assert root.parent_span_id is None
    assert root.is_sampled is True
    assert root.trace_state == "v=1"

    child = root.new_child()
    assert child.trace_id == root.trace_id
    assert child.parent_span_id == root.span_id
    assert child.span_id != root.span_id
    assert child.is_sampled is True
    assert child.trace_state == "v=1"

    child_unsampled = root.new_child(sampled=False)
    assert child_unsampled.is_sampled is False
    assert child_unsampled.trace_flags == "00"


def test_trace_context_serialization() -> None:
    ctx = TraceContext(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        parent_span_id="5fb397be34d23b0f",
        trace_flags="01",
        trace_state="congo=t61rcWkgMzE",
    )
    d = ctx.to_dict()
    assert d["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert d["parent_span_id"] == "5fb397be34d23b0f"
    assert d["is_sampled"] is True

    restored = TraceContext.from_dict(d)
    assert restored == ctx


def test_span_lifecycle_and_finish() -> None:
    span = SpanRecord(
        span_id="1111222233334444",
        trace_id="aaaabbbbccccddddeeeeffff00001111",
        name="unit_task",
        kind=SpanKind.SERVER,
    )
    assert span.end_time is None
    assert span.duration_seconds == 0.0
    assert span.status == SpanStatus.UNSET

    span.finish(status=SpanStatus.OK, message="finished cleanly")
    assert span.end_time is not None
    assert span.duration_seconds >= 0.0
    assert span.status == SpanStatus.OK
    assert span.status_message == "finished cleanly"


def test_span_attributes_and_events() -> None:
    span = SpanRecord(
        span_id="1111222233334444",
        trace_id="aaaabbbbccccddddeeeeffff00001111",
        name="test_op",
    )
    span.set_attribute("http.method", "GET")
    span.set_attribute("http.status_code", 200)
    span.set_attribute("is_active", True)
    span.set_attribute("score", 0.95)

    assert span.attributes["http.method"] == "GET"
    assert span.attributes["http.status_code"] == 200
    assert span.attributes["is_active"] is True
    assert span.attributes["score"] == 0.95

    evt = span.add_event("cache_hit", {"key": "artist:10"})
    assert evt.name == "cache_hit"
    assert len(span.events) == 1
    assert span.events[0].attributes["key"] == "artist:10"


def test_span_record_exception() -> None:
    span = SpanRecord(
        span_id="1111222233334444",
        trace_id="aaaabbbbccccddddeeeeffff00001111",
        name="failing_op",
    )
    try:
        raise ValueError("database unavailable")
    except ValueError as exc:
        span.record_exception(exc, {"db.system": "sqlite"})

    assert span.status == SpanStatus.ERROR
    assert span.status_message == "ValueError: database unavailable"
    assert len(span.events) == 1
    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.type"] == "ValueError"
    assert span.events[0].attributes["exception.message"] == "database unavailable"
    assert span.events[0].attributes["db.system"] == "sqlite"


def test_span_serialization_roundtrip() -> None:
    span = SpanRecord(
        span_id="1111222233334444",
        trace_id="aaaabbbbccccddddeeeeffff00001111",
        parent_span_id="5555666677778888",
        name="recommend_pipeline",
        kind=SpanKind.INTERNAL,
        attributes={"user_id": 42},
    )
    span.add_event("stage_1", {"items": 10})
    span.finish(status=SpanStatus.OK)

    d = span.to_dict()
    assert d["span_id"] == "1111222233334444"
    assert d["kind"] == "internal"
    assert d["status"] == "ok"
    assert len(d["events"]) == 1

    restored = SpanRecord.from_dict(d)
    assert restored.span_id == span.span_id
    assert restored.trace_id == span.trace_id
    assert restored.parent_span_id == span.parent_span_id
    assert restored.kind == SpanKind.INTERNAL
    assert restored.status == SpanStatus.OK
    assert restored.attributes["user_id"] == 42
    assert len(restored.events) == 1
    assert restored.events[0].name == "stage_1"


def test_span_otlp_dict_structure() -> None:
    span = SpanRecord(
        span_id="1111222233334444",
        trace_id="aaaabbbbccccddddeeeeffff00001111",
        parent_span_id="5555666677778888",
        name="recommend_api",
        kind=SpanKind.SERVER,
    )
    span.set_attribute("endpoint", "/api/v1/recommend")
    span.set_attribute("user.id", 100)
    span.set_attribute("cached", False)
    span.set_attribute("latency_ms", 12.34)
    span.add_event("render_start", {"step": 1})
    span.finish(status=SpanStatus.OK, message="Success")

    otlp = span.to_otlp_dict()
    assert otlp["traceId"] == "aaaabbbbccccddddeeeeffff00001111"
    assert otlp["spanId"] == "1111222233334444"
    assert otlp["parentSpanId"] == "5555666677778888"
    assert otlp["name"] == "recommend_api"
    assert otlp["kind"] == 2  # SpanKind.SERVER
    assert "startTimeUnixNano" in otlp
    assert "endTimeUnixNano" in otlp
    assert otlp["status"]["code"] == 1  # SpanStatus.OK
    assert otlp["status"]["message"] == "Success"

    # Verify typed attributes
    attrs_by_key = {item["key"]: item["value"] for item in otlp["attributes"]}
    assert attrs_by_key["endpoint"] == {"stringValue": "/api/v1/recommend"}
    assert attrs_by_key["user.id"] == {"intValue": "100"}
    assert attrs_by_key["cached"] == {"boolValue": False}
    assert attrs_by_key["latency_ms"] == {"doubleValue": 12.34}

    # Verify event
    assert len(otlp["events"]) == 1
    assert otlp["events"][0]["name"] == "render_start"


def test_distributed_tracer_nested_spans() -> None:
    tracer = DistributedTracer()
    with tracer.span("outer_op", {"level": "outer"}) as s_outer:
        assert tracer.active_span is s_outer
        assert tracer.active_context is not None
        assert tracer.active_context.span_id == s_outer.span_id

        with tracer.span("inner_op", {"level": "inner"}) as s_inner:
            assert tracer.active_span is s_inner
            assert s_inner.trace_id == s_outer.trace_id
            assert s_inner.parent_span_id == s_outer.span_id

        # Returned back to outer
        assert tracer.active_span is s_outer

    # Outside span context
    assert tracer.active_span is None
    assert tracer.active_context is None

    trace_spans = tracer.get_trace(s_outer.trace_id)
    assert len(trace_spans) == 2
    span_names = [s.name for s in trace_spans]
    assert "outer_op" in span_names
    assert "inner_op" in span_names


def test_distributed_tracer_custom_context() -> None:
    tracer = DistributedTracer()
    ctx = TraceContext.new_root()

    with tracer.span("custom_op", context=ctx) as span:
        assert span.trace_id == ctx.trace_id
        assert span.parent_span_id == ctx.span_id

    spans = tracer.get_spans(trace_id=ctx.trace_id)
    assert len(spans) == 1
    assert spans[0].name == "custom_op"


def test_distributed_tracer_exception_handling() -> None:
    tracer = DistributedTracer()

    with (
        pytest.raises(RuntimeError, match="pipeline breakdown"),
        tracer.span("faulty_op"),
    ):
        raise RuntimeError("pipeline breakdown")

    err_spans = tracer.get_spans(status=SpanStatus.ERROR)
    assert len(err_spans) == 1
    span = err_spans[0]
    assert span.name == "faulty_op"
    assert span.status == SpanStatus.ERROR
    assert "RuntimeError: pipeline breakdown" in (span.status_message or "")
    assert any(e.name == "exception" for e in span.events)


def test_distributed_tracer_queries_and_clear() -> None:
    tracer = DistributedTracer(max_spans=50)

    for i in range(10):
        with tracer.span(f"op_{i % 2}"):
            pass

    assert len(tracer.get_spans(name="op_0")) == 5
    assert len(tracer.get_spans(name="op_1")) == 5
    assert len(tracer.get_spans(limit=3)) == 3

    tracer.clear()
    assert len(tracer.get_spans()) == 0


def test_distributed_tracer_otlp_export() -> None:
    tracer = DistributedTracer()
    with tracer.span("server_req", kind=SpanKind.SERVER):
        pass

    export = tracer.to_otlp_export(
        service_name="test-service",
        service_version="1.0.0",
    )
    assert "resourceSpans" in export
    assert len(export["resourceSpans"]) == 1
    resource_attrs = export["resourceSpans"][0]["resource"]["attributes"]
    svc_attr = next(a for a in resource_attrs if a["key"] == "service.name")
    assert svc_attr["value"]["stringValue"] == "test-service"
    scope_spans = export["resourceSpans"][0]["scopeSpans"]
    assert len(scope_spans[0]["spans"]) == 1
    assert scope_spans[0]["spans"][0]["name"] == "server_req"
