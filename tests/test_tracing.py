"""Unit tests for distributed tracing models and W3C trace context."""

from __future__ import annotations

import pytest

from music_recommender.tracing import (
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
