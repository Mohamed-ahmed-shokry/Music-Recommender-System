"""Distributed tracing, W3C Trace Context propagation, and OTel interop."""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

_TRACE_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_SPAN_ID_RE = re.compile(r"^[0-9a-fA-F]{16}$")
_TRACE_FLAGS_RE = re.compile(r"^[0-9a-fA-F]{2}$")
_TRACEPARENT_RE = re.compile(
    r"^([0-9a-fA-F]{2})-([0-9a-fA-F]{32})-([0-9a-fA-F]{16})-([0-9a-fA-F]{2})$"
)

INVALID_TRACE_ID = "0" * 32
INVALID_SPAN_ID = "0" * 16


def generate_trace_id() -> str:
    """Generate a random 16-byte (32-hex-character) trace identifier."""
    while True:
        tid = secrets.token_hex(16).lower()
        if tid != INVALID_TRACE_ID:
            return tid


def generate_span_id() -> str:
    """Generate a random 8-byte (16-hex-character) span identifier."""
    while True:
        sid = secrets.token_hex(8).lower()
        if sid != INVALID_SPAN_ID:
            return sid


@dataclass(frozen=True)
class TraceContext:
    """W3C Trace Context representation for distributed trace propagation."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    trace_flags: str = "01"
    trace_state: str = ""

    def __post_init__(self) -> None:
        tid = self.trace_id.lower()
        sid = self.span_id.lower()
        flags = self.trace_flags.lower()

        if not _TRACE_ID_RE.match(tid) or tid == INVALID_TRACE_ID:
            raise ValueError(f"Invalid W3C trace_id: '{self.trace_id}'")
        if not _SPAN_ID_RE.match(sid) or sid == INVALID_SPAN_ID:
            raise ValueError(f"Invalid W3C span_id: '{self.span_id}'")
        if not _TRACE_FLAGS_RE.match(flags):
            raise ValueError(f"Invalid W3C trace_flags: '{self.trace_flags}'")

        if self.parent_span_id is not None:
            psid = self.parent_span_id.lower()
            if not _SPAN_ID_RE.match(psid) or psid == INVALID_SPAN_ID:
                raise ValueError(f"Invalid W3C parent_span_id: '{self.parent_span_id}'")

        object.__setattr__(self, "trace_id", tid)
        object.__setattr__(self, "span_id", sid)
        object.__setattr__(self, "trace_flags", flags)
        if self.parent_span_id is not None:
            object.__setattr__(self, "parent_span_id", self.parent_span_id.lower())

    @property
    def is_sampled(self) -> bool:
        """Return True if the sampled flag (0x01) is set in trace_flags."""
        try:
            return bool(int(self.trace_flags, 16) & 1)
        except ValueError:
            return False

    def to_traceparent(self) -> str:
        """Serialize context according to W3C traceparent header format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"

    def new_child(self, sampled: bool | None = None) -> TraceContext:
        """Create a child context inheriting trace_id and trace_state."""
        flags = ("01" if sampled else "00") if sampled is not None else self.trace_flags
        return TraceContext(
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            trace_flags=flags,
            trace_state=self.trace_state,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert trace context to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
            "traceparent": self.to_traceparent(),
            "is_sampled": self.is_sampled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceContext:
        """Reconstruct trace context from dictionary representation."""
        return cls(
            trace_id=str(data.get("trace_id", "")),
            span_id=str(data.get("span_id", "")),
            parent_span_id=(
                str(data["parent_span_id"])
                if data.get("parent_span_id") is not None
                else None
            ),
            trace_flags=str(data.get("trace_flags", "01")),
            trace_state=str(data.get("trace_state", "")),
        )

    @classmethod
    def from_traceparent(cls, header: str, trace_state: str = "") -> TraceContext:
        """Parse W3C traceparent header string (00-{trace_id}-{span_id}-{flags})."""
        raw = header.strip()
        match = _TRACEPARENT_RE.match(raw)
        if not match:
            raise ValueError(f"Malformed W3C traceparent header: '{header}'")
        version, tid, sid, flags = match.groups()
        if version == "ff":
            raise ValueError(f"Unsupported W3C traceparent version 0xff in '{header}'")
        return cls(
            trace_id=tid,
            span_id=sid,
            trace_flags=flags,
            trace_state=trace_state.strip(),
        )

    @classmethod
    def new_root(cls, sampled: bool = True, trace_state: str = "") -> TraceContext:
        """Generate a new root trace context."""
        return cls(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            parent_span_id=None,
            trace_flags="01" if sampled else "00",
            trace_state=trace_state.strip(),
        )


class SpanKind(StrEnum):
    """OpenTelemetry span kind classifications."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"

    def to_otlp_code(self) -> int:
        """Map span kind to standard OpenTelemetry numeric enum code."""
        mapping = {
            SpanKind.INTERNAL: 1,
            SpanKind.SERVER: 2,
            SpanKind.CLIENT: 3,
            SpanKind.PRODUCER: 4,
            SpanKind.CONSUMER: 5,
        }
        return mapping[self]


class SpanStatus(StrEnum):
    """OpenTelemetry span completion status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"

    def to_otlp_code(self) -> int:
        """Map span status to standard OpenTelemetry numeric code."""
        mapping = {
            SpanStatus.UNSET: 0,
            SpanStatus.OK: 1,
            SpanStatus.ERROR: 2,
        }
        return mapping[self]


def _attributes_to_otlp(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Python dictionary of attributes to OTLP KeyValue format."""
    result: list[dict[str, Any]] = []
    for k, v in attrs.items():
        if isinstance(v, bool):
            val: dict[str, Any] = {"boolValue": v}
        elif isinstance(v, int):
            val = {"intValue": str(v)}
        elif isinstance(v, float):
            val = {"doubleValue": v}
        else:
            val = {"stringValue": str(v)}
        result.append({"key": str(k), "value": val})
    return result


@dataclass
class SpanEvent:
    """An event or annotation occurring during a span's lifetime."""

    name: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert span event to dictionary representation."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanEvent:
        """Reconstruct span event from dictionary representation."""
        return cls(
            name=str(data.get("name", "")),
            timestamp=str(data.get("timestamp", "")),
            attributes=dict(data.get("attributes", {})),
        )

    def to_otlp_dict(self) -> dict[str, Any]:
        """Convert span event to OpenTelemetry OTLP JSON format."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            nano = int(dt.timestamp() * 1e9)
        except Exception:
            nano = int(time.time() * 1e9)
        return {
            "timeUnixNano": str(nano),
            "name": self.name,
            "attributes": _attributes_to_otlp(self.attributes),
        }


@dataclass
class SpanRecord:
    """A trace span record representing a single unit of execution."""

    span_id: str
    trace_id: str
    name: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time: str | None = None
    duration_seconds: float = 0.0
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    def finish(
        self,
        status: SpanStatus | str | None = None,
        message: str | None = None,
    ) -> SpanRecord:
        """Finish span execution, setting end_time and duration_seconds."""
        now_dt = datetime.now(UTC)
        self.end_time = now_dt.isoformat()
        try:
            start_dt = datetime.fromisoformat(self.start_time)
            self.duration_seconds = max(0.0, (now_dt - start_dt).total_seconds())
        except Exception:
            self.duration_seconds = 0.0
        if status is not None:
            self.status = SpanStatus(status) if isinstance(status, str) else status
        if message is not None:
            self.status_message = message
        return self

    def set_attribute(self, key: str, value: Any) -> SpanRecord:
        """Set a single span attribute."""
        self.attributes[str(key)] = value
        return self

    def add_event(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> SpanEvent:
        """Append an event annotation to this span."""
        event = SpanEvent(
            name=name,
            attributes=attributes or {},
        )
        self.events.append(event)
        return event

    def record_exception(
        self, exc: BaseException, attributes: dict[str, Any] | None = None
    ) -> SpanRecord:
        """Record an unhandled exception, setting status to ERROR."""
        self.status = SpanStatus.ERROR
        self.status_message = f"{type(exc).__name__}: {exc}"
        event_attrs: dict[str, Any] = {
            "exception.type": type(exc).__name__,
            "exception.message": str(exc),
        }
        if attributes:
            event_attrs.update(attributes)
        self.add_event(name="exception", attributes=event_attrs)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert span record to dictionary representation."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration_seconds, 6),
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": dict(self.attributes),
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanRecord:
        """Reconstruct span record from dictionary representation."""
        kind_str = str(data.get("kind", SpanKind.INTERNAL.value))
        try:
            kind = SpanKind(kind_str)
        except ValueError:
            kind = SpanKind.INTERNAL

        status_str = str(data.get("status", SpanStatus.UNSET.value))
        try:
            status = SpanStatus(status_str)
        except ValueError:
            status = SpanStatus.UNSET

        events_raw = data.get("events", [])
        events = (
            [SpanEvent.from_dict(e) for e in events_raw]
            if isinstance(events_raw, list)
            else []
        )

        return cls(
            span_id=str(data["span_id"]),
            trace_id=str(data["trace_id"]),
            name=str(data.get("name", "")),
            parent_span_id=(
                str(data["parent_span_id"])
                if data.get("parent_span_id") is not None
                else None
            ),
            kind=kind,
            start_time=str(data.get("start_time", "")),
            end_time=(
                str(data["end_time"]) if data.get("end_time") is not None else None
            ),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            status=status,
            status_message=(
                str(data["status_message"])
                if data.get("status_message") is not None
                else None
            ),
            attributes=dict(data.get("attributes", {})),
            events=events,
        )

    def to_otlp_dict(self) -> dict[str, Any]:
        """Convert span record to canonical OpenTelemetry OTLP JSON format."""
        try:
            start_nano = int(datetime.fromisoformat(self.start_time).timestamp() * 1e9)
        except Exception:
            start_nano = 0
        try:
            end_nano = (
                int(datetime.fromisoformat(self.end_time).timestamp() * 1e9)
                if self.end_time
                else start_nano
            )
        except Exception:
            end_nano = start_nano

        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": self.kind.to_otlp_code(),
            "startTimeUnixNano": str(start_nano),
            "endTimeUnixNano": str(end_nano),
            "attributes": _attributes_to_otlp(self.attributes),
            "events": [event.to_otlp_dict() for event in self.events],
            "status": {
                "code": self.status.to_otlp_code(),
                "message": self.status_message or "",
            },
        }


_CURRENT_SPAN: ContextVar[SpanRecord | None] = ContextVar("_CURRENT_SPAN", default=None)
_CURRENT_CONTEXT: ContextVar[TraceContext | None] = ContextVar(
    "_CURRENT_CONTEXT", default=None
)
_GLOBAL_TRACER: DistributedTracer | None = None


def get_global_tracer() -> DistributedTracer:
    """Return or lazily initialize the singleton distributed tracer."""
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = DistributedTracer()
    return _GLOBAL_TRACER


def set_global_tracer(tracer: DistributedTracer | None) -> None:
    """Explicitly set or reset the singleton distributed tracer."""
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = tracer


class DistributedTracer:
    """In-memory thread-safe distributed tracer with W3C and OpenTelemetry support."""

    def __init__(self, max_spans: int = 5000) -> None:
        self.max_spans = max(10, max_spans)
        self._lock = threading.Lock()
        self._spans: deque[SpanRecord] = deque(maxlen=self.max_spans)

    @property
    def active_span(self) -> SpanRecord | None:
        """Return the active span for the current async task or thread."""
        return _CURRENT_SPAN.get()

    @property
    def active_context(self) -> TraceContext | None:
        """Return the active trace context for the current async task or thread."""
        return _CURRENT_CONTEXT.get()

    def record_span(self, span: SpanRecord) -> None:
        """Record a completed or in-flight span in the ring buffer."""
        with self._lock:
            self._spans.append(span)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        context: TraceContext | None = None,
    ) -> Iterator[SpanRecord]:
        """Context manager creating and activating a traced span."""
        parent_span = _CURRENT_SPAN.get()
        active_ctx = context or _CURRENT_CONTEXT.get()

        if active_ctx is not None:
            span_ctx = active_ctx.new_child()
        else:
            span_ctx = TraceContext.new_root()

        parent_span_id = (
            parent_span.span_id
            if parent_span is not None
            else (active_ctx.span_id if active_ctx is not None else None)
        )

        record = SpanRecord(
            span_id=span_ctx.span_id,
            trace_id=span_ctx.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            attributes=dict(attributes or {}),
        )

        token_span = _CURRENT_SPAN.set(record)
        token_ctx = _CURRENT_CONTEXT.set(span_ctx)

        try:
            yield record
        except BaseException as exc:
            record.record_exception(exc)
            raise
        finally:
            if record.end_time is None:
                record.finish(
                    status=SpanStatus.OK
                    if record.status == SpanStatus.UNSET
                    else record.status
                )
            self.record_span(record)
            _CURRENT_SPAN.reset(token_span)
            _CURRENT_CONTEXT.reset(token_ctx)

    def get_spans(
        self,
        trace_id: str | None = None,
        name: str | None = None,
        status: SpanStatus | str | None = None,
        limit: int = 100,
    ) -> list[SpanRecord]:
        """Query recorded spans with filtering criteria."""
        status_val: str | None = None
        if status is not None:
            status_val = status.value if isinstance(status, SpanStatus) else str(status)

        with self._lock:
            spans = list(self._spans)

        results: list[SpanRecord] = []
        for s in reversed(spans):
            if trace_id and s.trace_id.lower() != trace_id.lower():
                continue
            if name and s.name != name:
                continue
            if status_val and s.status.value != status_val:
                continue
            results.append(s)
            if len(results) >= limit:
                break
        return results

    def get_trace(self, trace_id: str) -> list[SpanRecord]:
        """Retrieve all spans belonging to a trace, ordered chronologically."""
        target = trace_id.strip().lower()
        with self._lock:
            spans = [s for s in self._spans if s.trace_id.lower() == target]
        return sorted(spans, key=lambda s: s.start_time)

    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self._spans.clear()

    def to_otlp_export(
        self,
        trace_id: str | None = None,
        limit: int = 100,
        service_name: str = "music-recommender",
        service_version: str = "0.42.0",
    ) -> dict[str, Any]:
        """Export stored spans into OpenTelemetry standard JSON payload."""
        spans = self.get_trace(trace_id) if trace_id else self.get_spans(limit=limit)
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": service_name},
                            },
                            {
                                "key": "service.version",
                                "value": {"stringValue": service_version},
                            },
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "music-recommender-tracer"},
                            "spans": [span.to_otlp_dict() for span in spans],
                        }
                    ],
                }
            ]
        }


@dataclass
class TraceCorrelationResult:
    """Correlation findings linking an alert or incident to distributed traces."""

    incident_id: str | None
    alert_name: str
    time_window_start: str
    time_window_end: str
    matched_traces_count: int
    matched_spans_count: int
    error_spans_count: int
    candidate_root_cause_spans: list[dict[str, Any]]
    sample_trace_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize correlation result to dictionary."""
        return {
            "incident_id": self.incident_id,
            "alert_name": self.alert_name,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "matched_traces_count": self.matched_traces_count,
            "matched_spans_count": self.matched_spans_count,
            "error_spans_count": self.error_spans_count,
            "candidate_root_cause_spans": self.candidate_root_cause_spans,
            "sample_trace_ids": self.sample_trace_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceCorrelationResult:
        """Reconstruct correlation result from dictionary."""
        return cls(
            incident_id=(
                str(data["incident_id"])
                if data.get("incident_id") is not None
                else None
            ),
            alert_name=str(data.get("alert_name", "")),
            time_window_start=str(data.get("time_window_start", "")),
            time_window_end=str(data.get("time_window_end", "")),
            matched_traces_count=int(data.get("matched_traces_count", 0)),
            matched_spans_count=int(data.get("matched_spans_count", 0)),
            error_spans_count=int(data.get("error_spans_count", 0)),
            candidate_root_cause_spans=list(data.get("candidate_root_cause_spans", [])),
            sample_trace_ids=list(data.get("sample_trace_ids", [])),
        )


class AlertTraceCorrelator:
    """Correlate alerts and incidents with candidate traces and error spans."""

    def __init__(self, tracer: DistributedTracer) -> None:
        self.tracer = tracer

    def correlate(
        self,
        alert_name: str,
        timestamp: float | str | datetime | None = None,
        window_seconds: float = 300.0,
        incident_id: str | None = None,
        labels: dict[str, Any] | None = None,
    ) -> TraceCorrelationResult:
        """Find candidate traces and error root causes within anomaly time window."""
        window_secs = max(1.0, float(window_seconds))

        if timestamp is None:
            ref_ts = time.time()
        elif isinstance(timestamp, (int, float)):
            ref_ts = float(timestamp)
        elif isinstance(timestamp, datetime):
            ref_ts = timestamp.timestamp()
        else:
            try:
                ref_ts = datetime.fromisoformat(str(timestamp)).timestamp()
            except Exception:
                try:
                    ref_ts = float(timestamp)
                except Exception:
                    ref_ts = time.time()

        start_ts = ref_ts - window_secs
        end_ts = ref_ts + (window_secs / 2.0)

        dt_start = datetime.fromtimestamp(start_ts, tz=UTC)
        dt_end = datetime.fromtimestamp(end_ts, tz=UTC)

        all_spans = self.tracer.get_spans(limit=self.tracer.max_spans)

        matched_spans: list[SpanRecord] = []
        for s in all_spans:
            try:
                s_ts = datetime.fromisoformat(s.start_time).timestamp()
            except Exception:
                continue
            if start_ts <= s_ts <= end_ts:
                matched_spans.append(s)

        matched_trace_ids = sorted({s.trace_id for s in matched_spans})
        error_spans = [s for s in matched_spans if s.status == SpanStatus.ERROR]

        candidates = sorted(
            matched_spans,
            key=lambda s: (s.status == SpanStatus.ERROR, s.duration_seconds),
            reverse=True,
        )

        candidate_dicts = [s.to_dict() for s in candidates[:10]]

        return TraceCorrelationResult(
            incident_id=incident_id,
            alert_name=alert_name,
            time_window_start=dt_start.isoformat(),
            time_window_end=dt_end.isoformat(),
            matched_traces_count=len(matched_trace_ids),
            matched_spans_count=len(matched_spans),
            error_spans_count=len(error_spans),
            candidate_root_cause_spans=candidate_dicts,
            sample_trace_ids=matched_trace_ids[:10],
        )

    def correlate_incident(
        self,
        incident: Any,
        window_seconds: float = 300.0,
    ) -> TraceCorrelationResult:
        """Correlate an IncidentLifecycleRecord or dictionary with traces."""
        if hasattr(incident, "rule_name"):
            alert_name = str(incident.rule_name)
            incident_id: str | None = str(incident.incident_id)
            started_at = float(incident.started_at)
            labels = getattr(incident, "labels", {})
        elif isinstance(incident, dict):
            alert_name = str(
                incident.get("rule_name") or incident.get("alert_name", "unknown_alert")
            )
            raw_id = incident.get("incident_id")
            incident_id = str(raw_id) if raw_id is not None else None
            started_at = float(incident.get("started_at", time.time()))
            labels = incident.get("labels", {})
        else:
            alert_name = "unknown_alert"
            incident_id = None
            started_at = time.time()
            labels = {}

        return self.correlate(
            alert_name=alert_name,
            timestamp=started_at,
            window_seconds=window_seconds,
            incident_id=incident_id,
            labels=labels,
        )
