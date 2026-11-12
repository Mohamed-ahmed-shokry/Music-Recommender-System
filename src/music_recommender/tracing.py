"""Distributed tracing, W3C Trace Context propagation, and OTel interop."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
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
                raise ValueError(
                    f"Invalid W3C parent_span_id: '{self.parent_span_id}'"
                )

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
        flags = (
            ("01" if sampled else "00")
            if sampled is not None
            else self.trace_flags
        )
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
    def from_traceparent(
        cls, header: str, trace_state: str = ""
    ) -> TraceContext:
        """Parse W3C traceparent header string (00-{trace_id}-{span_id}-{flags})."""
        raw = header.strip()
        match = _TRACEPARENT_RE.match(raw)
        if not match:
            raise ValueError(f"Malformed W3C traceparent header: '{header}'")
        version, tid, sid, flags = match.groups()
        if version == "ff":
            raise ValueError(
                f"Unsupported W3C traceparent version 0xff in '{header}'"
            )
        return cls(
            trace_id=tid,
            span_id=sid,
            trace_flags=flags,
            trace_state=trace_state.strip(),
        )

    @classmethod
    def new_root(
        cls, sampled: bool = True, trace_state: str = ""
    ) -> TraceContext:
        """Generate a new root trace context."""
        return cls(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            parent_span_id=None,
            trace_flags="01" if sampled else "00",
            trace_state=trace_state.strip(),
        )
