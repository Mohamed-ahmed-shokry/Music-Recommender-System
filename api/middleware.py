"""Transport-level safeguards and request correlation for the API."""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from music_recommender.tracing import DistributedTracer

logger = logging.getLogger("music_recommender.api.middleware")

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


class _RequestBodyTooLarge(Exception):
    """Signal that a streamed request exceeded the configured body limit."""


class RequestSafetyMiddleware:
    """Bound request bodies and add correlation metadata to HTTP responses."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be at least 1")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = _request_id(headers.get("x-request-id"))
        started_at = perf_counter()

        received_size = 0
        response_started = False

        async def send_with_context(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                duration = perf_counter() - started_at
                response_headers = MutableHeaders(scope=message)
                response_headers["x-request-id"] = request_id
                response_headers["x-process-time"] = f"{duration:.6f}"
                status_code = int(message.get("status", -1))
                logger.info(
                    "request method=%s path=%s status=%d request_id=%s duration=%.3fs",
                    scope.get("method"),
                    scope.get("path"),
                    status_code,
                    request_id,
                    duration,
                )
                try:
                    from music_recommender.telemetry import (
                        get_global_request_tracker,
                    )

                    get_global_request_tracker().record_request(
                        endpoint=str(scope.get("path") or "/"),
                        method=str(scope.get("method") or "GET"),
                        status_code=status_code,
                        duration_seconds=duration,
                    )
                except Exception:
                    pass
            await send(message)

        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                await self._reject(
                    scope,
                    receive,
                    send_with_context,
                    status_code=400,
                    detail="Content-Length must be a non-negative integer.",
                )
                return
            if declared_size < 0:
                await self._reject(
                    scope,
                    receive,
                    send_with_context,
                    status_code=400,
                    detail="Content-Length must be a non-negative integer.",
                )
                return
            if declared_size > self.max_body_bytes:
                await self._reject_too_large(scope, receive, send_with_context)
                return

        async def receive_with_limit() -> Message:
            nonlocal received_size
            message = await receive()
            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))
                if received_size > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, receive_with_limit, send_with_context)
        except _RequestBodyTooLarge:
            if response_started:
                return
            await self._reject_too_large(scope, receive, send_with_context)

    async def _reject_too_large(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await self._reject(
            scope,
            receive,
            send,
            status_code=413,
            detail=(f"Request body exceeds the {self.max_body_bytes}-byte limit."),
        )

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int,
        detail: str,
    ) -> None:
        response = JSONResponse({"detail": detail}, status_code=status_code)
        await response(scope, receive, send)


def _request_id(candidate: str | None) -> str:
    if candidate is not None and _REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid4().hex


class TraceContextMiddleware:
    """Propagate W3C Trace Context and wrap requests in OpenTelemetry server spans."""

    def __init__(self, app: ASGIApp, tracer: DistributedTracer | None = None) -> None:
        self.app = app
        self._tracer = tracer

    @property
    def tracer(self) -> DistributedTracer:
        if self._tracer is not None:
            return self._tracer
        from music_recommender.tracing import get_global_tracer

        return get_global_tracer()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from music_recommender.tracing import (
            SpanKind,
            SpanStatus,
            TraceContext,
        )

        headers = Headers(scope=scope)
        raw_traceparent = headers.get("traceparent")
        raw_tracestate = headers.get("tracestate", "")

        req_ctx: TraceContext | None = None
        if raw_traceparent:
            try:
                req_ctx = TraceContext.from_traceparent(
                    raw_traceparent, trace_state=raw_tracestate
                )
            except Exception:
                req_ctx = None

        method = str(scope.get("method") or "GET")
        path = str(scope.get("path") or "/")
        span_name = f"{method} {path}"

        initial_attrs: dict[str, str | int | float | bool] = {
            "http.method": method,
            "http.target": path,
            "http.route": path,
            "http.scheme": str(scope.get("scheme") or "http"),
        }
        client = scope.get("client")
        if client and isinstance(client, (list, tuple)) and len(client) > 0:
            initial_attrs["http.client_ip"] = str(client[0])

        with self.tracer.span(
            name=span_name,
            attributes=initial_attrs,
            kind=SpanKind.SERVER,
            context=req_ctx,
        ) as span:
            span_traceparent = (
                self.tracer.active_context.to_traceparent()
                if self.tracer.active_context is not None
                else f"00-{span.trace_id}-{span.span_id}-01"
            )

            async def send_with_trace(message: Message) -> None:
                if message["type"] == "http.response.start":
                    response_headers = MutableHeaders(scope=message)
                    response_headers["traceparent"] = span_traceparent
                    response_headers["x-trace-id"] = span.trace_id
                    response_headers["x-span-id"] = span.span_id
                    if (
                        self.tracer.active_context
                        and self.tracer.active_context.trace_state
                    ):
                        response_headers["tracestate"] = (
                            self.tracer.active_context.trace_state
                        )

                    status_code = int(message.get("status", 200))
                    span.set_attribute("http.status_code", status_code)
                    if status_code >= 500:
                        span.finish(
                            status=SpanStatus.ERROR,
                            message=f"HTTP {status_code}",
                        )

                await send(message)

            try:
                await self.app(scope, receive, send_with_trace)
            except Exception as exc:
                span.record_exception(exc)
                raise
