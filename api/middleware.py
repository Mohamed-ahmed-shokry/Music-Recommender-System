"""Transport-level safeguards and request correlation for the API."""

from __future__ import annotations

import re
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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

        async def send_with_context(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers["x-request-id"] = request_id
                response_headers["x-process-time"] = (
                    f"{perf_counter() - started_at:.6f}"
                )
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

        received_size = 0

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
