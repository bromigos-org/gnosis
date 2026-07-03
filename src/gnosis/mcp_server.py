"""Streamable-HTTP MCP surface for the gnosis memory gateway.

The MCP module stays thin: every tool builds a server-side scope, delegates to
the shared ``MemoryBackend`` operations, and returns redacted gateway payloads.
Bearer auth is enforced by an ASGI wrapper before the MCP transport runs, and
the whole mount is feature-flagged behind ``GNOSIS_MCP_ENABLED``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from secrets import compare_digest
from typing import Final

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, TypeAdapter
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from gnosis.backend import BackendRequestError, MemoryBackend, MemoryNotFoundError
from gnosis.models import (
    BackendReadiness,
    JsonObject,
    JsonValue,
    MemoryAddRequest,
    MemoryContextRequest,
    MemoryDeleteRequest,
    MemoryListRequest,
    MemoryMessage,
    MemoryScope,
    MemorySearchRequest,
    MemoryVisibility,
)
from gnosis.redaction import redact_secrets
from gnosis.settings import Settings

_JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)
_MCP_SPACE_ID: Final[str] = "mcp"
_EDIT_DISABLED_DETAIL: Final[str] = "Memory editing is disabled by service policy."


def build_mcp_server(  # noqa: C901 - tool grouping is intentional.
    settings: Settings,
    get_backend: Callable[[], MemoryBackend],
) -> FastMCP:
    server = FastMCP(
        name="gnosis",
        instructions="Scoped long-term memory tools backed by the gnosis gateway.",
        streamable_http_path="/",
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )

    @server.tool(
        description=(
            "Store a memory for a user. Set infer=true to extract from "
            "conversation text instead of storing it verbatim."
        ),
    )
    async def add_memory(
        content: str,
        user_id: str,
        metadata: dict[str, JsonValue] | None = None,
        infer: bool = False,
    ) -> JsonObject:
        request = MemoryAddRequest(
            scope=_mcp_scope(settings, user_id),
            messages=([MemoryMessage(role="user", content=content)] if infer else []),
            content=None if infer else content,
            infer=infer,
            metadata=metadata or {},
        )
        try:
            return _payload(await get_backend().add_memories(request))
        except BackendRequestError as error:
            raise ToolError(error.detail) from error

    @server.tool(
        description="Semantic search over a user's long-term memories.",
    )
    async def search_memory(
        query: str,
        user_id: str,
        limit: int = 8,
    ) -> JsonObject:
        request = MemorySearchRequest(
            scope=_mcp_scope(settings, user_id),
            query=query,
            limit=limit,
        )
        try:
            return _payload(await get_backend().search_memories(request))
        except BackendRequestError as error:
            raise ToolError(error.detail) from error

    @server.tool(
        description="Assemble combined prompt-safe memory context for a query.",
    )
    async def get_context(
        query: str,
        user_id: str,
        max_items: int = 8,
    ) -> JsonObject:
        request = MemoryContextRequest(
            scope=_mcp_scope(settings, user_id),
            query=query,
            max_items=max_items,
        )
        return _payload(await get_backend().get_memory_context(request))

    @server.tool(
        description="List a user's memories, newest first.",
    )
    async def list_memories(user_id: str, page: int = 1) -> JsonObject:
        request = MemoryListRequest(
            scope=_mcp_scope(settings, user_id),
            page=page,
        )
        try:
            return _payload(await get_backend().list_memories(request))
        except BackendRequestError as error:
            raise ToolError(error.detail) from error

    @server.tool(
        description="Delete one memory owned by the user.",
    )
    async def delete_memory(memory_id: str, user_id: str) -> JsonObject:
        if not settings.gnosis_memory_edit_enabled:
            raise ToolError(_EDIT_DISABLED_DETAIL)
        request = MemoryDeleteRequest(scope=_mcp_scope(settings, user_id))
        try:
            return _payload(await get_backend().delete_memory(memory_id, request))
        except MemoryNotFoundError as error:
            raise ToolError(error.detail) from error

    @server.tool(
        description="Report service readiness and redacted backend diagnostics.",
    )
    async def get_status() -> JsonObject:
        memory = get_backend()
        readiness = await memory.readiness()
        diagnostics = memory.diagnostics(readiness)
        return {
            "service": "gnosis",
            "status": "ready" if _is_ready(readiness) else "unavailable",
            "diagnostics": _redacted_payload(diagnostics),
        }

    return server


@dataclass(frozen=True, slots=True)
class BearerTokenMiddleware:
    app: ASGIApp
    token: str

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not self._is_authorized(Headers(scope=scope).get("authorization")):
            response = JSONResponse(
                status_code=401,
                content={"detail": "invalid bearer token"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _is_authorized(self, header: str | None) -> bool:
        if header is None:
            return False
        prefix, _, credentials = header.partition(" ")
        return (
            prefix.lower() == "bearer"
            and bool(credentials)
            and compare_digest(credentials, self.token)
        )


def _mcp_scope(settings: Settings, user_id: str) -> MemoryScope:
    return MemoryScope(
        tenant_id=settings.gnosis_tenant_id,
        space_id=_MCP_SPACE_ID,
        agent_id=settings.gnosis_mcp_agent_id,
        session_id=f"{_MCP_SPACE_ID}:{user_id}",
        user_id=user_id,
        visibility=MemoryVisibility.PRIVATE_USER,
    )


def _is_ready(readiness: BackendReadiness) -> bool:
    return (
        readiness.graph == "ready"
        and readiness.schema_status == "ready"
        and readiness.buffer_status == "ready"
    )


def _payload(model: BaseModel) -> JsonObject:
    """Dump a gateway response that the backend already redacted outbound."""
    return _JSON_OBJECT_ADAPTER.validate_python(model.model_dump(mode="json"))


def _redacted_payload(model: BaseModel) -> JsonObject:
    redacted = redact_secrets(_payload(model))
    if isinstance(redacted, dict):
        return redacted
    return {}
