from collections.abc import Callable
from dataclasses import dataclass, field
from os import environ
from typing import cast

import pytest

_ = environ.setdefault("GNOSIS_TOKEN", "mcp-access")
_ = environ.setdefault("GNOSIS_READ_OPERATOR_TOKEN", "read-operator-token")
_ = environ.setdefault("GNOSIS_EXPORT_OPERATOR_TOKEN", "export-operator-token")
_ = environ.setdefault("GNOSIS_WRITE_OPERATOR_TOKEN", "write-operator-token")
_ = environ.setdefault("GNOSIS_ADMIN_OPERATOR_TOKEN", "admin-operator-token")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "inert-password")
_ = environ.setdefault("LITELLM_BASE_URL", "http://litellm.local/v1")
_ = environ.setdefault("LITELLM_API_KEY", "inert-litellm-key")

from fastapi.testclient import TestClient  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

from gnosis.backend import MemoryBackend  # noqa: E402
from gnosis.main import create_app  # noqa: E402
from gnosis.mcp_server import build_mcp_server  # noqa: E402
from gnosis.models import (  # noqa: E402
    JsonObject,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryAddResult,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryVisibility,
)
from gnosis.settings import Settings  # noqa: E402

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_EXPECTED_TOOLS = {
    "add_memory",
    "search_memory",
    "get_context",
    "list_memories",
    "delete_memory",
    "get_status",
}


@pytest.mark.anyio
async def test_mcp_server_registers_exactly_six_tools() -> None:
    # Given: the MCP server built against the shared backend seam.
    server = build_mcp_server(Settings(), _get_backend(StubMemoryBackend()))

    # When: the tool registry is listed.
    tools = await server.list_tools()

    # Then: exactly the six contract tools are exposed with descriptions.
    assert {tool.name for tool in tools} == _EXPECTED_TOOLS
    assert all(tool.description for tool in tools)


@pytest.mark.anyio
async def test_add_memory_tool_round_trip_builds_server_side_scope() -> None:
    # Given: an MCP caller adding a verbatim memory for a user.
    stub = StubMemoryBackend()
    server = build_mcp_server(Settings(), _get_backend(stub))

    # When: the tool is called through the MCP tool manager.
    result = await server.call_tool(
        "add_memory",
        {"content": "Cartman prefers cheesy poofs", "user_id": "789"},
    )

    # Then: the scope is constructed server-side and results flow back.
    request = stub.add_requests[0]
    assert request.scope.tenant_id == "bromigos"
    assert request.scope.space_id == "mcp"
    assert request.scope.agent_id == "mcp-client"
    assert request.scope.session_id == "mcp:789"
    assert request.scope.user_id == "789"
    assert request.scope.visibility is MemoryVisibility.PRIVATE_USER
    assert request.infer is False
    assert request.content == "Cartman prefers cheesy poofs"
    assert request.messages == []
    structured = _structured(result)
    assert structured == {
        "result": {
            "results": [
                {
                    "memory_id": "00000000-0000-0000-0000-0000000000aa",
                    "content": "Cartman prefers cheesy poofs",
                    "event": "ADD",
                    "metadata": {},
                },
            ],
        },
    }


@pytest.mark.anyio
async def test_add_memory_tool_when_infer_routes_to_extraction_mode() -> None:
    # Given: an MCP caller opting into extraction mode.
    stub = StubMemoryBackend()
    server = build_mcp_server(Settings(), _get_backend(stub))

    # When: the tool is called with infer=true.
    _ = await server.call_tool(
        "add_memory",
        {"content": "I love cheesy poofs", "user_id": "789", "infer": True},
    )

    # Then: the content rides the conversation extraction path.
    request = stub.add_requests[0]
    assert request.infer is True
    assert request.content is None
    assert [message.content for message in request.messages] == [
        "I love cheesy poofs",
    ]


@pytest.mark.anyio
async def test_delete_memory_tool_respects_edit_flag_default() -> None:
    # Given: the default safe rollout posture with editing disabled.
    stub = StubMemoryBackend()
    server = build_mcp_server(Settings(), _get_backend(stub))

    # When / Then: the delete tool refuses with a clear policy error.
    with pytest.raises(ToolError, match="disabled by service policy"):
        _ = await server.call_tool(
            "delete_memory",
            {"memory_id": "00000000-0000-0000-0000-0000000000aa", "user_id": "789"},
        )
    assert stub.delete_requests == []


def test_mcp_mount_is_disabled_by_default() -> None:
    # Given: an app created with default settings.
    client = TestClient(
        create_app(settings_factory=Settings, backend=_backend_protocol()),
    )

    # When: a caller posts to the MCP mount.
    response = client.post("/mcp", json={})

    # Then: the surface does not exist.
    assert response.status_code == 404


def test_mcp_mount_requires_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: the MCP surface is enabled.
    monkeypatch.setenv("GNOSIS_MCP_ENABLED", "true")
    app = create_app(settings_factory=Settings, backend=_backend_protocol())

    # When / Then: unauthenticated and wrongly authenticated calls are rejected.
    with TestClient(app) as client:
        assert client.post("/mcp", json={}).status_code == 401
        assert (
            client.post(
                "/mcp",
                json={},
                headers={"Authorization": "Bearer wrong-token"},
            ).status_code
            == 401
        )

        # And: a bearer-authenticated call reaches the MCP transport.
        response = client.post(
            "/mcp",
            json={},
            headers={"Authorization": f"Bearer {Settings().gnosis_token}"},
        )
        assert response.status_code != 401
        assert response.status_code != 404


def _structured(result: object) -> JsonObject:
    assert isinstance(result, tuple)
    content = cast("tuple[object, object]", result)
    return _JSON_OBJECT_ADAPTER.validate_python(content[1])


def _get_backend(stub: "StubMemoryBackend") -> Callable[[], MemoryBackend]:
    backend = cast("object", stub)
    return lambda: cast("MemoryBackend", backend)


def _backend_protocol() -> MemoryBackend:
    backend = cast("object", StubMemoryBackend())
    return cast("MemoryBackend", backend)


@dataclass(slots=True)
class StubMemoryBackend:
    add_requests: list[MemoryAddRequest] = field(default_factory=list)
    search_requests: list[MemorySearchRequest] = field(default_factory=list)
    list_requests: list[MemoryListRequest] = field(default_factory=list)
    context_requests: list[MemoryContextRequest] = field(default_factory=list)
    delete_requests: list[tuple[str, MemoryDeleteRequest]] = field(
        default_factory=list,
    )

    async def add_memories(self, request: MemoryAddRequest) -> MemoryAddResponse:
        self.add_requests.append(request)
        content = request.content or request.messages[0].content
        return MemoryAddResponse(
            results=[
                MemoryAddResult(
                    memory_id="00000000-0000-0000-0000-0000000000aa",
                    content=content,
                    event="ADD",
                ),
            ],
        )

    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        self.search_requests.append(request)
        return MemorySearchResponse()

    async def list_memories(self, request: MemoryListRequest) -> MemoryListResponse:
        self.list_requests.append(request)
        return MemoryListResponse(
            total=0,
            page=request.page,
            page_size=request.page_size,
        )

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        self.context_requests.append(request)
        return MemoryContextResponse()

    async def delete_memory(
        self,
        memory_id: str,
        request: MemoryDeleteRequest,
    ) -> MemoryDeleteResponse:
        self.delete_requests.append((memory_id, request))
        return MemoryDeleteResponse(memory_id=memory_id)

    async def shutdown(self) -> None:
        return None
