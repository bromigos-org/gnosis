import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Response
from fastapi.responses import JSONResponse

from gnosis.auth import Authenticator, build_authenticator
from gnosis.backend import (
    BackendCapabilityUnavailable,
    MemoryBackend,
    Neo4jAgentMemoryBackend,
)
from gnosis.federation import (
    FederationGateway,
    FederationTransport,
    PeerNotAllowedError,
    PeerTokenUnavailableError,
    UnknownPeerError,
)
from gnosis.mcp_server import BearerTokenMiddleware, build_mcp_server
from gnosis.models import (
    ContextRequest,
    ContextResponse,
    GraphContextRequest,
    GraphContextResponse,
    MemoryContextRequest,
    MemoryContextResponse,
)
from gnosis.routes.events_skills import register_event_routes, register_skill_routes
from gnosis.routes.memory_provider import (
    register_memory_provider_routes,
    register_message_routes,
)
from gnosis.routes.operator import register_operator_routes
from gnosis.routes.reasoning import register_reasoning_routes
from gnosis.routes.system import register_health_route, register_readiness_routes
from gnosis.settings import Settings, load_settings

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_LEGACY_CONTEXT_ROUTE: Final = "/v1/context"
_MEMORY_CONTEXT_ROUTE: Final = "/v1/memory/context"
_LEGACY_CONTEXT_SUCCESSOR_LINK: Final = (
    f'<{_MEMORY_CONTEXT_ROUTE}>; rel="successor-version"'
)


def _legacy_context_warning() -> Callable[[], None]:
    """Build a warner that logs deprecated /v1/context usage once per process."""
    emitted = False

    def warn() -> None:
        nonlocal emitted
        if emitted:
            return
        emitted = True
        _LOGGER.warning(
            "Deprecated route %s was called; migrate to %s.",
            _LEGACY_CONTEXT_ROUTE,
            _MEMORY_CONTEXT_ROUTE,
            extra={
                "deprecated_route": _LEGACY_CONTEXT_ROUTE,
                "successor_route": _MEMORY_CONTEXT_ROUTE,
            },
        )

    return warn


def create_app(
    settings_factory: Callable[[], Settings] = load_settings,
    backend: MemoryBackend | None = None,
    federation_transport: FederationTransport | None = None,
) -> FastAPI:
    settings = settings_factory()
    memory_backend = backend or Neo4jAgentMemoryBackend(settings)
    authenticator = build_authenticator(settings)
    federation = FederationGateway(settings, transport=federation_transport)

    def get_backend() -> MemoryBackend:
        return memory_backend

    mcp_server = (
        build_mcp_server(settings, get_backend) if settings.gnosis_mcp_enabled else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        try:
            if mcp_server is None:
                yield
            else:
                async with mcp_server.session_manager.run():
                    yield
        finally:
            await memory_backend.shutdown()

    app = FastAPI(title="gnosis", lifespan=lifespan)
    _register_exception_handlers(app)

    if mcp_server is not None:
        app.mount(
            "/mcp",
            BearerTokenMiddleware(
                app=mcp_server.streamable_http_app(),
                token=settings.gnosis_token,
            ),
        )

    register_health_route(app)
    register_readiness_routes(app, settings, authenticator, get_backend)
    register_message_routes(app, authenticator, get_backend)
    register_memory_provider_routes(
        app,
        settings,
        authenticator,
        get_backend,
        federation,
    )
    register_event_routes(app, authenticator, get_backend)
    _register_context_routes(app, authenticator, get_backend)
    register_operator_routes(app, settings, authenticator, get_backend)
    register_reasoning_routes(app, authenticator, get_backend)
    register_skill_routes(app, authenticator, get_backend)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BackendCapabilityUnavailable)
    async def capability_unavailable(
        _request: object,
        error: BackendCapabilityUnavailable,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={"detail": "capability_unavailable", "message": error.detail},
        )

    @app.exception_handler(UnknownPeerError)
    async def unknown_peer(
        _request: object,
        error: UnknownPeerError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": error.detail})

    @app.exception_handler(PeerNotAllowedError)
    async def peer_not_allowed(
        _request: object,
        error: PeerNotAllowedError,
    ) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": error.detail})

    @app.exception_handler(PeerTokenUnavailableError)
    async def peer_token_unavailable(
        _request: object,
        error: PeerTokenUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": error.detail})


def _register_context_routes(
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    warn_legacy_context_route_used = _legacy_context_warning()

    @app.post(
        _LEGACY_CONTEXT_ROUTE,
        dependencies=[Depends(authenticator.require_token)],
        deprecated=True,
    )
    async def get_context(
        request: ContextRequest,
        response: Response,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ContextResponse:
        authenticator.require_scope(request.scope)
        warn_legacy_context_route_used()
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = _LEGACY_CONTEXT_SUCCESSOR_LINK
        return await memory.get_context(request)

    @app.post(
        _MEMORY_CONTEXT_ROUTE,
        dependencies=[Depends(authenticator.require_token)],
    )
    async def get_memory_context(
        request: MemoryContextRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryContextResponse:
        authenticator.require_scope(request.scope)
        return await memory.get_memory_context(request)

    @app.post(
        "/v1/graph/context",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def get_graph_context(
        request: GraphContextRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> GraphContextResponse:
        authenticator.require_scope(request.scope)
        return await memory.get_graph_context(request)


app = create_app()
