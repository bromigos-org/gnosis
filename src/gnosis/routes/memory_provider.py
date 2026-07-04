"""Provider-surface memory routes: add, search, list, promote, edit.

These are the endpoints that federated peers may also call, so this module
owns the federated-caller guards (provenance on writes, no fan-out, and
shareable-only filters) alongside the routes themselves.
"""

import logging
from collections.abc import Callable
from typing import Annotated, Final

from fastapi import Depends, FastAPI, HTTPException

from gnosis.auth import Authenticator, MemoryCaller
from gnosis.backend import (
    BackendRequestError,
    ExtractionPreviewBackend,
    MemoryBackend,
    MemoryNotFoundError,
    RecallFilteringBackend,
)
from gnosis.federation import (
    FederationGateway,
    merged_search_results,
    shareable_filters,
)
from gnosis.models import (
    ExtractionPreviewRequest,
    ExtractionPreviewResponse,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemoryPromoteCandidate,
    MemoryPromoteRequest,
    MemoryPromoteResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MessageWriteRequest,
    MessageWriteResponse,
)
from gnosis.settings import Settings

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def register_message_routes(
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    @app.post(
        "/v1/messages",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def write_message(
        request: MessageWriteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MessageWriteResponse:
        authenticator.require_scope(request.scope)
        try:
            return await memory.add_message(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post(
        "/v1/memory/extraction/preview",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def preview_extraction(
        request: ExtractionPreviewRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ExtractionPreviewResponse:
        authenticator.require_scope(request.scope)
        if not isinstance(memory, ExtractionPreviewBackend):
            raise HTTPException(
                status_code=501,
                detail="Extraction preview is unavailable.",
            )
        try:
            return await memory.preview_extraction(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error


def register_memory_provider_routes(  # noqa: C901, PLR0915 - route grouping is intentional.
    app: FastAPI,
    settings: Settings,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
    federation: FederationGateway,
) -> None:
    @app.post("/v1/memories")
    async def add_memories(
        request: MemoryAddRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
        caller: Annotated[MemoryCaller, Depends(authenticator.resolve_memory_caller)],
    ) -> MemoryAddResponse:
        authenticator.require_scope(request.scope)
        if caller is MemoryCaller.FEDERATED:
            _require_federated_add_provenance(request)
        try:
            return await memory.add_memories(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post("/v1/memories/search")
    async def search_memories(
        request: MemorySearchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
        caller: Annotated[MemoryCaller, Depends(authenticator.resolve_memory_caller)],
    ) -> MemorySearchResponse:
        authenticator.require_scope(request.scope)
        if caller is MemoryCaller.FEDERATED:
            _require_no_federated_fanout(request)
            request = request.model_copy(
                update={"filters": shareable_filters(request.filters)},
            )
        for peer_name in request.peers:
            _ = federation.require_pull_peer(peer_name)
        try:
            local = await memory.search_memories(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error
        if not request.peers:
            return local
        remote, peer_errors = await federation.search_peers(request)
        results = merged_search_results(local.results, remote, request.limit)
        if isinstance(memory, RecallFilteringBackend):
            # Federated searches run the recall filter here, once over the
            # merged result set; remote results are already shareable-only
            # and the filter can only remove or keep records.
            results = await memory.filter_recalled_memories(request.query, results)
        return MemorySearchResponse(results=results, peer_errors=peer_errors)

    @app.post("/v1/memories/list")
    async def list_memories(
        request: MemoryListRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
        caller: Annotated[MemoryCaller, Depends(authenticator.resolve_memory_caller)],
    ) -> MemoryListResponse:
        authenticator.require_scope(request.scope)
        if caller is MemoryCaller.FEDERATED:
            request = request.model_copy(
                update={"filters": shareable_filters(request.filters)},
            )
        try:
            return await memory.list_memories(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post(
        "/v1/memories/promote",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def promote_memories(
        request: MemoryPromoteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryPromoteResponse:
        authenticator.require_scope(request.scope)
        peer = federation.require_push_peer(request.peer)
        try:
            listing = await memory.list_memories(
                MemoryListRequest(
                    scope=request.scope,
                    filters=shareable_filters(request.filters),
                    page=1,
                    page_size=request.limit,
                ),
            )
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error
        candidates = [
            MemoryPromoteCandidate(
                memory_id=record.memory_id,
                content=record.content,
                metadata=record.metadata,
            )
            for record in listing.results
        ]
        if request.dry_run:
            return MemoryPromoteResponse(
                peer=peer.name,
                count=len(candidates),
                dry_run=True,
                candidates=candidates,
            )
        outcome = await federation.promote(peer, request.scope, candidates)
        _LOGGER.info(
            "memory promotion applied",
            extra={
                "peer": peer.name,
                "tenant_id": request.scope.tenant_id,
                "user_id": request.scope.user_id,
                "promoted": len(outcome.promoted),
                "failed": len(outcome.failed),
            },
        )
        return MemoryPromoteResponse(
            peer=peer.name,
            count=len(candidates),
            dry_run=False,
            promoted=outcome.promoted,
            failed=outcome.failed,
        )

    @app.patch(
        "/v1/memories/{memory_id}",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def update_memory(
        memory_id: str,
        request: MemoryUpdateRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryUpdateResponse:
        _require_memory_edit_enabled(settings)
        authenticator.require_scope(request.scope)
        try:
            return await memory.update_memory(memory_id, request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail=error.detail) from error

    @app.delete(
        "/v1/memories/{memory_id}",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def delete_memory(
        memory_id: str,
        request: MemoryDeleteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryDeleteResponse:
        _require_memory_edit_enabled(settings)
        authenticator.require_scope(request.scope)
        try:
            return await memory.delete_memory(memory_id, request)
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail=error.detail) from error


def _require_memory_edit_enabled(settings: Settings) -> None:
    if not settings.gnosis_memory_edit_enabled:
        raise HTTPException(
            status_code=403,
            detail="Memory editing is disabled by service policy.",
        )


def _require_federated_add_provenance(request: MemoryAddRequest) -> None:
    promoted_from = request.metadata.get("promoted_from")
    if not (isinstance(promoted_from, str) and promoted_from):
        raise HTTPException(
            status_code=403,
            detail="federated writes require metadata.promoted_from",
        )


def _require_no_federated_fanout(request: MemorySearchRequest) -> None:
    if request.peers:
        raise HTTPException(
            status_code=403,
            detail="federated callers cannot request peer fan-out",
        )
