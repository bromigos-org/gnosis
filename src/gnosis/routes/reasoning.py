"""Reasoning trace routes: trace lifecycle, recall, and operator inspection."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from gnosis.auth import Authenticator
from gnosis.backend import MemoryBackend
from gnosis.models import (
    JsonObject,
    ReasoningContextRequest,
    ReasoningContextResponse,
    ReasoningSimilarTracesRequest,
    ReasoningSimilarTracesResponse,
    ReasoningStepRequest,
    ReasoningStepResponse,
    ReasoningStepSearchRequest,
    ReasoningStepSearchResponse,
    ReasoningToolCallRequest,
    ReasoningToolCallResponse,
    ReasoningToolStatsRequest,
    ReasoningToolStatsResponse,
    ReasoningTraceCompleteRequest,
    ReasoningTraceCompleteResponse,
    ReasoningTraceDetailRequest,
    ReasoningTraceDetailResponse,
    ReasoningTraceListRequest,
    ReasoningTraceListResponse,
    ReasoningTraceStartRequest,
    ReasoningTraceStartResponse,
)
from gnosis.redaction import redact_secrets


def register_reasoning_routes(  # noqa: C901 - FastAPI route grouping is intentional.
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    @app.post(
        "/v1/reasoning/traces",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def start_reasoning_trace(
        request: ReasoningTraceStartRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceStartResponse:
        authenticator.require_scope(request.scope)
        return await memory.start_reasoning_trace(request)

    @app.post(
        "/v1/reasoning/traces/{trace_id}/steps",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def add_reasoning_step(
        trace_id: str,
        request: ReasoningStepRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningStepResponse:
        require_matching_identifier(
            path_value=trace_id,
            body_value=request.trace_id,
            field_name="trace_id",
        )
        authenticator.require_scope(request.scope)
        return await memory.add_reasoning_step(request)

    @app.post(
        "/v1/reasoning/steps/{step_id}/tool-calls",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def record_reasoning_tool_call(
        step_id: str,
        request: ReasoningToolCallRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningToolCallResponse:
        require_matching_identifier(
            path_value=step_id,
            body_value=request.step_id,
            field_name="step_id",
        )
        authenticator.require_scope(request.scope)
        return await memory.record_reasoning_tool_call(request)

    @app.post(
        "/v1/reasoning/traces/{trace_id}/complete",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def complete_reasoning_trace(
        trace_id: str,
        request: ReasoningTraceCompleteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceCompleteResponse:
        require_matching_identifier(
            path_value=trace_id,
            body_value=request.trace_id,
            field_name="trace_id",
        )
        authenticator.require_scope(request.scope)
        return await memory.complete_reasoning_trace(request)

    @app.post(
        "/v1/reasoning/context",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def get_reasoning_context(
        request: ReasoningContextRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningContextResponse:
        authenticator.require_scope(request.scope)
        response = await memory.get_reasoning_context(request)
        return ReasoningContextResponse(
            context=_redacted_context(response.context),
            traces=_redacted_traces(response.traces),
        )

    @app.post(
        "/v1/reasoning/traces/list",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def list_reasoning_traces(
        request: ReasoningTraceListRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceListResponse:
        authenticator.require_scope(request.scope)
        return await memory.list_reasoning_traces(request)

    @app.post(
        "/v1/reasoning/traces/{trace_id}/detail",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def get_reasoning_trace(
        trace_id: str,
        request: ReasoningTraceDetailRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceDetailResponse:
        authenticator.require_scope(request.scope)
        require_matching_identifier(
            path_value=trace_id,
            body_value=request.trace_id,
            field_name="trace_id",
        )
        return await memory.get_reasoning_trace(request)

    @app.post(
        "/v1/reasoning/traces/similar",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def find_similar_reasoning_traces(
        request: ReasoningSimilarTracesRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningSimilarTracesResponse:
        authenticator.require_scope(request.scope)
        return await memory.find_similar_reasoning_traces(request)

    @app.post(
        "/v1/reasoning/steps/search",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def search_reasoning_steps(
        request: ReasoningStepSearchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningStepSearchResponse:
        authenticator.require_scope(request.scope)
        return await memory.search_reasoning_steps(request)

    @app.post(
        "/v1/reasoning/tools/stats",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def get_reasoning_tool_stats(
        request: ReasoningToolStatsRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningToolStatsResponse:
        authenticator.require_scope(request.scope)
        return await memory.get_reasoning_tool_stats(request)


def require_matching_identifier(
    *,
    path_value: str,
    body_value: str,
    field_name: str,
) -> None:
    if path_value != body_value:
        raise HTTPException(
            status_code=400,
            detail=f"Path {field_name} must match request body {field_name}.",
        )


def _redacted_context(context: str) -> str:
    redacted = redact_secrets(context)
    if isinstance(redacted, str):
        return redacted
    return context


def _redacted_traces(traces: list[JsonObject]) -> list[JsonObject]:
    redacted_traces: list[JsonObject] = []
    for trace in traces:
        redacted = redact_secrets(trace)
        if isinstance(redacted, dict):
            redacted_traces.append(redacted)
    return redacted_traces
