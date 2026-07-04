"""Prompt-safe views over SDK reasoning traces, steps, and tool calls.

Reasoning memory is auditable, not free-form hidden thought: recall must
never expose chain-of-thought style fields (``thought``,
``chain_of_thought``, embeddings) and must redact credential-shaped
metadata. This module converts SDK reasoning objects into the public
response records, enforces per-scope visibility on traces and steps, and
strips or redacts the unsafe keys on every rendered object.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Final
from uuid import UUID

from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolStats

from gnosis.backend_protocols import BackendCapabilityUnavailable
from gnosis.dedup_consolidation import UNSAFE_CONSOLIDATION_REPORT_KEYS
from gnosis.json_redaction import (
    JSON_OBJECT_ADAPTER,
    json_object,
    redacted_optional_text,
    redacted_text,
    string_metadata,
)
from gnosis.models import (
    JsonObject,
    JsonValue,
    MemoryScope,
    ReasoningStepRecord,
    ReasoningToolStatsRecord,
    ReasoningTraceDetailRequest,
    ReasoningTraceSummary,
)
from gnosis.scope_policy import SCOPE_METADATA_KEYS, scope_metadata
from gnosis.sdk_client import ReasoningMemory

REASONING_READ_UNAVAILABLE_DETAIL: Final[str] = "SDK reasoning read is unavailable."
_UNSAFE_REASONING_KEYS: Final[frozenset[str]] = UNSAFE_CONSOLIDATION_REPORT_KEYS | {
    "task_embedding",
    "tool_calls",
}
_REDACT_REASONING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "client_secret",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
    },
)


def safe_reasoning_context(context: str) -> str:
    lines = [
        line
        for line in context.splitlines()
        if not _reasoning_line_exposes_hidden_trace(line)
    ]
    return redacted_text("\n".join(lines))


def _reasoning_line_exposes_hidden_trace(line: str) -> bool:
    normalized = line.casefold()
    return "thought:" in normalized or "chain_of_thought" in normalized


async def get_reasoning_trace(
    reasoning: ReasoningMemory,
    request: ReasoningTraceDetailRequest,
) -> SdkReasoningTrace | None:
    if request.include_steps:
        return await reasoning.get_trace_with_steps(UUID(request.trace_id))
    return await reasoning.get_trace(request.trace_id)


def scoped_reasoning_traces(
    scope: MemoryScope,
    traces: Sequence[SdkReasoningTrace],
) -> list[ReasoningTraceSummary]:
    return [
        reasoning_trace_summary(trace)
        for trace in traces
        if reasoning_trace_matches_scope(trace, scope)
    ]


def reasoning_trace_matches_scope(
    trace: SdkReasoningTrace,
    scope: MemoryScope,
) -> bool:
    metadata = string_metadata(json_object(trace.metadata))
    expected_metadata = scope_metadata(scope)
    return trace.session_id == scope.session_id and all(
        metadata.get(field_name) == expected_value
        for field_name, expected_value in expected_metadata.items()
        if field_name != "session_id"
    )


def reasoning_step_matches_scope(step_like: object, scope: MemoryScope) -> bool:
    step = _reasoning_step_from_result(step_like)
    if step is None:
        return False
    metadata = string_metadata(json_object(step.metadata))
    expected_metadata = scope_metadata(scope)
    return bool(metadata) and all(
        metadata.get(field_name) == expected_value
        for field_name, expected_value in expected_metadata.items()
    )


def reasoning_trace_summary(trace: SdkReasoningTrace) -> ReasoningTraceSummary:
    return ReasoningTraceSummary(
        trace_id=str(trace.id),
        session_id=trace.session_id,
        task=redacted_text(trace.task),
        outcome=redacted_optional_text(trace.outcome),
        success=trace.success,
        started_at=_optional_datetime_text(trace.started_at),
        completed_at=_optional_datetime_text(trace.completed_at),
        metadata=_public_reasoning_metadata(json_object(trace.metadata)),
    )


def reasoning_step_record(step_like: object) -> ReasoningStepRecord:
    step = _reasoning_step_from_result(step_like)
    if step is None:
        raise BackendCapabilityUnavailable(REASONING_READ_UNAVAILABLE_DETAIL)
    return ReasoningStepRecord(
        step_id=str(step.id),
        trace_id=str(step.trace_id),
        step_number=step.step_number,
        action=redacted_optional_text(step.action),
        observation=redacted_optional_text(step.observation),
        tool_calls=[
            _reasoning_tool_call_record(tool_call) for tool_call in step.tool_calls
        ],
        metadata=redacted_reasoning_object(json_object(step.metadata)),
    )


def _reasoning_step_from_result(step_like: object) -> SdkReasoningStep | None:
    if isinstance(step_like, SdkReasoningStep):
        return step_like
    step = getattr(step_like, "step", None)
    if isinstance(step, SdkReasoningStep):
        return step
    return None


def _reasoning_tool_call_record(tool_call: ToolCall) -> JsonObject:
    return redacted_reasoning_object(
        {
            "tool_call_id": str(tool_call.id),
            "step_id": (
                str(tool_call.step_id) if tool_call.step_id is not None else None
            ),
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "result": tool_call.result,
            "status": tool_call.status.value,
            "duration_ms": tool_call.duration_ms,
            "error": tool_call.error,
            "metadata": tool_call.metadata,
        },
    )


def reasoning_tool_stats_record(tool_stats: ToolStats) -> ReasoningToolStatsRecord:
    return ReasoningToolStatsRecord(
        name=redacted_text(tool_stats.name),
        description=redacted_optional_text(tool_stats.description),
        total_calls=tool_stats.total_calls,
        successful_calls=tool_stats.successful_calls,
        failed_calls=tool_stats.failed_calls,
        success_rate=tool_stats.success_rate,
        avg_duration_ms=tool_stats.avg_duration_ms,
        last_used_at=_optional_datetime_text(tool_stats.last_used_at),
    )


def _optional_datetime_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def redacted_reasoning_object(value: object) -> JsonObject:
    return _redacted_reasoning_json(json_object(value))


def _redacted_reasoning_json(value: JsonObject) -> JsonObject:
    redacted: dict[str, JsonValue] = {}
    for key, member in value.items():
        normalized_key = key.casefold()
        if normalized_key in _UNSAFE_REASONING_KEYS:
            continue
        if normalized_key in _REDACT_REASONING_KEYS:
            redacted[key] = "[REDACTED]"
            continue
        redacted[key] = _redacted_reasoning_value(member)
    return JSON_OBJECT_ADAPTER.validate_python(redacted)


def _redacted_reasoning_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return redacted_text(value)
    if isinstance(value, dict):
        return _redacted_reasoning_json(json_object(value))
    if isinstance(value, list):
        return [_redacted_reasoning_value(item) for item in value]
    return value


def _public_reasoning_metadata(metadata: JsonObject) -> JsonObject:
    return _redacted_reasoning_json(
        {
            key: value
            for key, value in metadata.items()
            if key not in SCOPE_METADATA_KEYS
        },
    )
