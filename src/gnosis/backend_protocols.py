"""Service-facing backend protocols and error types.

The HTTP routes (:mod:`gnosis.main`), the MCP server, and the federation
client all talk to memory through these structural protocols; the errors
carry an operator-safe ``detail`` that route handlers map onto HTTP status
codes. :class:`gnosis.backend.Neo4jAgentMemoryBackend` is the production
implementation.
"""

from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

from gnosis.models import (
    BackendReadiness,
    BufferFlushResponse,
    BufferStatus,
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ConsolidationApplyRequest,
    ConsolidationApplyResponse,
    ConsolidationDryRunRequest,
    ConsolidationDryRunResponse,
    ContextRequest,
    ContextResponse,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidateRequest,
    DedupCandidateResponse,
    DedupStatsRequest,
    DedupStatsResponse,
    DiagnosticsResponse,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    EventIngestResult,
    ExtractionPreviewRequest,
    ExtractionPreviewResponse,
    FactRecord,
    FactSearchRequest,
    FactSearchResponse,
    FactWriteRequest,
    GraphContextRequest,
    GraphContextResponse,
    GraphExportRequest,
    GraphExportResponse,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemoryRecord,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MessageWriteRequest,
    MessageWriteResponse,
    PreferenceRecord,
    PreferenceSearchRequest,
    PreferenceSearchResponse,
    PreferenceWriteRequest,
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
    SdkStatsRequest,
    SdkStatsResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
)

_MEMORY_NOT_FOUND_DETAIL: Final[str] = "memory not found in scope"


class MemoryBackend(Protocol):
    async def readiness(self) -> BackendReadiness: ...
    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse: ...
    async def buffer_status(self) -> BufferStatus: ...
    async def flush_buffer(self) -> BufferFlushResponse: ...
    async def shutdown(self) -> None: ...
    async def add_message(
        self,
        request: MessageWriteRequest,
    ) -> MessageWriteResponse: ...
    async def get_context(self, request: ContextRequest) -> ContextResponse: ...
    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse: ...
    async def add_memories(self, request: MemoryAddRequest) -> MemoryAddResponse: ...
    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse: ...
    async def list_memories(
        self,
        request: MemoryListRequest,
    ) -> MemoryListResponse: ...
    async def update_memory(
        self,
        memory_id: str,
        request: MemoryUpdateRequest,
    ) -> MemoryUpdateResponse: ...
    async def delete_memory(
        self,
        memory_id: str,
        request: MemoryDeleteRequest,
    ) -> MemoryDeleteResponse: ...
    async def ingest_event(self, event: ClientEvent) -> EventIngestResult: ...
    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse: ...
    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse: ...
    async def get_sdk_stats(self, request: SdkStatsRequest) -> SdkStatsResponse: ...
    async def get_dedup_stats(
        self,
        request: DedupStatsRequest,
    ) -> DedupStatsResponse: ...
    async def find_dedup_candidates(
        self,
        request: DedupCandidateRequest,
    ) -> DedupCandidateResponse: ...
    async def apply_dedup_candidate(
        self,
        request: DedupApplyRequest,
    ) -> DedupApplyResponse: ...
    async def dry_run_consolidation(
        self,
        request: ConsolidationDryRunRequest,
    ) -> ConsolidationDryRunResponse: ...
    async def apply_consolidation(
        self,
        request: ConsolidationApplyRequest,
    ) -> ConsolidationApplyResponse: ...
    async def export_graph(
        self,
        request: GraphExportRequest,
    ) -> GraphExportResponse: ...
    async def search_entities(
        self,
        request: EntitySearchRequest,
    ) -> EntitySearchResponse: ...
    async def search_facts(self, request: FactSearchRequest) -> FactSearchResponse: ...
    async def search_preferences(
        self,
        request: PreferenceSearchRequest,
    ) -> PreferenceSearchResponse: ...
    async def add_entity(self, request: EntityWriteRequest) -> EntityRecord: ...
    async def add_fact(self, request: FactWriteRequest) -> FactRecord: ...
    async def add_preference(
        self,
        request: PreferenceWriteRequest,
    ) -> PreferenceRecord: ...
    async def list_skills(self, request: SkillListRequest) -> SkillListResponse: ...
    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal: ...
    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult: ...
    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse: ...
    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse: ...
    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse: ...
    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse: ...
    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse: ...
    async def list_reasoning_traces(
        self,
        request: ReasoningTraceListRequest,
    ) -> ReasoningTraceListResponse: ...
    async def get_reasoning_trace(
        self,
        request: ReasoningTraceDetailRequest,
    ) -> ReasoningTraceDetailResponse: ...
    async def find_similar_reasoning_traces(
        self,
        request: ReasoningSimilarTracesRequest,
    ) -> ReasoningSimilarTracesResponse: ...
    async def search_reasoning_steps(
        self,
        request: ReasoningStepSearchRequest,
    ) -> ReasoningStepSearchResponse: ...
    async def get_reasoning_tool_stats(
        self,
        request: ReasoningToolStatsRequest,
    ) -> ReasoningToolStatsResponse: ...


@runtime_checkable
class ExtractionPreviewBackend(Protocol):
    async def preview_extraction(
        self,
        request: ExtractionPreviewRequest,
    ) -> ExtractionPreviewResponse: ...


@runtime_checkable
class RecallFilteringBackend(Protocol):
    async def filter_recalled_memories(
        self,
        query: str,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]: ...


class BackendRequestError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail: str
        self.detail = detail
        super().__init__(detail)


class MemoryNotFoundError(Exception):
    def __init__(self, detail: str = _MEMORY_NOT_FOUND_DETAIL) -> None:
        self.detail: str
        self.detail = detail
        super().__init__(detail)


class BackendCapabilityUnavailable(Exception):  # noqa: N818 - Public API name.
    def __init__(self, detail: str) -> None:
        self.detail: str
        self.detail = detail
        super().__init__(detail)
