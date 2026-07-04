import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import (
    Final,
    Literal,
    Protocol,
    Self,
    TypedDict,
    assert_never,
    cast,
    runtime_checkable,
)
from uuid import UUID, uuid4

from neo4j.exceptions import Neo4jError
from neo4j_agent_memory import MemoryClient, MemoryConfig, MemorySettings, Neo4jConfig
from neo4j_agent_memory.llm.adapters.litellm import (
    LiteLLMEmbeddingProvider,
    LiteLLMProvider,
)
from neo4j_agent_memory.memory.long_term import EntityType
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus, ToolStats
from neo4j_agent_memory.schema.models import EntityRef
from openai import OpenAIError
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError

from gnosis.bridge_traversal import (
    BRIDGE_MENTION_CYPHER,
    BridgeNamer,
    LiteLLMBridgeNamer,
    bridge_parameters,
    parse_bridge_names,
)
from gnosis.entity_graph import (
    CREATE_ENTITY_SCOPE_INDEX_CYPHER,
    RelationTriple,
    entity_graph_statements,
    normalize_entity_name,
)
from gnosis.entity_traversal import (
    ENTITY_TRAVERSAL_CYPHER,
    query_seed_candidates,
    traversal_parameters,
)
from gnosis.event_facts import EventFactPromoter
from gnosis.extraction_worker import BackgroundExtractionQueue
from gnosis.fact_extraction import (
    EXTRACTION_VERSION,
    ConversationTurn,
    LiteLLMMemoryUnitExtractor,
    MemoryUnit,
    MemoryUnitExtractor,
    extract_memory_units,
    unit_relations,
)
from gnosis.graph_probe import StructuredGraphStore, direct_neo4j_driver_factory
from gnosis.graph_query_qa import LiteLLMGraphQueryPlanner
from gnosis.graph_store import DirectNeo4jGraphStore, Neo4jGraphExecutor
from gnosis.memory_filters import (
    FilterValidationError,
    MemoryFilter,
    build_cypher_filter,
    matches_filters,
    parse_filters,
)
from gnosis.memory_provider import (
    CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER,
    CREATE_MEMORY_CYPHER,
    DELETE_MEMORY_CYPHER,
    EXTRACTED_FACT_PREDICATE,
    LEXICAL_MEMORY_SEARCH_CYPHER,
    LOOKUP_LATEST_MEMORY_CYPHER,
    LOOKUP_MEMORIES_BY_IDS_CYPHER,
    LOOKUP_MEMORY_CYPHER,
    RECENT_TURN_MEMORIES_CYPHER,
    SCOPED_DENSE_MEMORY_SEARCH_CYPHER,
    TURN_MEMORY_PREDICATE_PREFIX,
    UPDATE_MEMORY_CYPHER,
    VERBATIM_MEMORY_PREDICATE,
    StoredMemory,
    fuse_memory_rankings,
    lexical_stored_memory,
    list_memories_cypher,
    memory_add_event,
    memory_filter_fields,
    memory_matches_scope,
    memory_record,
    memory_score,
    merged_memory_metadata,
    public_memory_metadata,
    sanitize_lucene_query,
    scope_read_fragments,
    scored_stored_memory_from_row,
    session_read_fragments,
    stored_memories_from_sdk,
    stored_memory_from_row,
    stored_memory_from_sdk,
)
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
    ConsolidationOperationName,
    ContextRequest,
    ContextResponse,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidate,
    DedupCandidateRequest,
    DedupCandidateResponse,
    DedupEntitySnapshot,
    DedupOperationName,
    DedupStatsRequest,
    DedupStatsResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    EventIngestResult,
    ExtractionCandidate,
    ExtractionPreviewMetrics,
    ExtractionPreviewProvenance,
    ExtractionPreviewRequest,
    ExtractionPreviewResponse,
    FactRecord,
    FactSearchRequest,
    FactSearchResponse,
    FactWriteRequest,
    GraphContextRequest,
    GraphContextResponse,
    GraphExportNode,
    GraphExportRelationship,
    GraphExportRequest,
    GraphExportResponse,
    JsonObject,
    JsonValue,
    MemoryAddEvent,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryAddResult,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemoryRecord,
    MemoryScope,
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
    ReasoningStepRecord,
    ReasoningStepRequest,
    ReasoningStepResponse,
    ReasoningStepSearchRequest,
    ReasoningStepSearchResponse,
    ReasoningToolCallRequest,
    ReasoningToolCallResponse,
    ReasoningToolStatsRecord,
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
    ReasoningTraceSummary,
    RustFSSourceReference,
    SdkStatsRequest,
    SdkStatsResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
    SufficiencyAssessment,
)
from gnosis.query_router import LiteLLMQueryRouter, QueryRouter, RouteDecision
from gnosis.recall_filter import (
    LiteLLMRecallFilter,
    RecallFilter,
    keep_relevant_candidates,
)
from gnosis.redaction import redact_secrets
from gnosis.settings import Settings
from gnosis.skill_registry import InMemorySkillRegistry, SkillRegistry
from gnosis.sufficiency import (
    LiteLLMSufficiencyAssessor,
    SufficiencyAssessor,
    bounded_reason,
)
from gnosis.supersession import FactFreshness, drop_superseded, slot_key

_JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)
_ENTITY_RECORD_ADAPTER: Final[TypeAdapter[EntityRecord]] = TypeAdapter(EntityRecord)
_ENTITY_RECORDS_ADAPTER: Final[TypeAdapter[list[EntityRecord]]] = TypeAdapter(
    list[EntityRecord],
)
_FACT_RECORD_ADAPTER: Final[TypeAdapter[FactRecord]] = TypeAdapter(FactRecord)
_FACT_RECORDS_ADAPTER: Final[TypeAdapter[list[FactRecord]]] = TypeAdapter(
    list[FactRecord],
)
_PREFERENCE_RECORD_ADAPTER: Final[TypeAdapter[PreferenceRecord]] = TypeAdapter(
    PreferenceRecord,
)
_PREFERENCE_RECORDS_ADAPTER: Final[TypeAdapter[list[PreferenceRecord]]] = TypeAdapter(
    list[PreferenceRecord]
)
_PREVIEW_WRITE_DETAIL: Final[str] = (
    "Use /v1/memory/extraction/preview for dry-run previews."
)
_RELATION_ENTITY_DETAIL: Final[str] = "Relation extraction requires entity extraction."
_RAW_TEXT_PREVIEW_DETAIL: Final[str] = (
    "Raw text extraction is available only through preview."
)
_OCR_PREVIEW_ONLY_DETAIL: Final[str] = (
    "OCR extraction is available only through preview."
)
_PREVIEW_DISABLED_DETAIL: Final[str] = (
    "Extraction preview is disabled by service policy."
)
_OCR_DISABLED_DETAIL: Final[str] = "OCR preview is disabled by service policy."
_OCR_SIZE_DETAIL: Final[str] = "OCR image exceeds service policy size limit."
_RUSTFS_DISABLED_DETAIL: Final[str] = (
    "RustFS source references are disabled by service policy."
)
_RUSTFS_BUCKET_DETAIL: Final[str] = "RustFS source bucket is outside service policy."
_RUSTFS_KEY_DETAIL: Final[str] = "RustFS source key is outside service policy."
_SDK_STATS_UNAVAILABLE_DETAIL: Final[str] = "SDK stats are unavailable."
_SDK_BUFFER_FLUSH_UNAVAILABLE_DETAIL: Final[str] = "SDK buffer flush is unavailable."
_SDK_BUFFER_WAIT_UNAVAILABLE_DETAIL: Final[str] = "SDK buffer wait is unavailable."
# Bounded shutdown drain for background extraction: enough for in-flight
# extraction LLM calls to finish, short enough not to stall pod rotation.
_EXTRACTION_DRAIN_TIMEOUT_SECONDS: Final[float] = 10.0
_NO_EXCLUDED_MEMORY_IDS: Final[frozenset[str]] = frozenset()
_SDK_GRAPH_UNAVAILABLE_DETAIL: Final[str] = "SDK graph export is unavailable."
_DEDUP_UNAVAILABLE_DETAIL: Final[str] = "SDK deduplication is unavailable."
_QUERY_EMBEDDER_UNAVAILABLE_DETAIL: Final[str] = "SDK query embedder is unavailable."
_DEDUP_APPLY_REQUIRED_DETAIL: Final[str] = (
    "Deduplication apply requests require apply=true."
)
_DEDUP_TOKEN_DETAIL: Final[str] = "Deduplication dry-run token is invalid or expired."  # noqa: S105
_DEDUP_STALE_DETAIL: Final[str] = "Deduplication candidate is stale."
_DEDUP_IDEMPOTENCY_DETAIL: Final[str] = (
    "Idempotency key was already used for a different deduplication request."
)
_DEDUP_TOKEN_TTL: Final[timedelta] = timedelta(minutes=15)
_DEDUP_TOKEN_PARTS: Final[int] = 2
_DEDUP_PENDING_TOKEN: Final[str] = "pending"  # noqa: S105
_CONSOLIDATION_UNAVAILABLE_DETAIL: Final[str] = "SDK consolidation is unavailable."
_CONSOLIDATION_APPLY_REQUIRED_DETAIL: Final[str] = (
    "Consolidation apply requests require apply=true."
)
_CONSOLIDATION_TOKEN_DETAIL: Final[str] = (
    "Consolidation dry-run token is invalid or expired."  # noqa: S105
)
_CONSOLIDATION_STALE_DETAIL: Final[str] = "Consolidation dry-run report is stale."
_CONSOLIDATION_IDEMPOTENCY_DETAIL: Final[str] = (
    "Idempotency key was already used for a different consolidation request."
)
_CONSOLIDATION_TOKEN_TTL: Final[timedelta] = timedelta(minutes=15)
_REASONING_READ_UNAVAILABLE_DETAIL: Final[str] = "SDK reasoning read is unavailable."
_MEMORY_MODE_DETAIL: Final[str] = (
    "Provide messages with infer=true or content with infer=false."
)
_MEMORY_MESSAGES_INFER_DETAIL: Final[str] = "messages require infer=true."
_MEMORY_CONTENT_INFER_DETAIL: Final[str] = "content requires infer=false."
_MEMORY_UPDATE_FIELDS_DETAIL: Final[str] = "Memory updates require content or metadata."
_MEMORY_WRITE_UNAVAILABLE_DETAIL: Final[str] = "SDK graph writes are unavailable."
_MEMORY_ID_UNAVAILABLE_DETAIL: Final[str] = "SDK did not expose a stable memory id."
_MEMORY_NOT_FOUND_DETAIL: Final[str] = "memory not found in scope"
_MEMORY_LIST_SCAN_LIMIT: Final[int] = 2000
_MEMORY_SEARCH_CANDIDATE_LIMIT: Final[int] = 100
_UNSAFE_CONSOLIDATION_REPORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "chain_of_thought",
        "credentials",
        "cypher",
        "embedding",
        "embeddings",
        "password",
        "prompt",
        "raw_prompt",
        "rustfs_credentials",
        "secret",
        "thought",
        "token",
    },
)
_UNSAFE_REASONING_KEYS: Final[frozenset[str]] = _UNSAFE_CONSOLIDATION_REPORT_KEYS | {
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
_SCOPE_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "tenant_id",
        "space_id",
        "agent_id",
        "session_id",
        "user_id",
        "visibility",
        "guild_id",
        "channel_id",
    },
)
_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_ABSTENTION_INSTRUCTION: Final[str] = (
    "Answer only from the memories below; if they do not contain the answer, "
    "say you don't know."
)
# Chain-of-Note read-then-reason (arXiv 2311.09210; LongMemEval arXiv
# 2410.10813 measures up to +10 QA points from structured reading on
# conversational memory). The note step lets the reader reject retrieved
# similar-but-wrong memories instead of answering from them, and raises
# rejection of unanswerable questions - reading is the lever, not retrieval.
# The attribution and yes/no clauses target the two residual adversarial
# failure patterns measured in Run 15 (2026-07-04, 24 misses): answering
# with a *different person's* similar fact, and answering yes/no about a
# thing the memories never mention. The likelihood clause carves the one
# measured over-abstention cost back out (Run 17: open-domain "would X
# likely..." questions expect an inference from known traits, and the bare
# never-guess rule refused them).
_CHAIN_OF_NOTE_INSTRUCTION: Final[str] = (
    "Before answering, silently take notes on each memory below: state "
    "whether it is relevant to the question, what it says, who it is about, "
    "and whether it contradicts another memory. Ignore memories that are "
    "merely similar to the question but do not answer it, especially "
    "memories about a different person than the question asks about. Then "
    "answer using only the relevant memories; if no memory states the "
    "answer, say you don't know - never guess, and never answer yes or no "
    "about something the memories never mention. Only when the question "
    "itself asks what is likely or probable, infer the most plausible "
    "answer from the relevant memories instead of saying you don't know."
)


class MemoryConfigKwargs(TypedDict, total=False):
    multi_tenant: bool
    write_mode: Literal["sync", "buffered"]
    max_pending: int
    conversation_ttl_days: int | None
    audit_read: bool
    fact_deduplication_enabled: bool
    trace_embedding_enabled: bool


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


class MemoryClientFactory(Protocol):
    def __call__(self, settings: MemorySettings) -> "MemoryClientContext": ...


class ShortTermMemory(Protocol):
    async def add_message(  # noqa: PLR0913
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        user_identifier: str,
        metadata: dict[str, str],
        extract_entities: bool,
        extract_relations: bool,
    ) -> object: ...

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str: ...


class LongTermFactMemory(Protocol):
    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> object: ...

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> object: ...

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> object: ...

    async def add_entity(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        name: str,
        entity_type: EntityType | str,
        *,
        subtype: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: JsonObject | None = None,
        resolve: bool = True,
        generate_embedding: bool = True,
        deduplicate: bool = True,
        geocode: bool = True,
        enrich: bool = True,
        coordinates: tuple[float, float] | None = None,
        metadata: JsonObject | None = None,
    ) -> object: ...

    async def add_fact(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
    ) -> object: ...

    async def add_preference(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        category: str,
        preference: str,
        *,
        context: str | None = None,
        confidence: float = 1.0,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
        user_identifier: str | None = None,
        applies_to: object | None = None,
    ) -> object: ...

    async def get_context(self, query: str, *, max_items: int) -> str: ...


class LongTermMemory(Protocol):
    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[EntityRecord]: ...

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[FactRecord]: ...

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[PreferenceRecord]: ...

    async def add_entity(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        name: str,
        entity_type: EntityType | str,
        *,
        subtype: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: JsonObject | None = None,
        resolve: bool = True,
        generate_embedding: bool = True,
        deduplicate: bool = True,
        geocode: bool = True,
        enrich: bool = True,
        coordinates: tuple[float, float] | None = None,
        metadata: JsonObject | None = None,
    ) -> EntityRecord: ...

    async def add_fact(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
    ) -> FactRecord: ...

    async def add_preference(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        category: str,
        preference: str,
        *,
        context: str | None = None,
        confidence: float = 1.0,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
        user_identifier: str | None = None,
        applies_to: object | None = None,
    ) -> PreferenceRecord: ...

    async def get_preferences_for(
        self,
        user_identifier: str,
        *,
        applies_to: object | None = None,
        active_only: bool = True,
        as_of: datetime | None = None,
    ) -> list[PreferenceRecord]: ...

    async def get_facts_about(
        self,
        subject: str,
        *,
        limit: int = 100,
    ) -> list[FactRecord]: ...

    async def link_entity_to_message(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        entity: EntityRecord | UUID,
        message_id: UUID | str,
        *,
        confidence: float = 1.0,
        start_pos: int | None = None,
        end_pos: int | None = None,
        context: str | None = None,
    ) -> bool: ...

    async def link_entity_to_extractor(
        self,
        entity: EntityRecord | UUID,
        extractor_name: str,
        *,
        confidence: float = 1.0,
        extraction_time_ms: float | None = None,
    ) -> bool: ...

    async def get_context(self, query: str, *, max_items: int) -> str: ...


@runtime_checkable
class DedupCapableLongTermMemory(Protocol):
    def get_deduplication_stats(self) -> Awaitable[object]: ...
    def find_potential_duplicates(
        self,
        *,
        limit: int = 100,
    ) -> Awaitable[list[tuple[object, object, float]]]: ...
    def review_duplicate(
        self,
        source_id: UUID,
        target_id: UUID,
        *,
        confirm: bool,
    ) -> Awaitable[bool]: ...
    def merge_duplicate_entities(
        self,
        source_id: UUID,
        target_id: UUID,
    ) -> Awaitable[tuple[object, object] | None]: ...


@runtime_checkable
class ConsolidationMemory(Protocol):
    def archive_expired_conversations(
        self,
        *,
        ttl_days: int | None = None,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...

    def dedupe_entities(
        self,
        *,
        similarity_threshold: float = 0.95,
        max_pairs: int = 10000,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...

    def detect_superseded_preferences(
        self,
        *,
        user_identifier: str | None = None,
        similarity_threshold: float = 0.92,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...

    def summarize_long_traces(
        self,
        *,
        min_steps: int = 20,
        max_traces: int = 1000,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...


@runtime_checkable
class ConsolidationCapableMemoryClient(Protocol):
    @property
    def consolidation(self) -> ConsolidationMemory: ...


@dataclass(frozen=True, slots=True)
class DedupCandidateState:
    candidate_id: str
    version: int
    scope: MemoryScope
    source_id: UUID
    target_id: UUID
    graph_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class DedupTokenClaims:
    scope: MemoryScope
    candidate_id: str
    candidate_version: int
    graph_snapshot_hash: str
    operation: DedupOperationName
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class DedupIdempotencyRecord:
    request_fingerprint: str
    response: DedupApplyResponse


@dataclass(frozen=True, slots=True)
class ConsolidationDryRunState:
    scope: MemoryScope
    operation: ConsolidationOperationName
    graph_snapshot_hash: str
    request_fingerprint: str


@dataclass(frozen=True, slots=True)
class ConsolidationTokenClaims:
    scope: MemoryScope
    operation: ConsolidationOperationName
    graph_snapshot_hash: str
    request_fingerprint: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ConsolidationIdempotencyRecord:
    request_fingerprint: str
    response: ConsolidationApplyResponse


class ReasoningMemory(Protocol):
    async def get_context(self, query: str, *, max_traces: int) -> str: ...
    async def start_trace(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        session_id: str,
        task: str,
        *,
        generate_embedding: bool,
        metadata: JsonObject | None,
        triggered_by_message_id: str | None,
        user_identifier: str,
    ) -> SdkReasoningTrace: ...
    async def add_step(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        trace_id: UUID,
        *,
        thought: None,
        action: str | None,
        observation: str | None,
        generate_embedding: bool,
        metadata: JsonObject | None,
    ) -> SdkReasoningStep: ...
    async def record_tool_call(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        step_id: UUID,
        tool_name: str,
        arguments: JsonObject,
        *,
        result: JsonValue | None,
        status: ToolCallStatus,
        duration_ms: int | None,
        error: str | None,
        message_id: str | None,
        touched_entities: list[EntityRef],
    ) -> ToolCall: ...
    async def complete_trace(
        self,
        trace_id: UUID,
        *,
        outcome: str | None,
        success: bool | None,
        generate_step_embeddings: bool,
    ) -> SdkReasoningTrace: ...
    async def list_traces(
        self,
        *,
        session_id: str | None = None,
        success_only: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SdkReasoningTrace]: ...
    async def get_trace(self, trace_id: UUID | str) -> SdkReasoningTrace | None: ...
    async def get_trace_with_steps(
        self,
        trace_id: UUID,
    ) -> SdkReasoningTrace | None: ...
    async def get_similar_traces(
        self,
        task: str,
        *,
        limit: int = 5,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[SdkReasoningTrace]: ...
    async def search_steps(
        self,
        query: str,
        *,
        limit: int = 10,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> Sequence[object]: ...
    async def get_tool_stats(
        self,
        tool_name: str | None = None,
    ) -> list[ToolStats]: ...


class CypherQuery(Protocol):
    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]: ...


class MemoryClientContext(Protocol):
    @property
    def short_term(self) -> ShortTermMemory: ...
    @property
    def long_term(self) -> LongTermFactMemory: ...
    @property
    def reasoning(self) -> ReasoningMemory: ...
    @property
    def query(self) -> CypherQuery: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


@runtime_checkable
class StatsCapableMemoryClient(Protocol):
    def get_stats(self) -> Awaitable[object]: ...


@runtime_checkable
class BufferFlushCapableMemoryClient(Protocol):
    def flush(self) -> Awaitable[object]: ...


@runtime_checkable
class BufferPendingCapableMemoryClient(Protocol):
    def wait_for_pending(self) -> Awaitable[object]: ...


@runtime_checkable
class BufferErrorCapableMemoryClient(Protocol):
    @property
    def write_errors(self) -> Sequence[object]: ...


class GraphNodeLike(Protocol):
    id: str
    labels: Sequence[str]
    properties: object


class GraphRelationshipLike(Protocol):
    id: str
    type: str
    from_node: str
    to_node: str
    properties: object


class MemoryGraphLike(Protocol):
    nodes: Sequence[GraphNodeLike]
    relationships: Sequence[GraphRelationshipLike]
    metadata: object


@runtime_checkable
class GraphCapableMemoryClient(Protocol):
    def get_graph(
        self,
        *,
        memory_types: list[Literal["short_term", "long_term", "reasoning"]] | None,
        session_id: str | None,
        include_embeddings: bool,
        limit: int,
    ) -> Awaitable[MemoryGraphLike]: ...


@runtime_checkable
class GraphWriteQuery(Protocol):
    def execute_write(
        self,
        query: str,
        parameters: dict[str, JsonValue] | None = None,
    ) -> Awaitable[list[JsonObject]]: ...


@runtime_checkable
class TextEmbedder(Protocol):
    def embed(self, text: str) -> Awaitable[list[float]]: ...


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


@dataclass(frozen=True, slots=True)
class ExtractionPolicy:
    extract_entities: bool
    extract_relations: bool


@dataclass(frozen=True, slots=True)
class LongTermFactsContext:
    context: str = ""
    markers: set[str] = field(default_factory=set)


def build_direct_graph_store(settings: Settings) -> DirectNeo4jGraphStore:
    embedding_provider = LiteLLMEmbeddingProvider(
        litellm_embedding_model(settings.gnosis_embedding),
        dimensions=settings.gnosis_embedding_dimensions,
        api_base=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    return DirectNeo4jGraphStore(
        executor=Neo4jGraphExecutor(
            driver_factory=direct_neo4j_driver_factory(settings),
            embedding_dimensions=settings.gnosis_embedding_dimensions,
            embedding_provider=embedding_provider,
            graph_query_planner=LiteLLMGraphQueryPlanner(
                model=settings.gnosis_llm,
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
            ),
        ),
    )


class Neo4jAgentMemoryBackend:
    def __init__(  # noqa: PLR0913 - One injection seam per LLM collaborator.
        self,
        settings: Settings,
        memory_client_factory: MemoryClientFactory | None = None,
        graph_store: StructuredGraphStore | None = None,
        skill_registry: SkillRegistry | None = None,
        recall_filter: RecallFilter | None = None,
        fact_extractor: MemoryUnitExtractor | None = None,
        sufficiency_assessor: SufficiencyAssessor | None = None,
        query_router: QueryRouter | None = None,
        bridge_namer: BridgeNamer | None = None,
    ) -> None:
        self._app_settings: Settings = settings
        self._settings: MemorySettings = _build_memory_settings(settings)
        self._memory_client_factory: MemoryClientFactory | None = memory_client_factory
        self._graph_store: StructuredGraphStore = (
            graph_store or build_direct_graph_store(settings)
        )
        self._recall_filter: RecallFilter = recall_filter or LiteLLMRecallFilter(
            model=settings.gnosis_llm,
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._fact_extractor: MemoryUnitExtractor = (
            fact_extractor
            or LiteLLMMemoryUnitExtractor(
                model=_fact_extraction_model(settings),
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
                # The entity graph needs (head, relation, tail) triples, so the
                # extractor emits them only when materialization is enabled;
                # off, the edu-v1 prompt and schema stay byte-identical.
                emit_relations=settings.gnosis_entity_graph_enabled,
            )
        )
        self._sufficiency_assessor: SufficiencyAssessor = (
            sufficiency_assessor
            or LiteLLMSufficiencyAssessor(
                model=_sufficiency_model(settings),
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
            )
        )
        self._query_router: QueryRouter = query_router or LiteLLMQueryRouter(
            model=_routing_model(settings),
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._bridge_namer: BridgeNamer = bridge_namer or LiteLLMBridgeNamer(
            model=_routing_model(settings),
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._skill_registry: SkillRegistry = skill_registry or InMemorySkillRegistry()
        self._extraction_queue: BackgroundExtractionQueue = BackgroundExtractionQueue(
            max_concurrency=settings.gnosis_fact_extraction_max_concurrency,
            max_pending=settings.gnosis_fact_extraction_max_pending,
        )
        self._event_fact_promoter: EventFactPromoter = EventFactPromoter()
        self._fulltext_index_ready: bool = False
        self._entity_graph_schema_ready: bool = False
        self._dedup_candidates: dict[str, DedupCandidateState] = {}
        self._dedup_idempotency: dict[str, DedupIdempotencyRecord] = {}
        self._consolidation_dry_runs: dict[str, ConsolidationDryRunState] = {}
        self._consolidation_idempotency: dict[
            str,
            ConsolidationIdempotencyRecord,
        ] = {}

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        metadata = _scope_metadata(request.scope)
        policy = _message_extraction_policy(request, self._app_settings)
        _require_ingestion_sources_allowed(request, self._app_settings)
        extraction_enabled = self._app_settings.gnosis_fact_extraction_enabled
        extract_inline = extraction_enabled and not self._background_extraction()
        source_memory_ids: list[str] = []
        async with self._memory_client() as client:
            context_turns: list[ConversationTurn] = []
            if extract_inline:
                context_turns = await self._recent_session_turns(
                    client,
                    request.scope,
                )
            _ = await client.short_term.add_message(
                session_id=_session_id(request.scope),
                role=request.role.value,
                content=request.content,
                user_identifier=_user_identifier(request.scope),
                metadata=metadata,
                extract_entities=policy.extract_entities,
                extract_relations=policy.extract_relations,
            )
            raw_record = await client.long_term.add_fact(
                subject=_user_identifier(request.scope),
                predicate=f"said_{request.role.value}",
                obj=request.content,
                metadata=_scope_json_metadata(request.scope),
                generate_embedding=True,
            )
            if extraction_enabled:
                memory = stored_memory_from_sdk(raw_record)
                source_memory_ids = [memory.memory_id] if memory is not None else []
            if extract_inline:
                _ = await self._extracted_unit_results(
                    client,
                    request.scope,
                    caller_metadata={},
                    context_turns=context_turns,
                    new_turns=[
                        ConversationTurn(
                            speaker=request.role.value,
                            content=request.content,
                        ),
                    ],
                    source_memory_ids=source_memory_ids,
                )
        if extraction_enabled and not extract_inline:
            self._enqueue_background_extraction(
                request.scope,
                caller_metadata={},
                new_turns=[
                    ConversationTurn(
                        speaker=request.role.value,
                        content=request.content,
                    ),
                ],
                source_memory_ids=source_memory_ids,
            )
        return MessageWriteResponse(accepted=True)

    async def preview_extraction(
        self,
        request: ExtractionPreviewRequest,
    ) -> ExtractionPreviewResponse:
        _require_preview_enabled(self._app_settings)
        policy = _preview_extraction_policy(request, self._app_settings)
        _require_preview_sources_allowed(request, self._app_settings)
        documents = _preview_document_count(request)
        candidates = _preview_candidates(request, self._app_settings)
        return ExtractionPreviewResponse(
            candidates=candidates,
            metrics=ExtractionPreviewMetrics(
                documents=documents,
                chunks=max(documents, 1),
                ocr_images=len(request.ocr_image_references),
                rustfs_objects=len(request.rustfs_source_references),
                batch_size=self._app_settings.gnosis_extraction_batch_size,
                max_concurrency=self._app_settings.gnosis_extraction_max_concurrency,
            ),
            provenance=ExtractionPreviewProvenance(
                source_ids=_preview_source_ids(request),
                rustfs_objects=request.rustfs_source_references,
            ),
            extract_entities=policy.extract_entities,
            extract_relations=policy.extract_relations,
        )

    async def readiness(self) -> BackendReadiness:
        graph_readiness = await self._graph_store.readiness()
        buffer_status = await self.buffer_status()
        return graph_readiness.model_copy(
            update={"buffer_status": buffer_status.status},
        )

    async def buffer_status(self) -> BufferStatus:
        write_errors = await self._buffer_write_error_count()
        return BufferStatus(
            write_mode=self._app_settings.gnosis_write_mode,
            max_pending=self._app_settings.gnosis_max_pending,
            pending_writes=None,
            write_errors=write_errors,
            status=_buffer_readiness_status(write_errors),
        )

    async def flush_buffer(self) -> BufferFlushResponse:
        async with self._memory_client() as client:
            if not isinstance(client, BufferFlushCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_BUFFER_FLUSH_UNAVAILABLE_DETAIL)
            _ = await client.flush()
        return BufferFlushResponse(flushed=True, status=await self.buffer_status())

    async def shutdown(self) -> None:
        await self._extraction_queue.drain(
            drain_window_seconds=_EXTRACTION_DRAIN_TIMEOUT_SECONDS,
        )
        if self._app_settings.gnosis_write_mode != "buffered":
            return
        async with self._memory_client() as client:
            if not isinstance(client, BufferPendingCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_BUFFER_WAIT_UNAVAILABLE_DETAIL)
            _ = await client.wait_for_pending()

    async def _buffer_write_error_count(self) -> int:
        async with self._memory_client() as client:
            if not isinstance(client, BufferErrorCapableMemoryClient):
                return 0
            return len(client.write_errors)

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        return DiagnosticsResponse(
            tenant_id=self._app_settings.gnosis_tenant_id,
            config=DiagnosticsConfig(
                neo4j_uri=self._app_settings.neo4j_uri,
                neo4j_username=self._app_settings.neo4j_username,
                litellm_base_url=self._app_settings.litellm_base_url,
                gnosis_llm=self._app_settings.gnosis_llm,
                gnosis_embedding=self._app_settings.gnosis_embedding,
                gnosis_embedding_dimensions=(
                    self._app_settings.gnosis_embedding_dimensions
                ),
                gnosis_audit_read=self._app_settings.gnosis_audit_read,
                gnosis_conversation_ttl_days=(
                    self._app_settings.gnosis_conversation_ttl_days
                ),
                gnosis_write_mode=self._app_settings.gnosis_write_mode,
                gnosis_max_pending=self._app_settings.gnosis_max_pending,
                gnosis_fact_deduplication_enabled=(
                    self._app_settings.gnosis_fact_deduplication_enabled
                ),
                gnosis_trace_embedding_enabled=(
                    self._app_settings.gnosis_trace_embedding_enabled
                ),
                gnosis_extract_entities_enabled=(
                    self._app_settings.gnosis_extract_entities_enabled
                ),
                gnosis_extract_relations_enabled=(
                    self._app_settings.gnosis_extract_relations_enabled
                ),
                gnosis_extraction_preview_enabled=(
                    self._app_settings.gnosis_extraction_preview_enabled
                ),
                gnosis_extraction_batch_size=(
                    self._app_settings.gnosis_extraction_batch_size
                ),
                gnosis_extraction_max_concurrency=(
                    self._app_settings.gnosis_extraction_max_concurrency
                ),
                gnosis_extraction_chunk_size=(
                    self._app_settings.gnosis_extraction_chunk_size
                ),
                gnosis_extraction_chunk_overlap=(
                    self._app_settings.gnosis_extraction_chunk_overlap
                ),
                gnosis_fact_extraction_enabled=(
                    self._app_settings.gnosis_fact_extraction_enabled
                ),
                gnosis_fact_extraction_model=(
                    self._app_settings.gnosis_fact_extraction_model
                ),
                gnosis_fact_extraction_context_turns=(
                    self._app_settings.gnosis_fact_extraction_context_turns
                ),
                gnosis_fact_extraction_mode=(
                    self._app_settings.gnosis_fact_extraction_mode
                ),
                gnosis_fact_extraction_max_concurrency=(
                    self._app_settings.gnosis_fact_extraction_max_concurrency
                ),
                gnosis_fact_extraction_max_pending=(
                    self._app_settings.gnosis_fact_extraction_max_pending
                ),
                gnosis_ocr_enabled=self._app_settings.gnosis_ocr_enabled,
                gnosis_ocr_model=self._app_settings.gnosis_ocr_model,
                gnosis_ocr_max_image_bytes=(
                    self._app_settings.gnosis_ocr_max_image_bytes
                ),
                gnosis_rustfs_enabled=self._app_settings.gnosis_rustfs_enabled,
                gnosis_rustfs_bucket=self._app_settings.gnosis_rustfs_bucket,
                gnosis_rustfs_prefix=self._app_settings.gnosis_rustfs_prefix,
                gnosis_rustfs_endpoint=self._app_settings.gnosis_rustfs_endpoint,
                gnosis_rustfs_retention_days=(
                    self._app_settings.gnosis_rustfs_retention_days
                ),
                gnosis_prompt_entities_enabled=(
                    self._app_settings.gnosis_prompt_entities_enabled
                ),
                gnosis_prompt_preferences_enabled=(
                    self._app_settings.gnosis_prompt_preferences_enabled
                ),
                gnosis_prompt_reasoning_enabled=(
                    self._app_settings.gnosis_prompt_reasoning_enabled
                ),
                gnosis_consolidation_schedule_enabled=(
                    self._app_settings.gnosis_consolidation_schedule_enabled
                ),
            ),
            backend=readiness,
            extraction_queue=self._extraction_queue.status(
                mode=self._app_settings.gnosis_fact_extraction_mode,
            ),
        )

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        combined = await self.get_memory_context(_legacy_context_request(request))
        return _legacy_context_response(combined)

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        sections: list[MemoryContextSection] = []
        long_term_facts = LongTermFactsContext()
        decision = await self._route_decision(request.query)
        async with self._memory_client() as client:
            if request.include_short_term:
                try:
                    short_term = await client.short_term.get_context(
                        request.query,
                        session_id=_session_id(request.scope),
                        max_messages=request.max_items,
                        metadata_filters=_scope_metadata(request.scope),
                    )
                except (ValueError, ValidationError) as error:
                    _LOGGER.warning("short-term memory context skipped: %s", error)
                else:
                    _append_context_section(sections, "short_term", short_term)

            if request.include_long_term:
                long_term_facts = await self._get_long_term_facts_context(
                    request,
                    client,
                    decision,
                )
                _append_context_section(
                    sections,
                    "long_term_facts",
                    long_term_facts.context,
                )

                if _long_term_enrichment_enabled(self._app_settings):
                    long_term = await client.long_term.get_context(
                        request.query,
                        max_items=request.max_items,
                    )
                    _append_context_section(
                        sections,
                        "long_term_preferences_entities",
                        _redacted_text(long_term),
                    )

            if (
                request.include_reasoning
                and self._app_settings.gnosis_prompt_reasoning_enabled
            ):
                reasoning = await client.reasoning.get_context(
                    request.query,
                    max_traces=request.max_items,
                )
                _append_context_section(
                    sections,
                    "reasoning",
                    _safe_reasoning_context(reasoning),
                )

        if request.include_graph:
            graph = await self.get_graph_context(
                GraphContextRequest(
                    scope=request.scope,
                    query=request.query,
                    limit=request.graph_limit,
                ),
            )
            graph = _dedupe_graph_context(graph, long_term_facts.markers)
            if graph.context:
                sections.append(
                    MemoryContextSection(
                        source="graph",
                        content=graph.context,
                        facts=graph.facts,
                    ),
                )

        sufficiency = await self._assess_sufficiency(request.query, sections)
        sections = self._with_abstention_instruction(sections, decision)
        return MemoryContextResponse(sections=sections, sufficiency=sufficiency)

    async def _route_decision(self, query: str) -> RouteDecision:
        """Resolve the effective read-path feature set for one query.

        The globally configured flags while GNOSIS_ADAPTIVE_ROUTING_ENABLED is
        off or the query is empty (byte-identical unrouted behavior). With the
        flag on, one cheap structured-output LLM call classifies the query and
        the route's measured-best feature set replaces the global toggles for
        this request; any classifier failure degrades to the global flags with
        a structured warning, so routing can never fail a read.
        """
        unrouted = RouteDecision.from_settings(self._app_settings)
        if not self._app_settings.gnosis_adaptive_routing_enabled or not query:
            return unrouted
        try:
            verdict = await self._query_router.classify(query)
        except (RuntimeError, OSError, OpenAIError, ValidationError) as error:
            _LOGGER.warning(
                "query routing failed; using globally configured flags",
                extra={"error_type": type(error).__name__},
            )
            return unrouted
        if verdict is None:
            _LOGGER.warning(
                "query router returned no verdict; using globally configured flags",
            )
            return unrouted
        return RouteDecision.for_route(verdict.route, self._app_settings)

    def _with_abstention_instruction(
        self,
        sections: list[MemoryContextSection],
        decision: RouteDecision,
    ) -> list[MemoryContextSection]:
        """Prepend the standing reading instruction as a leading section.

        Two prompt-only reading aids share this seam, both added as a leading
        section so existing section parsing stays intact and both no-ops
        (byte-identical output) when off or when no memory content was
        assembled:

        * the decision's ``chain_of_note`` - Chain-of-Note read-then-reason
          (arXiv 2311.09210): note each memory's relevance first, then answer
          from the relevant ones or abstain. Subsumes the abstention line, so
          it takes precedence when the effective decision also asks for the
          bare abstention instruction. Carried on the route decision because
          it is route-aware: stacked globally with routing it cost temporal
          8.9 points (Run 14 - the note step parrots hybrid's relative-dated
          raw turns), so routed requests skip it on the temporal route.
        * the decision's ``abstention_prompt`` - the bare grounding
          instruction (AbstentionBench, arXiv 2506.09038), globally via
          GNOSIS_ABSTENTION_PROMPT_ENABLED or routed for unanswerable-risk
          queries.
        """
        if not sections:
            return sections
        if decision.chain_of_note:
            content = _CHAIN_OF_NOTE_INSTRUCTION
        elif decision.abstention_prompt:
            content = _ABSTENTION_INSTRUCTION
        else:
            return sections
        instruction = MemoryContextSection(
            source="instructions",
            content=content,
        )
        return [instruction, *sections]

    async def _assess_sufficiency(
        self,
        query: str,
        sections: list[MemoryContextSection],
    ) -> SufficiencyAssessment | None:
        """Judge whether the assembled context can answer the query.

        One structured-output LLM call behind GNOSIS_SUFFICIENCY_CHECK_ENABLED;
        absent (``None``) while the flag is off or the query is empty. Any
        failure degrades to ``assessed=False`` so the check never blocks the
        context response.
        """
        if not self._app_settings.gnosis_sufficiency_check_enabled or not query:
            return None
        context = "\n\n".join(section.content for section in sections)
        try:
            verdict = await self._sufficiency_assessor.assess(query, context)
        except (RuntimeError, OSError, OpenAIError) as error:
            _LOGGER.warning(
                "sufficiency check failed; reporting not assessed",
                extra={"error_type": type(error).__name__},
            )
            return SufficiencyAssessment(assessed=False)
        if verdict is None:
            return SufficiencyAssessment(assessed=False)
        return SufficiencyAssessment(
            assessed=True,
            sufficient=verdict.sufficient,
            reason=bounded_reason(verdict.reason),
        )

    async def _get_long_term_facts_context(
        self,
        request: MemoryContextRequest,
        client: MemoryClientContext,
        decision: RouteDecision,
    ) -> "LongTermFactsContext":
        """Render scoped long-term facts with the same read reach as search.

        Reads narrow by the cross-session scope (long-term memory spans
        sessions, so session_id never prunes recall). When the request has a
        query, candidates come from the same embedding-similarity search that
        /v1/memories/search uses so relevance decides which facts fit the
        ``max_items`` budget; recency ordering is the fallback when similarity
        search has nothing to rank (no query or no embedder). With
        GNOSIS_RECALL_FILTER_ENABLED on, one LLM call then screens the top
        candidates against the query before the item budget applies. The
        ``decision`` carries the effective per-request read-path feature set
        (the global flags unrouted, or the route's measured-best set).
        """
        metadata = _scope_metadata(request.scope)
        facts, graph_facts, traversal_facts = await asyncio.gather(
            self._query_ranked_facts(client, request.query, metadata, decision),
            self._graphqa_fused_facts(request, decision),
            self._traversal_facts(client, request, metadata, decision),
        )
        if not facts:
            facts = await _query_recent_facts(client, metadata)
        # The directed bridge hop reads hop-1's dense evidence, so it runs
        # after the parallel retrieval legs, not among them.
        bridge_facts = await self._bridge_traversal_facts(
            client,
            request,
            metadata,
            decision,
            facts,
        )
        facts = _fuse_graph_facts(
            facts,
            [*graph_facts, *traversal_facts, *bridge_facts],
        )
        facts = await self._recall_filtered_facts(request.query, facts)
        facts = self._superseded_facts(facts)
        facts = _cut_with_graph_reserve(
            facts,
            request.max_items * decision.budget_multiplier,
        )
        if not facts:
            return LongTermFactsContext()
        expansion = await self._verbatim_expansion(client, facts, metadata, decision)
        lines = ["### Long-Term Facts"]
        for fact in facts:
            lines.append(_fact_context_line(fact))
            fact_id = fact.get("id")
            if isinstance(fact_id, str):
                lines.extend(
                    f"  quote: {_redacted_text(quote)}"
                    for quote in expansion.get(fact_id, ())
                )
        return LongTermFactsContext(
            context="\n".join(lines),
            markers=_fact_markers(facts),
        )

    async def _verbatim_expansion(
        self,
        client: MemoryClientContext,
        facts: list[JsonObject],
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
    ) -> dict[str, list[str]]:
        """Map each top extracted fact to its source verbatim turn text(s).

        Frontier technique (EverMemOS facts->episodes; True Memory verbatim):
        rank on the compact extracted fact for precision, then surface the
        linked raw turn(s) for nuance. A no-op empty mapping while the
        effective decision leaves expansion off (byte-identical output).
        Only the highest-ranked GNOSIS_FACT_VERBATIM_EXPANSION_MAX extracted
        facts expand; a verbatim turn already present in the ranked facts (or
        already attached to an earlier fact) is never double-rendered; the
        scoped batch lookup re-checks scope on every fetched row so a source
        id can never leak a cross-scope turn. Any lookup failure or empty
        source set degrades to the compact fact alone.
        """
        if not decision.verbatim_expansion:
            return {}
        targets = _verbatim_expansion_targets(
            facts,
            cap=self._app_settings.gnosis_fact_verbatim_expansion_max,
        )
        wanted_ids: set[str] = {
            source_id for _, source_ids in targets for source_id in source_ids
        }
        if not wanted_ids:
            return {}
        verbatim = await self._fetch_verbatim_turns(
            client,
            sorted(wanted_ids),
            scope_metadata,
        )
        if not verbatim:
            return {}
        rendered: set[str] = set()
        expansion: dict[str, list[str]] = {}
        for fact_id, source_ids in targets:
            quotes: list[str] = []
            for source_id in source_ids:
                if source_id not in verbatim or source_id in rendered:
                    continue
                rendered.add(source_id)
                quotes.append(verbatim[source_id])
            if quotes:
                expansion[fact_id] = quotes
        if expansion:
            _LOGGER.info(
                "facts-to-verbatim expansion attached source turns",
                extra={"expanded": len(expansion), "candidates": len(facts)},
            )
        return expansion

    async def _fetch_verbatim_turns(
        self,
        client: MemoryClientContext,
        memory_ids: list[str],
        scope_metadata: Mapping[str, JsonValue],
    ) -> dict[str, str]:
        """Batch-fetch source verbatim turns by id, re-checking scope.

        One parameterized Cypher lookup narrowed by the scope metadata
        fragments; every returned row is scope-re-checked with the same
        predicate the ranked read uses, so a cross-scope turn is dropped and
        never rendered. A lookup failure degrades to no expansion.
        """
        try:
            rows = await client.query.cypher(
                LOOKUP_MEMORIES_BY_IDS_CYPHER,
                {
                    "memory_ids": list(memory_ids),
                    "scope_fragments": _metadata_fragments(scope_metadata),
                },
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "verbatim expansion lookup failed; keeping compact facts",
                extra={"error_type": type(error).__name__},
            )
            return {}
        verbatim: dict[str, str] = {}
        for row in rows:
            memory = stored_memory_from_row(row)
            if memory is None:
                continue
            fact = _fact_from_memory(memory)
            if not _fact_matches_scope(fact, scope_metadata):
                continue
            verbatim[memory.memory_id] = memory.content
        return verbatim

    async def add_memories(self, request: MemoryAddRequest) -> MemoryAddResponse:
        _require_memory_add_mode(request)
        metadata = _write_metadata(request.scope, request.metadata, None)
        results: list[MemoryAddResult] = []
        async with self._memory_client() as client:
            if request.content is not None:
                results.append(
                    await self._add_memory_fact(
                        client,
                        request.scope,
                        predicate=VERBATIM_MEMORY_PREDICATE,
                        content=_redacted_text(request.content),
                        metadata=metadata,
                    ),
                )
            else:
                results.extend(
                    await self._add_turn_memories(client, request, metadata),
                )
        return MemoryAddResponse(results=results)

    async def _add_turn_memories(
        self,
        client: MemoryClientContext,
        request: MemoryAddRequest,
        metadata: JsonObject,
    ) -> list[MemoryAddResult]:
        policy = _extraction_policy(
            extract_entities=None,
            extract_relations=None,
            settings=self._app_settings,
        )
        extraction_enabled = self._app_settings.gnosis_fact_extraction_enabled
        extract_inline = extraction_enabled and not self._background_extraction()
        context_turns: list[ConversationTurn] = []
        if extract_inline:
            # Fetched before the verbatim writes so the just-added turns
            # cannot leak into their own extraction context.
            context_turns = await self._recent_session_turns(client, request.scope)
        results: list[MemoryAddResult] = []
        for message in request.messages:
            _ = await client.short_term.add_message(
                session_id=_session_id(request.scope),
                role=message.role,
                content=message.content,
                user_identifier=_user_identifier(request.scope),
                metadata=_scope_metadata(request.scope),
                extract_entities=policy.extract_entities,
                extract_relations=policy.extract_relations,
            )
            results.append(
                await self._add_memory_fact(
                    client,
                    request.scope,
                    predicate=f"{TURN_MEMORY_PREDICATE_PREFIX}{message.role}",
                    content=message.content,
                    metadata=metadata,
                ),
            )
        new_turns = [
            ConversationTurn(speaker=message.role, content=message.content)
            for message in request.messages
        ]
        if extract_inline:
            results.extend(
                await self._extracted_unit_results(
                    client,
                    request.scope,
                    caller_metadata=request.metadata,
                    context_turns=context_turns,
                    new_turns=new_turns,
                    source_memory_ids=[result.memory_id for result in results],
                ),
            )
        elif extraction_enabled:
            # Background mode: the response carries only the verbatim
            # results; extraction runs on the queue after the request.
            self._enqueue_background_extraction(
                request.scope,
                caller_metadata=request.metadata,
                new_turns=new_turns,
                source_memory_ids=[result.memory_id for result in results],
            )
        return results

    async def _add_memory_fact(
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        predicate: str,
        content: str,
        metadata: JsonObject,
    ) -> MemoryAddResult:
        raw_record = await client.long_term.add_fact(
            _user_identifier(scope),
            predicate,
            content,
            metadata=metadata,
            generate_embedding=True,
        )
        memory = stored_memory_from_sdk(raw_record)
        event: MemoryAddEvent = "ADD"
        if memory is not None:
            event = memory_add_event(memory)
        else:
            memory = await _lookup_latest_memory(
                client,
                scope,
                predicate=predicate,
                content=content,
            )
        if memory is None:
            raise BackendCapabilityUnavailable(_MEMORY_ID_UNAVAILABLE_DETAIL)
        return MemoryAddResult(
            memory_id=memory.memory_id,
            content=_redacted_text(memory.content),
            event=event,
            metadata=public_memory_metadata(memory),
        )

    async def _recent_session_turns(
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        exclude_memory_ids: frozenset[str] = _NO_EXCLUDED_MEMORY_IDS,
    ) -> list[ConversationTurn]:
        """Read the extraction context window: recent turns of this session.

        Returns the last ``GNOSIS_FACT_EXTRACTION_CONTEXT_TURNS`` verbatim
        ``said_*`` facts for the scope's session in chronological order. In
        sync mode the window is read before the verbatim writes; in
        background mode the turns being extracted are already stored when
        the job runs, so ``exclude_memory_ids`` (their fact ids) filters
        them out - a pair is never its own context. Any read failure
        degrades to an empty context window - extraction still runs, it
        just resolves references less well - because extraction may never
        fail an add.
        """
        limit = self._app_settings.gnosis_fact_extraction_context_turns
        if limit <= 0:
            return []
        try:
            rows = await client.query.cypher(
                RECENT_TURN_MEMORIES_CYPHER,
                {
                    "subject": _user_identifier(scope),
                    "predicate_prefix": TURN_MEMORY_PREDICATE_PREFIX,
                    "scope_fragments": session_read_fragments(scope),
                    # Over-fetch by the excluded rows so exclusion never
                    # shrinks the effective window.
                    "limit": limit + len(exclude_memory_ids),
                },
            )
        except (RuntimeError, OSError, Neo4jError) as error:
            _LOGGER.warning(
                "fact extraction context read failed; extracting without context",
                extra={"error_type": type(error).__name__},
            )
            return []
        recent = [
            memory
            for row in rows
            if (memory := stored_memory_from_row(row)) is not None
            and memory.memory_id not in exclude_memory_ids
            and memory_matches_scope(memory, scope)
        ]
        turns = [
            ConversationTurn(
                speaker=memory.predicate.removeprefix(TURN_MEMORY_PREDICATE_PREFIX),
                content=memory.content,
            )
            for memory in recent[:limit]
        ]
        turns.reverse()
        return turns

    def _background_extraction(self) -> bool:
        return self._app_settings.gnosis_fact_extraction_mode == "background"

    def _enqueue_background_extraction(
        self,
        scope: MemoryScope,
        *,
        caller_metadata: JsonObject,
        new_turns: list[ConversationTurn],
        source_memory_ids: list[str],
    ) -> None:
        """Queue extraction for turns whose verbatim facts already landed.

        The job opens its own memory client and re-reads the context window
        at processing time, excluding the just-written verbatim facts. On
        queue overflow the extraction is dropped with a structured warning -
        never backpressure for message ingestion.
        """

        async def job() -> None:
            async with self._memory_client() as client:
                context_turns = await self._recent_session_turns(
                    client,
                    scope,
                    exclude_memory_ids=frozenset(source_memory_ids),
                )
                _ = await self._extracted_unit_results(
                    client,
                    scope,
                    caller_metadata=caller_metadata,
                    context_turns=context_turns,
                    new_turns=new_turns,
                    source_memory_ids=source_memory_ids,
                )

        _ = self._extraction_queue.submit(
            job,
            source_memory_ids=source_memory_ids,
        )

    async def _extracted_unit_results(  # noqa: PLR0913 - One argument per prompt input.
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        caller_metadata: JsonObject,
        context_turns: list[ConversationTurn],
        new_turns: list[ConversationTurn],
        source_memory_ids: list[str],
    ) -> list[MemoryAddResult]:
        """Extract memory units from the new turns and write them as facts.

        Strictly additive: extraction and per-unit write failures log a
        structured warning and leave the verbatim add untouched.
        """
        units = await extract_memory_units(
            self._fact_extractor,
            conversation_date=_conversation_date(caller_metadata),
            context_turns=context_turns,
            new_turns=new_turns,
        )
        results: list[MemoryAddResult] = []
        for unit in units:
            try:
                results.append(
                    await self._add_extracted_fact(
                        client,
                        scope,
                        unit=unit,
                        caller_metadata=caller_metadata,
                        source_memory_ids=source_memory_ids,
                    ),
                )
            except (
                RuntimeError,
                OSError,
                Neo4jError,
                OpenAIError,
                ValidationError,
                BackendCapabilityUnavailable,
            ) as error:
                _LOGGER.warning(
                    "extracted fact write failed; keeping verbatim add",
                    extra={"error_type": type(error).__name__},
                )
        return results

    async def _add_extracted_fact(
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        unit: MemoryUnit,
        caller_metadata: JsonObject,
        source_memory_ids: list[str],
    ) -> MemoryAddResult:
        """Write one extracted unit as an ordinary long-term ``Fact`` node.

        The write is a direct parameterized CREATE so it bypasses the SDK's
        write-time dedup, which can silently swallow a near-duplicate add
        into an existing fact - unacceptable for distinct dated events.
        """
        extraction_metadata: JsonObject = {
            "extracted": True,
            "extraction_version": EXTRACTION_VERSION,
            "extraction_model": _fact_extraction_model(self._app_settings),
            "entities": list(unit.entities),
            "source_turn_ids": list(unit.source_turn_ids),
        }
        if unit.event_date is not None:
            extraction_metadata["event_date"] = unit.event_date
        metadata = _write_metadata(scope, caller_metadata | extraction_metadata, None)
        # Provenance ids are gateway-generated fact UUIDs, added after
        # redaction because the opaque-value secret pattern matches UUIDs.
        metadata["source_memory_ids"] = list(source_memory_ids)
        memory_id = str(uuid4())
        content = _redacted_text(unit.text)
        embedding = await _memory_embedding(client, content)
        _ = await _graph_write_query(client).execute_write(
            CREATE_MEMORY_CYPHER,
            {
                "memory_id": memory_id,
                "subject": _user_identifier(scope),
                "predicate": EXTRACTED_FACT_PREDICATE,
                "object": content,
                "embedding": embedding,
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "metadata": json.dumps(metadata),
            },
        )
        await self._materialize_entity_graph(client, scope, memory_id, unit)
        stored = StoredMemory(
            memory_id=memory_id,
            subject=_user_identifier(scope),
            predicate=EXTRACTED_FACT_PREDICATE,
            content=content,
            metadata=metadata,
            created_at=None,
            updated_at=None,
        )
        return MemoryAddResult(
            memory_id=memory_id,
            content=content,
            event="ADD",
            metadata=public_memory_metadata(stored),
        )

    async def _materialize_entity_graph(
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        fact_id: str,
        unit: MemoryUnit,
    ) -> None:
        """Materialize the entity graph for a just-written extracted fact.

        Behind GNOSIS_ENTITY_GRAPH_ENABLED: MERGE a scope-keyed ``:Entity`` per
        named entity, link the fact to each with ``:MENTIONS``, and connect the
        entities with directed ``:RELATES`` edges from the unit's extracted
        triples (HippoRAG-2 / Graphiti). A no-op while the flag is off, so the
        extracted-fact write is byte-identical to before. The fact already
        landed, so any failure here degrades to "no graph materialized" with a
        structured warning and never fails the add or drops the fact.
        """
        if not self._app_settings.gnosis_entity_graph_enabled:
            return
        statements = entity_graph_statements(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            fact_id=fact_id,
            entities=unit.entities,
            relations=[
                RelationTriple(
                    head=relation.head,
                    relation=relation.relation,
                    tail=relation.tail,
                )
                for relation in unit_relations(unit)
            ],
            event_date=unit.event_date,
        )
        if not statements:
            return
        try:
            await self._ensure_entity_graph_schema(client)
            graph_write = _graph_write_query(client)
            for cypher, parameters in statements:
                _ = await graph_write.execute_write(cypher, parameters)
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "entity graph materialization failed; fact kept without graph",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": scope.tenant_id,
                },
            )

    async def _ensure_entity_graph_schema(self, client: MemoryClientContext) -> None:
        """Create the entity scope-key range index if absent.

        Idempotent ``CREATE INDEX ... IF NOT EXISTS`` through the same graph
        write handle the fact writes use, once per backend instance; a failed
        attempt retries on the next entity write.
        """
        if self._entity_graph_schema_ready:
            return
        _ = await _graph_write_query(client).execute_write(
            CREATE_ENTITY_SCOPE_INDEX_CYPHER,
            {},
        )
        self._entity_graph_schema_ready = True

    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        filters = _parsed_memory_filters(request.filters)
        decision = await self._route_decision(request.query)
        async with self._memory_client() as client:
            dense = await self._dense_memory_candidates(
                client,
                request.query,
                scope_read_fragments(request.scope),
            )
            candidates = await self._hybrid_memory_candidates(
                client,
                request.query,
                scope_read_fragments(request.scope),
                dense,
                decision,
            )
        budget = self._search_match_budget(request)
        matches: list[StoredMemory] = []
        for memory in candidates:
            if not memory_matches_scope(memory, request.scope):
                continue
            if not matches_filters(filters, memory_filter_fields(memory)):
                continue
            if not _meets_min_score(memory, request.min_score):
                continue
            matches.append(memory)
            if len(matches) == budget:
                break
        matches = await self._recall_filtered_matches(request, matches)
        matches = self._superseded_matches(matches)
        return MemorySearchResponse(
            results=[
                memory_record(memory, include_score=True)
                for memory in matches[: request.limit]
            ],
        )

    def _superseded_facts(self, facts: list[JsonObject]) -> list[JsonObject]:
        """Drop same-slot older facts from the ranked context candidates."""
        if not self._app_settings.gnosis_read_supersession_enabled:
            return facts
        kept, dropped = drop_superseded(facts, _fact_freshness)
        _log_supersession(dropped, len(facts), surface="context")
        return kept

    def _superseded_matches(
        self,
        matches: list[StoredMemory],
    ) -> list[StoredMemory]:
        """Drop same-slot older facts from the ranked search matches."""
        if not self._app_settings.gnosis_read_supersession_enabled:
            return matches
        kept, dropped = drop_superseded(matches, _memory_freshness)
        _log_supersession(dropped, len(matches), surface="search")
        return kept

    async def _graphqa_fused_facts(
        self,
        request: MemoryContextRequest,
        decision: RouteDecision,
    ) -> list[JsonObject]:
        """Run the planned graph-QA route as a parallel fusion candidate leg.

        Frontier technique (Mnemis dual-route, frontier-2026 T3): the existing
        LLM-planned, validated, scope-safe, read-only graph-QA path runs on
        every context query in parallel with dense long-term retrieval; its
        derived nodes join the long-term candidate set before ranking is cut so
        multi-hop traversal facts survive the item budget. A no-op empty list
        while the effective decision leaves fusion off (byte-identical
        dense-only output) or with no query. The route is bounded by
        GNOSIS_GRAPHQA_FUSION_TIMEOUT_SECONDS and any planner/execution failure
        (LLM, Neo4j, validation rejection, timeout) degrades to dense-only with
        a structured warning - the context request never fails on the graph leg.

        Distinct from the per-request ``include_graph`` flag, which renders a
        separate graph section rather than fusing into the ranked facts.
        """
        if not decision.graphqa_fusion or not request.query:
            return []
        try:
            graph = await asyncio.wait_for(
                self._graph_store.get_context(
                    GraphContextRequest(
                        scope=request.scope,
                        query=request.query,
                        limit=request.graph_limit,
                    ),
                ),
                timeout=self._app_settings.gnosis_graphqa_fusion_timeout_seconds,
            )
        except (
            TimeoutError,
            OpenAIError,
            Neo4jError,
            OSError,
            RuntimeError,
            ValidationError,
        ) as error:
            _LOGGER.warning(
                "graph-QA fusion route failed; assembling dense-only context",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": request.scope.tenant_id,
                    "guild_id": request.scope.guild_id,
                    "channel_id": request.scope.channel_id,
                },
            )
            return []
        return _graph_facts_to_candidates(graph.facts)

    async def _traversal_facts(
        self,
        client: MemoryClientContext,
        request: MemoryContextRequest,
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
    ) -> list[JsonObject]:
        """Entity-anchored graph traversal candidates for context fusion (T1).

        Pins the query's entity mentions as ``:Entity`` seed nodes, expands
        1-2 ``RELATES`` hops to reach bridge entities the query never names,
        and follows edge provenance back to the dated extracted facts - all
        in one fixed parameterized Cypher read, zero extra LLM calls. The
        provenance facts join the candidate pool tagged as graph-derived so
        they hold the reserved graph slots of the item budget. A no-op empty
        list while the effective decision leaves traversal off or the query
        pins nothing; any read failure degrades to dense-only with a
        structured warning - the context request never fails on this leg.
        """
        if not decision.graph_traversal or not request.query:
            return []
        seeds = query_seed_candidates(request.query)
        if not seeds:
            return []
        try:
            rows = await client.query.cypher(
                ENTITY_TRAVERSAL_CYPHER,
                traversal_parameters(
                    tenant_id=request.scope.tenant_id,
                    user_id=request.scope.user_id,
                    seeds=seeds,
                    scope_fragments=_metadata_fragments(scope_metadata),
                    limit=request.graph_limit,
                ),
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "entity traversal failed; assembling dense-only context",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": request.scope.tenant_id,
                },
            )
            return []
        candidates: list[JsonObject] = []
        for row in rows:
            memory = stored_memory_from_row(row)
            if memory is None:
                continue
            fact = _fact_from_memory(memory)
            if not _fact_matches_scope(fact, scope_metadata):
                continue
            fact["graphqa"] = True
            candidates.append(fact)
        if candidates:
            _LOGGER.info(
                "entity traversal fused provenance facts",
                extra={"count": len(candidates), "seeds": len(seeds)},
            )
        return candidates

    async def _bridge_traversal_facts(
        self,
        client: MemoryClientContext,
        request: MemoryContextRequest,
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
        dense_facts: list[JsonObject],
    ) -> list[JsonObject]:
        """Directed bridge-hop candidates for context fusion (T1-directed).

        One LLM call reads the query plus hop-1's dense evidence and names
        the bridge entities the question needs but never names; a fixed
        Cypher then fetches the dated extracted facts MENTIONing those
        entities - hop 2's evidence, unreachable by any ranking of the query
        text. The bridge facts join the pool tagged graph-derived so they
        hold the reserved graph slots. Entities the query itself names are
        dropped from the bridge list (they are hop 1, already dense-covered).
        A no-op empty list while the effective decision leaves the hop off,
        the query or evidence is empty, or the namer finds no bridge; any
        namer or read failure degrades to dense-only with a structured
        warning - the context request never fails on this leg.
        """
        if not decision.bridge_traversal or not request.query or not dense_facts:
            return []
        evidence = [_fact_context_line(fact) for fact in dense_facts]
        try:
            reply = await self._bridge_namer.name_bridges(request.query, evidence)
        except (RuntimeError, OSError, OpenAIError) as error:
            _LOGGER.warning(
                "bridge namer failed; assembling dense-only context",
                extra={"error_type": type(error).__name__},
            )
            return []
        query_mentions = set(query_seed_candidates(request.query))
        bridges = [
            normalized
            for name in parse_bridge_names(reply)
            if (normalized := normalize_entity_name(name))
            and normalized not in query_mentions
        ]
        if not bridges:
            return []
        try:
            rows = await client.query.cypher(
                BRIDGE_MENTION_CYPHER,
                bridge_parameters(
                    tenant_id=request.scope.tenant_id,
                    user_id=request.scope.user_id,
                    bridges=bridges,
                    scope_fragments=_metadata_fragments(scope_metadata),
                    limit=request.graph_limit,
                ),
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "bridge traversal read failed; assembling dense-only context",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": request.scope.tenant_id,
                },
            )
            return []
        candidates: list[JsonObject] = []
        for row in rows:
            memory = stored_memory_from_row(row)
            if memory is None:
                continue
            fact = _fact_from_memory(memory)
            if not _fact_matches_scope(fact, scope_metadata):
                continue
            fact["graphqa"] = True
            candidates.append(fact)
        if candidates:
            _LOGGER.info(
                "bridge traversal fused mention facts",
                extra={"count": len(candidates), "bridges": len(bridges)},
            )
        return candidates

    async def _query_ranked_facts(
        self,
        client: MemoryClientContext,
        query: str,
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
    ) -> list[JsonObject]:
        """Long-term fact candidates ranked by embedding similarity.

        Reuses the /v1/memories/search candidate path (SDK ``search_facts``
        plus the same hybrid lexical fusion when the effective decision has
        hybrid retrieval on) so context and search rank identical
        stored data the same way, then re-checks scope on the deserialized
        records. Nothing ranked (no query, no embedder, no lexical hits)
        tells the caller to fall back to recency ordering.
        """
        if not query:
            return []
        dense = await self._dense_memory_candidates(
            client,
            query,
            _metadata_fragments(scope_metadata),
        )
        candidates = await self._hybrid_memory_candidates(
            client,
            query,
            _metadata_fragments(scope_metadata),
            dense,
            decision,
        )
        return [
            fact
            for memory in candidates
            if _fact_matches_scope(fact := _fact_from_memory(memory), scope_metadata)
        ]

    async def _dense_memory_candidates(
        self,
        client: MemoryClientContext,
        query: str,
        scope_fragments: list[JsonValue],
    ) -> list[StoredMemory]:
        """Embedding-similarity candidates for one scope, best score first.

        With GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED off (default) this is the
        SDK's global search_facts ranking, byte-identical to the historical
        contract. On, the candidates come from a scope-narrowed vector query
        instead, so other scopes' facts can never crowd this scope out of the
        candidate pool (multi-user stores with near-duplicate content). Any
        failure on the scoped path - no embedder, embedding call failure, or
        the vector query itself - degrades to the SDK ranking with a warning
        rather than failing the read.
        """
        if not self._app_settings.gnosis_scoped_dense_retrieval_enabled:
            return await self._sdk_dense_candidates(client, query)
        try:
            embedding = _required_query_embedding(
                await _memory_embedding(client, query),
            )
            rows = await client.query.cypher(
                SCOPED_DENSE_MEMORY_SEARCH_CYPHER,
                {
                    "embedding": embedding,
                    "vector_pool": self._app_settings.gnosis_dense_scope_pool,
                    "scope_fragments": scope_fragments,
                    "candidate_limit": _MEMORY_SEARCH_CANDIDATE_LIMIT,
                },
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "scoped dense search failed; degrading to global dense retrieval",
                extra={"error_type": type(error).__name__},
            )
            return await self._sdk_dense_candidates(client, query)
        return [
            memory
            for row in rows
            if (memory := scored_stored_memory_from_row(row)) is not None
        ]

    @staticmethod
    async def _sdk_dense_candidates(
        client: MemoryClientContext,
        query: str,
    ) -> list[StoredMemory]:
        raw_records = await client.long_term.search_facts(
            query,
            limit=_MEMORY_SEARCH_CANDIDATE_LIMIT,
        )
        return stored_memories_from_sdk(raw_records)

    async def _hybrid_memory_candidates(
        self,
        client: MemoryClientContext,
        query: str,
        scope_fragments: list[JsonValue],
        dense: list[StoredMemory],
        decision: RouteDecision,
    ) -> list[StoredMemory]:
        """Fuse the dense ranking with BM25 lexical candidates via RRF.

        A no-op passthrough of the dense ranking while the effective decision
        leaves hybrid retrieval off (safe default) and whenever the
        lexical leg has nothing to add, so the dense-only behavior is
        byte-identical with hybrid off.
        """
        if not decision.hybrid_retrieval:
            return dense
        lexical = await self._lexical_memory_candidates(
            client,
            query,
            scope_fragments,
        )
        if not lexical:
            return dense
        return fuse_memory_rankings(dense, lexical)

    async def _lexical_memory_candidates(
        self,
        client: MemoryClientContext,
        query: str,
        scope_fragments: list[JsonValue],
    ) -> list[StoredMemory]:
        """BM25 full-text candidates over Fact content, best score first.

        The Lucene query string is sanitized so user input can never inject
        Lucene operators, and any full-text failure (index bootstrap or
        query) degrades to an empty lexical leg with a structured warning -
        the read never fails because of the lexical path.
        """
        lucene_query = sanitize_lucene_query(query)
        if not lucene_query:
            return []
        try:
            await self._ensure_fulltext_index(client)
            rows = await client.query.cypher(
                LEXICAL_MEMORY_SEARCH_CYPHER,
                {
                    "query": lucene_query,
                    "scope_fragments": scope_fragments,
                    "candidate_limit": _MEMORY_SEARCH_CANDIDATE_LIMIT,
                },
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "lexical memory search failed; degrading to dense-only retrieval",
                extra={"error_type": type(error).__name__},
            )
            return []
        return [
            lexical_stored_memory(memory)
            for row in rows
            if (memory := stored_memory_from_row(row)) is not None
        ]

    async def _ensure_fulltext_index(self, client: MemoryClientContext) -> None:
        """Create the gateway-owned Fact full-text index if absent.

        The SDK owns the rest of the Fact schema, so this idempotent
        ``CREATE FULLTEXT INDEX ... IF NOT EXISTS`` goes through the same
        graph write handle the direct Fact writes use, once per backend
        instance; a failed attempt retries on the next lexical read.
        """
        if self._fulltext_index_ready:
            return
        _ = await _graph_write_query(client).execute_write(
            CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER,
            {},
        )
        self._fulltext_index_ready = True

    async def filter_recalled_memories(
        self,
        query: str,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Screen already-scoped memory records with the recall filter.

        The federated search route uses this over the merged local+remote
        result set; the filter only removes or keeps records, so the
        shareable-only scope of remote results is untouched. A no-op while
        GNOSIS_RECALL_FILTER_ENABLED is off.
        """
        if not self._app_settings.gnosis_recall_filter_enabled or not query:
            return list(records)
        return await keep_relevant_candidates(
            self._recall_filter,
            query=query,
            items=records,
            render=_memory_record_line,
            max_candidates=self._app_settings.gnosis_recall_filter_candidates,
        )

    async def _recall_filtered_facts(
        self,
        query: str,
        facts: list[JsonObject],
    ) -> list[JsonObject]:
        """Screen ranked long-term fact candidates for context assembly."""
        if not self._app_settings.gnosis_recall_filter_enabled or not query:
            return facts
        return await keep_relevant_candidates(
            self._recall_filter,
            query=query,
            items=facts,
            render=_fact_context_line,
            max_candidates=self._app_settings.gnosis_recall_filter_candidates,
        )

    def _search_recall_filter_active(self, request: MemorySearchRequest) -> bool:
        """Whether this search call runs the recall filter in the backend.

        Federated searches (``peers`` named) are filtered once over the
        merged local+remote result set by the route instead, keeping the
        budget at one LLM call per request.
        """
        return self._app_settings.gnosis_recall_filter_enabled and not request.peers

    def _search_match_budget(self, request: MemorySearchRequest) -> int:
        if not self._search_recall_filter_active(request):
            return request.limit
        return max(
            request.limit,
            self._app_settings.gnosis_recall_filter_candidates,
        )

    async def _recall_filtered_matches(
        self,
        request: MemorySearchRequest,
        matches: list[StoredMemory],
    ) -> list[StoredMemory]:
        if not self._search_recall_filter_active(request):
            return matches
        return await keep_relevant_candidates(
            self._recall_filter,
            query=request.query,
            items=matches,
            render=_stored_memory_line,
            max_candidates=self._app_settings.gnosis_recall_filter_candidates,
        )

    async def list_memories(self, request: MemoryListRequest) -> MemoryListResponse:
        filters = _parsed_memory_filters(request.filters)
        narrowing = build_cypher_filter(filters)
        parameters: dict[str, JsonValue] = {
            "scope_fragments": scope_read_fragments(request.scope),
            "scan_limit": _MEMORY_LIST_SCAN_LIMIT,
        }
        parameters.update(narrowing.parameters)
        async with self._memory_client() as client:
            rows = await client.query.cypher(
                list_memories_cypher(narrowing.fragment),
                parameters,
            )
        memories = [
            memory
            for row in rows
            if (memory := stored_memory_from_row(row)) is not None
            and memory_matches_scope(memory, request.scope)
            and matches_filters(filters, memory_filter_fields(memory))
        ]
        start = (request.page - 1) * request.page_size
        return MemoryListResponse(
            results=[
                memory_record(memory, include_score=False)
                for memory in memories[start : start + request.page_size]
            ],
            total=len(memories),
            page=request.page,
            page_size=request.page_size,
        )

    async def update_memory(
        self,
        memory_id: str,
        request: MemoryUpdateRequest,
    ) -> MemoryUpdateResponse:
        if request.content is None and request.metadata is None:
            raise BackendRequestError(_MEMORY_UPDATE_FIELDS_DETAIL)
        async with self._memory_client() as client:
            memory = await _require_scoped_memory(client, memory_id, request.scope)
            content = None
            embedding = None
            if request.content is not None:
                content = _redacted_text(request.content)
                embedding = await _memory_embedding(client, content)
            metadata = None
            if request.metadata is not None:
                metadata = merged_memory_metadata(memory, request.metadata)
            parameters: dict[str, JsonValue] = {
                "memory_id": memory.memory_id,
                "content": content,
                "metadata": metadata,
                "embedding": embedding,
            }
            rows = await _graph_write_query(client).execute_write(
                UPDATE_MEMORY_CYPHER,
                parameters,
            )
        _LOGGER.info(
            "memory update applied",
            extra=_memory_edit_audit(memory.memory_id, request.scope),
        )
        return MemoryUpdateResponse(
            memory_id=memory.memory_id,
            content=_updated_memory_content(rows, content or memory.content),
        )

    async def delete_memory(
        self,
        memory_id: str,
        request: MemoryDeleteRequest,
    ) -> MemoryDeleteResponse:
        async with self._memory_client() as client:
            memory = await _require_scoped_memory(client, memory_id, request.scope)
            _ = await _graph_write_query(client).execute_write(
                DELETE_MEMORY_CYPHER,
                {"memory_id": memory.memory_id},
            )
        _LOGGER.info(
            "memory delete applied",
            extra=_memory_edit_audit(memory.memory_id, request.scope),
        )
        return MemoryDeleteResponse(memory_id=memory.memory_id)

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        result = await self._graph_store.ingest_event(event)
        await self._event_fact_promoter.promote_for_result(
            event,
            result,
            self._memory_client(),
        )
        return result

    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse:
        results = [await self.ingest_event(event) for event in request.events]
        return ClientEventBatchResponse(results=results)

    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse:
        return await self._graph_store.get_context(request)

    async def get_sdk_stats(self, request: SdkStatsRequest) -> SdkStatsResponse:
        async with self._memory_client() as client:
            if not isinstance(client, StatsCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_STATS_UNAVAILABLE_DETAIL)
            stats = await client.get_stats()
        return SdkStatsResponse(
            scope=request.scope,
            stats=_redacted_object(_json_object(stats)),
        )

    async def get_dedup_stats(
        self,
        request: DedupStatsRequest,
    ) -> DedupStatsResponse:
        async with self._memory_client() as client:
            if not isinstance(client.long_term, DedupCapableLongTermMemory):
                raise BackendCapabilityUnavailable(_DEDUP_UNAVAILABLE_DETAIL)
            stats = await client.long_term.get_deduplication_stats()
        return DedupStatsResponse(
            scope=request.scope,
            stats=_redacted_object(_json_object(_dedup_stats_payload(stats))),
        )

    async def find_dedup_candidates(
        self,
        request: DedupCandidateRequest,
    ) -> DedupCandidateResponse:
        async with self._memory_client() as client:
            if not isinstance(client.long_term, DedupCapableLongTermMemory):
                raise BackendCapabilityUnavailable(_DEDUP_UNAVAILABLE_DETAIL)
            raw_candidates = await client.long_term.find_potential_duplicates(
                limit=request.limit,
            )
        candidates = [
            _dedup_candidate(source, target, similarity)
            for source, target, similarity in raw_candidates
        ]
        snapshot_hash = _dedup_snapshot_hash(request.scope, candidates)
        expires_at = datetime.now(UTC) + _CONSOLIDATION_TOKEN_TTL
        for candidate in candidates:
            self._dedup_candidates[candidate.candidate_id] = DedupCandidateState(
                candidate_id=candidate.candidate_id,
                version=candidate.version,
                scope=request.scope,
                source_id=UUID(candidate.source.id),
                target_id=UUID(candidate.target.id),
                graph_snapshot_hash=snapshot_hash,
            )
        return DedupCandidateResponse(
            scope=request.scope,
            candidates=[
                candidate.model_copy(
                    update={
                        "reject_dry_run_token": _dedup_token(
                            self._app_settings,
                            DedupTokenClaims(
                                scope=request.scope,
                                candidate_id=candidate.candidate_id,
                                candidate_version=candidate.version,
                                graph_snapshot_hash=snapshot_hash,
                                operation="reject",
                                expires_at=expires_at,
                            ),
                        ),
                        "merge_dry_run_token": _dedup_token(
                            self._app_settings,
                            DedupTokenClaims(
                                scope=request.scope,
                                candidate_id=candidate.candidate_id,
                                candidate_version=candidate.version,
                                graph_snapshot_hash=snapshot_hash,
                                operation="merge",
                                expires_at=expires_at,
                            ),
                        ),
                    },
                )
                for candidate in candidates
            ],
            graph_snapshot_hash=snapshot_hash,
            expires_at=expires_at.isoformat(),
        )

    async def apply_dedup_candidate(
        self,
        request: DedupApplyRequest,
    ) -> DedupApplyResponse:
        if not request.apply:
            raise BackendRequestError(_DEDUP_APPLY_REQUIRED_DETAIL)
        fingerprint = request.model_dump_json(exclude={"dry_run_token"})
        if (record := self._dedup_idempotency.get(request.idempotency_key)) is not None:
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise BackendRequestError(_DEDUP_IDEMPOTENCY_DETAIL)
            return record.response
        state = _require_current_dedup_candidate(
            request,
            self._dedup_candidates.get(request.candidate_id),
        )
        _require_dedup_token(self._app_settings, request)
        async with self._memory_client() as client:
            if not isinstance(client.long_term, DedupCapableLongTermMemory):
                raise BackendCapabilityUnavailable(_DEDUP_UNAVAILABLE_DETAIL)
            result = await _apply_dedup_operation(client.long_term, request, state)
        response = DedupApplyResponse(
            scope=request.scope,
            operation=request.operation,
            candidate_id=request.candidate_id,
            candidate_version=request.candidate_version,
            applied=True,
            result=_redacted_object(result),
            audit=request.audit,
        )
        self._dedup_idempotency[request.idempotency_key] = DedupIdempotencyRecord(
            request_fingerprint=fingerprint,
            response=response,
        )
        return response

    async def dry_run_consolidation(
        self,
        request: ConsolidationDryRunRequest,
    ) -> ConsolidationDryRunResponse:
        async with self._memory_client() as client:
            if not isinstance(client, ConsolidationCapableMemoryClient):
                raise BackendCapabilityUnavailable(_CONSOLIDATION_UNAVAILABLE_DETAIL)
            report = await _run_consolidation_operation(
                client.consolidation,
                request,
                dry_run=True,
            )
        report_payload = _safe_consolidation_report(report)
        fingerprint = _consolidation_request_fingerprint(request)
        graph_snapshot_hash = _hash_json(
            {
                "scope": _json_object(request.scope.model_dump(mode="json")),
                "operation": request.operation,
                "request_fingerprint": fingerprint,
                "report": report_payload,
            },
        )
        expires_at = datetime.now(UTC) + _DEDUP_TOKEN_TTL
        self._consolidation_dry_runs[graph_snapshot_hash] = ConsolidationDryRunState(
            scope=request.scope,
            operation=request.operation,
            graph_snapshot_hash=graph_snapshot_hash,
            request_fingerprint=fingerprint,
        )
        return ConsolidationDryRunResponse(
            scope=request.scope,
            operation=request.operation,
            dry_run=True,
            report=report_payload,
            graph_snapshot_hash=graph_snapshot_hash,
            dry_run_token=_consolidation_token(
                self._app_settings,
                ConsolidationTokenClaims(
                    scope=request.scope,
                    operation=request.operation,
                    graph_snapshot_hash=graph_snapshot_hash,
                    request_fingerprint=fingerprint,
                    expires_at=expires_at,
                ),
            ),
            expires_at=expires_at.isoformat(),
        )

    async def apply_consolidation(
        self,
        request: ConsolidationApplyRequest,
    ) -> ConsolidationApplyResponse:
        if not request.apply:
            raise BackendRequestError(_CONSOLIDATION_APPLY_REQUIRED_DETAIL)
        fingerprint = _consolidation_apply_fingerprint(request)
        if (
            record := self._consolidation_idempotency.get(request.idempotency_key)
        ) is not None:
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise BackendRequestError(_CONSOLIDATION_IDEMPOTENCY_DETAIL)
            return record.response
        state = _require_current_consolidation_dry_run(
            request,
            self._consolidation_dry_runs.get(request.graph_snapshot_hash),
        )
        _require_consolidation_token(self._app_settings, request, state)
        async with self._memory_client() as client:
            if not isinstance(client, ConsolidationCapableMemoryClient):
                raise BackendCapabilityUnavailable(_CONSOLIDATION_UNAVAILABLE_DETAIL)
            result = await _run_consolidation_operation(
                client.consolidation,
                request,
                dry_run=False,
            )
        response = ConsolidationApplyResponse(
            scope=request.scope,
            operation=request.operation,
            applied=True,
            result=_safe_consolidation_report(result),
            audit=request.audit,
        )
        self._consolidation_idempotency[request.idempotency_key] = (
            ConsolidationIdempotencyRecord(
                request_fingerprint=fingerprint,
                response=response,
            )
        )
        return response

    async def export_graph(self, request: GraphExportRequest) -> GraphExportResponse:
        async with self._memory_client() as client:
            if not isinstance(client, GraphCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_GRAPH_UNAVAILABLE_DETAIL)
            graph = await client.get_graph(
                memory_types=request.memory_types,
                session_id=request.scope.session_id,
                include_embeddings=False,
                limit=request.limit,
            )
        return _graph_export_response(request, graph)

    async def search_entities(
        self,
        request: EntitySearchRequest,
    ) -> EntitySearchResponse:
        filters = _scoped_filters(request.scope, request.metadata)
        async with self._memory_client() as client:
            raw_records = await client.long_term.search_entities(
                request.query,
                limit=request.limit,
            )
        records = _ENTITY_RECORDS_ADAPTER.validate_python(raw_records)
        return EntitySearchResponse(
            entities=[
                _redacted_entity(record)
                for record in records
                if _record_matches_filters(record.metadata, filters)
            ],
        )

    async def search_facts(self, request: FactSearchRequest) -> FactSearchResponse:
        filters = _scoped_filters(request.scope, request.metadata)
        async with self._memory_client() as client:
            raw_records = await client.long_term.search_facts(
                request.query,
                limit=request.limit,
            )
        records = _FACT_RECORDS_ADAPTER.validate_python(raw_records)
        return FactSearchResponse(
            facts=[
                _redacted_fact(record)
                for record in records
                if _record_matches_filters(record.metadata, filters)
            ],
        )

    async def search_preferences(
        self,
        request: PreferenceSearchRequest,
    ) -> PreferenceSearchResponse:
        filters = _scoped_filters(request.scope, request.metadata)
        async with self._memory_client() as client:
            raw_records = await client.long_term.search_preferences(
                request.query,
                category=request.category,
                limit=request.limit,
            )
        records = _PREFERENCE_RECORDS_ADAPTER.validate_python(raw_records)
        return PreferenceSearchResponse(
            preferences=[
                _redacted_preference(record)
                for record in records
                if _record_matches_filters(record.metadata, filters)
            ],
        )

    async def add_entity(self, request: EntityWriteRequest) -> EntityRecord:
        async with self._memory_client() as client:
            raw_record = await client.long_term.add_entity(
                request.name,
                request.type,
                subtype=request.subtype,
                description=_redacted_optional_text(request.description),
                aliases=request.aliases,
                attributes=_redacted_object(request.attributes),
                resolve=request.resolve,
                generate_embedding=request.generate_embedding,
                deduplicate=request.deduplicate,
                geocode=False,
                enrich=False,
                metadata=_write_metadata(
                    request.scope,
                    request.metadata,
                    request.provenance,
                ),
            )
        record = _ENTITY_RECORD_ADAPTER.validate_python(raw_record)
        return _redacted_entity(record)

    async def add_fact(self, request: FactWriteRequest) -> FactRecord:
        async with self._memory_client() as client:
            raw_record = await client.long_term.add_fact(
                _redacted_text(request.subject),
                _redacted_text(request.predicate),
                _redacted_text(request.object),
                confidence=request.confidence,
                generate_embedding=request.generate_embedding,
                metadata=_write_metadata(
                    request.scope,
                    request.metadata,
                    request.provenance,
                ),
            )
        record = _FACT_RECORD_ADAPTER.validate_python(raw_record)
        return _redacted_fact(record)

    async def add_preference(self, request: PreferenceWriteRequest) -> PreferenceRecord:
        async with self._memory_client() as client:
            raw_record = await client.long_term.add_preference(
                request.category,
                _redacted_text(request.preference),
                context=_redacted_optional_text(request.context),
                confidence=request.confidence,
                generate_embedding=request.generate_embedding,
                metadata=_write_metadata(
                    request.scope,
                    request.metadata,
                    request.provenance,
                ),
                user_identifier=(
                    request.user_identifier or _user_identifier(request.scope)
                ),
            )
        record = _PREFERENCE_RECORD_ADAPTER.validate_python(raw_record)
        return _redacted_preference(record)

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        await self._graph_store.require_available()
        return await self._skill_registry.list_skills(request)

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        await self._graph_store.require_available()
        return await self._skill_registry.propose_skill(proposal)

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        await self._graph_store.require_available()
        return await self._skill_registry.record_skill_usage(usage)

    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse:
        async with self._memory_client() as client:
            trace = await client.reasoning.start_trace(
                request.session_id,
                _redacted_text(request.task),
                generate_embedding=True,
                metadata=_reasoning_write_metadata(request.scope, request.metadata),
                triggered_by_message_id=request.triggered_by_message_id,
                user_identifier=(
                    request.user_identifier or _user_identifier(request.scope)
                ),
            )
        return ReasoningTraceStartResponse(
            trace_id=str(trace.id),
            session_id=trace.session_id,
            task=trace.task,
        )

    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse:
        async with self._memory_client() as client:
            step = await client.reasoning.add_step(
                UUID(request.trace_id),
                thought=None,
                action=_redacted_optional_text(request.action),
                observation=_redacted_optional_text(request.observation),
                generate_embedding=True,
                metadata=_reasoning_write_metadata(request.scope, request.metadata),
            )
        return ReasoningStepResponse(
            step_id=str(step.id),
            trace_id=str(step.trace_id),
            step_number=step.step_number,
        )

    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse:
        async with self._memory_client() as client:
            tool_call = await client.reasoning.record_tool_call(
                UUID(request.step_id),
                request.tool_name,
                _redacted_object(request.arguments),
                result=redact_secrets(request.result),
                status=ToolCallStatus(request.status),
                duration_ms=request.duration_ms,
                error=_redacted_optional_text(request.error),
                message_id=request.message_id,
                touched_entities=[
                    EntityRef(id=entity.id, name=entity.name, type=entity.type)
                    for entity in request.touched_entities
                ],
            )
        return ReasoningToolCallResponse(
            tool_call_id=str(tool_call.id),
            trace_id=request.trace_id,
            step_id=request.step_id,
        )

    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse:
        async with self._memory_client() as client:
            trace = await client.reasoning.complete_trace(
                UUID(request.trace_id),
                outcome=_redacted_optional_text(request.outcome),
                success=request.success,
                generate_step_embeddings=False,
            )
        completed_at = None
        if trace.completed_at is not None:
            completed_at = trace.completed_at.isoformat()
        return ReasoningTraceCompleteResponse(
            trace_id=str(trace.id),
            success=trace.success,
            outcome=trace.outcome,
            completed_at=completed_at,
        )

    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse:
        async with self._memory_client() as client:
            context = await client.reasoning.get_context(
                request.query,
                max_traces=request.max_items,
            )
        redacted_context = _redacted_text(context)
        if not redacted_context:
            redacted_context = "No similar reasoning traces found."
        return ReasoningContextResponse(context=redacted_context, traces=[])

    async def list_reasoning_traces(
        self,
        request: ReasoningTraceListRequest,
    ) -> ReasoningTraceListResponse:
        async with self._memory_client() as client:
            try:
                traces = await client.reasoning.list_traces(
                    session_id=request.session_id or request.scope.session_id,
                    success_only=request.success_only,
                    limit=request.limit,
                    offset=request.offset,
                )
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        return ReasoningTraceListResponse(
            scope=request.scope,
            traces=_scoped_reasoning_traces(request.scope, traces),
        )

    async def get_reasoning_trace(
        self,
        request: ReasoningTraceDetailRequest,
    ) -> ReasoningTraceDetailResponse:
        async with self._memory_client() as client:
            try:
                trace = await _get_reasoning_trace(client.reasoning, request)
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        if trace is None or not _reasoning_trace_matches_scope(trace, request.scope):
            return ReasoningTraceDetailResponse(scope=request.scope)
        return ReasoningTraceDetailResponse(
            scope=request.scope,
            trace=_reasoning_trace_summary(trace),
            steps=[_reasoning_step_record(step) for step in trace.steps],
        )

    async def find_similar_reasoning_traces(
        self,
        request: ReasoningSimilarTracesRequest,
    ) -> ReasoningSimilarTracesResponse:
        async with self._memory_client() as client:
            try:
                traces = await client.reasoning.get_similar_traces(
                    _redacted_text(request.task),
                    limit=request.limit,
                    success_only=request.success_only,
                    threshold=request.threshold,
                )
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        return ReasoningSimilarTracesResponse(
            scope=request.scope,
            traces=_scoped_reasoning_traces(request.scope, traces),
        )

    async def search_reasoning_steps(
        self,
        request: ReasoningStepSearchRequest,
    ) -> ReasoningStepSearchResponse:
        async with self._memory_client() as client:
            try:
                steps = await client.reasoning.search_steps(
                    _redacted_text(request.query),
                    limit=request.limit,
                    success_only=request.success_only,
                    threshold=request.threshold,
                )
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        return ReasoningStepSearchResponse(
            scope=request.scope,
            steps=[
                _reasoning_step_record(step)
                for step in steps
                if _reasoning_step_matches_scope(step, request.scope)
            ],
        )

    async def get_reasoning_tool_stats(
        self,
        request: ReasoningToolStatsRequest,
    ) -> ReasoningToolStatsResponse:
        _ = request.tool_name
        return ReasoningToolStatsResponse(
            scope=request.scope,
            tools=[],
        )

    def _memory_client(self) -> MemoryClientContext:
        if self._memory_client_factory is not None:
            return self._memory_client_factory(self._settings)
        return _memory_client_context(MemoryClient(self._settings))


def _require_memory_add_mode(request: MemoryAddRequest) -> None:
    has_messages = bool(request.messages)
    has_content = request.content is not None
    if has_messages == has_content:
        raise BackendRequestError(_MEMORY_MODE_DETAIL)
    if has_messages and not request.infer:
        raise BackendRequestError(_MEMORY_MESSAGES_INFER_DETAIL)
    if has_content and request.infer:
        raise BackendRequestError(_MEMORY_CONTENT_INFER_DETAIL)


def _parsed_memory_filters(filters: JsonObject | None) -> MemoryFilter | None:
    if filters is None:
        return None
    try:
        return parse_filters(filters)
    except FilterValidationError as error:
        raise BackendRequestError(error.detail) from error


def _meets_min_score(memory: StoredMemory, min_score: float | None) -> bool:
    if min_score is None:
        return True
    score = memory_score(memory)
    return score is not None and score >= min_score


async def _lookup_latest_memory(
    client: MemoryClientContext,
    scope: MemoryScope,
    *,
    predicate: str,
    content: str,
) -> StoredMemory | None:
    rows = await client.query.cypher(
        LOOKUP_LATEST_MEMORY_CYPHER,
        {
            "subject": _user_identifier(scope),
            "predicate": predicate,
            "object": content,
            "scope_fragments": scope_read_fragments(scope),
        },
    )
    for row in rows:
        memory = stored_memory_from_row(row)
        if memory is not None and memory_matches_scope(memory, scope):
            return memory
    return None


async def _require_scoped_memory(
    client: MemoryClientContext,
    memory_id: str,
    scope: MemoryScope,
) -> StoredMemory:
    rows = await client.query.cypher(
        LOOKUP_MEMORY_CYPHER,
        {"memory_id": memory_id},
    )
    for row in rows:
        memory = stored_memory_from_row(row)
        if memory is not None and memory_matches_scope(memory, scope):
            return memory
    raise MemoryNotFoundError


def _graph_write_query(client: MemoryClientContext) -> GraphWriteQuery:
    """Acquire the graph write handle used by memory update and delete.

    The runtime-protocol ``isinstance`` check alone is too strict for the
    installed SDK: since Python 3.12 it resolves members with
    ``inspect.getattr_static``, and ``neo4j-agent-memory==0.5.0`` returns a
    ``client.graph`` proxy that forwards ``execute_write`` to the same driver
    the read routes use only through dynamic ``__getattr__`` delegation. Fall
    back to a duck-typed check so that proxy stays usable.
    """
    graph: object = getattr(client, "graph", None)
    if isinstance(graph, GraphWriteQuery):
        return graph
    execute_write: object = getattr(graph, "execute_write", None)
    if graph is not None and callable(execute_write):
        return cast("GraphWriteQuery", graph)
    raise BackendCapabilityUnavailable(_MEMORY_WRITE_UNAVAILABLE_DETAIL)


def _required_query_embedding(
    embedding: list[JsonValue] | None,
) -> list[JsonValue]:
    if embedding is None:
        raise BackendCapabilityUnavailable(_QUERY_EMBEDDER_UNAVAILABLE_DETAIL)
    return embedding


async def _memory_embedding(
    client: MemoryClientContext,
    text: str,
) -> list[JsonValue] | None:
    embedder: object = getattr(client.long_term, "embedder", None)
    if not isinstance(embedder, TextEmbedder):
        return None
    return [float(item) for item in await embedder.embed(text)]


def _updated_memory_content(rows: list[JsonObject], fallback: str) -> str:
    for row in rows:
        value = row.get("object")
        if isinstance(value, str) and value:
            return _redacted_text(value)
    return _redacted_text(fallback)


def _memory_edit_audit(memory_id: str, scope: MemoryScope) -> dict[str, str]:
    return {
        "memory_id": memory_id,
        "tenant_id": scope.tenant_id,
        "agent_id": scope.agent_id,
        "user_id": scope.user_id,
    }


def _build_memory_settings(settings: Settings) -> MemorySettings:
    return MemorySettings(
        backend="bolt",
        neo4j=Neo4jConfig(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=SecretStr(settings.neo4j_password),
        ),
        llm=LiteLLMProvider(
            settings.gnosis_llm,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        embedding=LiteLLMEmbeddingProvider(
            litellm_embedding_model(settings.gnosis_embedding),
            dimensions=settings.gnosis_embedding_dimensions,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        memory=_build_memory_config(settings),
    )


def _memory_client_context(client: object) -> MemoryClientContext:
    return cast("MemoryClientContext", client)


def _buffer_readiness_status(write_errors: int) -> Literal["ready", "degraded"]:
    if write_errors == 0:
        return "ready"
    return "degraded"


def _build_memory_config(settings: Settings) -> MemoryConfig:
    config = MemoryConfigKwargs(multi_tenant=True)
    supported_fields = MemoryConfig.model_fields
    if "write_mode" in supported_fields:
        config["write_mode"] = settings.gnosis_write_mode
    if "max_pending" in supported_fields:
        config["max_pending"] = settings.gnosis_max_pending
    if "conversation_ttl_days" in supported_fields:
        config["conversation_ttl_days"] = settings.gnosis_conversation_ttl_days
    if "audit_read" in supported_fields:
        config["audit_read"] = settings.gnosis_audit_read
    if "fact_deduplication_enabled" in supported_fields:
        config["fact_deduplication_enabled"] = (
            settings.gnosis_fact_deduplication_enabled
        )
    if "trace_embedding_enabled" in supported_fields:
        config["trace_embedding_enabled"] = settings.gnosis_trace_embedding_enabled
    return MemoryConfig(**config)


def litellm_embedding_model(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def _message_extraction_policy(
    request: MessageWriteRequest,
    settings: Settings,
) -> ExtractionPolicy:
    if request.preview_extraction:
        raise BackendRequestError(_PREVIEW_WRITE_DETAIL)
    return _extraction_policy(
        extract_entities=request.extract_entities,
        extract_relations=request.extract_relations,
        settings=settings,
    )


def _preview_extraction_policy(
    request: ExtractionPreviewRequest,
    settings: Settings,
) -> ExtractionPolicy:
    return _extraction_policy(
        extract_entities=request.extract_entities,
        extract_relations=request.extract_relations,
        settings=settings,
    )


def _extraction_policy(
    *,
    extract_entities: bool | None,
    extract_relations: bool | None,
    settings: Settings,
) -> ExtractionPolicy:
    if extract_relations is True and extract_entities is not True:
        raise BackendRequestError(_RELATION_ENTITY_DETAIL)
    entities_enabled = (
        extract_entities is True and settings.gnosis_extract_entities_enabled
    )
    relations_enabled = (
        extract_relations is True
        and entities_enabled
        and settings.gnosis_extract_relations_enabled
    )
    return ExtractionPolicy(
        extract_entities=entities_enabled,
        extract_relations=relations_enabled,
    )


def _fact_extraction_model(settings: Settings) -> str:
    return settings.gnosis_fact_extraction_model or settings.gnosis_llm


def _sufficiency_model(settings: Settings) -> str:
    return settings.gnosis_sufficiency_model or settings.gnosis_llm


def _routing_model(settings: Settings) -> str:
    return settings.gnosis_routing_model or settings.gnosis_llm


def _conversation_date(caller_metadata: JsonObject) -> str:
    """Resolve the prompt's conversation date for relative-date resolution.

    Callers that replay historical sessions (membench) supply
    ``metadata.session_date``; live ingestion falls back to the ingest date.
    """
    session_date = caller_metadata.get("session_date")
    if isinstance(session_date, str) and session_date:
        return session_date
    return datetime.now(UTC).date().isoformat()


def _require_ingestion_sources_allowed(
    request: MessageWriteRequest,
    settings: Settings,
) -> None:
    if request.raw_text_documents:
        raise BackendRequestError(_RAW_TEXT_PREVIEW_DETAIL)
    if request.ocr_image_references:
        raise BackendRequestError(_OCR_PREVIEW_ONLY_DETAIL)
    _require_rustfs_references_allowed(request.rustfs_source_references, settings)


def _require_preview_enabled(settings: Settings) -> None:
    if not settings.gnosis_extraction_preview_enabled:
        raise BackendRequestError(_PREVIEW_DISABLED_DETAIL)


def _require_preview_sources_allowed(
    request: ExtractionPreviewRequest,
    settings: Settings,
) -> None:
    if request.ocr_image_references and not settings.gnosis_ocr_enabled:
        raise BackendRequestError(_OCR_DISABLED_DETAIL)
    for image in request.ocr_image_references:
        if image.size_bytes > settings.gnosis_ocr_max_image_bytes:
            raise BackendRequestError(_OCR_SIZE_DETAIL)
        if image.rustfs is not None:
            _require_rustfs_references_allowed([image.rustfs], settings)
    _require_rustfs_references_allowed(request.rustfs_source_references, settings)


def _require_rustfs_references_allowed(
    references: list[RustFSSourceReference],
    settings: Settings,
) -> None:
    if not references:
        return
    if not settings.gnosis_rustfs_enabled:
        raise BackendRequestError(_RUSTFS_DISABLED_DETAIL)
    for reference in references:
        if (
            settings.gnosis_rustfs_bucket
            and reference.bucket != settings.gnosis_rustfs_bucket
        ):
            raise BackendRequestError(_RUSTFS_BUCKET_DETAIL)
        if settings.gnosis_rustfs_prefix and not reference.object_key.startswith(
            settings.gnosis_rustfs_prefix,
        ):
            raise BackendRequestError(_RUSTFS_KEY_DETAIL)


def _preview_document_count(request: ExtractionPreviewRequest) -> int:
    count = len(request.raw_text_documents)
    if request.content is not None:
        count += 1
    return count


def _preview_source_ids(request: ExtractionPreviewRequest) -> list[str]:
    source_ids: list[str] = []
    if request.content is not None:
        source_ids.append("message.content")
    source_ids.extend(document.source_id for document in request.raw_text_documents)
    source_ids.extend(image.source_id for image in request.ocr_image_references)
    source_ids.extend(
        f"rustfs://{reference.bucket}/{reference.object_key}"
        for reference in request.rustfs_source_references
    )
    return source_ids


def _preview_candidates(
    request: ExtractionPreviewRequest,
    settings: Settings,
) -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
    if request.content is not None:
        candidates.append(
            ExtractionCandidate(
                kind="text_chunk",
                text=_preview_text(request.content),
                source_id="message.content",
                confidence=1.0,
            ),
        )
    candidates.extend(
        ExtractionCandidate(
            kind="text_chunk",
            text=_preview_text(document.text),
            source_id=document.source_id,
            confidence=1.0,
        )
        for document in request.raw_text_documents
    )
    if settings.gnosis_ocr_enabled:
        candidates.extend(
            ExtractionCandidate(
                kind="ocr_text",
                text=f"OCR preview placeholder via {settings.gnosis_ocr_model}",
                source_id=image.source_id,
                confidence=0.0,
            )
            for image in request.ocr_image_references
        )
    return candidates


def _preview_text(text: str) -> str:
    return text[:240]


def _session_id(scope: MemoryScope) -> str:
    return scope.session_id


def _user_identifier(scope: MemoryScope) -> str:
    return (
        f"{scope.tenant_id}:{scope.space_id}:{scope.visibility.value}:"
        f"{scope.agent_id}:{scope.user_id}"
    )


def _scope_metadata(scope: MemoryScope) -> dict[str, str]:
    metadata = {
        "tenant_id": scope.tenant_id,
        "space_id": scope.space_id,
        "agent_id": scope.agent_id,
        "session_id": scope.session_id,
        "user_id": scope.user_id,
        "visibility": scope.visibility.value,
    }
    if scope.guild_id is not None:
        metadata["guild_id"] = scope.guild_id
    if scope.channel_id is not None:
        metadata["channel_id"] = scope.channel_id
    return metadata


def _scope_json_metadata(scope: MemoryScope) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(_scope_metadata(scope))


def _scoped_filters(scope: MemoryScope, metadata: JsonObject) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(metadata | _scope_metadata(scope))


def _write_metadata(
    scope: MemoryScope,
    metadata: JsonObject,
    provenance: object | None,
) -> JsonObject:
    base = metadata | _scope_metadata(scope)
    if isinstance(provenance, BaseModel):
        base |= _json_object(provenance.model_dump(exclude_none=True))
    return _redacted_object(_JSON_OBJECT_ADAPTER.validate_python(base))


def _reasoning_write_metadata(scope: MemoryScope, metadata: JsonObject) -> JsonObject:
    return _redacted_object(
        _JSON_OBJECT_ADAPTER.validate_python(metadata | _scope_metadata(scope)),
    )


def _record_matches_filters(metadata: JsonObject, filters: JsonObject) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def _redacted_entity(record: EntityRecord) -> EntityRecord:
    return record.model_copy(
        update={
            "description": _redacted_optional_text(record.description),
            "attributes": _redacted_object(record.attributes),
            "metadata": _redacted_object(record.metadata),
        },
    )


def _redacted_fact(record: FactRecord) -> FactRecord:
    return record.model_copy(
        update={
            "subject": _redacted_text(record.subject),
            "predicate": _redacted_text(record.predicate),
            "object": _redacted_text(record.object),
            "metadata": _redacted_object(record.metadata),
        },
    )


def _redacted_preference(record: PreferenceRecord) -> PreferenceRecord:
    return record.model_copy(
        update={
            "preference": _redacted_text(record.preference),
            "context": _redacted_optional_text(record.context),
            "metadata": _redacted_object(record.metadata),
        },
    )


def _dedup_stats_payload(stats: object) -> JsonObject:
    if isinstance(stats, BaseModel):
        return _json_object(stats.model_dump(mode="json"))
    try:
        return _json_compatible_object(vars(stats))
    except TypeError:
        return _json_compatible_object(stats)


def _dedup_candidate(
    source: object,
    target: object,
    similarity: float,
) -> DedupCandidate:
    source_snapshot = _dedup_entity_snapshot(source)
    target_snapshot = _dedup_entity_snapshot(target)
    fingerprint = _hash_json(
        {
            "source_id": source_snapshot.id,
            "target_id": target_snapshot.id,
            "similarity": similarity,
        },
    )
    return DedupCandidate(
        candidate_id=f"dedup-{fingerprint[:24]}",
        version=1,
        source=source_snapshot,
        target=target_snapshot,
        similarity=similarity,
        reject_dry_run_token=_DEDUP_PENDING_TOKEN,
        merge_dry_run_token=_DEDUP_PENDING_TOKEN,
    )


def _dedup_entity_snapshot(entity: object) -> DedupEntitySnapshot:
    record = _dedup_entity_payload(entity)
    return DedupEntitySnapshot(
        id=_required_text(record, "id"),
        name=_required_text(record, "name"),
        type=_required_text(record, "type"),
        subtype=_optional_text(record, "subtype"),
        description=_redacted_optional_text(_optional_text(record, "description")),
        confidence=_optional_float(record, "confidence", 1.0),
        aliases=_string_list(record.get("aliases")),
        attributes=_redacted_object(_json_member_object(record, "attributes")),
        metadata=_redacted_object(_json_member_object(record, "metadata")),
    )


def _dedup_entity_payload(entity: object) -> JsonObject:
    if isinstance(entity, BaseModel):
        return _json_object(entity.model_dump(mode="json"))
    try:
        return _json_compatible_object(vars(entity))
    except TypeError:
        return _json_compatible_object(entity)


def _dedup_snapshot_hash(
    scope: MemoryScope,
    candidates: list[DedupCandidate],
) -> str:
    return _hash_json(
        {
            "scope": _json_object(scope.model_dump(mode="json")),
            "candidates": [
                candidate.model_dump(
                    mode="json",
                    exclude={"reject_dry_run_token", "merge_dry_run_token"},
                )
                for candidate in candidates
            ],
        },
    )


def _dedup_token(
    settings: Settings,
    claims: DedupTokenClaims,
) -> str:
    payload = _json_object(
        {
            "scope": claims.scope.model_dump(mode="json"),
            "candidate_id": claims.candidate_id,
            "candidate_version": claims.candidate_version,
            "graph_snapshot_hash": claims.graph_snapshot_hash,
            "operation": claims.operation,
            "expires_at": claims.expires_at.isoformat(),
        },
    )
    payload_bytes = _canonical_json(payload).encode()
    encoded_payload = _urlsafe_b64encode(payload_bytes)
    signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded_payload}.{signature}"


def _require_current_dedup_candidate(
    request: DedupApplyRequest,
    state: DedupCandidateState | None,
) -> DedupCandidateState:
    if state is None:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    if state.version != request.candidate_version:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    if state.scope != request.scope:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    if state.graph_snapshot_hash != request.graph_snapshot_hash:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    return state


def _require_dedup_token(settings: Settings, request: DedupApplyRequest) -> None:
    payload = _dedup_token_payload(settings, request.dry_run_token)
    expected = _json_object(
        {
            "scope": request.scope.model_dump(mode="json"),
            "candidate_id": request.candidate_id,
            "candidate_version": request.candidate_version,
            "graph_snapshot_hash": request.graph_snapshot_hash,
            "operation": request.operation,
        },
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    expires_at = _optional_text(payload, "expires_at")
    if expires_at is None:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    try:
        expiry = datetime.fromisoformat(expires_at)
    except (ValueError, binascii.Error) as error:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL) from error
    if expiry <= datetime.now(UTC):
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)


def _dedup_token_payload(settings: Settings, token: str) -> JsonObject:
    parts = token.split(".", maxsplit=1)
    if len(parts) != _DEDUP_TOKEN_PARTS:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    encoded_payload, signature = parts
    try:
        payload_bytes = _urlsafe_b64decode(encoded_payload)
    except ValueError as error:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL) from error
    expected_signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(payload_bytes)
    except ValidationError as error:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL) from error


async def _run_consolidation_operation(
    consolidation: ConsolidationMemory,
    request: ConsolidationDryRunRequest | ConsolidationApplyRequest,
    *,
    dry_run: bool,
) -> object:
    match request.operation:
        case "archive_expired_conversations":
            return await consolidation.archive_expired_conversations(
                ttl_days=request.ttl_days,
                dry_run=dry_run,
            )
        case "dedupe_entities":
            return await consolidation.dedupe_entities(
                similarity_threshold=request.similarity_threshold or 0.95,
                max_pairs=request.max_pairs or 10000,
                dry_run=dry_run,
            )
        case "detect_superseded_preferences":
            return await consolidation.detect_superseded_preferences(
                user_identifier=request.user_identifier,
                similarity_threshold=request.similarity_threshold or 0.92,
                dry_run=dry_run,
            )
        case "summarize_long_traces":
            return await consolidation.summarize_long_traces(
                min_steps=request.min_steps or 20,
                max_traces=request.max_traces or 1000,
                dry_run=dry_run,
            )
    assert_never(request.operation)


def _safe_consolidation_report(report: object) -> JsonObject:
    if isinstance(report, BaseModel):
        payload = _json_object(report.model_dump(mode="json"))
    else:
        try:
            payload = _json_compatible_object(vars(report))
        except TypeError:
            payload = _json_compatible_object(report)
    return _strip_unsafe_consolidation_fields(_redacted_object(payload))


def _strip_unsafe_consolidation_fields(value: JsonObject) -> JsonObject:
    safe: JsonObject = {}
    for key, item in value.items():
        normalized = key.casefold()
        if normalized in _UNSAFE_CONSOLIDATION_REPORT_KEYS:
            continue
        safe[key] = _strip_unsafe_consolidation_value(item)
    return safe


def _strip_unsafe_consolidation_value(value: JsonValue) -> JsonValue:
    match value:
        case dict():
            return _strip_unsafe_consolidation_fields(_json_object(value))
        case list():
            return [_strip_unsafe_consolidation_value(item) for item in value]
        case _:
            return value


def _consolidation_request_fingerprint(
    request: ConsolidationDryRunRequest | ConsolidationApplyRequest,
) -> str:
    return _hash_json(
        _json_object(
            request.model_dump(
                mode="json",
                exclude={
                    "audit",
                    "apply",
                    "dry_run_token",
                    "graph_snapshot_hash",
                    "idempotency_key",
                },
            ),
        ),
    )


def _consolidation_apply_fingerprint(request: ConsolidationApplyRequest) -> str:
    return _hash_json(
        _json_object(
            request.model_dump(mode="json", exclude={"idempotency_key"}),
        ),
    )


def _consolidation_token(settings: Settings, claims: ConsolidationTokenClaims) -> str:
    payload = _json_object(
        {
            "scope": claims.scope.model_dump(mode="json"),
            "operation": claims.operation,
            "graph_snapshot_hash": claims.graph_snapshot_hash,
            "request_fingerprint": claims.request_fingerprint,
            "expires_at": claims.expires_at.isoformat(),
        },
    )
    payload_bytes = _canonical_json(payload).encode()
    encoded_payload = _urlsafe_b64encode(payload_bytes)
    signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded_payload}.{signature}"


def _require_current_consolidation_dry_run(
    request: ConsolidationApplyRequest,
    state: ConsolidationDryRunState | None,
) -> ConsolidationDryRunState:
    if state is None:
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    if state.scope != request.scope:
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    if state.operation != request.operation:
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    if state.request_fingerprint != _consolidation_request_fingerprint(request):
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    return state


def _require_consolidation_token(
    settings: Settings,
    request: ConsolidationApplyRequest,
    state: ConsolidationDryRunState,
) -> None:
    payload = _consolidation_token_payload(settings, request.dry_run_token)
    expected = _json_object(
        {
            "scope": request.scope.model_dump(mode="json"),
            "operation": request.operation,
            "graph_snapshot_hash": request.graph_snapshot_hash,
            "request_fingerprint": state.request_fingerprint,
        },
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    expires_at = _optional_text(payload, "expires_at")
    if expires_at is None:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    try:
        expiry = datetime.fromisoformat(expires_at)
    except (ValueError, binascii.Error) as error:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL) from error
    if expiry <= datetime.now(UTC):
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)


def _consolidation_token_payload(settings: Settings, token: str) -> JsonObject:
    parts = token.split(".", maxsplit=1)
    if len(parts) != _DEDUP_TOKEN_PARTS:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    encoded_payload, signature = parts
    try:
        payload_bytes = _urlsafe_b64decode(encoded_payload)
    except ValueError as error:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL) from error
    expected_signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(payload_bytes)
    except ValidationError as error:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL) from error


async def _apply_dedup_operation(
    long_term: DedupCapableLongTermMemory,
    request: DedupApplyRequest,
    state: DedupCandidateState,
) -> JsonObject:
    match request.operation:
        case "reject":
            rejected = await long_term.review_duplicate(
                state.source_id,
                state.target_id,
                confirm=False,
            )
            return {"rejected": rejected}
        case "merge":
            merged = await long_term.merge_duplicate_entities(
                state.source_id,
                state.target_id,
            )
            return {"merged": merged is not None}


def _required_text(record: JsonObject, field_name: str) -> str:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return value
    raise BackendRequestError(_DEDUP_UNAVAILABLE_DETAIL)


def _optional_text(record: JsonObject, field_name: str) -> str | None:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return value
    return None


def _optional_float(record: JsonObject, field_name: str, default: float) -> float:
    value = record.get(field_name)
    if isinstance(value, int | float):
        return float(value)
    return default


def _string_list(value: JsonValue | None) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _json_member_object(record: JsonObject, field_name: str) -> JsonObject:
    value = record.get(field_name)
    if isinstance(value, dict):
        return _json_object(value)
    return {}


def _hash_json(value: JsonObject) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _json_compatible_object(value: object) -> JsonObject:
    parsed = _json_object(value)
    if parsed:
        return parsed
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(json.dumps(value, default=str))
    except (TypeError, ValidationError):
        return {}


def _canonical_json(value: JsonObject) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


async def _query_recent_facts(
    client: MemoryClientContext,
    scope_metadata: Mapping[str, JsonValue],
) -> list[JsonObject]:
    """Most recently written scoped facts, newest first.

    Scans a candidate pool sized like /v1/memories/search and re-checks scope
    on the deserialized rows so the item budget only sees in-scope facts.
    """
    params: JsonObject = {
        "metadata_fragments": _metadata_fragments(scope_metadata),
        "candidate_limit": _MEMORY_SEARCH_CANDIDATE_LIMIT,
    }
    rows = await client.query.cypher(
        """
        MATCH (f:Fact)
        WHERE f.metadata IS NOT NULL
          AND all(
            fragment IN $metadata_fragments WHERE f.metadata CONTAINS fragment
          )
        WITH f
        ORDER BY f.created_at DESC, f.subject ASC, f.predicate ASC, f.object ASC
        LIMIT $candidate_limit
        RETURN f{
            .id, .subject, .predicate, .object, .metadata,
            created_at: toString(f.created_at)
        } AS f
        """,
        params,
    )
    return [
        fact
        for row in rows
        if (fact := _fact_from_row(row)) is not None
        and _fact_matches_scope(fact, scope_metadata)
    ]


def _fact_from_row(row: JsonObject) -> JsonObject | None:
    fact = row.get("f")
    if not isinstance(fact, dict):
        return None
    required_fields = ("subject", "predicate", "object")
    if all(isinstance(fact.get(field_name), str) for field_name in required_fields):
        return fact
    return None


def _fact_from_memory(memory: StoredMemory) -> JsonObject:
    return {
        "id": memory.memory_id,
        "subject": memory.subject,
        "predicate": memory.predicate,
        "object": memory.content,
        "metadata": dict(memory.metadata),
        "created_at": memory.created_at,
    }


def _fact_freshness(fact: JsonObject) -> FactFreshness:
    metadata = _fact_raw_metadata(fact)
    created_at = fact.get("created_at")
    return FactFreshness(
        slot_key=slot_key(
            str(fact.get("subject", "")),
            str(fact.get("predicate", "")),
            _metadata_entities(metadata),
        ),
        event_date=_metadata_event_date(metadata),
        created_at=created_at if isinstance(created_at, str) and created_at else None,
    )


def _memory_freshness(memory: StoredMemory) -> FactFreshness:
    return FactFreshness(
        slot_key=slot_key(
            memory.subject,
            memory.predicate,
            _metadata_entities(memory.metadata),
        ),
        event_date=_metadata_event_date(memory.metadata),
        created_at=memory.created_at,
    )


def _fact_raw_metadata(fact: JsonObject) -> JsonObject:
    metadata = fact.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return _JSON_OBJECT_ADAPTER.validate_json(metadata)
        except ValidationError:
            return {}
    return {}


def _verbatim_expansion_targets(
    facts: list[JsonObject],
    *,
    cap: int,
) -> list[tuple[str, list[str]]]:
    """Highest-ranked extracted facts to expand, with novel source turn ids.

    Source ids already present as ranked facts are dropped so a verbatim turn
    independently in the result set is never double-rendered; at most ``cap``
    facts (rank order) are returned.
    """
    present_ids = {
        fact_id
        for fact in facts
        if isinstance(fact_id := fact.get("id"), str) and fact_id
    }
    targets: list[tuple[str, list[str]]] = []
    for fact in facts:
        if len(targets) >= cap:
            break
        fact_id = fact.get("id")
        if str(fact.get("predicate")) != EXTRACTED_FACT_PREDICATE or not isinstance(
            fact_id,
            str,
        ):
            continue
        source_ids = [
            source_id
            for source_id in _fact_source_memory_ids(fact)
            if source_id not in present_ids
        ]
        if source_ids:
            targets.append((fact_id, source_ids))
    return targets


def _fact_source_memory_ids(fact: JsonObject) -> list[str]:
    """Provenance ids of the verbatim turns an extracted fact was derived from."""
    source_ids = _fact_raw_metadata(fact).get("source_memory_ids")
    if not isinstance(source_ids, list):
        return []
    return [
        source_id
        for source_id in source_ids
        if isinstance(source_id, str) and source_id
    ]


def _metadata_entities(metadata: Mapping[str, JsonValue]) -> list[str]:
    entities = metadata.get("entities")
    if not isinstance(entities, list):
        return []
    return [entity for entity in entities if isinstance(entity, str)]


def _metadata_event_date(metadata: Mapping[str, JsonValue]) -> str | None:
    event_date = metadata.get("event_date")
    if isinstance(event_date, str) and event_date:
        return event_date
    return None


def _log_supersession(dropped: int, candidates: int, *, surface: str) -> None:
    if dropped == 0:
        return
    _LOGGER.info(
        "read-time supersession dropped superseded facts",
        extra={"surface": surface, "dropped": dropped, "candidates": candidates},
    )


def _fact_context_line(fact: JsonObject) -> str:
    """Render one fact as a single dated line of prompt-facing content.

    Provenance stays out of the prompt: it crowds the answer model's
    attention and remains available through the audit read paths. Subject
    and predicate only render when they carry signal beyond conversation
    plumbing (verbatim ``memory`` and turn ``said_*`` predicates repeat the
    speaker, not knowledge).
    """
    predicate = str(fact["predicate"])
    content = _redacted_text(str(fact["object"]))
    if predicate != VERBATIM_MEMORY_PREDICATE and not predicate.startswith(
        TURN_MEMORY_PREDICATE_PREFIX,
    ):
        subject = _redacted_text(str(fact["subject"]))
        content = f"{subject} {_redacted_text(predicate)}: {content}"
    fact_date = _fact_date(fact, _fact_metadata(fact))
    if fact_date:
        return f"- [{_redacted_text(fact_date)}] {content}"
    return f"- {content}"


def _stored_memory_line(memory: StoredMemory) -> str:
    """Render one search candidate for the recall filter prompt."""
    return _fact_context_line(_fact_from_memory(memory))


def _memory_record_line(record: MemoryRecord) -> str:
    """Render one already-public memory record for the recall filter prompt."""
    metadata = _string_metadata(record.metadata)
    record_date = next(
        (metadata[key] for key in _FACT_DATE_METADATA_KEYS if metadata.get(key)),
        (record.created_at or "")[:10],
    )
    if record_date:
        return f"- [{record_date}] {record.content}"
    return f"- {record.content}"


def _fact_matches_scope(
    fact: JsonObject,
    scope_metadata: Mapping[str, JsonValue],
) -> bool:
    metadata = _fact_metadata(fact)
    requested_scope = {
        field_name: scope_value
        for field_name, scope_value in scope_metadata.items()
        if field_name in _FACT_SCOPE_FIELDS
    }
    return all(
        metadata.get(field_name) == requested_value
        for field_name, requested_value in requested_scope.items()
    ) and all(
        requested_scope.get(field_name) == fact_value
        for field_name, fact_value in metadata.items()
        if field_name in _FACT_SCOPE_FIELDS
    )


def _fact_metadata(fact: JsonObject) -> dict[str, str]:
    metadata = fact.get("metadata")
    if isinstance(metadata, str):
        return _metadata_from_json(metadata)
    if isinstance(metadata, dict):
        return _string_metadata(metadata)
    return {}


def _fact_date(fact: JsonObject, metadata: Mapping[str, str]) -> str:
    for key in _FACT_DATE_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            return value
    created_at = fact.get("created_at")
    if isinstance(created_at, str):
        # The YYYY-MM-DD prefix of the ISO timestamp.
        return created_at[:10]
    return ""


def _fact_markers(facts: list[JsonObject]) -> set[str]:
    markers: set[str] = set()
    for fact in facts:
        for field_name in ("id", "subject", "object"):
            value = fact.get(field_name)
            if isinstance(value, str) and value != "":
                markers.add(value)
    return markers


def _graph_facts_to_candidates(facts: Sequence[JsonObject]) -> list[JsonObject]:
    """Adapt graph-QA nodes to long-term fact candidates for fused ranking.

    Each live graph node's summary becomes a verbatim-predicate candidate so
    ``_fact_context_line`` renders it as a uniform dated line and read-time
    supersession never claims a slot for it (graph summaries are traversal
    evidence, not knowledge slots).
    """
    candidates: list[JsonObject] = []
    for fact in facts:
        summary = fact.get("summary")
        if not isinstance(summary, str) or not summary or fact.get("deleted") is True:
            continue
        identifier = fact.get("id")
        candidates.append(
            {
                "id": identifier if isinstance(identifier, str) else "",
                "subject": "",
                "predicate": VERBATIM_MEMORY_PREDICATE,
                "object": summary,
                "metadata": {},
                "graphqa": True,
            },
        )
    return candidates


# At most this fraction of the context item budget is reserved for
# graph-derived candidates, so dense relevance keeps the bulk of the slots.
_GRAPH_RESERVE_DIVISOR: Final[int] = 4


def _cut_with_graph_reserve(
    facts: list[JsonObject],
    max_items: int,
) -> list[JsonObject]:
    """Apply the item budget while reserving tail slots for graph candidates.

    ``_fuse_graph_facts`` appends graph-derived candidates after the dense
    ranking, so a plain ``facts[:max_items]`` cut silently dropped every one
    of them whenever dense retrieval filled the candidate pool (always, on a
    populated store) - the fusion leg ran but never rendered. Up to a quarter
    of the budget now goes to the highest-ranked graph candidates; dense
    candidates keep the rest, in ranking order. A pure passthrough cut when
    no graph candidate is present.
    """
    graph = [fact for fact in facts if fact.get("graphqa") is True]
    if not graph or len(facts) <= max_items:
        return facts[:max_items]
    dense = [fact for fact in facts if fact.get("graphqa") is not True]
    reserve = min(len(graph), max(1, max_items // _GRAPH_RESERVE_DIVISOR))
    return [*dense[: max_items - reserve], *graph[:reserve]]


def _fuse_graph_facts(
    dense: list[JsonObject],
    graph: list[JsonObject],
) -> list[JsonObject]:
    """Union graph-derived candidates into the dense set, dropping duplicates.

    A graph candidate is dropped when it shares a memory id with a dense fact
    or renders to the same dated line, so a node already surfaced by dense
    retrieval is never double-added. Dense ranking order is preserved; graph
    candidates append after it (Mnemis unions the unordered structured route).
    """
    if not graph:
        return dense
    existing_ids = {
        identifier
        for fact in dense
        if isinstance(identifier := fact.get("id"), str) and identifier
    }
    existing_lines = {_fact_context_line(fact) for fact in dense}
    fused = list(dense)
    for candidate in graph:
        identifier = candidate.get("id")
        if isinstance(identifier, str) and identifier and identifier in existing_ids:
            continue
        line = _fact_context_line(candidate)
        if line in existing_lines:
            continue
        existing_lines.add(line)
        fused.append(candidate)
    return fused


def _dedupe_graph_context(
    graph: GraphContextResponse,
    long_term_markers: set[str],
) -> GraphContextResponse:
    if not long_term_markers or not graph.facts:
        return graph

    facts = [
        fact
        for fact in graph.facts
        if not _graph_fact_matches_markers(fact, long_term_markers)
    ]
    if len(facts) == len(graph.facts):
        return graph

    return GraphContextResponse(
        context="\n".join(_graph_fact_summary(fact) for fact in facts),
        facts=facts,
    )


def _graph_fact_matches_markers(fact: JsonObject, markers: set[str]) -> bool:
    return any(
        isinstance(value, str) and value in markers
        for field_name in ("id", "summary")
        if (value := fact.get(field_name)) is not None
    )


def _graph_fact_summary(fact: JsonObject) -> str:
    summary = fact.get("summary")
    if isinstance(summary, str):
        return summary
    return ""


def _metadata_from_json(metadata: str) -> dict[str, str]:
    try:
        parsed = _JSON_OBJECT_ADAPTER.validate_json(metadata)
    except ValidationError:
        return {}
    return _string_metadata(parsed)


def _metadata_fragments(metadata: Mapping[str, JsonValue]) -> list[JsonValue]:
    fragments: list[JsonValue] = []
    for key, value in metadata.items():
        if key not in _FACT_READ_EXCLUDED_FIELDS and isinstance(value, str):
            fragments.append(f'"{key}": {json.dumps(value)}')
    return fragments


def _string_metadata(metadata: dict[str, JsonValue]) -> dict[str, str]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, str) and value != ""
    }


def _redacted_text(value: str) -> str:
    redacted = redact_secrets(value)
    if isinstance(redacted, str):
        return redacted
    return value


def _long_term_enrichment_enabled(settings: Settings) -> bool:
    return (
        settings.gnosis_prompt_entities_enabled
        or settings.gnosis_prompt_preferences_enabled
    )


def _safe_reasoning_context(context: str) -> str:
    lines = [
        line
        for line in context.splitlines()
        if not _reasoning_line_exposes_hidden_trace(line)
    ]
    return _redacted_text("\n".join(lines))


def _reasoning_line_exposes_hidden_trace(line: str) -> bool:
    normalized = line.casefold()
    return "thought:" in normalized or "chain_of_thought" in normalized


def _redacted_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _redacted_text(value)


def _redacted_object(value: JsonObject) -> JsonObject:
    redacted = redact_secrets(value)
    if isinstance(redacted, dict):
        return redacted
    return {}


def _json_object(value: object) -> JsonObject:
    try:
        return _JSON_OBJECT_ADAPTER.validate_python(value)
    except ValidationError:
        return {}


async def _get_reasoning_trace(
    reasoning: ReasoningMemory,
    request: ReasoningTraceDetailRequest,
) -> SdkReasoningTrace | None:
    if request.include_steps:
        return await reasoning.get_trace_with_steps(UUID(request.trace_id))
    return await reasoning.get_trace(request.trace_id)


def _scoped_reasoning_traces(
    scope: MemoryScope,
    traces: Sequence[SdkReasoningTrace],
) -> list[ReasoningTraceSummary]:
    return [
        _reasoning_trace_summary(trace)
        for trace in traces
        if _reasoning_trace_matches_scope(trace, scope)
    ]


def _reasoning_trace_matches_scope(
    trace: SdkReasoningTrace,
    scope: MemoryScope,
) -> bool:
    metadata = _string_metadata(_json_object(trace.metadata))
    scope_metadata = _scope_metadata(scope)
    return trace.session_id == scope.session_id and all(
        metadata.get(field_name) == expected_value
        for field_name, expected_value in scope_metadata.items()
        if field_name != "session_id"
    )


def _reasoning_step_matches_scope(step_like: object, scope: MemoryScope) -> bool:
    step = _reasoning_step_from_result(step_like)
    if step is None:
        return False
    metadata = _string_metadata(_json_object(step.metadata))
    scope_metadata = _scope_metadata(scope)
    return bool(metadata) and all(
        metadata.get(field_name) == expected_value
        for field_name, expected_value in scope_metadata.items()
    )


def _reasoning_trace_summary(trace: SdkReasoningTrace) -> ReasoningTraceSummary:
    return ReasoningTraceSummary(
        trace_id=str(trace.id),
        session_id=trace.session_id,
        task=_redacted_text(trace.task),
        outcome=_redacted_optional_text(trace.outcome),
        success=trace.success,
        started_at=_optional_datetime_text(trace.started_at),
        completed_at=_optional_datetime_text(trace.completed_at),
        metadata=_public_reasoning_metadata(_json_object(trace.metadata)),
    )


def _reasoning_step_record(step_like: object) -> ReasoningStepRecord:
    step = _reasoning_step_from_result(step_like)
    if step is None:
        raise BackendCapabilityUnavailable(_REASONING_READ_UNAVAILABLE_DETAIL)
    return ReasoningStepRecord(
        step_id=str(step.id),
        trace_id=str(step.trace_id),
        step_number=step.step_number,
        action=_redacted_optional_text(step.action),
        observation=_redacted_optional_text(step.observation),
        tool_calls=[
            _reasoning_tool_call_record(tool_call) for tool_call in step.tool_calls
        ],
        metadata=_redacted_reasoning_object(_json_object(step.metadata)),
    )


def _reasoning_step_from_result(step_like: object) -> SdkReasoningStep | None:
    if isinstance(step_like, SdkReasoningStep):
        return step_like
    step = getattr(step_like, "step", None)
    if isinstance(step, SdkReasoningStep):
        return step
    return None


def _reasoning_tool_call_record(tool_call: ToolCall) -> JsonObject:
    return _redacted_reasoning_object(
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


def _reasoning_tool_stats_record(tool_stats: ToolStats) -> ReasoningToolStatsRecord:
    return ReasoningToolStatsRecord(
        name=_redacted_text(tool_stats.name),
        description=_redacted_optional_text(tool_stats.description),
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


def _redacted_reasoning_object(value: object) -> JsonObject:
    return _redacted_reasoning_json(_json_object(value))


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
    return _JSON_OBJECT_ADAPTER.validate_python(redacted)


def _public_reasoning_metadata(metadata: JsonObject) -> JsonObject:
    return _redacted_reasoning_json(
        {
            key: value
            for key, value in metadata.items()
            if key not in _SCOPE_METADATA_KEYS
        },
    )


def _redacted_reasoning_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return _redacted_text(value)
    if isinstance(value, dict):
        return _redacted_reasoning_json(_json_object(value))
    if isinstance(value, list):
        return [_redacted_reasoning_value(item) for item in value]
    return value


def _graph_export_response(
    request: GraphExportRequest,
    graph: MemoryGraphLike,
) -> GraphExportResponse:
    return GraphExportResponse(
        scope=request.scope,
        nodes=[_graph_export_node(node) for node in graph.nodes],
        relationships=[
            _graph_export_relationship(relationship)
            for relationship in graph.relationships
        ],
        metadata=_redacted_object(_json_object(graph.metadata)),
    )


def _graph_export_node(node: GraphNodeLike) -> GraphExportNode:
    return GraphExportNode(
        id=node.id,
        labels=list(node.labels),
        properties=_redacted_object(_json_object(node.properties)),
    )


def _graph_export_relationship(
    relationship: GraphRelationshipLike,
) -> GraphExportRelationship:
    return GraphExportRelationship(
        id=relationship.id,
        type=relationship.type,
        from_node=relationship.from_node,
        to_node=relationship.to_node,
        properties=_redacted_object(_json_object(relationship.properties)),
    )


# Long-term memory is durable across conversations, so reads never narrow by
# session_id: pinning recall to the requesting session starves the context
# budget while /v1/memories/search over the same data spans sessions.
_FACT_READ_EXCLUDED_FIELDS = {
    "space_id",
    "session_id",
}

_FACT_SCOPE_FIELDS = {
    "tenant_id",
    "agent_id",
    "user_id",
    "visibility",
    "guild_id",
    "channel_id",
}

# Temporal anchor for rendered facts: a stored date tag wins over created_at.
_FACT_DATE_METADATA_KEYS = ("session_date", "date")


def _append_context_section(
    sections: list[MemoryContextSection],
    source: str,
    content: str,
) -> None:
    if content:
        sections.append(MemoryContextSection(source=source, content=content))


def _legacy_context_request(request: ContextRequest) -> MemoryContextRequest:
    """Map a legacy /v1/context request onto the combined memory-context path."""
    return MemoryContextRequest(
        scope=request.scope,
        query=request.query,
        include_short_term=True,
        include_long_term=False,
        include_reasoning=False,
        include_graph=False,
        max_items=request.limit,
    )


def _legacy_context_response(response: MemoryContextResponse) -> ContextResponse:
    """Reduce a combined memory-context response to the legacy short-term shape."""
    context = next(
        (
            section.content
            for section in response.sections
            if section.source == "short_term"
        ),
        "",
    )
    return ContextResponse(context=context)
