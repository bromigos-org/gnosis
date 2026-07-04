from collections.abc import Sized
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

type JsonValue = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
type JsonObject = dict[str, JsonValue]


def _is_none(value: bool | None) -> bool:
    return value is None


def _is_false(value: bool) -> bool:
    return value is False


def _is_empty(value: Sized) -> bool:
    return len(value) == 0


class ContractModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class MemoryVisibility(StrEnum):
    PRIVATE_USER = "private_user"
    AGENT_PRIVATE = "agent_private"
    AGENT_SHARED = "agent_shared"
    CHANNEL = "channel"
    GUILD = "guild"
    TENANT = "tenant"
    GLOBAL = "global"


class SourceClient(StrEnum):
    DISCORD = "discord"


class ClientEventType(StrEnum):
    MESSAGE_CREATED = "message_created"
    MESSAGE_UPDATED = "message_updated"
    MESSAGE_DELETED = "message_deleted"
    REACTION_ADDED = "reaction_added"
    REACTION_REMOVED = "reaction_removed"
    CHANNEL_CREATED = "channel_created"
    CHANNEL_UPDATED = "channel_updated"
    CHANNEL_DELETED = "channel_deleted"
    THREAD_CREATED = "thread_created"
    THREAD_UPDATED = "thread_updated"
    THREAD_DELETED = "thread_deleted"
    ROLE_CREATED = "role_created"
    ROLE_UPDATED = "role_updated"
    ROLE_DELETED = "role_deleted"
    MEMBER_UPDATED = "member_updated"
    USER_DISCOVERED = "user_discovered"
    MEMBER_ROLE_ASSIGNED = "member_role_assigned"
    MEMBER_ROLE_UNASSIGNED = "member_role_unassigned"
    ATTACHMENT_DISCOVERED = "attachment_discovered"
    LINK_DISCOVERED = "link_discovered"
    TOPIC_UPDATED = "topic_updated"
    SKILL_PROPOSED = "skill_proposed"
    SKILL_APPROVED = "skill_approved"
    SKILL_USED = "skill_used"


class EventIngestStatus(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    FAILED = "failed"


class SkillStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISABLED = "disabled"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryScope(ContractModel):
    tenant_id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    visibility: MemoryVisibility
    guild_id: str | None = Field(default=None, min_length=1)
    channel_id: str | None = Field(default=None, min_length=1)


class MessageWriteRequest(ContractModel):
    scope: MemoryScope
    role: MessageRole
    content: str = Field(min_length=1)
    extract_entities: bool | None = Field(default=None, exclude_if=_is_none)
    extract_relations: bool | None = Field(default=None, exclude_if=_is_none)
    preview_extraction: bool = Field(default=False, exclude_if=_is_false)
    raw_text_documents: list["RawTextDocument"] = Field(
        default_factory=list,
        exclude_if=_is_empty,
    )
    ocr_image_references: list["OcrImageReference"] = Field(
        default_factory=list,
        exclude_if=_is_empty,
    )
    rustfs_source_references: list["RustFSSourceReference"] = Field(
        default_factory=list,
        exclude_if=_is_empty,
    )


class MessageWriteResponse(ContractModel):
    accepted: bool


class RawTextDocument(ContractModel):
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    content_type: str = "text/plain"
    checksum_sha256: str | None = Field(default=None, min_length=1)


class OcrImageReference(ContractModel):
    source_id: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    checksum_sha256: str = Field(min_length=1)
    rustfs: "RustFSSourceReference | None" = None


class RustFSSourceReference(ContractModel):
    bucket: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    checksum_sha256: str = Field(min_length=1)


class ExtractionPreviewRequest(ContractModel):
    scope: MemoryScope
    content: str | None = Field(default=None, min_length=1)
    raw_text_documents: list[RawTextDocument] = Field(default_factory=list)
    ocr_image_references: list[OcrImageReference] = Field(default_factory=list)
    rustfs_source_references: list[RustFSSourceReference] = Field(default_factory=list)
    extract_entities: bool | None = None
    extract_relations: bool | None = None


class ExtractionCandidate(ContractModel):
    kind: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExtractionPreviewMetrics(ContractModel):
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    ocr_images: int = Field(ge=0)
    rustfs_objects: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    max_concurrency: int = Field(ge=1)


class ExtractionPreviewProvenance(ContractModel):
    source_ids: list[str] = Field(default_factory=list)
    rustfs_objects: list[RustFSSourceReference] = Field(default_factory=list)


class ExtractionPreviewResponse(ContractModel):
    candidates: list[ExtractionCandidate] = Field(default_factory=list)
    metrics: ExtractionPreviewMetrics
    provenance: ExtractionPreviewProvenance
    extract_entities: bool
    extract_relations: bool


class ContextRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=30)


class ContextResponse(ContractModel):
    context: str


class MemoryContextSection(ContractModel):
    source: str = Field(min_length=1)
    memory_type: str | None = Field(default=None, min_length=1, exclude_if=_is_none)
    content: str = Field(min_length=1)
    facts: list[JsonObject] = Field(default_factory=list)


class MemoryContextRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    include_short_term: bool = True
    include_long_term: bool = True
    include_reasoning: bool = True
    include_graph: bool = True
    max_items: int = Field(default=8, ge=1, le=100)
    graph_limit: int = Field(default=8, ge=1, le=100)


class MemoryContextResponse(ContractModel):
    sections: list[MemoryContextSection] = Field(default_factory=list)


class MemoryProvenance(ContractModel):
    source: str = Field(min_length=1)
    source_id: str | None = Field(default=None, min_length=1)


class MemorySearchUnavailable(ContractModel):
    capability: str = Field(min_length=1)
    reason: str = Field(default="unsupported_by_installed_sdk", min_length=1)


class EntitySearchRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    metadata: JsonObject = Field(default_factory=dict)


class FactSearchRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    metadata: JsonObject = Field(default_factory=dict)


class PreferenceSearchRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    category: str | None = Field(default=None, min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class EntityRecord(ContractModel):
    id: str | None = Field(default=None, min_length=1)
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    subtype: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    aliases: list[str] = Field(default_factory=list)
    attributes: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)
    provenance: MemoryProvenance | None = None


class FactRecord(ContractModel):
    id: str | None = Field(default=None, min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    metadata: JsonObject = Field(default_factory=dict)
    provenance: MemoryProvenance | None = None


class PreferenceRecord(ContractModel):
    id: str | None = Field(default=None, min_length=1)
    category: str = Field(min_length=1)
    preference: str = Field(min_length=1)
    context: str | None = Field(default=None, min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    user_identifier: str | None = Field(default=None, min_length=1)
    metadata: JsonObject = Field(default_factory=dict)
    provenance: MemoryProvenance | None = None


class EntitySearchResponse(ContractModel):
    entities: list[EntityRecord] = Field(default_factory=list)
    unavailable: list[MemorySearchUnavailable] = Field(default_factory=list)


class FactSearchResponse(ContractModel):
    facts: list[FactRecord] = Field(default_factory=list)
    unavailable: list[MemorySearchUnavailable] = Field(default_factory=list)


class PreferenceSearchResponse(ContractModel):
    preferences: list[PreferenceRecord] = Field(default_factory=list)
    unavailable: list[MemorySearchUnavailable] = Field(default_factory=list)


class EntityWriteRequest(ContractModel):
    scope: MemoryScope
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    subtype: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    aliases: list[str] = Field(default_factory=list)
    attributes: JsonObject = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)
    provenance: MemoryProvenance | None = None
    metadata: JsonObject = Field(default_factory=dict)
    resolve: bool = True
    generate_embedding: bool = True
    deduplicate: bool = True


class FactWriteRequest(ContractModel):
    scope: MemoryScope
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    provenance: MemoryProvenance | None = None
    metadata: JsonObject = Field(default_factory=dict)
    generate_embedding: bool = True


class PreferenceWriteRequest(ContractModel):
    scope: MemoryScope
    category: str = Field(min_length=1)
    preference: str = Field(min_length=1)
    context: str | None = Field(default=None, min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    user_identifier: str | None = Field(default=None, min_length=1)
    provenance: MemoryProvenance | None = None
    metadata: JsonObject = Field(default_factory=dict)
    generate_embedding: bool = True


class EntityMessageLinkRequest(ContractModel):
    scope: MemoryScope
    entity_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    context: str | None = Field(default=None, min_length=1)


class TouchedEntityRef(ContractModel):
    id: str | None = Field(default=None, min_length=1)
    name: str = Field(min_length=1)
    type: str | None = Field(default=None, min_length=1)


class ReasoningTraceStartRequest(ContractModel):
    scope: MemoryScope
    session_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    metadata: JsonObject = Field(default_factory=dict)
    triggered_by_message_id: str | None = Field(default=None, min_length=1)
    user_identifier: str | None = Field(default=None, min_length=1)


class ReasoningTraceStartResponse(ContractModel):
    trace_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task: str = Field(min_length=1)


class ReasoningStepRequest(ContractModel):
    scope: MemoryScope
    trace_id: str = Field(min_length=1)
    action: str | None = Field(default=None, min_length=1)
    observation: str | None = Field(default=None, min_length=1)
    step_number: int | None = Field(default=None, ge=1)
    metadata: JsonObject = Field(default_factory=dict)


class ReasoningStepResponse(ContractModel):
    step_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    step_number: int = Field(ge=1)


class ReasoningToolCallRequest(ContractModel):
    scope: MemoryScope
    trace_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    result: JsonValue | None = None
    status: str = Field(min_length=1)
    duration_ms: int | None = Field(default=None, ge=0)
    error: str | None = Field(default=None, min_length=1)
    message_id: str | None = Field(default=None, min_length=1)
    touched_entities: list[TouchedEntityRef] = Field(default_factory=list)


class ReasoningToolCallResponse(ContractModel):
    tool_call_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)


class ReasoningTraceCompleteRequest(ContractModel):
    scope: MemoryScope
    trace_id: str = Field(min_length=1)
    outcome: str | None = Field(default=None, min_length=1)
    success: bool | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ReasoningTraceCompleteResponse(ContractModel):
    trace_id: str = Field(min_length=1)
    success: bool | None = None
    outcome: str | None = Field(default=None, min_length=1)
    completed_at: str | None = Field(default=None, min_length=1)


class ReasoningContextRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    include_traces: bool = True
    include_steps: bool = True
    include_tool_calls: bool = True
    max_items: int = Field(default=8, ge=1, le=100)


class ReasoningContextResponse(ContractModel):
    context: str = Field(min_length=1)
    traces: list[JsonObject] = Field(default_factory=list)


class ReasoningTraceSummary(ContractModel):
    trace_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    outcome: str | None = Field(default=None, min_length=1)
    success: bool | None = None
    started_at: str | None = Field(default=None, min_length=1)
    completed_at: str | None = Field(default=None, min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class ReasoningStepRecord(ContractModel):
    step_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    step_number: int = Field(ge=1)
    action: str | None = Field(default=None, min_length=1)
    observation: str | None = Field(default=None, min_length=1)
    tool_calls: list[JsonObject] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)


class ReasoningToolStatsRecord(ContractModel):
    name: str = Field(min_length=1)
    description: str | None = Field(default=None, min_length=1)
    total_calls: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    avg_duration_ms: float | None = Field(default=None, ge=0)
    last_used_at: str | None = Field(default=None, min_length=1)


class ReasoningTraceListRequest(ContractModel):
    scope: MemoryScope
    session_id: str | None = Field(default=None, min_length=1)
    success_only: bool | None = None
    limit: int = Field(default=100, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ReasoningTraceListResponse(ContractModel):
    scope: MemoryScope
    traces: list[ReasoningTraceSummary] = Field(default_factory=list)


class ReasoningTraceDetailRequest(ContractModel):
    scope: MemoryScope
    trace_id: str = Field(min_length=1)
    include_steps: bool = True


class ReasoningTraceDetailResponse(ContractModel):
    scope: MemoryScope
    trace: ReasoningTraceSummary | None = None
    steps: list[ReasoningStepRecord] = Field(default_factory=list)


class ReasoningSimilarTracesRequest(ContractModel):
    scope: MemoryScope
    task: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=25)
    success_only: bool = True
    threshold: float = Field(default=0.7, ge=0, le=1)


class ReasoningSimilarTracesResponse(ContractModel):
    scope: MemoryScope
    traces: list[ReasoningTraceSummary] = Field(default_factory=list)


class ReasoningStepSearchRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    success_only: bool = True
    threshold: float = Field(default=0.7, ge=0, le=1)


class ReasoningStepSearchResponse(ContractModel):
    scope: MemoryScope
    steps: list[ReasoningStepRecord] = Field(default_factory=list)


class ReasoningToolStatsRequest(ContractModel):
    scope: MemoryScope
    tool_name: str | None = Field(default=None, min_length=1)


class ReasoningToolStatsResponse(ContractModel):
    scope: MemoryScope
    tools: list[ReasoningToolStatsRecord] = Field(default_factory=list)


class SkillListRequest(ContractModel):
    tenant_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)


class SkillListResponse(ContractModel):
    skills: list["SkillRecord"] = Field(default_factory=list)


class ClientEventActor(ContractModel):
    id: str = Field(min_length=1)
    display_name: str | None = Field(default=None, min_length=1)
    is_bot: bool = False


class ClientEventSubject(ContractModel):
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    parent_id: str | None = Field(default=None, min_length=1)


class DiscordEventContext(ContractModel):
    guild_id: str | None = Field(default=None, min_length=1)
    channel_id: str | None = Field(default=None, min_length=1)
    thread_id: str | None = Field(default=None, min_length=1)
    message_id: str | None = Field(default=None, min_length=1)


class ClientEvent(ContractModel):
    tenant_id: str = Field(min_length=1)
    source_client: SourceClient
    agent_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_type: ClientEventType
    occurred_at: str = Field(min_length=1)
    observed_at: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    scope: MemoryScope
    actor: ClientEventActor
    subject: ClientEventSubject
    payload: JsonObject = Field(default_factory=dict)
    discord: DiscordEventContext | None = None


class ClientEventBatchRequest(ContractModel):
    events: list[ClientEvent] = Field(min_length=1, max_length=100)


class EventIngestResult(ContractModel):
    event_id: str = Field(min_length=1)
    status: EventIngestStatus
    reason: str | None = Field(default=None, min_length=1)


class ClientEventBatchResponse(ContractModel):
    results: list[EventIngestResult]


type MemoryAddEvent = Literal["ADD", "UPDATE", "NONE"]
type MemoryMessageRole = Literal["user", "assistant"]


class MemoryMessage(ContractModel):
    role: MemoryMessageRole
    content: str = Field(min_length=1)


class MemoryAddRequest(ContractModel):
    scope: MemoryScope
    messages: list[MemoryMessage] = Field(default_factory=list, max_length=20)
    content: str | None = Field(default=None, min_length=1)
    infer: bool = True
    metadata: JsonObject = Field(default_factory=dict)


class MemoryAddResult(ContractModel):
    memory_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    event: MemoryAddEvent
    metadata: JsonObject = Field(default_factory=dict)


class MemoryAddResponse(ContractModel):
    results: list[MemoryAddResult] = Field(default_factory=list)


class MemorySearchRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    filters: JsonObject | None = None
    limit: int = Field(default=8, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0, le=1)
    peers: list[str] = Field(default_factory=list, exclude_if=_is_empty)


class MemoryRecord(ContractModel):
    memory_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0, le=1, exclude_if=_is_none)
    metadata: JsonObject = Field(default_factory=dict)
    created_at: str | None = Field(default=None, min_length=1)
    updated_at: str | None = Field(default=None, min_length=1)
    origin: str | None = Field(default=None, min_length=1, exclude_if=_is_none)


class MemoryPeerError(ContractModel):
    peer: str = Field(min_length=1)
    error: str = Field(min_length=1)


class MemorySearchResponse(ContractModel):
    results: list[MemoryRecord] = Field(default_factory=list)
    peer_errors: list[MemoryPeerError] = Field(
        default_factory=list,
        exclude_if=_is_empty,
    )


class MemoryListRequest(ContractModel):
    scope: MemoryScope
    filters: JsonObject | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


class MemoryListResponse(ContractModel):
    results: list[MemoryRecord] = Field(default_factory=list)
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)


class MemoryUpdateRequest(ContractModel):
    scope: MemoryScope
    content: str | None = Field(default=None, min_length=1)
    metadata: JsonObject | None = None


class MemoryUpdateResponse(ContractModel):
    memory_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    event: Literal["UPDATE"] = "UPDATE"


class MemoryDeleteRequest(ContractModel):
    scope: MemoryScope


class MemoryDeleteResponse(ContractModel):
    memory_id: str = Field(min_length=1)
    event: Literal["DELETE"] = "DELETE"


class MemoryPromoteRequest(ContractModel):
    peer: str = Field(min_length=1)
    scope: MemoryScope
    filters: JsonObject | None = None
    limit: int = Field(default=50, ge=1, le=200)
    dry_run: bool = True


class MemoryPromoteCandidate(ContractModel):
    memory_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class MemoryPromotedRecord(ContractModel):
    source_memory_id: str = Field(min_length=1)
    peer_memory_id: str = Field(min_length=1)
    event: MemoryAddEvent


class MemoryPromoteFailure(ContractModel):
    source_memory_id: str = Field(min_length=1)
    error: str = Field(min_length=1)


class MemoryPromoteResponse(ContractModel):
    peer: str = Field(min_length=1)
    count: int = Field(ge=0)
    dry_run: bool
    candidates: list[MemoryPromoteCandidate] = Field(default_factory=list)
    promoted: list[MemoryPromotedRecord] = Field(default_factory=list)
    failed: list[MemoryPromoteFailure] = Field(default_factory=list)


class GraphContextRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=30)
    include_topology: bool = True
    include_skills: bool = True


class GraphContextResponse(ContractModel):
    context: str
    facts: list[JsonObject] = Field(default_factory=list)


type GraphMemoryType = Literal["short_term", "long_term", "reasoning"]
type DedupOperationName = Literal["reject", "merge"]
type ConsolidationOperationName = Literal[
    "archive_expired_conversations",
    "dedupe_entities",
    "detect_superseded_preferences",
    "summarize_long_traces",
]


class SdkStatsRequest(ContractModel):
    scope: MemoryScope


class SdkStatsResponse(ContractModel):
    scope: MemoryScope
    stats: JsonObject


class DedupStatsRequest(ContractModel):
    scope: MemoryScope


class DedupStatsResponse(ContractModel):
    scope: MemoryScope
    stats: JsonObject


class DedupCandidateRequest(ContractModel):
    scope: MemoryScope
    limit: int = Field(default=100, ge=1, le=500)


class DedupEntitySnapshot(ContractModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    subtype: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    aliases: list[str] = Field(default_factory=list)
    attributes: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)


class DedupCandidate(ContractModel):
    candidate_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    source: DedupEntitySnapshot
    target: DedupEntitySnapshot
    similarity: float = Field(ge=0, le=1)
    reject_dry_run_token: str = Field(min_length=1)
    merge_dry_run_token: str = Field(min_length=1)


class DedupCandidateResponse(ContractModel):
    scope: MemoryScope
    candidates: list[DedupCandidate] = Field(default_factory=list)
    graph_snapshot_hash: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)


class DedupOperatorAudit(ContractModel):
    operator_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    ticket: str | None = Field(default=None, min_length=1)


class DedupApplyRequest(ContractModel):
    scope: MemoryScope
    apply: bool
    operation: DedupOperationName
    candidate_id: str = Field(min_length=1)
    candidate_version: int = Field(ge=1)
    graph_snapshot_hash: str = Field(min_length=1)
    dry_run_token: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    audit: DedupOperatorAudit


class DedupApplyResponse(ContractModel):
    scope: MemoryScope
    operation: DedupOperationName
    candidate_id: str = Field(min_length=1)
    candidate_version: int = Field(ge=1)
    applied: bool
    result: JsonObject = Field(default_factory=dict)
    audit: DedupOperatorAudit


class ConsolidationDryRunRequest(ContractModel):
    scope: MemoryScope
    operation: ConsolidationOperationName
    ttl_days: int | None = Field(default=None, ge=1)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    max_pairs: int | None = Field(default=None, ge=1, le=10000)
    user_identifier: str | None = Field(default=None, min_length=1)
    min_steps: int | None = Field(default=None, ge=1)
    max_traces: int | None = Field(default=None, ge=1, le=10000)


class ConsolidationDryRunResponse(ContractModel):
    scope: MemoryScope
    operation: ConsolidationOperationName
    dry_run: bool
    report: JsonObject = Field(default_factory=dict)
    graph_snapshot_hash: str = Field(min_length=1)
    dry_run_token: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)


class ConsolidationApplyRequest(ContractModel):
    scope: MemoryScope
    apply: bool
    operation: ConsolidationOperationName
    graph_snapshot_hash: str = Field(min_length=1)
    dry_run_token: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    audit: DedupOperatorAudit
    ttl_days: int | None = Field(default=None, ge=1)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    max_pairs: int | None = Field(default=None, ge=1, le=10000)
    user_identifier: str | None = Field(default=None, min_length=1)
    min_steps: int | None = Field(default=None, ge=1)
    max_traces: int | None = Field(default=None, ge=1, le=10000)


class ConsolidationApplyResponse(ContractModel):
    scope: MemoryScope
    operation: ConsolidationOperationName
    applied: bool
    result: JsonObject = Field(default_factory=dict)
    audit: DedupOperatorAudit


class GraphExportRequest(ContractModel):
    scope: MemoryScope
    memory_types: list[GraphMemoryType] = Field(
        default_factory=lambda: ["short_term", "long_term", "reasoning"],
        min_length=1,
        max_length=3,
    )
    limit: int = Field(default=100, ge=1, le=1000)


class GraphExportNode(ContractModel):
    id: str = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    properties: JsonObject = Field(default_factory=dict)


class GraphExportRelationship(ContractModel):
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    from_node: str = Field(min_length=1)
    to_node: str = Field(min_length=1)
    properties: JsonObject = Field(default_factory=dict)


class GraphExportResponse(ContractModel):
    scope: MemoryScope
    nodes: list[GraphExportNode] = Field(default_factory=list)
    relationships: list[GraphExportRelationship] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)


class BackendReadiness(ContractModel):
    graph: str
    schema_status: str = Field(alias="schema")
    buffer_status: str = Field(default="ready", alias="buffer")


class BufferStatus(ContractModel):
    write_mode: Literal["sync", "buffered"]
    max_pending: int
    pending_writes: int | None = None
    write_errors: int
    status: Literal["ready", "degraded", "unavailable"]


class BufferFlushResponse(ContractModel):
    flushed: bool
    status: BufferStatus


class ReadinessResponse(ContractModel):
    status: str


class DiagnosticsConfig(ContractModel):
    neo4j_uri: str
    neo4j_username: str
    litellm_base_url: str
    gnosis_llm: str
    gnosis_embedding: str
    gnosis_embedding_dimensions: int
    gnosis_audit_read: bool
    gnosis_conversation_ttl_days: int | None
    gnosis_write_mode: str
    gnosis_max_pending: int
    gnosis_fact_deduplication_enabled: bool
    gnosis_trace_embedding_enabled: bool
    gnosis_extract_entities_enabled: bool
    gnosis_extract_relations_enabled: bool
    gnosis_extraction_preview_enabled: bool
    gnosis_extraction_batch_size: int
    gnosis_extraction_max_concurrency: int
    gnosis_extraction_chunk_size: int
    gnosis_extraction_chunk_overlap: int
    gnosis_ocr_enabled: bool
    gnosis_ocr_model: str
    gnosis_ocr_max_image_bytes: int
    gnosis_rustfs_enabled: bool
    gnosis_rustfs_bucket: str
    gnosis_rustfs_prefix: str
    gnosis_rustfs_endpoint: str
    gnosis_rustfs_retention_days: int | None
    gnosis_prompt_entities_enabled: bool
    gnosis_prompt_preferences_enabled: bool
    gnosis_prompt_reasoning_enabled: bool
    gnosis_consolidation_schedule_enabled: bool


class DiagnosticsResponse(ContractModel):
    tenant_id: str
    config: DiagnosticsConfig
    backend: BackendReadiness


class SkillRecord(ContractModel):
    skill_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: SkillStatus
    scope: MemoryVisibility = MemoryVisibility.AGENT_SHARED
    metadata: JsonObject = Field(default_factory=dict)


class SkillProposal(ContractModel):
    proposal_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    proposed_by: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scope: MemoryVisibility = MemoryVisibility.AGENT_SHARED
    metadata: JsonObject = Field(default_factory=dict)


class SkillUsage(ContractModel):
    skill_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    used_by: str = Field(min_length=1)
    used_at: str = Field(min_length=1)
    scope: MemoryVisibility = MemoryVisibility.AGENT_SHARED
    metadata: JsonObject = Field(default_factory=dict)


def default_event_visibility(event: ClientEvent) -> MemoryVisibility:
    match event.event_type:
        case (
            ClientEventType.CHANNEL_CREATED
            | ClientEventType.CHANNEL_UPDATED
            | ClientEventType.CHANNEL_DELETED
            | ClientEventType.THREAD_CREATED
            | ClientEventType.THREAD_UPDATED
            | ClientEventType.THREAD_DELETED
            | ClientEventType.ROLE_CREATED
            | ClientEventType.ROLE_UPDATED
            | ClientEventType.ROLE_DELETED
            | ClientEventType.MEMBER_UPDATED
            | ClientEventType.USER_DISCOVERED
            | ClientEventType.MEMBER_ROLE_ASSIGNED
            | ClientEventType.MEMBER_ROLE_UNASSIGNED
        ):
            return MemoryVisibility.GUILD
        case (
            ClientEventType.MESSAGE_CREATED
            | ClientEventType.MESSAGE_UPDATED
            | ClientEventType.MESSAGE_DELETED
            | ClientEventType.REACTION_ADDED
            | ClientEventType.REACTION_REMOVED
            | ClientEventType.ATTACHMENT_DISCOVERED
            | ClientEventType.LINK_DISCOVERED
            | ClientEventType.TOPIC_UPDATED
        ):
            if event.scope.guild_id is None:
                return MemoryVisibility.PRIVATE_USER
            return MemoryVisibility.CHANNEL
        case (
            ClientEventType.SKILL_PROPOSED
            | ClientEventType.SKILL_APPROVED
            | ClientEventType.SKILL_USED
        ):
            return MemoryVisibility.AGENT_SHARED


def default_skill_visibility(scope: MemoryVisibility | None = None) -> MemoryVisibility:
    match scope:
        case None:
            return MemoryVisibility.AGENT_SHARED
        case MemoryVisibility.GLOBAL:
            return MemoryVisibility.GLOBAL
        case (
            MemoryVisibility.PRIVATE_USER
            | MemoryVisibility.AGENT_PRIVATE
            | MemoryVisibility.AGENT_SHARED
            | MemoryVisibility.CHANNEL
            | MemoryVisibility.GUILD
            | MemoryVisibility.TENANT
        ):
            return MemoryVisibility.AGENT_SHARED


class HealthResponse(ContractModel):
    status: str
