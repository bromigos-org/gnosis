from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

type JsonValue = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
type JsonObject = dict[str, JsonValue]


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


class MessageWriteResponse(ContractModel):
    accepted: bool


class ContextRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=30)


class ContextResponse(ContractModel):
    context: str


class MemoryContextSection(ContractModel):
    source: str = Field(min_length=1)
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


class GraphContextRequest(ContractModel):
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=30)
    include_topology: bool = True
    include_skills: bool = True


class GraphContextResponse(ContractModel):
    context: str
    facts: list[JsonObject] = Field(default_factory=list)


class BackendReadiness(ContractModel):
    graph: str
    schema_status: str = Field(alias="schema")


class ReadinessResponse(ContractModel):
    status: str


class DiagnosticsConfig(ContractModel):
    neo4j_uri: str
    neo4j_username: str
    litellm_base_url: str
    memory_llm: str
    memory_embedding: str
    memory_embedding_dimensions: int


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
