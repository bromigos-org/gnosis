from dataclasses import dataclass, field
from os import environ
from typing import Self

import pytest
from pydantic import ValidationError

environ["AGENTS_MEMORY_TOKEN"] = "test-token"
environ["NEO4J_URI"] = "bolt://neo4j.neo4j.svc.cluster.local:7687"
environ["NEO4J_PASSWORD"] = "test-password"
environ["LITELLM_BASE_URL"] = "http://litellm.litellm.svc.cluster.local:4000/v1"
environ["LITELLM_API_KEY"] = "test-litellm-key"

from fastapi.testclient import TestClient
from neo4j_agent_memory import MemorySettings

from agents_memory.backend import Neo4jAgentMemoryBackend
from agents_memory.main import create_app
from agents_memory.models import (
    ClientEvent,
    ClientEventType,
    ContextRequest,
    ContextResponse,
    MemoryVisibility,
    MessageWriteRequest,
    MessageWriteResponse,
    SkillRecord,
    SkillStatus,
    default_event_visibility,
    default_skill_visibility,
)
from agents_memory.settings import Settings


def test_health_when_called_without_auth() -> None:
    # Given: an app configured with a memory token.
    client = TestClient(create_app(settings_factory=_settings))

    # When: a client calls the unauthenticated health endpoint.
    response = client.get("/health")

    # Then: the service reports healthy.
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_write_message_when_token_is_missing() -> None:
    # Given: an app with protected memory endpoints.
    client = TestClient(create_app(settings_factory=_settings))

    # When: a client writes memory without credentials.
    response = client.post("/v1/messages", json=_message_payload())

    # Then: the API rejects the request.
    assert response.status_code == 401


def test_write_message_when_request_is_scoped() -> None:
    # Given: an app with a fake memory backend.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client writes a scoped message.
    response = client.post(
        "/v1/messages",
        headers=_auth_header(),
        json=_message_payload(),
    )

    # Then: the API accepts and forwards the typed request.
    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert backend.messages[0].scope.tenant_id == "bromigos"


def test_write_message_when_scope_is_outside_policy() -> None:
    # Given: an app configured for the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller tries to write memory into another tenant.
    response = client.post(
        "/v1/messages",
        headers=_auth_header(),
        json=_message_payload(scope=_scope_payload(tenant_id="evil-corp")),
    )

    # Then: the API rejects the request before reaching the backend.
    assert response.status_code == 403
    assert backend.messages == []


def test_get_context_when_request_is_scoped() -> None:
    # Given: an app with a fake memory backend.
    backend = RecordingBackend(context="remembered facts")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client requests context with scope metadata.
    response = client.post(
        "/v1/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # Then: the scoped context is returned.
    assert response.status_code == 200
    assert response.json() == {"context": "remembered facts"}
    assert backend.context_requests[0].scope.agent_id == "pc-principal"


@pytest.mark.anyio
async def test_neo4j_backend_writes_short_and_long_term_memory() -> None:
    # Given: a Neo4j backend wired to a fake memory client.
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
    )

    # When: a scoped user message is written.
    response = await backend.add_message(
        MessageWriteRequest.model_validate(_message_payload()),
    )

    # Then: short-term conversation and long-term fact stores both receive scope.
    assert response.accepted is True
    assert fake_client.short_term.messages[0].user_identifier == (
        "bromigos:discord:channel:pc-principal:789"
    )
    assert fake_client.long_term.facts[0].metadata["session_id"] == (
        "guild:123:channel:456"
    )


@pytest.mark.anyio
async def test_neo4j_backend_context_uses_scoped_short_term_only() -> None:
    # Given: a Neo4j backend wired to a fake memory client.
    fake_client = RecordingMemoryClient(context="scoped short-term context")
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
    )

    # When: scoped context is requested.
    response = await backend.get_context(
        ContextRequest.model_validate(
            {"scope": _scope_payload(), "query": "what matters?", "limit": 4},
        ),
    )

    # Then: retrieval is constrained to the scoped session metadata filter.
    assert response.context == "scoped short-term context"
    assert (
        fake_client.short_term.context_queries[0].metadata_filters["user_id"] == "789"
    )
    assert fake_client.long_term.context_queries == []


def test_client_event_model_accepts_discord_message() -> None:
    # Given: a Discord message event with the canonical client envelope.
    payload = _client_event_payload()

    # When: the event crosses the API model boundary.
    event = ClientEvent.model_validate(payload)

    # Then: Pydantic parses strict enums, nested context, and arbitrary payload JSON.
    assert event.source_client == "discord"
    assert event.event_type == ClientEventType.MESSAGE_CREATED
    assert event.discord is not None
    assert event.discord.message_id == "message-999"
    assert event.payload["content"] == "remember this"


def test_client_event_model_rejects_unknown_fields() -> None:
    # Given: a client event containing an undeclared future field.
    payload = _client_event_payload()
    payload["unknown"] = "not allowed"

    # When / Then: the contract rejects the extra field at the boundary.
    with pytest.raises(ValidationError):
        ClientEvent.model_validate(payload)


def test_client_event_model_rejects_malformed_enum() -> None:
    # Given: a client event with an unsupported event type.
    payload = _client_event_payload()
    payload["event_type"] = "message_created_by_typo"

    # When / Then: enum validation rejects the malformed value.
    with pytest.raises(ValidationError):
        ClientEvent.model_validate(payload)


def test_existing_message_and_context_models_still_serialize() -> None:
    # Given: deployed message and context models.
    message = MessageWriteRequest.model_validate(_message_payload())
    context = ContextRequest.model_validate(
        {"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # When: current clients serialize them.
    message_payload = message.model_dump(mode="json")
    context_payload = context.model_dump(mode="json")

    # Then: existing wire shapes remain unchanged.
    assert message_payload == _message_payload()
    assert context_payload == {
        "scope": _scope_payload(),
        "query": "what matters?",
        "limit": 4,
    }


def test_privacy_defaults_match_discord_scope_policy() -> None:
    # Given: representative DM, channel, topology, and skill records.
    dm_event = ClientEvent.model_validate(
        _client_event_payload(scope=_scope_payload(guild_id=None, channel_id=None)),
    )
    channel_event = ClientEvent.model_validate(_client_event_payload())
    topology_event = ClientEvent.model_validate(
        _client_event_payload(
            event_type="channel_updated",
            subject={"id": "456", "type": "channel"},
        ),
    )
    skill = SkillRecord(
        skill_id="skill-1",
        tenant_id="bromigos",
        agent_id="pc-principal",
        name="Summarize channel",
        description="Summarizes visible Discord channel context for review.",
        status=SkillStatus.APPROVED,
    )

    # When / Then: default helpers keep privacy scoped by surface.
    assert default_event_visibility(dm_event) == MemoryVisibility.PRIVATE_USER
    assert default_event_visibility(channel_event) == MemoryVisibility.CHANNEL
    assert default_event_visibility(topology_event) == MemoryVisibility.GUILD
    assert skill.scope == MemoryVisibility.AGENT_SHARED
    assert default_skill_visibility() == MemoryVisibility.AGENT_SHARED
    assert default_skill_visibility(MemoryVisibility.GLOBAL) == MemoryVisibility.GLOBAL


@dataclass(slots=True)
class RecordingBackend:
    context: str = ""
    messages: list[MessageWriteRequest] = field(default_factory=list)
    context_requests: list[ContextRequest] = field(default_factory=list)

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        self.messages.append(request)
        return MessageWriteResponse(accepted=True)

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        self.context_requests.append(request)
        return ContextResponse(context=self.context)


@dataclass(frozen=True, slots=True)
class ShortTermWrite:
    user_identifier: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class LongTermFactWrite:
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class ShortTermContextQuery:
    metadata_filters: dict[str, str]


@dataclass(slots=True)
class RecordingShortTermMemory:
    context: str = ""
    messages: list[ShortTermWrite] = field(default_factory=list)
    context_queries: list[ShortTermContextQuery] = field(default_factory=list)

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
    ) -> None:
        _ = (session_id, role, content, extract_entities, extract_relations)
        self.messages.append(
            ShortTermWrite(user_identifier=user_identifier, metadata=metadata),
        )

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str:
        _ = (query, session_id, max_messages)
        self.context_queries.append(
            ShortTermContextQuery(metadata_filters=metadata_filters),
        )
        return self.context


@dataclass(slots=True)
class RecordingLongTermMemory:
    facts: list[LongTermFactWrite] = field(default_factory=list)
    context_queries: list[str] = field(default_factory=list)

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        metadata: dict[str, str],
    ) -> None:
        _ = (subject, predicate, obj)
        self.facts.append(LongTermFactWrite(metadata=metadata))

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = max_items
        self.context_queries.append(query)
        return "unscoped long-term context"


@dataclass(slots=True)
class RecordingMemoryClient:
    context: str = ""
    short_term: RecordingShortTermMemory = field(init=False)
    long_term: RecordingLongTermMemory = field(default_factory=RecordingLongTermMemory)

    def __post_init__(self) -> None:
        self.short_term = RecordingShortTermMemory(context=self.context)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


@dataclass(frozen=True, slots=True)
class MemoryClientFactory:
    client: RecordingMemoryClient

    def __call__(self, settings: MemorySettings) -> RecordingMemoryClient:
        _ = settings
        return self.client


def _settings() -> Settings:
    return Settings()


def _auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _scope_payload(
    *,
    tenant_id: str = "bromigos",
    guild_id: str | None = "123",
    channel_id: str | None = "456",
) -> dict[str, str]:
    payload = {
        "tenant_id": tenant_id,
        "space_id": "discord",
        "agent_id": "pc-principal",
        "session_id": "guild:123:channel:456",
        "user_id": "789",
        "visibility": "channel",
    }
    if guild_id is not None:
        payload["guild_id"] = guild_id
    if channel_id is not None:
        payload["channel_id"] = channel_id
    return payload


def _message_payload(
    *,
    scope: dict[str, str] | None = None,
) -> dict[str, str | dict[str, str]]:
    return {
        "scope": scope or _scope_payload(),
        "role": "user",
        "content": "remember this",
    }


def _client_event_payload(
    *,
    scope: dict[str, str] | None = None,
    event_type: str = "message_created",
    subject: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "tenant_id": "bromigos",
        "source_client": "discord",
        "agent_id": "pc-principal",
        "event_id": "discord-message-999",
        "event_type": event_type,
        "occurred_at": "2026-06-27T01:02:03Z",
        "observed_at": "2026-06-27T01:02:04Z",
        "idempotency_key": "discord:message:message-999:create",
        "scope": scope or _scope_payload(),
        "actor": {
            "id": "789",
            "display_name": "cartman",
            "is_bot": False,
        },
        "subject": subject or {"id": "message-999", "type": "message"},
        "payload": {
            "content": "remember this",
            "payload_version": 1,
        },
        "discord": {
            "guild_id": "123",
            "channel_id": "456",
            "message_id": "message-999",
        },
    }
