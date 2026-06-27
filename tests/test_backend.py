from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import pytest

from agents_memory.graph_probe import DirectNeo4jProbe, GraphPersistenceUnavailableError
from agents_memory.models import (
    ClientEvent,
    ClientEventActor,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ClientEventSubject,
    ClientEventType,
    ContextRequest,
    ContextResponse,
    DiscordEventContext,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    MemoryScope,
    MemoryVisibility,
    MessageWriteRequest,
    MessageWriteResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
    SourceClient,
)

if TYPE_CHECKING:
    from agents_memory.backend import MemoryBackend


@pytest.mark.anyio
async def test_backend_protocol_ingests_event_batch() -> None:
    # Given: a fake backend implementing the full persistence protocol.
    backend: MemoryBackend = RecordingBackend()
    event = _client_event()

    # When: a typed batch is ingested through the protocol seam.
    response = await backend.ingest_events(ClientEventBatchRequest(events=[event]))

    # Then: fake implementations can satisfy the seam without Neo4j.
    assert response == ClientEventBatchResponse(
        results=[
            EventIngestResult(
                event_id="discord-message-999",
                status=EventIngestStatus.ACCEPTED,
            ),
        ],
    )


@pytest.mark.anyio
async def test_direct_neo4j_probe_failure_degrades_to_clear_error() -> None:
    # Given: direct graph persistence is unavailable at startup.
    probe = DirectNeo4jProbe(driver_factory=FailingDriverFactory())

    # When / Then: the seam fails explicitly instead of silently no-oping.
    with pytest.raises(GraphPersistenceUnavailableError) as error:
        await probe.require_available()

    assert "Neo4j structured graph persistence is unavailable" in str(error.value)


@dataclass(slots=True)
class RecordingBackend:
    events: list[ClientEvent] = field(default_factory=list)

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        _ = request
        return MessageWriteResponse(accepted=True)

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        _ = request
        return ContextResponse(context="")

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        self.events.append(event)
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

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
        _ = request
        return GraphContextResponse(context="fake graph context")

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        _ = request
        return SkillListResponse()

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        return proposal

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        return EventIngestResult(
            event_id=usage.skill_id,
            status=EventIngestStatus.ACCEPTED,
        )


@dataclass(frozen=True, slots=True)
class FailingDriverFactory:
    def __call__(self) -> "FailingDriver":
        return FailingDriver()


@dataclass(frozen=True, slots=True)
class FailingDriver:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)

    async def verify_connectivity(self) -> None:
        reason = "connection refused"
        raise OSError(reason)


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="bromigos",
        space_id="discord",
        agent_id="pc-principal",
        session_id="guild:123:channel:456",
        user_id="789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="123",
        channel_id="456",
    )


def _client_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="discord-message-999",
        event_type=ClientEventType.MESSAGE_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="discord:message:message-999:create",
        scope=_scope(),
        actor=ClientEventActor(id="789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(id="message-999", type="message"),
        payload={"content": "remember this", "payload_version": 1},
        discord=DiscordEventContext(
            guild_id="123",
            channel_id="456",
            message_id="message-999",
        ),
    )
