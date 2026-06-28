from dataclasses import dataclass, field
from typing import Protocol, Self

from agents_memory.graph_events import PlannedGraphEvent, plan_event
from agents_memory.models import ClientEvent, EventIngestResult, EventIngestStatus


class LongTermFactWriter(Protocol):
    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        metadata: dict[str, str],
        generate_embedding: bool,
    ) -> object: ...


class MemoryClientContext(Protocol):
    @property
    def long_term(self) -> LongTermFactWriter: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


def event_metadata(event: ClientEvent) -> dict[str, str]:
    metadata = {
        "event_id": event.event_id,
        "idempotency_key": event.idempotency_key,
        "event_type": event.event_type.value,
        "tenant_id": event.tenant_id,
        "agent_id": event.agent_id,
        "session_id": event.scope.session_id,
        "user_id": event.scope.user_id,
        "visibility": event.scope.visibility.value,
    }
    if event.scope.guild_id is not None:
        metadata["guild_id"] = event.scope.guild_id
    if event.scope.channel_id is not None:
        metadata["channel_id"] = event.scope.channel_id
    return metadata


async def promote_event_fact(
    event: ClientEvent,
    planned_event: PlannedGraphEvent,
    memory_client: MemoryClientContext,
) -> None:
    async with memory_client as client:
        _ = await client.long_term.add_fact(
            subject=planned_event.node.id,
            predicate=f"{event.source_client.value}.{event.event_type.value}",
            obj=planned_event.node.summary,
            metadata=event_metadata(event),
            generate_embedding=True,
        )


@dataclass(slots=True)
class EventFactPromoter:
    _promoted_idempotency_keys: set[str] = field(default_factory=set)

    async def promote_for_result(
        self,
        event: ClientEvent,
        result: EventIngestResult,
        memory_client: MemoryClientContext,
    ) -> None:
        promotable_statuses = {
            EventIngestStatus.ACCEPTED,
            EventIngestStatus.DUPLICATE,
        }
        if result.status not in promotable_statuses:
            return
        if event.idempotency_key in self._promoted_idempotency_keys:
            return
        await promote_event_fact(event, plan_event(event), memory_client)
        self._promoted_idempotency_keys.add(event.idempotency_key)
