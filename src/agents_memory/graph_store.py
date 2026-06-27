from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from agents_memory.graph_cypher import (
    CONTEXT_CYPHER,
    UPSERT_EVENT_CYPHER,
    CypherParameters,
    context_parameters,
    is_duplicate_result,
    upsert_parameters,
)
from agents_memory.graph_events import (
    GraphNode,
    PlannedGraphEvent,
    context_allows_node,
    fact_from_node,
    node_from_row,
    plan_event,
)
from agents_memory.models import (
    ClientEvent,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    JsonValue,
)


class CypherDriver(Protocol):
    async def execute_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> Sequence[dict[str, JsonValue]]: ...
    async def verify_connectivity(self) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


class CypherDriverFactory(Protocol):
    def __call__(self) -> CypherDriver: ...


class GraphExecutor(Protocol):
    async def require_available(self) -> None: ...
    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult: ...
    async def get_context(
        self,
        request: GraphContextRequest,
    ) -> Sequence[GraphNode]: ...


@dataclass(frozen=True, slots=True)
class Neo4jGraphExecutor:
    driver_factory: CypherDriverFactory

    async def require_available(self) -> None:
        async with self.driver_factory() as driver:
            await driver.verify_connectivity()

    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult:
        async with self.driver_factory() as driver:
            rows = await driver.execute_query(
                UPSERT_EVENT_CYPHER,
                upsert_parameters(event),
            )
        if is_duplicate_result(rows):
            return EventIngestResult(
                event_id=event.event.event_id,
                status=EventIngestStatus.DUPLICATE,
                reason="idempotency key already ingested",
            )
        return EventIngestResult(
            event_id=event.event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> Sequence[GraphNode]:
        async with self.driver_factory() as driver:
            rows = await driver.execute_query(
                CONTEXT_CYPHER,
                context_parameters(request),
            )
        return tuple(node_from_row(row, request.scope) for row in rows)


@dataclass(frozen=True, slots=True)
class DirectNeo4jGraphStore:
    executor: GraphExecutor

    async def require_available(self) -> None:
        await self.executor.require_available()

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        await self.require_available()
        return await self.executor.upsert_event(plan_event(event))

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        await self.require_available()
        nodes = await self.executor.get_context(request)
        return GraphContextResponse(
            context="\n".join(node.summary for node in nodes),
            facts=[fact_from_node(node) for node in nodes],
        )

    async def event_count(self) -> int:
        if isinstance(self.executor, InMemoryGraphExecutor):
            return self.executor.event_count
        return 0


class InMemoryGraphExecutor:
    def __init__(self) -> None:
        self._idempotency_keys: set[str] = set()
        self._events: list[ClientEvent] = []
        self._nodes: dict[str, GraphNode] = {}

    @property
    def event_count(self) -> int:
        return len(self._events)

    async def require_available(self) -> None:
        return None

    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult:
        if event.event.idempotency_key in self._idempotency_keys:
            return EventIngestResult(
                event_id=event.event.event_id,
                status=EventIngestStatus.DUPLICATE,
                reason="idempotency key already ingested",
            )
        self._idempotency_keys.add(event.event.idempotency_key)
        self._events.append(event.event)
        self._nodes[event.node.id] = event.node
        return EventIngestResult(
            event_id=event.event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> Sequence[GraphNode]:
        scoped_nodes = [
            node for node in self._nodes.values() if context_allows_node(request, node)
        ]
        return scoped_nodes[: request.limit]
