from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from neo4j.exceptions import Neo4jError

from gnosis.graph_cypher import (
    CONTEXT_CYPHER,
    SEMANTIC_CONTEXT_CYPHER,
    UPSERT_EVENT_CYPHER,
    CypherParameters,
    context_parameters,
    is_duplicate_result,
    upsert_parameters,
)
from gnosis.graph_events import (
    GraphNode,
    PlannedGraphEvent,
    context_allows_node,
    fact_from_node,
    node_from_row,
    plan_event,
)
from gnosis.graph_schema import GRAPH_SCHEMA_CYPHER, graph_vector_schema_cypher
from gnosis.graph_types import vector_parameter
from gnosis.models import (
    BackendReadiness,
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


class TextEmbeddingProvider(Protocol):
    async def embed_one(self, text: str) -> list[float]: ...


class GraphExecutor(Protocol):
    async def require_available(self) -> None: ...
    async def readiness(self) -> BackendReadiness: ...
    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult: ...
    async def get_context(
        self,
        request: GraphContextRequest,
    ) -> Sequence[GraphNode]: ...


@dataclass(slots=True)
class Neo4jGraphExecutor:
    driver_factory: CypherDriverFactory
    embedding_dimensions: int
    embedding_provider: TextEmbeddingProvider | None = None
    _schema_bootstrapped: bool = False

    async def require_available(self) -> None:
        await self._bootstrap_schema()
        async with self.driver_factory() as driver:
            await driver.verify_connectivity()

    async def readiness(self) -> BackendReadiness:
        try:
            await self.require_available()
        except (Neo4jError, OSError):
            return BackendReadiness(graph="unavailable", schema="unavailable")
        return BackendReadiness(graph="ready", schema="ready")

    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult:
        await self._bootstrap_schema()
        parameters = upsert_parameters(event)
        if self.embedding_provider is not None:
            parameters["node_embedding"] = vector_parameter(
                await self.embedding_provider.embed_one(event.node.summary),
            )
        async with self.driver_factory() as driver:
            rows = await driver.execute_query(
                UPSERT_EVENT_CYPHER,
                parameters,
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
        await self._bootstrap_schema()
        query = CONTEXT_CYPHER
        parameters = context_parameters(request)
        if self.embedding_provider is not None:
            query = SEMANTIC_CONTEXT_CYPHER
            parameters = context_parameters(
                request,
                await self.embedding_provider.embed_one(request.query),
            )
        async with self.driver_factory() as driver:
            rows = await driver.execute_query(
                query,
                parameters,
            )
        return tuple(node_from_row(row, request.scope) for row in rows)

    async def _bootstrap_schema(self) -> None:
        if self._schema_bootstrapped:
            return
        async with self.driver_factory() as driver:
            for statement in GRAPH_SCHEMA_CYPHER:
                _ = await driver.execute_query(statement, {})
            _ = await driver.execute_query(
                graph_vector_schema_cypher(self.embedding_dimensions),
                {},
            )
        self._schema_bootstrapped = True


@dataclass(frozen=True, slots=True)
class DirectNeo4jGraphStore:
    executor: GraphExecutor

    async def require_available(self) -> None:
        await self.executor.require_available()

    async def readiness(self) -> BackendReadiness:
        return await self.executor.readiness()

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
        self._semantic_node_ids: set[str] = set()
        self._schema_bootstrapped: bool = False
        self._schema_bootstrap_count: int = 0

    @property
    def event_count(self) -> int:
        return len(self._events)

    async def require_available(self) -> None:
        await self._bootstrap_schema()

    async def readiness(self) -> BackendReadiness:
        await self.require_available()
        return BackendReadiness(graph="ready", schema="ready")

    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult:
        await self._bootstrap_schema()
        if event.event.idempotency_key in self._idempotency_keys:
            self._nodes[event.node.id] = event.node
            self._semantic_node_ids.update(event.semantic_node_ids)
            return EventIngestResult(
                event_id=event.event.event_id,
                status=EventIngestStatus.DUPLICATE,
                reason="idempotency key already ingested",
            )
        self._idempotency_keys.add(event.event.idempotency_key)
        self._events.append(event.event)
        self._nodes[event.node.id] = event.node
        self._semantic_node_ids.update(event.semantic_node_ids)
        return EventIngestResult(
            event_id=event.event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> Sequence[GraphNode]:
        await self._bootstrap_schema()
        scoped_nodes = [
            node for node in self._nodes.values() if context_allows_node(request, node)
        ]
        return scoped_nodes[: request.limit]

    def clear_current_nodes_for_test(self) -> None:
        self._nodes.clear()
        self._semantic_node_ids.clear()

    def semantic_node_ids_for_test(self) -> set[str]:
        return set(self._semantic_node_ids)

    def schema_bootstrap_count_for_test(self) -> int:
        return self._schema_bootstrap_count

    async def _bootstrap_schema(self) -> None:
        if self._schema_bootstrapped:
            return
        self._schema_bootstrapped = True
        self._schema_bootstrap_count += 1
