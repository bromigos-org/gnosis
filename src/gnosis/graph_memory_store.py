from collections.abc import Sequence

from gnosis.graph_activity import top_active_channel_nodes
from gnosis.graph_cypher import is_top_active_channels_request
from gnosis.graph_events import (
    GraphNode,
    PlannedGraphEvent,
    context_allows_node,
)
from gnosis.models import (
    BackendReadiness,
    ClientEvent,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
)


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
        if is_top_active_channels_request(request):
            return top_active_channel_nodes(request, self._events)
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
