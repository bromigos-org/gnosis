"""Process-scoped Neo4j drivers, schema setup and LiteLLM clients.

Every request used to open its own SDK ``MemoryClient`` (a new Neo4j driver
plus the SDK's ~34 schema queries), its own structured-graph driver, and its
own ``AsyncOpenAI`` HTTP client. These tests pin the replacement: one of each
per process, opened once (at startup or first use), reused by every request,
and closed once at shutdown.
"""

import asyncio
from dataclasses import dataclass, field
from os import environ
from typing import Self, cast

import pytest
from fastapi.testclient import TestClient
from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.core.exceptions import ConnectionError as SdkConnectionError

_ = environ.setdefault("GNOSIS_TOKEN", "test-token")
_ = environ.setdefault("GNOSIS_READ_OPERATOR_TOKEN", "read-operator-token")
_ = environ.setdefault("GNOSIS_EXPORT_OPERATOR_TOKEN", "export-operator-token")
_ = environ.setdefault("GNOSIS_WRITE_OPERATOR_TOKEN", "write-operator-token")
_ = environ.setdefault("GNOSIS_ADMIN_OPERATOR_TOKEN", "admin-operator-token")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "inert-password")
_ = environ.setdefault("LITELLM_BASE_URL", "http://litellm.local/v1")
_ = environ.setdefault("LITELLM_API_KEY", "inert-litellm-key")

from gnosis import backend as backend_module  # noqa: E402
from gnosis import graph_probe  # noqa: E402
from gnosis.backend import Neo4jAgentMemoryBackend  # noqa: E402
from gnosis.graph_probe import CypherParameters, CypherRow  # noqa: E402
from gnosis.graph_schema import GRAPH_SCHEMA_CYPHER  # noqa: E402
from gnosis.graph_store import DirectNeo4jGraphStore, Neo4jGraphExecutor  # noqa: E402
from gnosis.llm_clients import (  # noqa: E402
    close_shared_openai_clients,
    shared_openai_client,
)
from gnosis.main import create_app  # noqa: E402
from gnosis.models import (  # noqa: E402
    BackendReadiness,
    ClientEvent,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
)
from gnosis.sdk_client import MemoryClientContext, build_memory_settings  # noqa: E402
from gnosis.settings import Settings  # noqa: E402


@dataclass(slots=True)
class OpenCounter:
    """Counts what a real ``MemoryClient`` does once per open/close."""

    opens: int = 0
    closes: int = 0
    fail_next_opens: int = 0


@dataclass(slots=True)
class CountingMemoryClient:
    """Stands in for the SDK client: entering it is the driver + schema setup."""

    counter: OpenCounter
    write_errors: list[object] = field(default_factory=list)
    flush_calls: int = 0

    async def __aenter__(self) -> Self:
        # Yield to the loop so concurrent first borrowers really race the open.
        await asyncio.sleep(0)
        if self.counter.fail_next_opens > 0:
            self.counter.fail_next_opens -= 1
            msg = "Neo4j service unavailable"
            raise SdkConnectionError(msg)
        self.counter.opens += 1
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)
        self.counter.closes += 1

    async def flush(self) -> None:
        self.flush_calls += 1


@dataclass(slots=True)
class CountingClientFactory:
    counter: OpenCounter = field(default_factory=OpenCounter)
    calls: int = 0

    def __call__(self, settings: MemorySettings) -> MemoryClientContext:
        _ = settings
        self.calls += 1
        client = cast("object", CountingMemoryClient(self.counter))
        return cast("MemoryClientContext", client)


@dataclass(slots=True)
class ReadyGraphStore:
    available_calls: int = 0
    closes: int = 0

    async def require_available(self) -> None:
        self.available_calls += 1

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        _ = request
        return GraphContextResponse(context="")

    async def close(self) -> None:
        self.closes += 1


def _backend(
    factory: CountingClientFactory,
    graph_store: ReadyGraphStore | None = None,
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=factory,
        graph_store=graph_store or ReadyGraphStore(),
    )


@pytest.mark.anyio
async def test_concurrent_requests_share_one_sdk_client() -> None:
    # Given: a backend whose SDK client open (driver + schema setup) is counted.
    factory = CountingClientFactory()
    backend = _backend(factory)

    # When: many requests borrow the client at once, then again sequentially.
    _ = await asyncio.gather(*(backend.buffer_status() for _ in range(16)))
    for _ in range(4):
        _ = await backend.readiness()

    # Then: one client was built and opened, and no request closed it.
    assert factory.calls == 1
    assert factory.counter.opens == 1
    assert factory.counter.closes == 0


@pytest.mark.anyio
async def test_shutdown_closes_the_shared_client_and_graph_driver_once() -> None:
    # Given: a backend that has served requests.
    factory = CountingClientFactory()
    graph_store = ReadyGraphStore()
    backend = _backend(factory, graph_store)
    _ = await backend.buffer_status()

    # When: the process shuts down.
    await backend.shutdown()

    # Then: the shared SDK client and the graph store's driver close once.
    assert factory.counter.closes == 1
    assert graph_store.closes == 1


@pytest.mark.anyio
async def test_default_sdk_path_builds_one_memory_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the production path (no injected factory), with MemoryClient counted.
    counter = OpenCounter()
    built: list[MemorySettings] = []

    def memory_client(settings: MemorySettings) -> CountingMemoryClient:
        built.append(settings)
        return CountingMemoryClient(counter)

    monkeypatch.setattr(backend_module, "MemoryClient", memory_client)
    backend = Neo4jAgentMemoryBackend(Settings(), graph_store=ReadyGraphStore())

    # When: requests arrive concurrently.
    _ = await asyncio.gather(*(backend.buffer_status() for _ in range(8)))

    # Then: one SDK client (one driver, one schema setup) serves all of them.
    assert len(built) == 1
    assert counter.opens == 1


@pytest.mark.anyio
async def test_failed_open_is_retried_by_the_next_request() -> None:
    # Given: Neo4j is unreachable for the first open only.
    factory = CountingClientFactory()
    factory.counter.fail_next_opens = 1
    backend = _backend(factory)

    # When: the first request fails and a later one arrives.
    with pytest.raises(SdkConnectionError):
        _ = await backend.buffer_status()
    status = await backend.buffer_status()

    # Then: the failure was not cached; the retry opened the client once.
    assert status.status == "ready"
    assert factory.calls == 2
    assert factory.counter.opens == 1


@pytest.mark.anyio
async def test_startup_runs_schema_setup_once_before_requests() -> None:
    # Given: a fresh backend.
    factory = CountingClientFactory()
    graph_store = ReadyGraphStore()
    backend = _backend(factory, graph_store)

    # When: the app starts, then serves requests.
    await backend.startup()
    _ = await asyncio.gather(*(backend.buffer_status() for _ in range(8)))

    # Then: the client opened (schema setup ran) at startup only.
    assert factory.counter.opens == 1
    assert graph_store.available_calls == 1


@pytest.mark.anyio
async def test_startup_tolerates_unreachable_neo4j() -> None:
    # Given: Neo4j is down while the app starts.
    factory = CountingClientFactory()
    factory.counter.fail_next_opens = 1
    backend = _backend(factory)

    # When: the app starts, and later a request arrives.
    await backend.startup()
    status = await backend.buffer_status()

    # Then: startup did not crash, and the first request connected.
    assert status.status == "ready"
    assert factory.counter.opens == 1


def test_app_lifespan_opens_once_and_closes_on_shutdown() -> None:
    # Given: the app with a counted backend.
    factory = CountingClientFactory()
    graph_store = ReadyGraphStore()
    app = create_app(settings_factory=Settings, backend=_backend(factory, graph_store))

    # When: the app starts, serves readiness probes, and stops.
    with TestClient(app) as client:
        assert factory.counter.opens == 1
        for _ in range(5):
            assert client.get("/ready").status_code == 200
        assert factory.counter.closes == 0

    # Then: one open at startup, one close at shutdown.
    assert factory.counter.opens == 1
    assert factory.counter.closes == 1
    assert graph_store.closes == 1


@dataclass(slots=True)
class FakeRawDriver:
    queries: list[str] = field(default_factory=list)
    verifications: int = 0
    closes: int = 0

    async def verify_connectivity(self) -> None:
        self.verifications += 1

    async def execute_query(
        self,
        query_: str,
        parameters_: CypherParameters,
    ) -> tuple[list[CypherRow], object, object]:
        _ = parameters_
        self.queries.append(query_)
        return [], None, None

    async def close(self) -> None:
        self.closes += 1


@dataclass(slots=True)
class FakeGraphDatabase:
    drivers: list[FakeRawDriver] = field(default_factory=list)
    pool_sizes: list[int] = field(default_factory=list)

    def driver(
        self,
        uri: str,
        *,
        auth: tuple[str, str],
        max_connection_pool_size: int,
    ) -> FakeRawDriver:
        _ = (uri, auth)
        self.pool_sizes.append(max_connection_pool_size)
        driver = FakeRawDriver()
        self.drivers.append(driver)
        return driver


@pytest.mark.anyio
async def test_graph_store_reuses_one_driver_and_bootstraps_schema_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the default structured-graph driver factory over a counted driver.
    database = FakeGraphDatabase()
    monkeypatch.setattr(graph_probe, "AsyncGraphDatabase", database)
    settings = Settings(neo4j_max_connection_pool_size=12)
    store = DirectNeo4jGraphStore(
        executor=Neo4jGraphExecutor(
            driver_factory=graph_probe.direct_neo4j_driver_factory(settings),
            embedding_dimensions=8,
        ),
    )

    # When: readiness probes and graph requests run repeatedly.
    for _ in range(5):
        _ = await store.readiness()

    # Then: one driver with the configured pool serves them all; the schema
    # bootstrap ran once; no request closed the driver.
    assert len(database.drivers) == 1
    assert database.pool_sizes == [12]
    driver = database.drivers[0]
    assert len(driver.queries) == len(GRAPH_SCHEMA_CYPHER) + 1
    assert driver.verifications == 5
    assert driver.closes == 0

    # When: the process shuts down.
    await store.close()

    # Then: the driver closes exactly once.
    assert driver.closes == 1


def test_sdk_driver_pool_size_comes_from_settings() -> None:
    settings = Settings(neo4j_max_connection_pool_size=24)

    memory_settings = build_memory_settings(settings)

    assert memory_settings.neo4j.max_connection_pool_size == 24


def test_sdk_driver_pool_size_default_matches_the_sdk_default() -> None:
    memory_settings = build_memory_settings(Settings())

    assert memory_settings.neo4j.max_connection_pool_size == 50


@pytest.mark.anyio
async def test_litellm_client_is_reused_across_calls_and_closed_at_shutdown() -> None:
    # Given: two collaborators pointed at the same LiteLLM endpoint.
    first = shared_openai_client(api_key="k", base_url="http://litellm.local/v1")
    second = shared_openai_client(api_key="k", base_url="http://litellm.local/v1")
    other = shared_openai_client(api_key="k2", base_url="http://litellm.local/v1")

    # Then: they share one HTTP client per endpoint and key.
    assert first is second
    assert other is not first

    # When: the process shuts down.
    await close_shared_openai_clients()

    # Then: the clients are closed and a later call builds a fresh one.
    assert first.is_closed()
    assert other.is_closed()
    fresh = shared_openai_client(api_key="k", base_url="http://litellm.local/v1")
    assert fresh is not first
    await close_shared_openai_clients()
