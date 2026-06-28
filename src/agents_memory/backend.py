from typing import Protocol, Self

from neo4j_agent_memory import MemoryClient, MemoryConfig, MemorySettings, Neo4jConfig
from neo4j_agent_memory.llm.adapters.litellm import (
    LiteLLMEmbeddingProvider,
    LiteLLMProvider,
)
from pydantic import SecretStr

from agents_memory.graph_probe import StructuredGraphStore, direct_neo4j_driver_factory
from agents_memory.graph_store import DirectNeo4jGraphStore, Neo4jGraphExecutor
from agents_memory.models import (
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ContextRequest,
    ContextResponse,
    EventIngestResult,
    GraphContextRequest,
    GraphContextResponse,
    MemoryScope,
    MessageWriteRequest,
    MessageWriteResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
)
from agents_memory.settings import Settings
from agents_memory.skill_registry import InMemorySkillRegistry, SkillRegistry


class MemoryBackend(Protocol):
    async def add_message(
        self,
        request: MessageWriteRequest,
    ) -> MessageWriteResponse: ...
    async def get_context(self, request: ContextRequest) -> ContextResponse: ...
    async def ingest_event(self, event: ClientEvent) -> EventIngestResult: ...
    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse: ...
    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse: ...
    async def list_skills(self, request: SkillListRequest) -> SkillListResponse: ...
    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal: ...
    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult: ...


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


class LongTermMemory(Protocol):
    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        metadata: dict[str, str],
    ) -> object: ...


class MemoryClientContext(Protocol):
    @property
    def short_term(self) -> ShortTermMemory: ...
    @property
    def long_term(self) -> LongTermMemory: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


class Neo4jAgentMemoryBackend:
    def __init__(
        self,
        settings: Settings,
        memory_client_factory: MemoryClientFactory | None = None,
        graph_store: StructuredGraphStore | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self._settings: MemorySettings = _build_memory_settings(settings)
        self._memory_client_factory: MemoryClientFactory | None = memory_client_factory
        self._graph_store: StructuredGraphStore = graph_store or DirectNeo4jGraphStore(
            executor=Neo4jGraphExecutor(
                driver_factory=direct_neo4j_driver_factory(settings),
            ),
        )
        self._skill_registry: SkillRegistry = skill_registry or InMemorySkillRegistry()

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        metadata = _scope_metadata(request.scope)
        async with self._memory_client() as client:
            _ = await client.short_term.add_message(
                session_id=_session_id(request.scope),
                role=request.role.value,
                content=request.content,
                user_identifier=_user_identifier(request.scope),
                metadata=metadata,
                extract_entities=False,
                extract_relations=False,
            )
            _ = await client.long_term.add_fact(
                subject=_user_identifier(request.scope),
                predicate=f"said_{request.role.value}",
                obj=request.content,
                metadata=metadata,
            )
        return MessageWriteResponse(accepted=True)

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        async with self._memory_client() as client:
            context = await client.short_term.get_context(
                request.query,
                session_id=_session_id(request.scope),
                max_messages=request.limit,
                metadata_filters=_scope_metadata(request.scope),
            )
        return ContextResponse(context=context)

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return await self._graph_store.ingest_event(event)

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

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        await self._graph_store.require_available()
        return await self._skill_registry.list_skills(request)

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        await self._graph_store.require_available()
        return await self._skill_registry.propose_skill(proposal)

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        await self._graph_store.require_available()
        return await self._skill_registry.record_skill_usage(usage)

    def _memory_client(self) -> MemoryClientContext:
        if self._memory_client_factory is not None:
            return self._memory_client_factory(self._settings)
        return MemoryClient(self._settings)


def _build_memory_settings(settings: Settings) -> MemorySettings:
    return MemorySettings(
        backend="bolt",
        neo4j=Neo4jConfig(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=SecretStr(settings.neo4j_password),
        ),
        llm=LiteLLMProvider(
            settings.memory_llm,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        embedding=LiteLLMEmbeddingProvider(
            litellm_embedding_model(settings.memory_embedding),
            dimensions=settings.memory_embedding_dimensions,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        memory=MemoryConfig(multi_tenant=True),
    )


def litellm_embedding_model(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


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
