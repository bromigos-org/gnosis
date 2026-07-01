import logging
import time
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, Field

from gnosis.graph_types import CypherParameters
from gnosis.models import GraphContextRequest

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_GRAPH_SCHEMA_GUIDE: Final[str] = """
Use this read-only Neo4j schema for gnosis graph QA.
Labels: Tenant, Agent, Client, Guild, Channel, Category, User, Bot, Role,
Message, Link, Attachment, Event, GraphNode.
Relationships: OWNS_AGENT, OWNS_CLIENT, USES_CLIENT, OWNS_GUILD, IN_GUILD,
IN_CATEGORY, OWNS_ROLE, HAS_ROLE, AUTHORED, IN_CHANNEL, LINKED_FROM,
ATTACHED_TO, AFFECTS.
Every query must scope by tenant_id = $tenant_id. If a guild question has
$guild_id, also scope by guild_id = $guild_id or graph IN_GUILD membership.
If a channel question has $channel_id, scope by channel_id = $channel_id.
Return rows with id, type, summary, deleted. Use LIMIT $limit.
Never use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, LOAD, DROP, or procedures.
""".strip()


class GraphQueryPlan(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    cypher: str = Field(min_length=1)
    parameters: CypherParameters = Field(default_factory=dict)
    answer_kind: str = Field(min_length=1, max_length=64)


@dataclass(frozen=True, slots=True)
class ValidatedGraphQuery:
    cypher: str
    parameters: CypherParameters
    answer_kind: str


class GraphQueryPlanner(Protocol):
    async def plan_query(
        self,
        request: GraphContextRequest,
    ) -> GraphQueryPlan | None: ...


@dataclass(frozen=True, slots=True)
class LiteLLMGraphQueryPlanner:
    model: str
    base_url: str
    api_key: str

    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        start = time.perf_counter()
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.beta.chat.completions.parse(
                messages=_messages(request),
                model=self.model,
                temperature=0,
                max_tokens=700,
                response_format=GraphQueryPlan,
            )
        plan = response.choices[0].message.parsed
        if plan is None:
            _LOGGER.info(
                "graph QA planner returned no content",
                extra={"model": self.model},
            )
            return None
        _LOGGER.info(
            "graph QA planner produced plan",
            extra={
                "answer_kind": plan.answer_kind,
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
            },
        )
        return plan


def graph_schema_guide() -> str:
    return _GRAPH_SCHEMA_GUIDE


def _messages(request: GraphContextRequest) -> tuple[ChatCompletionMessageParam, ...]:
    scope = request.scope
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _GRAPH_SCHEMA_GUIDE,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": (
            f"Question: {request.query}\n"
            f"tenant_id: {scope.tenant_id}\n"
            f"agent_id: {scope.agent_id}\n"
            f"guild_id: {scope.guild_id or ''}\n"
            f"channel_id: {scope.channel_id or ''}\n"
            f"user_id: {scope.user_id}\n"
            f"limit: {request.limit}"
        ),
    }
    return (system_message, user_message)
